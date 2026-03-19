"""
Caser: Convolutional Sequence Embedding Recommendation Model
Implemented in JAX + Equinox

─────────────────────────────────────────────────────────────────────────────
Historical Context
─────────────────────────────────────────────────────────────────────────────
Before Caser (Tang & Wang, 2018), sequential recommendation was dominated by
RNN-based models (e.g. GRU4Rec, 2016). RNNs capture "what comes next given the
entire past", but suffer from:
  - Vanishing gradients over long sequences
  - Sequential computation (hard to parallelise)
  - No explicit notion of "order" within a local window

Caser's key insight: treat the user's last L interactions as a 2D "image"
(rows = time steps, cols = embedding dimensions) and apply CNNs.

  - Horizontal filters (h×D): n-gram detectors — capture LOCAL ORDER patterns
    e.g. "user bought A then B → likely wants C"
  - Vertical   filters (L×1): column-wise patterns — capture LATENT FEATURE
    co-occurrence across the full window regardless of order

This decomposition gives Caser two complementary inductive biases:
  1. Order-sensitive (horizontal) — sequential intent
  2. Order-insensitive (vertical) — set-like taste

Long-term taste is modelled separately via a user embedding P_u, which is
concatenated with the short-term CNN output before scoring.

Paper: Tang & Wang 2018 — "Personalized Top-N Sequential Recommendation via
        Convolutional Sequence Embedding"

─────────────────────────────────────────────────────────────────────────────
Noam Shape Notation (shapes written as variable suffixes, dims separated by x)
─────────────────────────────────────────────────────────────────────────────
  B   = batch size
  L   = lookback window length (sequence)
  D   = embedding / factor dim
  V   = item vocab size
  Vu  = user vocab size
  Fv  = num vertical filters   (paper's d_v)
  Fh  = num horizontal filters (paper's d_h, same for every height h)
  Lh  = L - h + 1  (horizontal conv output length for filter height h)
  F   = flat feature dim (used after concatenation/pooling)
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import einops
from typing import List
from jaxtyping import Float, Array, Int


# ─────────────────────────────────────────────────────────────────────────────
# HorizontalConvBlock
#
# One convolutional layer for a single filter height h.
# Conceptually this is an n-gram detector of length h operating on the
# item-embedding "image":
#   - A (h × D) filter slides down the L-row image
#   - This detects sequential patterns spanning h consecutive items
#   - Each position produces one scalar per filter → output shape (Fh, Lh)
#   - Global max-pool over Lh collapses WHERE the pattern fired → (Fh,)
#     (translation invariance: the pattern can appear anywhere in the window)
#
# With multiple heights h=1..L we get detectors at every granularity,
# analogous to unigrams, bigrams, trigrams in NLP text-CNNs (Kim 2014).
# ─────────────────────────────────────────────────────────────────────────────
class HorizontalConvBlock(eqx.Module):
    conv: eqx.nn.Conv2d

    def __init__(self, h: int, D: int, Fh: int, key: jax.random.PRNGKey):
        # kernel_size = (h, D): spans h rows (time) and all D embedding cols
        # in_channels=1  — the "image" has a single channel
        # out_channels=Fh — Fh independent filters, each learns a different pattern
        self.conv = eqx.nn.Conv2d(
            in_channels=1,
            out_channels=Fh,
            kernel_size=(h, D),
            key=key,
        )

    def __call__(self, img_x1xLxD: Float[Array, "1 L D"]) -> Float[Array, "Fh"]:
        # img_x1xLxD — (1, L, D): single-channel image, L time steps, D features

        # After valid conv with (h, D) kernel:
        #   height: L - h + 1 = Lh  (slides h-step window down L rows)
        #   width:  D - D + 1 = 1   (kernel spans full width, no slide)
        # → (Fh, Lh, 1)
        out_xFhxLhx1 = self.conv(img_x1xLxD)

        # Drop the trailing size-1 width dim → (Fh, Lh)
        out_xFhxLh = einops.rearrange(out_xFhxLhx1, "fh lh 1 -> fh lh")

        # ReLU: keep only positive activations (patterns that fired)
        out_xFhxLh = jax.nn.relu(out_xFhxLh)

        # Global max-pool over Lh positions → (Fh,)
        # Picks the strongest activation across the window (where doesn't matter)
        out_xFh = jnp.max(out_xFhxLh, axis=-1)

        return out_xFh  # Fh


# ─────────────────────────────────────────────────────────────────────────────
# CaserModel
#
# Architecture overview (single sample, no batch dim — use vmap for batching):
#
#   seq_xL (item ids)
#     ──→ Q (embedding lookup) ──→ seq_emb_xLxD
#         ──→ add channel dim   ──→ img_x1xLxD
#
#   Vertical path (captures set-like co-occurrence across full window):
#     img_x1xLxD  ──conv(L×1)──▶  out_xFvx1xD
#                  ──ReLU──▶  out_xFvxD
#                  ──flatten──▶  out_v_xF   (F = Fv×D)
#
#   Horizontal path (captures ordered sequential patterns):
#     img_x1xLxD  ──[conv(h×D) + maxpool for h=1..L]──▶  [out_xFh, ...] each Fh
#                  ──concat all heights──▶  out_h_xF   (F = Fh×L)
#
#   Short-term intent:
#     concat(out_v_xF, out_h_xF)  ──dropout──▶  combined_xF
#     combined_xF  ──Linear + ReLU──▶  z_xD      (short-term intent vector)
#
#   Long-term taste:
#     P(user_id)  ──▶  pu_xD
#
#   Scoring:
#     x_x2D = concat(z_xD, pu_xD)
#     score  = dot(x_x2D, Q_prime(item_id)) + b_item(item_id)
# ─────────────────────────────────────────────────────────────────────────────
class CaserModel(eqx.Module):
    # ── Embedding tables ──────────────────────────────────────────────────────
    Q:       eqx.nn.Embedding  # item input embeddings       (V  × D)
    P:       eqx.nn.Embedding  # user long-term embeddings   (Vu × D)
    Q_prime: eqx.nn.Embedding  # item scoring embeddings     (V  × 2D)
    b_item:  eqx.nn.Embedding  # per-item bias               (V  × 1)

    # ── Convolutional layers ──────────────────────────────────────────────────
    conv_v:       eqx.nn.Conv2d             # vertical conv: kernel (L × 1)
    conv_h_list:  List[HorizontalConvBlock] # one block per filter height h=1..L

    # ── Fully connected ───────────────────────────────────────────────────────
    # Maps concat(out_v, out_h) → z (short-term intent vector of size D)
    fc: eqx.nn.Linear

    # ── Regularisation ────────────────────────────────────────────────────────
    dropout: eqx.nn.Dropout

    # ── Stored hyperparams (static = not traced by JAX) ──────────────────────
    D:  int = eqx.field(static=True)
    L:  int = eqx.field(static=True)
    Fv: int = eqx.field(static=True)
    Fh: int = eqx.field(static=True)

    def __init__(
        self,
        num_users:  int,
        num_items:  int,
        D:          int,    # embedding dim
        L:          int,    # lookback window length
        Fv:         int,    # num vertical filters   (paper: d_v)
        Fh:         int,    # num horizontal filters (paper: d_h)
        drop_ratio: float,
        key:        jax.random.PRNGKey,
    ):
        self.D, self.L, self.Fv, self.Fh = D, L, Fv, Fh

        # One PRNGKey per sub-module: Q, P, Q_prime, b_item, conv_v, fc, + L for conv_h
        keys = jax.random.split(key, 6 + L)
        k_Q, k_P, k_Qp, k_b, k_conv_v, k_fc = keys[:6]
        keys_h = keys[6:]  # one key per horizontal block (height h=1..L)

        # ── Embedding tables ──────────────────────────────────────────────────
        # Q and Q_prime are separate tables (following the paper):
        #   Q      → embeds items *as context* (what the user has seen)
        #   Q_prime → embeds items *as targets* (what we're scoring against)
        # This asymmetry avoids a trivial solution where the model just maximises
        # self-similarity. (Same trick used in Word2Vec: input vs output embeddings.)
        self.Q       = eqx.nn.Embedding(num_items, D,     key=k_Q)
        self.P       = eqx.nn.Embedding(num_users, D,     key=k_P)
        self.Q_prime = eqx.nn.Embedding(num_items, D * 2, key=k_Qp)
        self.b_item  = eqx.nn.Embedding(num_items, 1,     key=k_b)

        # ── Vertical conv ─────────────────────────────────────────────────────
        # kernel = (L, 1): spans the full time axis, 1 column wide
        # Sliding across D columns produces D output positions per filter
        # Output after conv: (Fv, 1, D) — the "1" is because L - L + 1 = 1
        # Intuition: each filter learns a weighted combination of embeddings
        # across time for a single feature dimension — order-free co-occurrence.
        self.conv_v = eqx.nn.Conv2d(
            in_channels=1,
            out_channels=Fv,
            kernel_size=(L, 1),
            key=k_conv_v,
        )

        # ── Horizontal conv blocks (one per filter height h = 1 .. L) ─────────
        # h=1 → unigram patterns (single item signal)
        # h=2 → bigram  patterns (pairwise sequential signal)
        # h=L → full-window pattern (complete session signal)
        self.conv_h_list = [
            HorizontalConvBlock(h=h, D=D, Fh=Fh, key=keys_h[h - 1])
            for h in range(1, L + 1)
        ]

        # ── FC layer ──────────────────────────────────────────────────────────
        # Input = Fv*D (vertical path) + Fh*L (horizontal path, L heights × Fh each)
        # Output = D  (short-term intent vector z)
        fc_in = Fv * D + Fh * L
        self.fc = eqx.nn.Linear(fc_in, D, key=k_fc)

        self.dropout = eqx.nn.Dropout(p=drop_ratio)

    def __call__(
        self,
        user_id:  Int[Array, ""],    # scalar user index
        seq_xL:   Int[Array, "L"],   # L item ids (the lookback sequence)
        item_id:  Int[Array, ""],    # scalar target item index
        key:      jax.random.PRNGKey,
        training: bool = True,
    ) -> Float[Array, ""]:           # scalar score
        """
        Single-sample forward pass (no batch dim).
        Call via vmap for batched scoring — see batched_score() below.
        """

        # ── Step 1: Build the item-embedding "image" ───────────────────────────
        # Each item id in the sequence maps to a D-dim embedding row.
        # The L rows stacked form an image of shape (L, D).
        # vmap applies self.Q (which takes a scalar) across the L positions.
        seq_emb_xLxD = jax.vmap(self.Q)(seq_xL)  # L × D

        # Conv2d in Equinox expects (in_channels, height, width).
        # We treat the sequence as a single-channel image: (1, L, D).
        img_x1xLxD = einops.rearrange(seq_emb_xLxD, "l d -> 1 l d")  # 1 × L × D

        # ── Step 2: Vertical conv path ─────────────────────────────────────────
        # kernel (L, 1) spans the full time axis and one embedding column.
        # Sliding across D columns: height output = L-L+1 = 1, width output = D.
        # So: (Fv, 1, D) — one row per filter, D values across embedding dims.
        out_xFvx1xD = self.conv_v(img_x1xLxD)                             # Fv × 1 × D

        # Squeeze the size-1 height dim → (Fv, D)
        out_xFvxD = einops.rearrange(out_xFvx1xD, "fv 1 d -> fv d")       # Fv × D
        out_xFvxD = jax.nn.relu(out_xFvxD)

        # Flatten to a single vector of length Fv*D for the FC layer
        out_v_xF = einops.rearrange(out_xFvxD, "fv d -> (fv d)")          # Fv*D

        # ── Step 3: Horizontal conv path (all heights) ─────────────────────────
        # For each height h, HorizontalConvBlock returns a vector of length Fh.
        # We get L such vectors (h = 1..L) and concatenate them → Fh*L.
        h_outs = [block(img_x1xLxD) for block in self.conv_h_list]  # L × [Fh]
        out_h_xF = jnp.concatenate(h_outs, axis=0)                  # Fh*L

        # ── Step 4: Merge vertical + horizontal, project to intent vector ──────
        combined_xF = jnp.concatenate([out_v_xF, out_h_xF], axis=0)  # Fv*D + Fh*L

        # Dropout applied on the merged representation (as per the paper §3.2)
        key, subkey = jax.random.split(key)
        combined_xF = self.dropout(combined_xF, key=subkey, inference=not training)

        # Linear + ReLU → short-term intent z (size D)
        # z captures "what does this user want right now given their recent history"
        z_xD = jax.nn.relu(self.fc(combined_xF))  # D

        # ── Step 5: Combine short-term intent z with long-term user taste P_u ──
        # P_u is a learned "standing preference" vector, independent of the sequence.
        # Concatenating z and P_u gives a 2D vector that jointly represents
        # both "what the user usually likes" and "what they want right now".
        pu_xD = self.P(user_id)                          # D
        x_x2D = jnp.concatenate([z_xD, pu_xD], axis=0)  # 2D

        # ── Step 6: Score the target item ──────────────────────────────────────
        # Q_prime maps items to the *same 2D space* as x.
        # Score = dot(x, Q_prime_i) + b_i
        # Higher score → model predicts item i is more likely to be clicked next.
        qi_x2D = self.Q_prime(item_id)                                     # 2D
        bi     = self.b_item(item_id)[0]                                   # scalar

        # einsum "d,d->" = inner product of two 1D vectors → scalar
        score = jnp.einsum("d,d->", x_x2D, qi_x2D) + bi  # scalar

        return score


# ─────────────────────────────────────────────────────────────────────────────
# Batched scoring via vmap
#
# Equinox models are pure functions (no mutable state), so vmap just works.
# We vmap over (user_id, seq, item_id, key) simultaneously, giving each
# sample in the batch its own independent PRNG key for dropout.
# ─────────────────────────────────────────────────────────────────────────────
def batched_score(
    model:       CaserModel,
    user_id_xB:  Int[Array, "B"],
    seq_xBxL:    Int[Array, "B L"],
    item_id_xB:  Int[Array, "B"],
    key:         jax.random.PRNGKey,
    training:    bool = True,
) -> Float[Array, "B"]:
    """Score a batch of (user, sequence, target_item) triples. Returns B scalars."""
    # Split into B independent keys so dropout differs across samples
    keys_xB = jax.random.split(key, user_id_xB.shape[0])  # B × 2

    scores_xB = jax.vmap(
        lambda uid, seq, iid, k: model(uid, seq, iid, k, training)
    )(user_id_xB, seq_xBxL, item_id_xB, keys_xB)

    return scores_xB  # B


# ─────────────────────────────────────────────────────────────────────────────
# BPR Loss — Bayesian Personalised Ranking
#
# Historical context:
#   Rendle et al. 2009 introduced BPR for implicit feedback datasets (clicks,
#   views, purchases) where we only observe positive interactions — we never
#   truly know what a user dislikes, only what they haven't clicked *yet*.
#
# Core idea:
#   Rather than predicting absolute relevance scores, just require
#       score(pos_item) > score(neg_item)   for each (user, pos, neg) triple
#   where neg_item is *sampled* uniformly from unobserved items.
#
#   Loss = -mean( log σ( score_pos - score_neg ) )
#
#   When score_pos >> score_neg: argument → +∞, log σ → 0  (no loss)
#   When score_pos ≈ score_neg: argument ≈  0, log σ ≈ -0.69  (push apart)
#   When score_pos << score_neg: argument → -∞, log σ → -∞   (large loss)
# ─────────────────────────────────────────────────────────────────────────────
def bpr_loss(
    pos_scores_xB: Float[Array, "B"],
    neg_scores_xB: Float[Array, "B"],
) -> Float[Array, ""]:
    return -jnp.mean(jax.nn.log_sigmoid(pos_scores_xB - neg_scores_xB))


# ─────────────────────────────────────────────────────────────────────────────
# Training step — one gradient update with optax
#
# eqx.filter_jit: JIT-compiles only the JAX array leaves of the pytree,
# leaving Python-level metadata (like module structure) untouched.
# ─────────────────────────────────────────────────────────────────────────────
@eqx.filter_jit
def train_step(
    model:      CaserModel,
    optimizer:  "optax.GradientTransformation",
    opt_state,
    user_id_xB: Int[Array, "B"],
    seq_xBxL:   Int[Array, "B L"],
    pos_id_xB:  Int[Array, "B"],   # positive (ground-truth) item ids
    neg_id_xB:  Int[Array, "B"],   # negative (sampled)      item ids
    key:        jax.random.PRNGKey,
):
    import optax

    def loss_fn(model):
        # Use different keys for pos and neg scoring to keep dropout independent
        k1, k2 = jax.random.split(key)
        pos_scores_xB = batched_score(model, user_id_xB, seq_xBxL, pos_id_xB, k1)
        neg_scores_xB = batched_score(model, user_id_xB, seq_xBxL, neg_id_xB, k2)
        return bpr_loss(pos_scores_xB, neg_scores_xB)

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)

    # eqx.filter extracts only the trainable arrays from the model pytree
    updates, opt_state_new = optimizer.update(
        grads, opt_state, eqx.filter(model, eqx.is_array)
    )
    model_new = eqx.apply_updates(model, updates)
    return model_new, opt_state_new, loss


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import optax

    B         = 32
    NUM_USERS = 100
    NUM_ITEMS = 500
    D         = 16   # embedding dim
    L         = 5    # lookback window
    Fv        = 4    # num vertical filters
    Fh        = 16   # num horizontal filters per height

    key = jax.random.PRNGKey(42)
    key, model_key = jax.random.split(key)

    model = CaserModel(
        num_users=NUM_USERS,
        num_items=NUM_ITEMS,
        D=D, L=L, Fv=Fv, Fh=Fh,
        drop_ratio=0.05,
        key=model_key,
    )

    # Dummy batch
    key, data_key = jax.random.split(key)
    user_id_xB = jax.random.randint(data_key, (B,),    0, NUM_USERS)
    seq_xBxL   = jax.random.randint(data_key, (B, L),  0, NUM_ITEMS)
    pos_id_xB  = jax.random.randint(data_key, (B,),    0, NUM_ITEMS)
    neg_id_xB  = jax.random.randint(data_key, (B,),    0, NUM_ITEMS)

    optimizer = optax.adam(learning_rate=0.04)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    key, step_key = jax.random.split(key)
    model, opt_state, loss = train_step(
        model, optimizer, opt_state,
        user_id_xB, seq_xBxL, pos_id_xB, neg_id_xB,
        step_key,
    )
    print(f"Loss after one step: {loss:.4f}")

    key, score_key = jax.random.split(key)
    scores_xB = batched_score(model, user_id_xB, seq_xBxL, pos_id_xB, score_key, training=False)
    assert scores_xB.shape == (B,), f"Expected ({B},), got {scores_xB.shape}"
    print(f"Score shape : {scores_xB.shape}  ✓")
    print(f"Score sample: {scores_xB[:3]}")
