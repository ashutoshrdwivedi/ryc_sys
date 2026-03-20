"""
factorization_machine.dataset
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CTR dataset representation and the offset trick.

The Offset Trick (sparse features → single embedding table)
────────────────────────────────────────────────────────────
Consider 3 categorical fields with their own vocabularies:

    field 0: user_id    — 5 unique users  → indices {0,1,2,3,4}
    field 1: item_id    — 3 unique items  → indices {0,1,2}
    field 2: country    — 4 unique values → indices {0,1,2,3}

If stored naïvely, user 0 and item 0 both have index=0 — ambiguous!

The offset trick assigns each field a global offset = cumulative sum so far:

    field 0 offset = 0          → user 0  maps to global index 0
    field 1 offset = 5          → item 0  maps to global index 5
    field 2 offset = 5+3 = 8    → country 0 maps to global index 8

Now a single embedding table of size N = 5+3+4 = 12 covers all fields with
no collisions. A sample (user=2, item=1, country=3) becomes global indices
[0+2, 5+1, 8+3] = [2, 6, 11].

What does a real CTR dataset look like?
────────────────────────────────────────
The canonical public benchmark is the Criteo Ad Display dataset (~45M rows).
Each row is one ad impression:

    label | int_feat_1 … int_feat_13 | cat_feat_1 … cat_feat_26
      0/1      (numerical, dropped)        (hashed categorical IDs)

We keep only the 26 categorical fields (common practice) so F=26.
Each field is already integer-encoded but each field has its own vocabulary
{0, 1, …, field_dim-1}.  Before feeding to the embedding table we apply the
offset trick so all F integers live in a single global index space [0, N).

Dataset anatomy:
    X_NxF     : Int[Array, "N F"]   — raw (pre-offset) integer indices
    y_N       : Float[Array, "N"]   — binary labels  0.0 / 1.0
    offsets   : Int[Array, "F"]     — per-field offsets (cumulative field_dims)
    field_dims : list[int]          — vocab size of each field

At batch time the dataset adds the offsets:
    x_global = X[sample_indices] + offsets    # (B, F) ready for embedding

Synthetic dataset with a learnable signal
──────────────────────────────────────────
To demonstrate that FM *learns* interactions (not just random noise) we
construct a toy dataset with 4 categorical fields and a hand-crafted rule:

    Fields:
      0 — user_type : 3 categories (casual=0, regular=1, power=2)
      1 — item_cat  : 5 categories (sports=0, tech=1, fashion=2, food=3, travel=4)
      2 — time_slot : 4 categories (morning=0, afternoon=1, evening=2, night=3)
      3 — device    : 3 categories (mobile=0, tablet=1, desktop=2)

    Click probability rule (ground truth):
      base_ctr                            = 0.10
      + 0.30 if user_type == 2            (power users click more)
      + 0.20 if item_cat in {1, 3}        (tech and food are popular)
      + 0.30 if user_type==2 AND item_cat in {1,3}   ← INTERACTION TERM
      label ~ Bernoulli(clip(ctr, 0, 1))

The interaction bonus (+0.30) is what the FM's 2nd-order term must learn.
A pure linear model can only capture the first three additive terms.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Float, Int, Array


# ─────────────────────────────────────────────────────────────────────────────
# CTRDataset
# ─────────────────────────────────────────────────────────────────────────────
class CTRDataset:
    """
    Holds a CTR dataset as raw integer field indices + binary labels.
    Applies the offset trick lazily at batch time.

    Attributes
    ----------
    X_NxF     : raw categorical indices, shape (N, F)
    y_N       : binary labels, shape (N,)
    field_dims : vocab size of each field, length F
    offsets   : cumulative field_dims prefix sums, shape (F,)
                used to convert raw indices → global embedding-table indices
    """

    def __init__(
        self,
        X_NxF: Int[Array, "N F"],
        y_N: Float[Array, "N"],
        field_dims: list[int],
    ):
        # field_dims[i] = number of unique values in field i.
        # offsets[i]    = sum of field_dims[0..i-1], so each field's indices
        #                 start at a distinct position in the embedding table.
        #
        # Example: field_dims=[3,5,4,3]  → offsets=[0, 3, 8, 12]
        #          field 2's raw index 2 → global index 8+2=10
        self.X_NxF = X_NxF
        self.y_N = y_N
        self.field_dims = field_dims
        self.offsets = jnp.array(
            [0] + list(jnp.cumsum(jnp.array(field_dims[:-1])))
        )  # F

    def __len__(self) -> int:
        return int(self.X_NxF.shape[0])

    @property
    def F(self) -> int:
        return len(self.field_dims)

    @property
    def N_vocab(self) -> int:
        """Total embedding-table size = Σ field_dims."""
        return int(sum(self.field_dims))

    def get_batch(
        self,
        key: jax.random.PRNGKey,
        batch_size: int,
    ) -> tuple[Int[Array, "B F"], Float[Array, "B"]]:
        """
        Sample a random mini-batch and apply the offset trick.

        Returns
        -------
        x_BxF : globally-offset indices, shape (B, F) — ready for FMModel
        y_B   : binary labels, shape (B,)
        """
        N = len(self)
        # Random sample (with replacement — fine for large datasets)
        idx_B = jax.random.randint(key, (batch_size,), 0, N)  # B

        x_raw_BxF = self.X_NxF[idx_B]  # B × F (raw)
        # Add offsets so each field's indices are globally unique.
        # Broadcasting: (B,F) + (F,) → (B,F)
        x_BxF = (x_raw_BxF + self.offsets).astype(jnp.int32)  # B × F
        y_B = self.y_N[idx_B]  # B

        return x_BxF, y_B

    def summary(self) -> None:
        """Print a quick overview of the dataset."""
        N = len(self)
        ctr = float(jnp.mean(self.y_N))
        print(f"  Samples N    = {N:,}")
        print(f"  Fields  F    = {self.F}")
        print(f"  Vocab   N    = {self.N_vocab:,}  (Σ field_dims)")
        print(f"  field_dims   = {self.field_dims}")
        print(f"  offsets      = {self.offsets.tolist()}")
        print(f"  Overall CTR  = {ctr:.3f}  (fraction of positive labels)")


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic dataset factory
# ─────────────────────────────────────────────────────────────────────────────
def make_synthetic_ctr_dataset(
    n_samples: int,
    key: jax.random.PRNGKey,
) -> CTRDataset:
    """
    Generate a synthetic CTR dataset with a hard-coded feature-interaction rule
    so we can verify that FM actually learns the interaction term.

    The rule is described in the module docstring above.
    """
    field_dims = [3, 5, 4, 3]  # user_type, item_cat, time_slot, device

    key, xk, yk = jax.random.split(key, 3)

    # ── Generate raw categorical indices ──────────────────────────────────────
    # Each row: randomly pick one value per field in [0, field_dim)
    X_cols = [jax.random.randint(xk, (n_samples,), 0, d) for d in field_dims]
    X_NxF = jnp.stack(X_cols, axis=1).astype(jnp.int32)  # N × F

    # ── Compute per-sample click probability ──────────────────────────────────
    user_type_N = X_NxF[:, 0]  # N  — values in {0,1,2}
    item_cat_N = X_NxF[:, 1]  # N  — values in {0,1,2,3,4}

    # --- Create some made up interactions to create the labels ---- 
    # Linear effects
    is_power_user_N = (user_type_N == 2).astype(jnp.float32)  # N
    is_popular_item_N = jnp.isin(item_cat_N, jnp.array([1, 3])).astype(jnp.float32)  # N

    # Feature interaction: power user × popular item
    interaction_N = is_power_user_N * is_popular_item_N  # N

    ctr_N = (
        0.10  # base click-through rate, scaler broadcasted to N
        + 0.30 * is_power_user_N  # linear: user effect
        + 0.20 * is_popular_item_N  # linear: item effect
        + 0.30 * interaction_N  # 2nd-order: interaction bonus
    )
    ctr_N = jnp.clip(
        ctr_N, 0.0, 1.0
    )  # clip not needed with the above formula for ctr_N would be needed if formula chnages

    # ── Sample binary labels from the computed probabilities ──────────────────
    # Bernoulli(p): draw uniform, compare to p
    uniform_N = jax.random.uniform(yk, (n_samples,))
    y_N = (uniform_N < ctr_N).astype(jnp.float32)  # N

    return CTRDataset(X_NxF=X_NxF, y_N=y_N, field_dims=field_dims)
