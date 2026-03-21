"""
DeepFM (Guo et al. 2017).
Paper: https://arxiv.org/abs/1703.04247

DeepFM = FM component + DNN component sharing the SAME embedding table V.

    ŷ = FM_logit + DNN_logit     (summed before sigmoid)

    FM_logit  = bias + Σ wᵢxᵢ + ½ Σₖ[(Σᵢ vᵢₖ)² − Σᵢ vᵢₖ²]   ← exact FMModel
    DNN_logit = MLP([v₁ ‖ v₂ ‖ … ‖ vF])                       ← shared vᵢ = V[x[i]]

Why share V?
    Joint training lets FM's 2nd-order signal and DNN's high-order signal
    inform the same embedding. In the original paper this was the key
    improvement over Wide&Deep (Cheng 2016) which had separate embeddings
    for the wide (linear) and deep parts.

    FM captures *explicit* pairwise interactions: ⟨vᵢ, vⱼ⟩ is directly
    optimised. DNN captures *implicit* high-order interactions via depth.
    Together they complement each other, which is why DeepFM consistently
    outperformed either component alone in the original CTR benchmarks.

Implementation approach:
    DeepFMModel contains an FMModel instance (unchanged).
    The DNN reads fm.V directly — same Python object, no weight duplication.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Int, Array
from typing import Sequence

from .model import FMModel


# ─────────────────────────────────────────────────────────────────────────────
# DeepFMModel
# ─────────────────────────────────────────────────────────────────────────────
class DeepFMModel(eqx.Module):
    # fm holds V (N×K embedding), W (N×1 linear weights), bias — all shared.
    # The DNN reads fm.V, so there is exactly one copy of V in this pytree.
    fm: FMModel  # FM component; owns the shared embedding V
    layers: list  # MLP layers: [Linear, Linear, …] — hidden then output

    F: int = eqx.field(static=True)  # number of categorical fields per sample
    K: int = eqx.field(static=True)  # latent factor / embedding dimension

    def __init__(
        self,
        field_dims: Sequence[int],  # vocab size of each categorical field
        K: int,  # latent factor dim (same as FM)
        hidden_dims: Sequence[int],  # MLP hidden sizes, e.g. [256, 128, 64]
        key: jax.random.PRNGKey,
    ):
        # Split key: one for FM, one per MLP layer (hidden + output)
        n_layers = len(hidden_dims) + 1  # +1 for the final output layer
        k_fm, *k_layers = jax.random.split(key, 1 + n_layers)

        self.F = len(field_dims)
        self.K = K

        # ── FM component (completely unchanged) ───────────────────────────────
        # FMModel owns V and W; we never duplicate them.
        self.fm = FMModel(field_dims, K, k_fm)

        # ── MLP component ─────────────────────────────────────────────────────
        # Input to DNN: concatenated field embeddings, shape F*K.
        # Each field contributes its K-dim embedding vector → F*K total inputs.
        # Tower: [F*K → h₁ → h₂ → … → 1]  (final layer outputs a scalar logit)
        input_dim = self.F * K
        dims = [input_dim] + list(hidden_dims) + [1]  #  layers[-1] is Linear(64, 1)
        self.layers = [
            eqx.nn.Linear(d_in, d_out, key=k)
            for d_in, d_out, k in zip(dims[:-1], dims[1:], k_layers)
        ]

    def __call__(
        self,
        x_F: Int[Array, "F"],  # F global indices, one per field (offset-corrected)
    ) -> Float[Array, ""]:  # scalar logit (pre-sigmoid)
        """
        Single-sample forward. Call via vmap for batches.
        """

        # ── FM component ──────────────────────────────────────────────────────
        # Computes bias + linear + 2nd-order interaction in O(FK).
        # Internally calls jax.vmap(self.fm.V) to look up embeddings.
        fm_logit = self.fm(x_F)  # scalar

        # ── Shared embeddings for DNN ──────────────────────────────────────────
        # We re-use fm.V (the same embedding object) for the deep component.
        # This means gradients from both FM and DNN update the same V weights.
        emb_FxK = jax.vmap(self.fm.V)(x_F)  # F × K

        # Flatten all F embedding vectors into one input vector for the MLP.
        # emb_FxK = [[v₁₁,…,v₁ₖ], [v₂₁,…,v₂ₖ], …] → h_FK = [v₁₁,…,vFk]
        h_FK = emb_FxK.reshape(-1)  # F*K  (single flat dim)

        # ── MLP forward ───────────────────────────────────────────────────────
        # All hidden layers use ReLU. The final layer is linear (no activation)
        # so the output is an unbounded scalar logit, consistent with FM's logit.
        # (They are consistent with FM's logits means both components output unbounded
        #  scalar predictions in the same pre-sigmoid space, so they can be meaningfully
        # summed.)
        # h_D: D is the current hidden width — changes each layer (F*K → h₁ → h₂ → …)
        h_D = h_FK
        for layer in self.layers[:-1]:
            h_D = jax.nn.relu(layer(h_D))  # D (varies per layer)

        deep_logit = self.layers[-1](h_D).squeeze()  # scalar

        # ── Combine ───────────────────────────────────────────────────────────
        # Summing logits before sigmoid = multiplying the odds. Both terms
        # are in the same pre-sigmoid space so this is well-defined.
        # Apply jax.nn.sigmoid() for probabilities (kept outside for stable BCE).
        return fm_logit + deep_logit  # scalar


# ─────────────────────────────────────────────────────────────────────────────
# Batched forward via vmap
# ─────────────────────────────────────────────────────────────────────────────
def batched_forward(
    model: DeepFMModel,
    x_BxF: Int[Array, "B F"],
) -> Float[Array, "B"]:
    """Vectorise the single-sample forward over a batch of B samples."""
    return jax.vmap(model)(x_BxF)  # B
