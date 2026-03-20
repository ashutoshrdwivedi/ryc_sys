"""
Factorization Machines (Rendle 2010).
Paper: https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Int, Array
from typing import Sequence


# ─────────────────────────────────────────────────────────────────────────────
# FMModel
#
# PROBLEM: In CTR prediction, a sample is a sparse one-hot vector x ∈ {0,1}^F
# (e.g. [user_id=7823, item_id=42, day_of_week=Mon, …] one-hot encoded).
# A linear model misses feature interactions ("female user AND browsing shoes").
# Adding all pairwise terms Σᵢ Σⱼ>ᵢ wᵢⱼ xᵢ xⱼ costs O(F²) params — too many.
#
# FM's KEY IDEA: Factorise the interaction matrix W ∈ ℝ^{F×F} as wᵢⱼ ≈ ⟨vᵢ, vⱼ⟩
# where vᵢ ∈ ℝ^K (K ≪ F). Params drop from O(F²) to O(FK). Crucially, even if
# feature pair (i,j) never co-occurs, ⟨vᵢ, vⱼ⟩ generalises from shared co-occurrences.
# (This is the same idea as matrix factorisation for user×item, but for any features.)
#
# EFFICIENT FORM: The naive O(F²K) double-sum is equivalent to (Rendle eq. 5):
#   Σᵢ<ⱼ ⟨vᵢ, vⱼ⟩ = ½ Σₖ [ (Σᵢ vᵢₖ)² − Σᵢ vᵢₖ² ]
# This is O(FK) — just one pass over the F embeddings.
#
# Full formula (single sample x ∈ ℝ^F):
#   ŷ = bias  +  Σᵢ wᵢ xᵢ  +  ½ Σₖ [ (Σᵢ vᵢₖ xᵢ)² − Σᵢ (vᵢₖ xᵢ)² ]
#              └──linear──┘   └────────── interaction ──────────────┘
# ─────────────────────────────────────────────────────────────────────────────
class FMModel(eqx.Module):
    # V: latent factor embeddings, shape (N, K)
    #    V[i] is the "interaction signature" of vocab entry i.
    #    Two features interact with strength ⟨V[i], V[j]⟩.
    # W: linear weights, shape (N, 1) — one scalar per vocab entry.
    # bias: global prior (analogous to overall CTR before any features).
    V:    eqx.nn.Embedding   # N × K
    W:    eqx.nn.Embedding   # N × 1
    bias: Float[Array, ""]   # scalar

    K: int = eqx.field(static=True)

    def __init__(
        self,
        field_dims: Sequence[int],  # vocab size of each categorical field
        K:          int,            # latent factor dimension (paper uses 20)
        key:        jax.random.PRNGKey,
    ):
        # F = len(field_dims) = number of categorical fields in one sample
        #   e.g. [user_id, item_id, day_of_week, country] → F=4
        # field_dims[i] = number of unique values in field i (vocab size)
        #   e.g. [10000, 5000, 7, 50] for 10k users, 5k items, 7 days, 50 countries
        # N = Σ field_dims = total vocabulary size across all fields
        #   One embedding table handles all fields via the offset trick:
        #   field i's indices are shifted by Σ_{j<i} field_dims[j] to avoid collisions.
        self.K = K
        N = int(sum(field_dims))
        k_V, k_W = jax.random.split(key)
        self.V    = eqx.nn.Embedding(N, K, key=k_V)
        self.W    = eqx.nn.Embedding(N, 1, key=k_W)
        self.bias = jnp.zeros(())

    def __call__(
        self,
        x_F: Int[Array, "F"],      # F global indices, one per field (offset-corrected)
    ) -> Float[Array, ""]:          # scalar logit (pre-sigmoid)
        """
        Single-sample forward pass. Call via vmap for batches.

        x_F holds one global index per field, already shifted by cumulative offsets,
        so all F fields fit in a single embedding table of size N.
        e.g. for fields [user_id=7, item_id=42] with field_dims=[10000, 5000]:
             x_F = [7, 10000+42]
        """

        # ── Step 1: Linear term  Σᵢ wᵢ xᵢ ───────────────────────────────────
        # x is one-hot, so W(x[i]) directly selects the weight for field i's
        # active vocab entry — no explicit multiplication by xᵢ needed.
        linear_Fx1 = jax.vmap(self.W)(x_F)        # F × 1
        linear_term = jnp.sum(linear_Fx1)          # scalar

        # ── Step 2: Latent factor embeddings  vᵢₖ xᵢ ────────────────────────
        # emb_FxK[i] = V[x[i]] = the K-dim interaction vector for field i's
        # active feature (implicitly multiplied by xᵢ=1 since x is one-hot).
        emb_FxK = jax.vmap(self.V)(x_F)            # F × K

        # ── Step 3: Efficient O(FK) interaction term ──────────────────────────
        # Sum embeddings across F fields for each latent dim k.
        # Intuition: for 3 features (a+b+c)² = a²+b²+c² + 2(ab+ac+bc)
        # → ab+ac+bc = ½[(a+b+c)² − (a²+b²+c²)]   (the ½ cancels the factor 2)
        sum_K = jnp.sum(emb_FxK, axis=0)           # K  (sum over F fields)

        # ½(square-of-sums − sum-of-squares) = all pairwise ⟨vᵢ, vⱼ⟩ in O(FK)
        interaction_term = 0.5 * (
            jnp.sum(sum_K ** 2)      # Σₖ (Σᵢ vᵢₖ)²  — square of sums,  shape K → scalar
          - jnp.sum(emb_FxK ** 2)    # Σᵢ Σₖ vᵢₖ²    — sum of squares, shape F×K → scalar
        )

        # ── Step 4: Combine ───────────────────────────────────────────────────
        # Return the *logit* (pre-sigmoid) so the loss can use the numerically
        # stable log-sigmoid form. Apply jax.nn.sigmoid() for probabilities.
        return self.bias + linear_term + interaction_term           # scalar


# ─────────────────────────────────────────────────────────────────────────────
# Batched forward via vmap
# ─────────────────────────────────────────────────────────────────────────────
def batched_forward(
    model:  FMModel,
    x_BxF: Int[Array, "B F"],
) -> Float[Array, "B"]:
    """Vectorise the single-sample forward over a batch of B samples."""
    return jax.vmap(model)(x_BxF)   # B
