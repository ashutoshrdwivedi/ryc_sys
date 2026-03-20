"""
factorization_machine.training
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Loss, training step, and evaluation.

Loss — Binary Cross-Entropy (logit form, numerically stable)
─────────────────────────────────────────────────────────────
Standard BCE:  L = −[ y log p + (1−y) log(1−p) ]   where p = σ(ŷ)

Naïve: compute p = sigmoid(ŷ) first, then log(p).
  Problem: if ŷ is large and negative, sigmoid(ŷ) ≈ 0 → log(0) = −∞.

Stable form — use log_sigmoid directly:
  log σ(ŷ)    = log_sigmoid(ŷ)   (JAX handles the numerics)
  log(1−σ(ŷ)) = log σ(−ŷ)       (because 1−σ(ŷ) = σ(−ŷ))

So:  L = −[ y · log_sigmoid(ŷ) + (1−y) · log_sigmoid(−ŷ) ]

Training step — equinox + optax
────────────────────────────────
equinox separates *static* pytree structure from *dynamic* arrays.
  eqx.filter_value_and_grad  differentiates only through array leaves.
  eqx.filter_jit             JIT-compiles, treating static fields (e.g. K)
                             as compile-time constants.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from jaxtyping import Float, Int, Array

from .model import FMModel, batched_forward


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────
def bce_loss(
    logits_B: Float[Array, "B"],
    labels_B: Float[Array, "B"],   # 0.0 or 1.0
) -> Float[Array, ""]:
    loss_B = -(
        labels_B         * jax.nn.log_sigmoid(logits_B)
        + (1 - labels_B) * jax.nn.log_sigmoid(-logits_B)
    )
    return jnp.mean(loss_B)


# ─────────────────────────────────────────────────────────────────────────────
# Training step
# ─────────────────────────────────────────────────────────────────────────────
@eqx.filter_jit
def train_step(
    model:    FMModel,
    optimizer: optax.GradientTransformation,
    opt_state,
    x_BxF:   Int[Array, "B F"],
    y_B:     Float[Array, "B"],
):
    def loss_fn(model):
        logits_B = batched_forward(model, x_BxF)
        return bce_loss(logits_B, y_B)

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state_new = optimizer.update(
        grads, opt_state, eqx.filter(model, eqx.is_array)
    )
    model_new = eqx.apply_updates(model, updates)
    return model_new, opt_state_new, loss


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation — binary accuracy
# ─────────────────────────────────────────────────────────────────────────────
@eqx.filter_jit
def evaluate(
    model:  FMModel,
    x_BxF: Int[Array, "B F"],
    y_B:   Float[Array, "B"],
) -> Float[Array, ""]:
    logits_B = batched_forward(model, x_BxF)
    # Threshold probabilities at 0.5 for hard predictions
    preds_B = (jax.nn.sigmoid(logits_B) > 0.5).astype(jnp.float32)
    return jnp.mean(preds_B == y_B)
