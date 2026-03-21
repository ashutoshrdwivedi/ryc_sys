"""
factorization_machine — FM tutorial package
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Modules
-------
dataset    CTRDataset, make_synthetic_ctr_dataset
model      FMModel, batched_forward
deep_model DeepFMModel, batched_forward (DeepFM variant)
training   bce_loss, train_step, evaluate
demos      demo_* functions + main()

Run the full tutorial:
    python -m factorization_machine.demos

Noam Shape Notation
-------------------
B  = batch size
F  = number of fields (categorical features per sample)
K  = latent factor / embedding dimension
N  = total vocabulary size = Σ field_dims  (after offset trick)
"""

from .dataset    import CTRDataset, make_synthetic_ctr_dataset
from .model      import FMModel, batched_forward
from .deep_model import DeepFMModel
from .training   import bce_loss, train_step, evaluate

__all__ = [
    "CTRDataset",
    "make_synthetic_ctr_dataset",
    "FMModel",
    "DeepFMModel",
    "batched_forward",
    "bce_loss",
    "train_step",
    "evaluate",
]
