"""Inference-time activation steering helpers."""

from .hooks import WriteHookContext, decoder_layers
from .steer import ActivationPatcher, NormMatchedSteer
from .subspace import SubspaceMeanSub, fit_subspace_basis, subspace_subtract_hook

__all__ = [
    "ActivationPatcher",
    "NormMatchedSteer",
    "SubspaceMeanSub",
    "WriteHookContext",
    "decoder_layers",
    "fit_subspace_basis",
    "subspace_subtract_hook",
]
