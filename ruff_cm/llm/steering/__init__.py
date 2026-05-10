"""Inference-time activation steering helpers."""

from ruff_cm.llm.forward import add, norm_match, patch, subspace_subtract

from .hooks import WriteHookContext, decoder_layers
from .steer import ActivationPatcher, NormMatchedSteer
from .subspace import SubspaceMeanSub, fit_subspace_basis, subspace_subtract_hook

__all__ = [
    "ActivationPatcher",
    "NormMatchedSteer",
    "SubspaceMeanSub",
    "WriteHookContext",
    "add",
    "decoder_layers",
    "fit_subspace_basis",
    "norm_match",
    "patch",
    "subspace_subtract",
    "subspace_subtract_hook",
]
