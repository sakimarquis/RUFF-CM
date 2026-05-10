from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from ruff_cm.llm.extract_hiddens.hooks import decoder_layers

if TYPE_CHECKING:
    import torch


class WriteHookContext:
    """Attach write hooks to selected decoder layers and remove them on exit."""

    def __init__(
        self,
        model: Any,
        layer_indices: list[int] | None = None,
        hook_factory: Callable[[int], Callable] | None = None,
        *,
        layer_idx: int | None = None,
        mutation_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        if layer_indices is None:
            layer_indices = [layer_idx]
        if hook_factory is None:
            hook_factory = lambda _layer_idx: mutate_hidden_output(mutation_fn)
        self.model = model
        self.layer_indices = layer_indices
        self.hook_factory = hook_factory
        self.handles: list[Any] = []

    def __enter__(self):
        layers = decoder_layers(self.model)
        try:
            for layer_idx in self.layer_indices:
                self.handles.append(layers[layer_idx].register_forward_hook(self.hook_factory(layer_idx)))
        except Exception:
            self._remove_handles()
            raise
        return self

    def __exit__(self, exc_type, exc, tb):
        self._remove_handles()

    def _remove_handles(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles = []


def mutate_hidden_output(mutation_fn: Callable[[torch.Tensor], torch.Tensor]):
    """Wrap a hidden-state mutation so tuple outputs keep cache-like trailing fields."""

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        mutated = mutation_fn(hidden)
        if isinstance(output, tuple):
            return (mutated, *output[1:])
        return mutated

    return hook


__all__ = ["WriteHookContext", "decoder_layers", "mutate_hidden_output"]
