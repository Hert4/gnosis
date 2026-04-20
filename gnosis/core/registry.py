"""Plugin registry — layer-keyed name → class lookup.

Plugins register via @register decorator. At runtime, pipelines can look up
a plugin by layer + name. Supports both static registration (import-time
decorator) and dynamic registration (register_class function).
"""

from __future__ import annotations

from typing import Any

_LAYERS = frozenset({
    "parser", "indexer", "retriever", "channel",
    "ranker", "synthesizer", "router", "backend",
})


class PluginRegistry:
    """Singleton-style registry of plugin classes keyed by (layer, name)."""

    _registry: dict[str, dict[str, type]] = {layer: {} for layer in _LAYERS}

    @classmethod
    def register_class(cls, layer: str, name: str, plugin_cls: type) -> None:
        """Programmatic registration."""
        if layer not in _LAYERS:
            raise ValueError(
                f"Unknown layer '{layer}'. Valid: {sorted(_LAYERS)}"
            )
        cls._registry[layer][name] = plugin_cls

    @classmethod
    def get(cls, layer: str, name: str) -> type:
        """Look up a plugin class. Raises KeyError if not found."""
        if layer not in _LAYERS:
            raise ValueError(f"Unknown layer '{layer}'")
        if name not in cls._registry[layer]:
            raise KeyError(
                f"Plugin '{name}' not registered under layer '{layer}'. "
                f"Available: {sorted(cls._registry[layer])}"
            )
        return cls._registry[layer][name]

    @classmethod
    def list(cls, layer: str) -> list[str]:
        """List all registered plugin names in a layer."""
        if layer not in _LAYERS:
            raise ValueError(f"Unknown layer '{layer}'")
        return sorted(cls._registry[layer])

    @classmethod
    def all_layers(cls) -> list[str]:
        return sorted(_LAYERS)

    @classmethod
    def clear(cls, layer: str | None = None) -> None:
        """Clear registry (for tests). If layer is None, clears everything."""
        if layer is None:
            for lname in _LAYERS:
                cls._registry[lname] = {}
        elif layer in _LAYERS:
            cls._registry[layer] = {}


def register(layer: str, name: str):
    """Decorator for plugin classes.

    Usage:
        @register("parser", "ocr2")
        class OCR2Parser: ...
    """
    def _decorator(cls: type) -> type:
        PluginRegistry.register_class(layer, name, cls)
        # Attach name so instances have a `.name` property auto-set
        if not hasattr(cls, "name") or not isinstance(
            getattr(cls, "name", None), str
        ):
            cls.name = name  # type: ignore[attr-defined]
        return cls
    return _decorator


def make(layer: str, name: str, **kwargs: Any):
    """Convenience: look up a plugin and instantiate it.

    Example:
        ocr = make("parser", "ocr2", dpi=250)
    """
    cls = PluginRegistry.get(layer, name)
    return cls(**kwargs)
