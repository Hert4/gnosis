"""Pipeline configuration — dataclass + YAML/dict loader.

A PipelineConfig declares which plugins to use in each layer plus their
kwargs. The Pipeline builder consumes this to wire concrete instances via
the PluginRegistry.

Both YAML and Python dict are supported so users can choose their style.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class StageConfig:
    """One plugin entry in a pipeline stage."""
    type: str
    config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_any(cls, spec: str | dict[str, Any]) -> StageConfig:
        """Accept either 'plugin_name' or {'type': 'plugin_name', 'config': {...}}."""
        if isinstance(spec, str):
            return cls(type=spec)
        if isinstance(spec, dict):
            if "type" not in spec:
                raise ValueError(f"Stage spec missing 'type': {spec}")
            return cls(
                type=spec["type"],
                config={k: v for k, v in spec.items() if k not in ("type",)},
            )
        raise TypeError(f"Unsupported stage spec: {type(spec).__name__}")


@dataclass
class PipelineConfig:
    """Declarative pipeline configuration.

    Each stage is a list of plugin entries. Empty list means stage is skipped.
    Router is optional (for multi-doc pipelines).
    """
    parsers: list[StageConfig] = field(default_factory=list)
    indexers: list[StageConfig] = field(default_factory=list)
    retrievers: list[StageConfig] = field(default_factory=list)
    rankers: list[StageConfig] = field(default_factory=list)
    synthesizers: list[StageConfig] = field(default_factory=list)
    router: StageConfig | None = None

    globals_: dict[str, Any] = field(default_factory=dict)
    """Shared settings (api keys, storage dir, etc.) passed to all stages."""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineConfig:
        pipeline = data.get("pipeline", data)

        def _stages(key: str) -> list[StageConfig]:
            raw = pipeline.get(key, [])
            if not raw:
                return []
            if isinstance(raw, (str, dict)):
                raw = [raw]
            return [StageConfig.from_any(item) for item in raw]

        router_raw = pipeline.get("router")
        router = StageConfig.from_any(router_raw) if router_raw else None

        return cls(
            parsers=_stages("parsers") or _stages("parse"),
            indexers=_stages("indexers") or _stages("index"),
            retrievers=_stages("retrievers") or _stages("retrieve"),
            rankers=_stages("rankers") or _stages("rank"),
            synthesizers=_stages("synthesizers") or _stages("synthesize"),
            router=router,
            globals_=data.get("globals", {}),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        try:
            import yaml  # type: ignore
        except ImportError as e:
            raise ImportError(
                "PyYAML not installed. pip install pyyaml"
            ) from e
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
