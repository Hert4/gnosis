"""Example: declarative YAML pipeline config.

Shows the plugin registry dispatching types by name from a YAML file.

Usage:
    python examples/pipeline_yaml.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

# Import side-effect registers all plugins
import gnosis.parsers  # noqa: F401
import gnosis.indexers  # noqa: F401
import gnosis.retrievers  # noqa: F401
import gnosis.rankers  # noqa: F401
import gnosis.synthesizers  # noqa: F401

from gnosis.core.pipeline import Pipeline
from gnosis.core.registry import PluginRegistry

YAML_CONFIG = """
pipeline:
  parsers:
    - pdfplumber
    - table_normalizer
    - ellipsis_handler
  indexers:
    - type: page_bm25
      mode: bmx
    - tree_index
  rankers:
    - type: smart_truncate
      max_chars: 20000
"""


def main() -> None:
    import yaml
    cfg = yaml.safe_load(YAML_CONFIG)

    pipeline = Pipeline.from_config(cfg)
    print("Pipeline assembled from YAML:")
    print(f"  parsers:       {[p.name for p in pipeline.parsers]}")
    print(f"  indexers:      {[i.name for i in pipeline.indexers]}")
    print(f"  rankers:       {[r.name for r in pipeline.rankers]}")
    print()
    print(f"Registered plugin names by layer:")
    for layer in PluginRegistry.all_layers():
        names = PluginRegistry.list(layer)
        if names:
            print(f"  {layer:12s}: {', '.join(names)}")


if __name__ == "__main__":
    main()
