"""Shims — backward-compat facades that look like legacy classes.

Use these to swap the smartsearch-v4 engine implementation without
changing existing callers (smartsearch-api, test_pipeline.py).
"""

from gnosis.shims.smartsearch_v4_shim import SmartSearchV4Shim

__all__ = ["SmartSearchV4Shim"]
