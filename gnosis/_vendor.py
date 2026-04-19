"""Path bootstrap for optional vendor imports.

Until retrival-framework has a proper pip-installable dependency on
agent-search/smartsearch-v4, parsers and indexers import legacy modules
via sys.path manipulation. This helper is the ONE place that injects
those paths so plugin modules don't each do it themselves.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Default location relative to this repo: ../agent-search/smartsearch-v4
_FRAMEWORK_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_AGENT_SEARCH = _FRAMEWORK_ROOT.parent / "agent-search" / "smartsearch-v4"


def agent_search_path() -> Path:
    """Resolve the agent-search/smartsearch-v4 directory.

    Override with env var ``RETRIVAL_AGENT_SEARCH`` if located elsewhere.
    """
    env = os.environ.get("RETRIVAL_AGENT_SEARCH")
    if env:
        return Path(env).resolve()
    return _DEFAULT_AGENT_SEARCH.resolve()


def ensure_agent_search_on_path() -> Path:
    """Add agent-search/smartsearch-v4 to sys.path once. Idempotent."""
    p = agent_search_path()
    if not p.exists():
        raise ImportError(
            f"agent-search not found at {p}. Set RETRIVAL_AGENT_SEARCH env var "
            f"or install retrival-framework[smartsearch_v4]."
        )
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)
    return p
