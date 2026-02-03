# tracepipe/context.py
"""
Thread-safe context for TracePipe state.

Each thread gets its own context via threading.local().

Modes:
- CI: Fast stats and drop tracking for production/CI use.
- DEBUG: Full provenance with merge origin tracking and ghost row values.
"""

import threading
from typing import Optional

from .core import TracePipeConfig, TracePipeMode
from .storage.base import (
    LineageBackend,
    RowIdentityStrategy,
    create_default_backend,
    create_default_identity,
)

# Thread-local storage for context
_thread_local = threading.local()


class TracePipeContext:
    """
    Per-thread context for TracePipe state.

    Thread Safety:
    - Each thread gets its own context via threading.local()
    - Shared state (if needed) must use locks
    - This design supports concurrent notebook cells but NOT
      parallel pandas operations on shared DataFrames

    Extensibility:
    - Pass custom `backend` for alternative storage (SQLite, Delta Lake)
    - Pass custom `identity` for alternative engines (Polars, Spark)

    Mode System:
    - CI mode (default): Fast stats, drop tracking, contracts
    - DEBUG mode: Full merge provenance, ghost values, cell history
    """

    def __init__(
        self,
        config: Optional[TracePipeConfig] = None,
        backend: Optional[LineageBackend] = None,
        identity: Optional[RowIdentityStrategy] = None,
    ):
        self.config = config or TracePipeConfig()
        self.enabled: bool = False

        # Use provided backends or create defaults
        self.store: LineageBackend = backend or create_default_backend(self.config)
        self.row_manager: RowIdentityStrategy = identity or create_default_identity(self.config)

        self.watched_columns: set[str] = set()
        self.current_stage: Optional[str] = None

        # Nested filter operation tracking (prevents double-counting drops)
        # When > 0, __getitem__[mask] skips capture (parent op will capture)
        self._filter_op_depth: int = 0

        # GroupBy state stack (supports nesting)
        self._groupby_stack: list[dict] = []

    # === MODE CONVENIENCE PROPERTIES ===

    @property
    def is_debug_mode(self) -> bool:
        """True if running in DEBUG mode."""
        return self.config.mode == TracePipeMode.DEBUG

    @property
    def is_ci_mode(self) -> bool:
        """True if running in CI mode (default)."""
        return self.config.mode == TracePipeMode.CI

    def push_groupby(self, state: dict) -> None:
        """Push groupby state for nested operations."""
        self._groupby_stack.append(state)

    def pop_groupby(self) -> Optional[dict]:
        """Pop most recent groupby state."""
        return self._groupby_stack.pop() if self._groupby_stack else None

    def peek_groupby(self) -> Optional[dict]:
        """Peek at current groupby state without removing."""
        return self._groupby_stack[-1] if self._groupby_stack else None

    def clear_groupby_for_source(self, source_id: int) -> None:
        """
        Clear any groupby state for a given source DataFrame.

        Called when a new groupby() is performed on the same DataFrame,
        which invalidates any previous groupby state for that source.
        """
        self._groupby_stack = [s for s in self._groupby_stack if s.get("source_id") != source_id]


def get_context() -> TracePipeContext:
    """Get the current thread's TracePipe context."""
    if not hasattr(_thread_local, "context"):
        _thread_local.context = TracePipeContext()
    return _thread_local.context


def set_context(ctx: TracePipeContext) -> None:
    """Set context for current thread (used in testing)."""
    _thread_local.context = ctx


def reset_context() -> None:
    """Reset context for current thread."""
    if hasattr(_thread_local, "context"):
        del _thread_local.context
