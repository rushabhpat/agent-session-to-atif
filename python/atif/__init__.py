"""Export Cursor, Claude Code, and Codex sessions as ATIF trajectories.

The parsing core is compiled Rust (`atif._core`); this package provides the CLI,
the interactive picker, and the publish flow.
"""

from ._core import SCHEMA, __version__

__all__ = ["SCHEMA", "__version__"]
