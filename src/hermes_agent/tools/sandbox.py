"""Workspace confinement for file tools.

The rule is simple and enforced in exactly one place: every path a tool touches
is resolved to an absolute real path and must live under WORKSPACE_DIR. Because
we resolve *before* comparing, this defeats `../` traversal, absolute paths,
and symlinks that point outside the workspace.
"""

from __future__ import annotations

import os
from pathlib import Path

from hermes_agent.errors import SandboxViolationError

_WORKSPACE: Path = Path("./workspace").resolve()


def set_workspace(path: str | os.PathLike[str]) -> Path:
    """Point the sandbox at `path`, creating it if needed. Returns the resolved root."""
    global _WORKSPACE
    resolved = Path(path).expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    _WORKSPACE = resolved
    return resolved


def get_workspace() -> Path:
    _WORKSPACE.mkdir(parents=True, exist_ok=True)
    return _WORKSPACE


def resolve_in_workspace(path: str | os.PathLike[str], *, must_exist: bool = False) -> Path:
    """Resolve `path` inside the workspace or raise SandboxViolationError.

    Args:
        path: A path relative to the workspace, or an absolute path already
            inside it.
        must_exist: Raise FileNotFoundError when the target is missing.
    """
    workspace = get_workspace()
    raw = Path(os.fspath(path)).expanduser()

    candidate = raw if raw.is_absolute() else workspace / raw
    # strict=False so we can validate paths that do not exist yet (writes).
    resolved = candidate.resolve(strict=False)

    if resolved != workspace and workspace not in resolved.parents:
        raise SandboxViolationError(str(path), str(workspace))

    if must_exist and not resolved.exists():
        raise FileNotFoundError(
            f"No such path in workspace: {relative_to_workspace(resolved)}"
        )
    return resolved


def relative_to_workspace(path: Path) -> str:
    """Display form: workspace-relative and POSIX-style, so traces are portable."""
    try:
        return path.resolve().relative_to(get_workspace()).as_posix() or "."
    except ValueError:
        return str(path)
