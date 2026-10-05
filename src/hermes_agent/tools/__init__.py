"""Tool registry and the built-in tool set."""

from hermes_agent.tools import builtins as _builtins  # noqa: F401  (registers the tools)
from hermes_agent.tools.builtins import (
    BUILTIN_TOOL_NAMES,
    configure_builtins,
    safe_eval_math,
    set_search_backend,
)
from hermes_agent.tools.registry import (
    GLOBAL_REGISTRY,
    ToolRegistry,
    ToolResult,
    ToolSpec,
    tool,
)
from hermes_agent.tools.sandbox import (
    get_workspace,
    relative_to_workspace,
    resolve_in_workspace,
    set_workspace,
)

__all__ = [
    "BUILTIN_TOOL_NAMES",
    "GLOBAL_REGISTRY",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "configure_builtins",
    "get_workspace",
    "relative_to_workspace",
    "resolve_in_workspace",
    "safe_eval_math",
    "set_search_backend",
    "set_workspace",
    "tool",
]
