"""Starter tool set. Importing this module registers the tools.

Docstrings here are part of the prompt -- they become the tool descriptions the
model sees -- so they are written for the model, not just for humans.
"""

from __future__ import annotations

import ast
import asyncio
import math
import operator
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from hermes_agent.tools.registry import tool
from hermes_agent.tools.sandbox import (
    get_workspace,
    relative_to_workspace,
    resolve_in_workspace,
)

_RUNNER = Path(__file__).with_name("_pyrunner.py")

#: Overridable limits, wired from Settings by `configure_builtins`.
LIMITS: dict[str, Any] = {
    "python_timeout_s": 20.0,
    "python_max_output_chars": 20_000,
    "python_memory_mb": 512,
    "max_read_bytes": 200_000,
}


def configure_builtins(
    *,
    python_timeout_s: float | None = None,
    python_max_output_chars: int | None = None,
    max_read_bytes: int | None = None,
    python_memory_mb: int | None = None,
) -> None:
    """Apply runtime limits from Settings without re-registering the tools."""
    for key, value in (
        ("python_timeout_s", python_timeout_s),
        ("python_max_output_chars", python_max_output_chars),
        ("max_read_bytes", max_read_bytes),
        ("python_memory_mb", python_memory_mb),
    ):
        if value is not None:
            LIMITS[key] = value


def _truncate(text: str, limit: int, label: str = "output") -> str:
    if len(text) <= limit:
        return text
    return (
        text[:limit]
        + f"\n\n... [{label} truncated: {len(text)} chars total, showing first {limit}]"
    )


# --- file tools -------------------------------------------------------------------


@tool(tags=["fs", "read"])
def read_file(path: str, max_bytes: int | None = None) -> str:
    """Read a UTF-8 text file from the workspace.

    Args:
        path: Path relative to the workspace root, e.g. "notes/todo.md".
        max_bytes: Optional cap on how much to read; the default comes from config.
    """
    target = resolve_in_workspace(path, must_exist=True)
    if target.is_dir():
        raise IsADirectoryError(f"{relative_to_workspace(target)} is a directory; use list_dir")
    cap = int(max_bytes or LIMITS["max_read_bytes"])
    data = target.read_bytes()[: cap + 1]
    text = data.decode("utf-8", errors="replace")
    return _truncate(text, cap, "file")


@tool(destructive=True, parallel_safe=False, tags=["fs", "write"])
def write_file(path: str, content: str, append: bool = False) -> str:
    """Write a UTF-8 text file inside the workspace, creating parent directories.

    Args:
        path: Path relative to the workspace root.
        content: Full text to write.
        append: Append to the file instead of overwriting it.
    """
    target = resolve_in_workspace(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a" if append else "w", encoding="utf-8", newline="\n") as fh:
        fh.write(content)
    verb = "Appended to" if append else "Wrote"
    return f"{verb} {relative_to_workspace(target)} ({len(content)} chars)."


@tool(tags=["fs", "read"])
def list_dir(path: str = ".", recursive: bool = False, max_entries: int = 200) -> str:
    """List files and directories in the workspace.

    Args:
        path: Directory relative to the workspace root; defaults to the root.
        recursive: Walk subdirectories as well.
        max_entries: Stop after this many entries.
    """
    target = resolve_in_workspace(path, must_exist=True)
    if not target.is_dir():
        raise NotADirectoryError(f"{relative_to_workspace(target)} is not a directory")

    iterator = target.rglob("*") if recursive else target.glob("*")
    rows: list[str] = []
    truncated = False
    for entry in sorted(iterator, key=lambda p: (p.is_file(), str(p).lower())):
        if len(rows) >= max_entries:
            truncated = True
            break
        rel = relative_to_workspace(entry)
        if entry.is_dir():
            rows.append(f"{rel}/")
        else:
            rows.append(f"{rel}  ({entry.stat().st_size} bytes)")

    if not rows:
        return f"{relative_to_workspace(target)} is empty."
    body = "\n".join(rows)
    if truncated:
        body += f"\n... [truncated at {max_entries} entries]"
    return body


# --- code execution -----------------------------------------------------------------


def _run_python_blocking(code: str, timeout_s: float) -> str:
    workspace = get_workspace()
    with tempfile.TemporaryDirectory(prefix="hermes_py_") as tmpdir:
        script = Path(tmpdir) / "snippet.py"
        script.write_text(code, encoding="utf-8")

        # -I: isolated mode (ignores PYTHONPATH, user site-packages, and env
        # config) so the snippet cannot be influenced by the parent environment.
        cmd = [
            sys.executable,
            "-I",
            str(_RUNNER),
            str(script),
            str(int(LIMITS["python_memory_mb"])),
            str(max(1, int(timeout_s))),
        ]
        env = {
            k: v
            for k, v in os.environ.items()
            if k.upper() in {"PATH", "SYSTEMROOT", "TEMP", "TMP", "LANG", "LC_ALL", "TZ"}
        }
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONDONTWRITEBYTECODE"] = "1"

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(workspace),
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return (
                f"TIMEOUT: execution exceeded {timeout_s}s and was killed. "
                "Simplify the code or avoid long loops."
            )

    cap = int(LIMITS["python_max_output_chars"])
    parts = [f"exit_code: {proc.returncode}"]
    if proc.stdout.strip():
        parts.append("stdout:\n" + _truncate(proc.stdout.rstrip(), cap, "stdout"))
    if proc.stderr.strip():
        parts.append("stderr:\n" + _truncate(proc.stderr.rstrip(), cap // 2, "stderr"))
    if len(parts) == 1:
        parts.append("(no output -- remember to print() the result you want to see)")
    return "\n\n".join(parts)


@tool(destructive=True, parallel_safe=False, tags=["code"])
async def run_python(code: str, timeout_s: float | None = None) -> str:
    """Run a Python snippet in a sandboxed subprocess and return its output.

    The snippet runs with the workspace as its working directory, with no
    network access and with a wall-clock timeout. Print anything you want to
    see; return values are not captured.

    Args:
        code: The Python source to execute.
        timeout_s: Optional override for the wall-clock timeout.
    """
    limit = float(timeout_s or LIMITS["python_timeout_s"])
    limit = max(1.0, min(limit, 300.0))
    return await asyncio.to_thread(_run_python_blocking, code, limit)


# --- calculator -----------------------------------------------------------------------

_BIN_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_NAMES: dict[str, Any] = {"pi": math.pi, "e": math.e, "tau": math.tau, "inf": math.inf}
_FUNCS: dict[str, Callable[..., Any]] = {
    name: getattr(math, name)
    for name in (
        "sqrt", "log", "log2", "log10", "exp", "sin", "cos", "tan", "asin", "acos",
        "atan", "atan2", "floor", "ceil", "fabs", "factorial", "hypot", "degrees",
        "radians", "gcd", "dist", "trunc",
    )
}
_FUNCS.update({"abs": abs, "round": round, "min": min, "max": max, "sum": sum, "pow": pow})

#: Guard against `2**999999999` hanging the process on a huge integer.
_MAX_POW_EXPONENT = 10_000


def safe_eval_math(expression: str) -> float | int:
    """Evaluate an arithmetic expression with a whitelisted AST walker (no eval)."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Could not parse expression: {exc.msg}") from exc

    def visit(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                return node.value
            raise ValueError(f"Only numeric literals are allowed, got {node.value!r}")
        if isinstance(node, ast.BinOp):
            op = _BIN_OPS.get(type(node.op))
            if op is None:
                raise ValueError(f"Operator {type(node.op).__name__} is not allowed")
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > _MAX_POW_EXPONENT:
                raise ValueError(f"Exponent too large (max {_MAX_POW_EXPONENT})")
            return op(left, right)
        if isinstance(node, ast.UnaryOp):
            op_u = _UNARY_OPS.get(type(node.op))
            if op_u is None:
                raise ValueError(f"Operator {type(node.op).__name__} is not allowed")
            return op_u(visit(node.operand))
        if isinstance(node, ast.Name):
            if node.id in _NAMES:
                return _NAMES[node.id]
            raise ValueError(f"Unknown name {node.id!r}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
                raise ValueError("Only whitelisted math functions may be called")
            if node.keywords:
                raise ValueError("Keyword arguments are not supported")
            return _FUNCS[node.func.id](*[visit(a) for a in node.args])
        if isinstance(node, (ast.Tuple, ast.List)):
            return [visit(e) for e in node.elts]
        raise ValueError(f"Expression element {type(node).__name__} is not allowed")

    result = visit(tree)
    if not isinstance(result, (int, float)):
        raise ValueError("Expression did not evaluate to a number")
    return result


@tool(tags=["math"])
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression exactly. Use this instead of doing mental math.

    Supports + - * / // % **, parentheses, and math functions such as sqrt, log,
    sin, cos, floor, ceil, factorial, abs, round, min, max, sum.

    Args:
        expression: The expression to evaluate, e.g. "sqrt(2) * 17.5 ** 2".
    """
    value = safe_eval_math(expression)
    return f"{expression} = {value}"


# --- web search (stub) ------------------------------------------------------------------

SearchBackend = Callable[[str, int], list[dict[str, str]]]
_SEARCH_BACKEND: SearchBackend | None = None


def set_search_backend(backend: SearchBackend | None) -> None:
    """Install a search implementation: ``(query, max_results) -> [{title,url,snippet}]``.

    Left unset by default so the system stays fully offline. Point it at a local
    SearXNG instance, an offline index, or a cloud API if you accept the
    dependency.
    """
    global _SEARCH_BACKEND
    _SEARCH_BACKEND = backend


@tool(tags=["web"])
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for a query and return ranked titles, URLs and snippets.

    Args:
        query: What to search for.
        max_results: How many results to return (1-20).
    """
    if _SEARCH_BACKEND is None:
        return (
            "web_search is not configured: this deployment is offline and no search "
            "backend is installed. Answer from the conversation, the workspace files, "
            "or your own knowledge instead, and state that you could not search."
        )
    results = _SEARCH_BACKEND(query, max(1, min(int(max_results), 20)))
    if not results:
        return f"No results for {query!r}."
    lines = []
    for i, item in enumerate(results, 1):
        lines.append(
            f"{i}. {item.get('title', '(untitled)')}\n"
            f"   {item.get('url', '')}\n"
            f"   {item.get('snippet', '')}"
        )
    return "\n".join(lines)


BUILTIN_TOOL_NAMES = (
    "read_file",
    "write_file",
    "list_dir",
    "run_python",
    "calculator",
    "web_search",
)
