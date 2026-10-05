"""Templating and condition evaluation for workflows.

Two deliberately small languages instead of Jinja + `eval`:

* ``render("... {{ steps.plan.output.title }} ...", context)`` -- substitution only.
* ``evaluate("steps.critic.output.approved == false", context)`` -- comparisons,
  boolean operators and membership, walked over a whitelisted AST.

Workflow files are code-adjacent, but the values flowing through them are model
and tool output. Keeping both languages non-executable means a crafted document
cannot turn a branch condition into arbitrary code.
"""

from __future__ import annotations

import ast
import json
import operator
import re
from typing import Any

TEMPLATE_RE = re.compile(r"\{\{\s*(?P<expr>[^}]+?)\s*\}\}")

_COMPARATORS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}
_LITERALS = {"true": True, "false": False, "null": None, "none": None}


class ExpressionError(ValueError):
    """Raised for a malformed or disallowed expression."""


def resolve_path(path: str, context: dict[str, Any]) -> Any:
    """Look up a dotted path such as ``steps.plan.output.items`` in `context`.

    Missing keys resolve to None rather than raising: a branch that references a
    step which has not run yet should evaluate falsy, not explode.
    """
    current: Any = context
    for part in path.split("."):
        part = part.strip()
        if not part:
            return None
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, (list, tuple)):
            if part.lstrip("-").isdigit():
                index = int(part)
                current = current[index] if -len(current) <= index < len(current) else None
            else:
                return None
        else:
            current = getattr(current, part, None)
        if current is None:
            return None
    return current


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, indent=2, default=str)
    return str(value)


def render(template: str, context: dict[str, Any]) -> str:
    """Substitute ``{{ dotted.path }}`` placeholders. Unknown paths become empty."""
    if not template:
        return ""

    def replace(match: re.Match[str]) -> str:
        return _stringify(resolve_path(match.group("expr"), context))

    return TEMPLATE_RE.sub(replace, template)


def render_any(value: Any, context: dict[str, Any]) -> Any:
    """Render recursively through dicts and lists, leaving non-strings alone."""
    if isinstance(value, str):
        return render(value, context)
    if isinstance(value, dict):
        return {k: render_any(v, context) for k, v in value.items()}
    if isinstance(value, list):
        return [render_any(v, context) for v in value]
    return value


def _dotted_name(node: ast.AST) -> str | None:
    """Flatten ``a.b.c`` attribute chains back into a lookup path."""
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return ".".join(reversed(parts))
    return None


def evaluate(expression: str, context: dict[str, Any]) -> bool:
    """Evaluate a boolean condition against the workflow context.

    Supported: ``== != < <= > >=``, ``and or not``, ``in`` / ``not in``,
    literals (numbers, strings, true/false/null), dotted paths, and bare paths
    as truthiness tests.
    """
    expression = (expression or "").strip()
    if not expression:
        return True

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"Invalid condition {expression!r}: {exc.msg}") from exc

    def visit(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.BoolOp):
            values = [visit(v) for v in node.values]
            if isinstance(node.op, ast.And):
                return all(bool(v) for v in values)
            return any(bool(v) for v in values)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return not bool(visit(node.operand))
        if isinstance(node, ast.Compare):
            left = visit(node.left)
            for op, comparator in zip(node.ops, node.comparators, strict=True):
                right = visit(comparator)
                if isinstance(op, ast.In):
                    ok = _contains(right, left)
                elif isinstance(op, ast.NotIn):
                    ok = not _contains(right, left)
                else:
                    func = _COMPARATORS.get(type(op))
                    if func is None:
                        raise ExpressionError(
                            f"Comparison {type(op).__name__} is not allowed"
                        )
                    ok = _compare(func, left, right)
                if not ok:
                    return False
                left = right
            return True
        if isinstance(node, (ast.Name, ast.Attribute)):
            path = _dotted_name(node)
            if path is None:
                raise ExpressionError("Unsupported name expression")
            lowered = path.lower()
            if lowered in _LITERALS and "." not in path:
                return _LITERALS[lowered]
            return resolve_path(path, context)
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            target = visit(node.value)
            key = node.slice.value
            if isinstance(target, dict):
                return target.get(key)
            if isinstance(target, (list, tuple, str)) and isinstance(key, int):
                return target[key] if -len(target) <= key < len(target) else None
            return None
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            return [visit(e) for e in node.elts]
        raise ExpressionError(f"Expression element {type(node).__name__} is not allowed")

    return bool(visit(tree))


def _compare(func: Any, left: Any, right: Any) -> bool:
    """Compare, tolerating the int/str mismatches model output routinely produces."""
    try:
        return bool(func(left, right))
    except TypeError:
        if isinstance(left, (int, float)) and isinstance(right, str):
            try:
                return bool(func(left, float(right)))
            except ValueError:
                return False
        if isinstance(left, str) and isinstance(right, (int, float)):
            try:
                return bool(func(float(left), right))
            except ValueError:
                return False
        return False


def _contains(container: Any, needle: Any) -> bool:
    if container is None:
        return False
    try:
        return needle in container
    except TypeError:
        return False
