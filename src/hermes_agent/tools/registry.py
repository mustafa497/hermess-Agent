"""Decorator-based tool registry.

`@tool` turns a typed Python function into a JSON-schema tool definition:
parameter types come from annotations, descriptions from a Google-style
``Args:`` block in the docstring. Keeping the schema derived from the signature
means it cannot drift from the implementation.

Validation is the important behaviour here: bad arguments never raise out of
`execute()`. They come back as a `ToolResult` with `ok=False` and a message the
model can read, because "you passed a string where an int was needed" is
something an 8B model can usually fix on the next turn.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
import time
from collections.abc import Callable, Iterable, Sequence
from typing import Any, get_type_hints

from pydantic import BaseModel, Field, ValidationError, create_model

from hermes_agent.errors import DestructiveToolBlockedError, ToolNotFoundError

logger = logging.getLogger(__name__)

ToolFunc = Callable[..., Any]

_ARGS_HEADER_RE = re.compile(r"^\s*(Args|Arguments|Parameters)\s*:\s*$", re.IGNORECASE)
_SECTION_HEADER_RE = re.compile(
    r"^\s*(Returns|Raises|Yields|Examples?|Notes?|Warnings?)\s*:\s*$", re.IGNORECASE
)
_ARG_LINE_RE = re.compile(r"^\s*(?P<name>\*{0,2}\w+)\s*(?:\([^)]*\))?\s*:\s*(?P<desc>.*)$")


class ToolResult(BaseModel):
    """What the agent appends to the conversation as a tool message."""

    tool: str
    ok: bool = True
    content: str = ""
    error: str | None = None
    latency_ms: float = 0.0
    meta: dict[str, Any] = Field(default_factory=dict)

    def to_model_text(self) -> str:
        """Render for the model. Errors are labelled so failure is unambiguous."""
        if self.ok:
            return self.content
        return f"ERROR ({self.tool}): {self.error}"


def parse_docstring(doc: str | None) -> tuple[str, dict[str, str]]:
    """Split a docstring into (summary, {param: description}).

    Supports the Google style used throughout this codebase. Unrecognised
    formats degrade to "whole docstring is the summary", which is fine.
    """
    if not doc:
        return "", {}
    lines = inspect.cleandoc(doc).splitlines()

    summary_lines: list[str] = []
    params: dict[str, str] = {}
    in_args = False
    current: str | None = None

    for line in lines:
        if _ARGS_HEADER_RE.match(line):
            in_args = True
            current = None
            continue
        if _SECTION_HEADER_RE.match(line):
            in_args = False
            current = None
            continue
        if in_args:
            match = _ARG_LINE_RE.match(line)
            if match and line[: len(line) - len(line.lstrip())]:
                current = match.group("name").lstrip("*")
                params[current] = match.group("desc").strip()
            elif current and line.strip():
                params[current] = f"{params[current]} {line.strip()}".strip()
            continue
        summary_lines.append(line)

    summary = "\n".join(summary_lines).strip()
    # Keep the first paragraph: tool lists get long and every token is context.
    summary = summary.split("\n\n")[0].strip() if summary else ""
    return summary, params


def build_args_model(fn: ToolFunc, name: str, param_docs: dict[str, str]) -> type[BaseModel]:
    """Derive a Pydantic model for a function's call signature."""
    signature = inspect.signature(fn)
    try:
        hints = get_type_hints(fn)
    except Exception:  # unresolvable forward refs: fall back to raw annotations
        hints = getattr(fn, "__annotations__", {})

    fields: dict[str, tuple[Any, Any]] = {}
    for param_name, param in signature.parameters.items():
        if param_name in {"self", "cls"}:
            continue
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue  # *args/**kwargs cannot be expressed as a JSON schema
        annotation = hints.get(param_name, param.annotation)
        if annotation is inspect.Parameter.empty:
            annotation = str  # untyped params become strings rather than `Any`
        default = ... if param.default is inspect.Parameter.empty else param.default
        fields[param_name] = (
            annotation,
            Field(default, description=param_docs.get(param_name) or None),
        )

    model_name = f"{''.join(p.title() for p in name.split('_'))}Args"
    model = create_model(model_name, **fields)  # type: ignore[call-overload]

    # Build the schema now rather than lazily. With `from __future__ import
    # annotations`, a type that is not resolvable from the function's module
    # globals (typically a class defined inside another function) otherwise
    # fails much later with an opaque Pydantic rebuild error.
    try:
        model.model_json_schema()
    except Exception as exc:
        unresolved = sorted(
            str(hints.get(p, "?")) for p in fields if isinstance(hints.get(p), str)
        )
        raise TypeError(
            f"Cannot build a JSON schema for tool {name!r}: {exc}. "
            f"Unresolved annotations: {unresolved or 'unknown'}. "
            "Tool parameter types must be importable from the function's module "
            "-- define any Pydantic models at module level, not inside another "
            "function."
        ) from exc
    return model


class ToolSpec(BaseModel):
    """A registered tool: metadata, schema, and the callable."""

    model_config = {"arbitrary_types_allowed": True}

    name: str
    description: str
    args_model: type[BaseModel]
    fn: ToolFunc
    is_async: bool
    destructive: bool = False
    parallel_safe: bool = True
    tags: tuple[str, ...] = ()

    def json_schema(self) -> dict[str, Any]:
        schema = self.args_model.model_json_schema()
        schema.pop("title", None)
        schema.setdefault("type", "object")
        schema.setdefault("properties", {})
        # Hermes handles inlined schemas far more reliably than $ref/$defs.
        schema = _inline_refs(schema)
        # Pydantic's auto-generated titles ("Max Bytes") add tokens and no
        # information; on an 8B model that context is better spent elsewhere.
        for prop in schema.get("properties", {}).values():
            if isinstance(prop, dict):
                prop.pop("title", None)
        return schema

    def to_ollama_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.json_schema(),
            },
        }


def _type_label(prop: dict[str, Any]) -> str:
    """Human-readable type for `describe()`, flattening Optional/union schemas."""
    if "type" in prop:
        return str(prop["type"])
    variants = prop.get("anyOf") or prop.get("oneOf") or []
    names = [v.get("type") for v in variants if isinstance(v, dict) and v.get("type")]
    concrete = [n for n in names if n != "null"]
    if concrete:
        return "|".join(dict.fromkeys(concrete))
    return "any"


def _inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve local $refs against $defs and drop the $defs block.

    Small models follow flat schemas much better than ones with indirection.
    Cyclic schemas are left with their $defs intact rather than recursing forever.
    """
    defs = schema.get("$defs") or {}
    if not defs:
        return schema

    def resolve(node: Any, depth: int = 0) -> Any:
        if depth > 12:
            return node
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                target = defs.get(ref.split("/")[-1])
                if isinstance(target, dict):
                    merged = {**resolve(target, depth + 1)}
                    merged.update({k: v for k, v in node.items() if k != "$ref"})
                    return merged
            return {k: resolve(v, depth + 1) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [resolve(v, depth + 1) for v in node]
        return node

    resolved = resolve(schema)
    return resolved if isinstance(resolved, dict) else schema


class ToolRegistry:
    """Holds tool specs and executes them with validation."""

    def __init__(self, *, allow_destructive: bool = False) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self.allow_destructive = allow_destructive

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def register(self, spec: ToolSpec, *, replace: bool = False) -> ToolSpec:
        existing = self._tools.get(spec.name)
        if existing is not None:
            if not replace:
                raise ValueError(f"Tool {spec.name!r} is already registered")
            if existing.fn is not spec.fn:
                # Re-importing a module re-registers the same function, which is
                # harmless. Two *different* functions claiming one name is a
                # silent footgun -- the model would call whichever won.
                logger.warning(
                    "Tool %r was re-registered with a different function "
                    "(%s -> %s); the later definition wins.",
                    spec.name,
                    getattr(existing.fn, "__qualname__", existing.fn),
                    getattr(spec.fn, "__qualname__", spec.fn),
                )
        self._tools[spec.name] = spec
        return spec

    def get(self, name: str) -> ToolSpec:
        try:
            return self._tools[name]
        except KeyError:
            known = ", ".join(sorted(self._tools)) or "(none)"
            raise ToolNotFoundError(
                f"Unknown tool {name!r}. Available tools: {known}"
            ) from None

    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self) -> list[ToolSpec]:
        return [self._tools[n] for n in sorted(self._tools)]

    def subset(self, names: Iterable[str] | None) -> ToolRegistry:
        """A registry view limited to `names`. ``None`` or ``["*"]`` means everything."""
        chosen = list(names) if names is not None else ["*"]
        clone = ToolRegistry(allow_destructive=self.allow_destructive)
        if "*" in chosen:
            clone._tools = dict(self._tools)
            return clone
        for name in chosen:
            clone._tools[name] = self.get(name)
        return clone

    def to_ollama_tools(self) -> list[dict[str, Any]]:
        return [spec.to_ollama_tool() for spec in self.specs()]

    def describe(self) -> str:
        """Compact text listing, used in system prompts as a fallback hint."""
        lines = []
        for spec in self.specs():
            schema = spec.json_schema()
            required = set(schema.get("required", []))
            args = ", ".join(
                f"{k}: {_type_label(v)}{'' if k in required else '?'}"
                for k, v in schema.get("properties", {}).items()
            )
            flag = " [destructive]" if spec.destructive else ""
            lines.append(f"- {spec.name}({args}){flag}: {spec.description}")
        return "\n".join(lines)

    async def execute(self, name: str, arguments: dict[str, Any] | None = None) -> ToolResult:
        """Validate then run a tool. Never raises for model-caused problems."""
        started = time.perf_counter()

        def elapsed() -> float:
            return round((time.perf_counter() - started) * 1000, 2)

        try:
            spec = self.get(name)
        except ToolNotFoundError as exc:
            return ToolResult(tool=name, ok=False, error=str(exc), latency_ms=elapsed())

        if spec.destructive and not self.allow_destructive:
            return ToolResult(
                tool=name,
                ok=False,
                error=str(DestructiveToolBlockedError(name)),
                latency_ms=elapsed(),
            )

        try:
            validated = spec.args_model.model_validate(arguments or {})
        except ValidationError as exc:
            return ToolResult(
                tool=name,
                ok=False,
                error=_format_validation_error(spec, exc),
                latency_ms=elapsed(),
                meta={"validation": True},
            )

        kwargs = {k: v for k, v in validated.model_dump().items()}
        try:
            if spec.is_async:
                raw = await spec.fn(**kwargs)
            else:
                raw = await asyncio.to_thread(spec.fn, **kwargs)
        except Exception as exc:
            # Graceful degradation: a crashing tool is a message, not a dead run.
            return ToolResult(
                tool=name,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=elapsed(),
            )

        return ToolResult(
            tool=name, ok=True, content=stringify_result(raw), latency_ms=elapsed()
        )


def _format_validation_error(spec: ToolSpec, exc: ValidationError) -> str:
    """Turn Pydantic's error list into something a model can act on."""
    problems = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "(root)"
        problems.append(f"{loc}: {err['msg']}")
    schema = json.dumps(spec.json_schema(), separators=(",", ":"))
    return (
        f"Invalid arguments for {spec.name}: " + "; ".join(problems) + ". "
        f"Expected schema: {schema}. Call the tool again with corrected arguments."
    )


def stringify_result(value: Any) -> str:
    """Tool return values reach the model as text; keep the conversion predictable."""
    if value is None:
        return "(no output)"
    if isinstance(value, str):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump_json(indent=2)
    if isinstance(value, (dict, list, tuple, int, float, bool)):
        try:
            return json.dumps(value, indent=2, default=str)
        except (TypeError, ValueError):
            return str(value)
    return str(value)


#: Process-wide default registry, populated by `hermes_agent.tools.builtins`.
GLOBAL_REGISTRY = ToolRegistry()


def tool(
    _fn: ToolFunc | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    destructive: bool = False,
    parallel_safe: bool = True,
    tags: Sequence[str] = (),
    registry: ToolRegistry | None = GLOBAL_REGISTRY,
) -> Any:
    """Register a typed function as a tool.

    Args:
        name: Override the tool name (defaults to the function name).
        description: Override the description (defaults to the docstring summary).
        destructive: Requires `--allow-destructive` to run.
        parallel_safe: False makes the agent run this tool sequentially, after
            the concurrent batch, so writers cannot race each other.
        tags: Free-form labels, useful for building tool subsets in workflows.
        registry: Target registry; pass None to build a spec without registering.
    """

    def decorate(fn: ToolFunc) -> ToolFunc:
        tool_name = name or fn.__name__
        summary, param_docs = parse_docstring(fn.__doc__)
        spec = ToolSpec(
            name=tool_name,
            description=description or summary or f"Tool {tool_name}",
            args_model=build_args_model(fn, tool_name, param_docs),
            fn=fn,
            is_async=inspect.iscoroutinefunction(fn),
            destructive=destructive,
            parallel_safe=parallel_safe,
            tags=tuple(tags),
        )
        if registry is not None:
            registry.register(spec, replace=True)
        fn.tool_spec = spec  # type: ignore[attr-defined]
        return fn

    return decorate(_fn) if _fn is not None else decorate
