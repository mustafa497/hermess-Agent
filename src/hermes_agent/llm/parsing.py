"""Fallback parsing of Hermes-style tool calls emitted as text.

Why this exists: Ollama's native `tools` path populates `message.tool_calls`,
but small local models routinely (a) emit the raw ChatML tool-call tags as
content anyway, (b) truncate the closing tag, or (c) produce *almost* JSON.
Losing the call in those cases turns a recoverable turn into a dead run, so we
parse the text as a backstop and repair bounded classes of malformation.

Everything here is pure and synchronous -- it is the easiest part of the system
to unit-test, and it is where most local-model flakiness gets absorbed.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from typing import Any

#: <tool_call> ... </tool_call>, tolerating a missing closing tag (truncated output).
TOOL_CALL_TAG_RE = re.compile(
    r"<tool_call>\s*(?P<body>.*?)\s*(?:</tool_call>|$)",
    re.DOTALL | re.IGNORECASE,
)
#: Some fine-tunes wrap calls in <function_call> instead.
ALT_TAG_RE = re.compile(
    r"<(?:function_call|tool)>\s*(?P<body>.*?)\s*(?:</(?:function_call|tool)>|$)",
    re.DOTALL | re.IGNORECASE,
)
FENCE_RE = re.compile(r"```(?:json|tool_call)?\s*(?P<body>.*?)```", re.DOTALL | re.IGNORECASE)

_SMART_QUOTES = {
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
    "«": '"',
    "»": '"',
}
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")
_NAME_KEYS = ("name", "tool", "tool_name")
_ARG_KEYS = ("arguments", "parameters", "args", "input", "kwargs")


@dataclass(slots=True)
class ParsedCall:
    name: str
    arguments: dict[str, Any]
    raw: str
    repaired: bool = False


@dataclass(slots=True)
class ParseFailure:
    raw: str
    reason: str


@dataclass(slots=True)
class ParseResult:
    """Outcome of scanning model text for tool calls."""

    text: str
    """Content with recognised tool-call blocks stripped out."""

    calls: list[ParsedCall] = field(default_factory=list)
    failures: list[ParseFailure] = field(default_factory=list)

    @property
    def found_any(self) -> bool:
        return bool(self.calls or self.failures)


def _balance_brackets(text: str) -> str:
    """Append the closing brackets a truncated generation left off.

    Quote-aware, so braces inside string values do not confuse the counter.
    """
    stack: list[str] = []
    in_string = False
    escaped = False
    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]" and stack and stack[-1] == ("{" if ch == "}" else "["):
            stack.pop()
    out = text
    if in_string:
        out += '"'
    for opener in reversed(stack):
        out += "}" if opener == "{" else "]"
    return out


def repair_json(raw: str) -> tuple[Any | None, bool]:
    """Best-effort JSON recovery.

    Returns ``(value, was_repaired)``; ``(None, False)`` when unrecoverable.
    Only bounded, well-understood repairs are attempted -- fences, smart quotes,
    trailing commas, unbalanced brackets, and Python-literal syntax. Anything
    else is reported as a failure so the model gets told, rather than us
    guessing wrong on its behalf.
    """
    text = raw.strip()
    if not text:
        return None, False

    try:
        return json.loads(text), False
    except (json.JSONDecodeError, ValueError):
        pass

    candidate = text
    fenced = FENCE_RE.search(candidate)
    if fenced:
        candidate = fenced.group("body").strip()

    for bad, good in _SMART_QUOTES.items():
        candidate = candidate.replace(bad, good)

    # Drop any prose before the first opening bracket.
    starts = [i for i in (candidate.find("{"), candidate.find("[")) if i != -1]
    if starts and min(starts) > 0:
        candidate = candidate[min(starts) :]

    candidate = _TRAILING_COMMA_RE.sub(r"\1", candidate)

    for attempt in (candidate, _balance_brackets(candidate)):
        if not attempt:
            continue
        try:
            return json.loads(attempt), True
        except (json.JSONDecodeError, ValueError):
            pass
        try:
            # Handles single quotes and True/False/None -- common when a model
            # writes a Python dict instead of JSON.
            value = ast.literal_eval(attempt)
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            continue
        if isinstance(value, (dict, list)):
            return value, True

    return None, False


def _coerce_arguments(value: Any) -> dict[str, Any] | None:
    """Arguments arrive as a dict or as a JSON string; both occur in the wild."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed, _ = repair_json(value)
        return parsed if isinstance(parsed, dict) else None
    return None


def normalise_call_payload(payload: Any) -> ParsedCall | ParseFailure:
    """Turn a decoded blob into a ParsedCall, accepting the common key spellings."""
    try:
        raw = json.dumps(payload, default=str)[:2000]
    except (TypeError, ValueError):
        raw = str(payload)[:2000]

    if not isinstance(payload, dict):
        return ParseFailure(raw=raw, reason="tool call must be a JSON object")

    # OpenAI-style nesting: {"function": {"name": ..., "arguments": ...}}
    inner = payload.get("function")
    if isinstance(inner, dict):
        payload = {**payload, **inner}
    elif isinstance(inner, str) and inner.strip() and not payload.get("name"):
        payload = {**payload, "name": inner}

    name: str | None = None
    for key in _NAME_KEYS:
        candidate = payload.get(key)
        if isinstance(candidate, str) and candidate.strip():
            name = candidate.strip()
            break
    if not name:
        return ParseFailure(raw=raw, reason="missing a string 'name' field")

    arguments: dict[str, Any] | None = None
    for key in _ARG_KEYS:
        if key in payload:
            arguments = _coerce_arguments(payload[key])
            if arguments is None:
                return ParseFailure(
                    raw=raw,
                    reason=f"'{key}' is not a JSON object and could not be repaired",
                )
            break
    if arguments is None:
        arguments = {}  # a zero-argument tool call is legitimate

    return ParsedCall(name=name, arguments=arguments, raw=raw)


def _consume(
    pattern: re.Pattern[str],
    source: str,
    calls: list[ParsedCall],
    failures: list[ParseFailure],
) -> str:
    spans: list[tuple[int, int]] = []
    for match in pattern.finditer(source):
        body = match.group("body").strip()
        if not body:
            continue
        spans.append(match.span())
        payload, repaired = repair_json(body)
        if payload is None:
            failures.append(ParseFailure(raw=body[:2000], reason="could not decode as JSON"))
            continue
        items = payload if isinstance(payload, list) else [payload]
        for item in items:
            outcome = normalise_call_payload(item)
            if isinstance(outcome, ParsedCall):
                outcome.repaired = repaired
                calls.append(outcome)
            else:
                failures.append(outcome)
    for start, end in reversed(spans):
        source = source[:start] + source[end:]
    return source


def parse_tool_calls(text: str, known_names: set[str] | None = None) -> ParseResult:
    """Scan model text for tool calls, stripping recognised blocks from the content.

    Args:
        text: Raw assistant content.
        known_names: Registered tool names. Used *only* to gate the untagged
            bare-JSON path -- without it, a structured JSON answer that happens
            to contain a "name" field would be misread as a tool call. Calls
            inside explicit tags are returned whatever their name, so the
            registry can reply with a proper "unknown tool" correction.
    """
    if not text:
        return ParseResult(text="")

    calls: list[ParsedCall] = []
    failures: list[ParseFailure] = []

    remainder = _consume(TOOL_CALL_TAG_RE, text, calls, failures)
    if not calls and not failures:
        remainder = _consume(ALT_TAG_RE, remainder, calls, failures)

    # Last resort: the whole message is a bare JSON tool call with no tags.
    if not calls and not failures:
        stripped = remainder.strip()
        if stripped.startswith(("{", "[")) or FENCE_RE.search(stripped):
            payload, repaired = repair_json(stripped)
            items = payload if isinstance(payload, list) else [payload]
            accepted: list[ParsedCall] = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                if not any(k in item for k in (*_NAME_KEYS, "function")):
                    continue  # plain JSON answer, not a tool call
                outcome = normalise_call_payload(item)
                if not isinstance(outcome, ParsedCall):
                    continue
                if known_names is not None and outcome.name not in known_names:
                    continue  # looks like data, not a call
                outcome.repaired = repaired
                accepted.append(outcome)
            if accepted:
                calls.extend(accepted)
                remainder = ""

    return ParseResult(text=remainder.strip(), calls=calls, failures=failures)


def strip_thinking(text: str) -> tuple[str, str | None]:
    """Split out <think>-style blocks so they land in the trace, not the answer."""
    pattern = re.compile(
        r"<(?P<tag>think|thinking|scratchpad|reasoning)>(?P<body>.*?)</(?P=tag)>",
        re.DOTALL | re.IGNORECASE,
    )
    thoughts = [m.group("body").strip() for m in pattern.finditer(text)]
    cleaned = pattern.sub("", text).strip()
    return cleaned, ("\n\n".join(thoughts) if thoughts else None)
