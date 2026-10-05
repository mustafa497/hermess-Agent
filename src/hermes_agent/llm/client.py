"""Async Ollama client: /api/chat (streaming + non-streaming), /api/embed, /api/tags.

Uses httpx directly rather than the `ollama` package so that retry policy,
NDJSON stream handling, and error classification are all visible and testable
here. The distinction the caller cares about most is *why* a call failed:
daemon down, model not pulled, timeout, or a real HTTP error.
"""

from __future__ import annotations

import asyncio
import json
import random
import ssl
import uuid
from collections.abc import AsyncIterator, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import certifi
import httpx
from pydantic import BaseModel, Field

from hermes_agent.config import OllamaSettings
from hermes_agent.errors import (
    ModelNotPulledError,
    OllamaHTTPError,
    OllamaTimeoutError,
    OllamaUnavailableError,
)

Role = Literal["system", "user", "assistant", "tool"]

#: httpx builds a fresh SSL context per client (~0.4s: it loads the CA bundle).
#: We talk to localhost, but the host is configurable, so verification stays on
#: -- we just build the context once and share it.
_SSL_CONTEXT: ssl.SSLContext | None = None


def _shared_ssl_context() -> ssl.SSLContext:
    global _SSL_CONTEXT
    if _SSL_CONTEXT is None:
        _SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
    return _SSL_CONTEXT


class ToolCall(BaseModel):
    """A tool invocation requested by the model."""

    id: str = Field(default_factory=lambda: f"call_{uuid.uuid4().hex[:12]}")
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    source: Literal["native", "tag_fallback"] = "native"
    repaired: bool = False


class Message(BaseModel):
    """One conversation message in Ollama's wire shape."""

    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_name: str | None = None  # set on role="tool" so the model can match results
    name: str | None = None

    def to_wire(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            payload["tool_calls"] = [
                {"function": {"name": c.name, "arguments": c.arguments}} for c in self.tool_calls
            ]
        # Ollama echoes tool results back to the template via `tool_name`; some
        # builds read `name` instead, so send both -- harmless and more portable.
        if self.role == "tool" and self.tool_name:
            payload["tool_name"] = self.tool_name
            payload["name"] = self.tool_name
        elif self.name:
            payload["name"] = self.name
        return payload


class ChatResponse(BaseModel):
    """Non-streaming result of one /api/chat call."""

    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    model: str = ""
    done_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_duration_ms: float | None = None
    raw: dict[str, Any] = Field(default_factory=dict, repr=False)

    def as_message(self) -> Message:
        return Message(role="assistant", content=self.content, tool_calls=list(self.tool_calls))


@dataclass(slots=True)
class StreamChunk:
    """One incremental piece of a streamed response."""

    delta: str = ""
    done: bool = False
    response: ChatResponse | None = None


def _parse_native_tool_calls(message: dict[str, Any]) -> list[ToolCall]:
    """Read `message.tool_calls`, tolerating arguments delivered as a JSON string."""
    out: list[ToolCall] = []
    for entry in message.get("tool_calls") or []:
        if not isinstance(entry, dict):
            continue
        fn = entry.get("function") if isinstance(entry.get("function"), dict) else entry
        name = fn.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        raw_args = fn.get("arguments", {})
        if isinstance(raw_args, str):
            try:
                raw_args = json.loads(raw_args)
            except (json.JSONDecodeError, ValueError):
                raw_args = {}
        if not isinstance(raw_args, dict):
            raw_args = {}
        call_id = entry.get("id") or f"call_{uuid.uuid4().hex[:12]}"
        out.append(ToolCall(id=str(call_id), name=name.strip(), arguments=raw_args))
    return out


def _response_from_payload(payload: dict[str, Any]) -> ChatResponse:
    message = payload.get("message") or {}
    total_ns = payload.get("total_duration")
    return ChatResponse(
        content=message.get("content") or "",
        tool_calls=_parse_native_tool_calls(message),
        model=payload.get("model", ""),
        done_reason=payload.get("done_reason"),
        prompt_tokens=payload.get("prompt_eval_count"),
        completion_tokens=payload.get("eval_count"),
        total_duration_ms=(total_ns / 1e6) if isinstance(total_ns, (int, float)) else None,
        raw=payload,
    )


class OllamaClient:
    """Thin async wrapper over the local Ollama HTTP API."""

    def __init__(
        self,
        settings: OllamaSettings | None = None,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.settings = settings or OllamaSettings()
        timeout = httpx.Timeout(
            self.settings.request_timeout_s,
            connect=self.settings.connect_timeout_s,
        )
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=self.settings.host, timeout=timeout, verify=_shared_ssl_context()
        )

    async def __aenter__(self) -> OllamaClient:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    # -- infrastructure ---------------------------------------------------------

    async def list_models(self) -> list[str]:
        payload = await self._request_json("GET", "/api/tags")
        return [m.get("name", "") for m in payload.get("models", []) if m.get("name")]

    async def ensure_ready(self, model: str | None = None) -> None:
        """Fail fast with an actionable message before burning a whole run.

        Tag matching is prefix-based: `hermes3:8b` should satisfy a request for
        `hermes3` and vice versa for the implicit `:latest` suffix.
        """
        target = model or self.settings.model
        available = await self.list_models()
        base = target.split(":")[0]
        if any(t == target or t.split(":")[0] == base for t in available):
            return
        raise ModelNotPulledError(target, self.settings.host)

    # -- chat -------------------------------------------------------------------

    def _chat_payload(
        self,
        messages: Sequence[Message] | Sequence[dict[str, Any]],
        *,
        tools: Iterable[dict[str, Any]] | None,
        stream: bool,
        response_format: dict[str, Any] | str | None,
        model: str | None,
        options: dict[str, Any] | None,
    ) -> dict[str, Any]:
        wire = [m.to_wire() if isinstance(m, Message) else dict(m) for m in messages]
        payload: dict[str, Any] = {
            "model": model or self.settings.model,
            "messages": wire,
            "stream": stream,
            "keep_alive": self.settings.keep_alive,
            "options": self.settings.options(**(options or {})),
        }
        tool_list = list(tools or [])
        if tool_list:
            payload["tools"] = tool_list
        if response_format is not None:
            payload["format"] = response_format
        return payload

    async def chat(
        self,
        messages: Sequence[Message] | Sequence[dict[str, Any]],
        *,
        tools: Iterable[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | str | None = None,
        model: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> ChatResponse:
        payload = self._chat_payload(
            messages,
            tools=tools,
            stream=False,
            response_format=response_format,
            model=model,
            options=options,
        )
        data = await self._request_json("POST", "/api/chat", json=payload)
        return _response_from_payload(data)

    async def chat_stream(
        self,
        messages: Sequence[Message] | Sequence[dict[str, Any]],
        *,
        tools: Iterable[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | str | None = None,
        model: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Yield deltas as they arrive; the final chunk carries the assembled response.

        Streaming is not retried once bytes have been delivered -- replaying a
        half-emitted answer is worse than surfacing the error.
        """
        payload = self._chat_payload(
            messages,
            tools=tools,
            stream=True,
            response_format=response_format,
            model=model,
            options=options,
        )
        parts: list[str] = []
        tool_calls: list[ToolCall] = []
        final: dict[str, Any] = {}
        try:
            async with self._client.stream("POST", "/api/chat", json=payload) as resp:
                if resp.status_code >= 400:
                    body = (await resp.aread()).decode("utf-8", "replace")
                    self._raise_for_body(resp.status_code, body, payload["model"])
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # ignore keep-alive noise rather than kill the stream
                    if chunk.get("error"):
                        raise OllamaHTTPError(500, str(chunk["error"]))
                    message = chunk.get("message") or {}
                    delta = message.get("content") or ""
                    if delta:
                        parts.append(delta)
                    tool_calls.extend(_parse_native_tool_calls(message))
                    if chunk.get("done"):
                        final = chunk
                        break
                    if delta:
                        yield StreamChunk(delta=delta)
        except httpx.ConnectError as exc:
            raise OllamaUnavailableError(self.settings.host, str(exc)) from exc
        except httpx.TimeoutException as exc:
            raise OllamaTimeoutError(f"Streaming chat timed out: {exc}") from exc

        response = _response_from_payload(final or {})
        response.content = "".join(parts) or response.content
        # Dedupe by (name, arguments): some builds repeat the call on the done frame.
        seen: set[str] = set()
        merged: list[ToolCall] = []
        for call in [*tool_calls, *response.tool_calls]:
            key = f"{call.name}:{json.dumps(call.arguments, sort_keys=True, default=str)}"
            if key not in seen:
                seen.add(key)
                merged.append(call)
        response.tool_calls = merged
        yield StreamChunk(done=True, response=response)

    # -- embeddings ---------------------------------------------------------------

    async def embed(self, texts: Sequence[str], *, model: str | None = None) -> list[list[float]]:
        target = model or self.settings.embed_model
        payload = {"model": target, "input": list(texts)}
        try:
            data = await self._request_json("POST", "/api/embed", json=payload, model_hint=target)
        except (OllamaHTTPError, ModelNotPulledError) as exc:
            # A 404 here is ambiguous: either the model is not pulled, or this
            # is an older Ollama that only exposes the singular /api/embeddings
            # route. Retry on the legacy route and let *its* error stand, which
            # distinguishes the two cases correctly.
            status = getattr(exc, "status_code", 404)
            if status != 404:
                raise
            out: list[list[float]] = []
            for text in texts:
                single = await self._request_json(
                    "POST",
                    "/api/embeddings",
                    json={"model": target, "prompt": text},
                    model_hint=target,
                )
                out.append([float(x) for x in single.get("embedding", [])])
            return out
        return [[float(x) for x in vec] for vec in data.get("embeddings", [])]

    # -- transport ------------------------------------------------------------------

    def _raise_for_body(self, status: int, body: str, model: str) -> None:
        lowered = body.lower()
        if status == 404 and ("not found" in lowered or "pull" in lowered):
            raise ModelNotPulledError(model, self.settings.host)
        raise OllamaHTTPError(status, body)

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        model_hint: str | None = None,
    ) -> dict[str, Any]:
        """Issue a request with exponential backoff + jitter on transient failures.

        Retried: connect errors, timeouts, and 5xx/429. Not retried: 4xx other
        than 429 -- a bad request or a missing model will fail identically on
        every attempt, and retrying just delays a clear error message.
        """
        model = model_hint or (json or {}).get("model") or self.settings.model
        attempts = max(1, self.settings.max_retries)
        last_exc: Exception | None = None

        for attempt in range(attempts):
            try:
                resp = await self._client.request(method, path, json=json)
            except httpx.ConnectError as exc:
                last_exc = OllamaUnavailableError(self.settings.host, str(exc))
            except httpx.TimeoutException as exc:
                last_exc = OllamaTimeoutError(
                    f"{method} {path} timed out after "
                    f"{self.settings.request_timeout_s}s. Large models on CPU can exceed "
                    "this -- raise ollama.request_timeout_s."
                )
                last_exc.__cause__ = exc
            else:
                if resp.status_code < 400:
                    try:
                        payload = resp.json()
                    except ValueError as exc:
                        raise OllamaHTTPError(resp.status_code, resp.text) from exc
                    return payload if isinstance(payload, dict) else {"data": payload}
                if resp.status_code < 500 and resp.status_code != 429:
                    self._raise_for_body(resp.status_code, resp.text, str(model))
                last_exc = OllamaHTTPError(resp.status_code, resp.text)

            if attempt < attempts - 1:
                delay = min(
                    self.settings.backoff_base_s * (2**attempt),
                    self.settings.backoff_max_s,
                )
                await asyncio.sleep(delay * (0.5 + random.random()))

        assert last_exc is not None
        raise last_exc
