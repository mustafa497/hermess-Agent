"""Shared fixtures. The LLM is mocked everywhere except the `ollama` marker."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from hermes_agent.config import Settings
from hermes_agent.llm.client import ChatResponse, Message, StreamChunk, ToolCall
from hermes_agent.tools import GLOBAL_REGISTRY, set_workspace
from hermes_agent.tools.registry import ToolRegistry
from hermes_agent.trace import TraceRecorder


class ScriptedClient:
    """A stand-in for OllamaClient that replays a fixed list of responses.

    Records what it was sent so tests can assert on the prompt as well as the
    behaviour. Once the script is exhausted it returns a terminal answer, which
    keeps a runaway loop from hanging the suite.
    """

    def __init__(self, script: Sequence[ChatResponse] | None = None) -> None:
        self.script: list[ChatResponse] = list(script or [])
        self.requests: list[dict[str, Any]] = []
        self.embeddings: dict[str, list[float]] = {}

    async def chat(
        self,
        messages: Sequence[Any],
        *,
        tools: Any = None,
        response_format: Any = None,
        model: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> ChatResponse:
        self.requests.append(
            {
                "messages": [
                    m.to_wire() if isinstance(m, Message) else dict(m) for m in messages
                ],
                "tools": list(tools or []),
                "response_format": response_format,
                "model": model,
            }
        )
        if self.script:
            return self.script.pop(0)
        return ChatResponse(content="(script exhausted)", prompt_tokens=10, completion_tokens=5)

    async def chat_stream(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        response = await self.chat(messages, **kwargs)
        for token in response.content.split(" "):
            yield StreamChunk(delta=token + " ")
        yield StreamChunk(done=True, response=response)

    async def embed(
        self, texts: Sequence[str], *, model: str | None = None
    ) -> list[list[float]]:
        # Deterministic pseudo-embeddings: stable across runs, no model needed.
        out = []
        for text in texts:
            vector = [0.0] * 16
            for i, ch in enumerate(text.encode("utf-8")):
                vector[i % 16] += ch / 255.0
            out.append(vector)
        return out

    async def list_models(self) -> list[str]:
        return ["hermes3:8b", "nomic-embed-text"]

    async def ensure_ready(self, model: str | None = None) -> None:
        return None

    async def aclose(self) -> None:
        return None

    # -- helpers for building scripts ------------------------------------

    @staticmethod
    def tool_turn(name: str, arguments: dict[str, Any]) -> ChatResponse:
        return ChatResponse(
            content="",
            tool_calls=[ToolCall(name=name, arguments=arguments)],
            prompt_tokens=100,
            completion_tokens=20,
        )

    @staticmethod
    def answer(text: str) -> ChatResponse:
        return ChatResponse(content=text, prompt_tokens=120, completion_tokens=30)


@pytest.fixture
def workspace(tmp_path):
    """Point the sandbox at a temp dir for the duration of a test."""
    from hermes_agent.tools import sandbox

    previous = sandbox.get_workspace()
    target = tmp_path / "workspace"
    set_workspace(target)
    yield target
    set_workspace(previous)


@pytest.fixture
def settings(tmp_path, workspace) -> Settings:
    s = Settings()
    s.safety.workspace_dir = workspace
    s.observability.runs_dir = tmp_path / "runs"
    s.agent.max_iterations = 6
    return s


@pytest.fixture
def trace(settings) -> TraceRecorder:
    return TraceRecorder(settings.resolved_runs_dir(), enabled=False)


@pytest.fixture
def registry() -> ToolRegistry:
    """Built-in tools, destructive ones enabled for test convenience."""
    reg = GLOBAL_REGISTRY.subset(None)
    reg.allow_destructive = True
    return reg


@pytest.fixture
def client() -> ScriptedClient:
    return ScriptedClient()
