"""Integration tests against a real local Ollama + Hermes.

Skipped automatically when Ollama is unreachable or the model is not pulled, so
`pytest` stays green on a machine that has never run `ollama serve`.

    pytest -m ollama -v            run these
    pytest -m "not ollama"         skip them explicitly

These assert on *capabilities*, not on exact wording: a local 8B model is
non-deterministic, and a test that demands an exact sentence is a test that
fails for no reason. Where output must be exact we force it through a tool or a
JSON schema.
"""

from __future__ import annotations

import asyncio
import os

import pytest
from pydantic import BaseModel

from hermes_agent.agent.loop import Agent
from hermes_agent.config import Settings
from hermes_agent.errors import HermesAgentError
from hermes_agent.llm.client import Message, OllamaClient
from hermes_agent.tools import GLOBAL_REGISTRY
from hermes_agent.trace import TraceRecorder
from hermes_agent.workflows.engine import WorkflowEngine
from hermes_agent.workflows.schema import WorkflowSpec

pytestmark = pytest.mark.ollama

MODEL = os.environ.get("HERMES_TEST_MODEL", os.environ.get("HERMES_MODEL", "hermes3:8b"))
HOST = os.environ.get("HERMES_HOST", "http://localhost:11434")


def _probe() -> str | None:
    """Return a skip reason, or None when the live model is usable."""

    async def check() -> str | None:
        settings = Settings().ollama
        settings.host = HOST
        settings.model = MODEL
        settings.connect_timeout_s = 2.0
        settings.max_retries = 1
        client = OllamaClient(settings)
        try:
            await client.ensure_ready(MODEL)
            return None
        except HermesAgentError as exc:
            return str(exc)
        finally:
            await client.aclose()

    try:
        return asyncio.run(check())
    except Exception as exc:  # pragma: no cover - environment dependent
        return f"could not probe Ollama: {exc}"


SKIP_REASON = _probe()
pytestmark = [
    pytest.mark.ollama,
    pytest.mark.skipif(SKIP_REASON is not None, reason=f"live Ollama unavailable: {SKIP_REASON}"),
]


@pytest.fixture
def live_settings(tmp_path) -> Settings:
    settings = Settings()
    settings.ollama.model = MODEL
    settings.ollama.host = HOST
    settings.ollama.temperature = 0.0   # as deterministic as a local model gets
    settings.ollama.seed = 42
    settings.ollama.request_timeout_s = 300.0
    settings.safety.workspace_dir = tmp_path / "workspace"
    settings.observability.runs_dir = tmp_path / "runs"
    settings.agent.max_iterations = 6
    return settings


@pytest.fixture
async def live_client(live_settings):
    client = OllamaClient(live_settings.ollama)
    yield client
    await client.aclose()


@pytest.fixture
def live_registry(live_settings):
    from hermes_agent.tools import set_workspace

    set_workspace(live_settings.resolved_workspace())
    registry = GLOBAL_REGISTRY.subset(["calculator", "read_file", "write_file", "list_dir"])
    registry.allow_destructive = True
    return registry


class TestLiveClient:
    async def test_plain_chat(self, live_client):
        response = await live_client.chat(
            [Message(role="user", content="Reply with exactly the word: pong")]
        )
        assert response.content.strip()
        assert response.prompt_tokens and response.prompt_tokens > 0
        assert response.completion_tokens and response.completion_tokens > 0

    async def test_streaming_matches_final(self, live_client):
        deltas: list[str] = []
        final = None
        async for chunk in live_client.chat_stream(
            [Message(role="user", content="Count from 1 to 5, digits only.")]
        ):
            if chunk.done:
                final = chunk.response
            else:
                deltas.append(chunk.delta)
        assert deltas, "expected at least one streamed delta"
        assert final is not None
        assert "".join(deltas) == final.content

    async def test_embeddings(self, live_client):
        try:
            vectors = await live_client.embed(["hello world", "goodbye"])
        except HermesAgentError as exc:
            pytest.skip(f"embedding model unavailable: {exc}")
        assert len(vectors) == 2
        assert len(vectors[0]) == len(vectors[1]) > 0


class TestLiveToolCalling:
    async def test_model_calls_the_calculator(self, live_client, live_registry, live_settings):
        """The exact number must come from the tool, not from the model's head."""
        agent = Agent(
            live_client,
            live_registry,
            live_settings,
            trace=TraceRecorder(live_settings.resolved_runs_dir()),
        )
        result = await agent.run(
            "Use the calculator tool to compute 1234 * 5678. Report the exact result."
        )
        tool_calls = [s for s in result.steps if s.kind == "tool_call"]
        assert tool_calls, "model never called a tool"
        assert any(s.name == "calculator" for s in tool_calls)
        assert "7006652" in result.answer.replace(",", "")

    async def test_file_round_trip(self, live_client, live_registry, live_settings):
        agent = Agent(
            live_client,
            live_registry,
            live_settings,
            trace=TraceRecorder(live_settings.resolved_runs_dir()),
        )
        result = await agent.run(
            "Write the exact text 'integration-ok' to a file called probe.txt "
            "using write_file, then read it back with read_file and tell me "
            "what it contains."
        )
        workspace = live_settings.resolved_workspace()
        assert (workspace / "probe.txt").exists()
        assert "integration-ok" in (workspace / "probe.txt").read_text()
        assert result.stop_reason in {"final_answer", "max_iterations"}

    async def test_answers_without_tools_when_unnecessary(
        self, live_client, live_registry, live_settings
    ):
        """A model that calls tools for everything is as broken as one that never does."""
        agent = Agent(
            live_client,
            live_registry,
            live_settings,
            trace=TraceRecorder(live_settings.resolved_runs_dir()),
        )
        result = await agent.run("What is the capital city of France? Answer in one word.")
        assert "paris" in result.answer.lower()

    async def test_recovers_from_a_bad_tool_name(
        self, live_client, live_registry, live_settings
    ):
        agent = Agent(
            live_client,
            live_registry,
            live_settings,
            trace=TraceRecorder(live_settings.resolved_runs_dir()),
        )
        result = await agent.run(
            "First try calling a tool called 'nonexistent_tool'. When that fails, "
            "use the calculator to compute 2+2 and report the answer."
        )
        assert result.answer.strip()
        assert "4" in result.answer


class TestLiveStructuredOutput:
    async def test_schema_is_honoured(self, live_client, live_registry, live_settings):
        class CityFact(BaseModel):
            city: str
            country: str
            population_millions: float

        agent = Agent(
            live_client,
            live_registry,
            live_settings,
            trace=TraceRecorder(live_settings.resolved_runs_dir()),
        )
        result = await agent.run(
            "Give one fact about Tokyo: its country and approximate population in millions.",
            response_model=CityFact,
        )
        assert result.structured is not None
        parsed = CityFact.model_validate(result.structured)
        assert parsed.city and parsed.country
        assert parsed.population_millions > 0


class TestLiveWorkflow:
    async def test_two_step_workflow_threads_context(
        self, live_client, live_registry, live_settings
    ):
        spec = WorkflowSpec.model_validate(
            {
                "name": "live_probe",
                "inputs": {"topic": {"default": "the number seven"}},
                "steps": [
                    {
                        "id": "fact",
                        "system": "You are terse.",
                        "prompt": "State one short fact about {{ inputs.topic }}.",
                        "tools": [],
                        "max_iterations": 2,
                    },
                    {
                        "id": "rephrase",
                        "system": "You rewrite text.",
                        "prompt": "Rewrite this as a single question: {{ steps.fact.answer }}",
                        "tools": [],
                        "max_iterations": 2,
                    },
                ],
            }
        )
        engine = WorkflowEngine(
            live_client,
            live_registry,
            live_settings,
            trace=TraceRecorder(live_settings.resolved_runs_dir()),
        )
        result = await engine.run(spec)
        assert result.completed
        assert len(result.steps) == 2
        assert result.steps[0].answer.strip()
        assert result.final_answer.strip()

    async def test_branch_on_structured_output(
        self, live_client, live_registry, live_settings
    ):
        spec = WorkflowSpec.model_validate(
            {
                "name": "live_branch",
                "steps": [
                    {
                        "id": "classify",
                        "system": "You classify text. Answer only with JSON.",
                        "prompt": "Is the sentence 'I love this' positive? ",
                        "tools": [],
                        "max_iterations": 2,
                        "output": {
                            "schema": {
                                "type": "object",
                                "properties": {"positive": {"type": "boolean"}},
                                "required": ["positive"],
                            }
                        },
                        "next": [
                            {"when": "steps.classify.output.positive == true", "goto": "happy"},
                            {"goto": "sad"},
                        ],
                    },
                    {"id": "happy", "prompt": "Say 'GOOD'.", "tools": [],
                     "max_iterations": 1, "next": [{"goto": "end"}]},
                    {"id": "sad", "prompt": "Say 'BAD'.", "tools": [], "max_iterations": 1},
                ],
            }
        )
        engine = WorkflowEngine(
            live_client,
            live_registry,
            live_settings,
            trace=TraceRecorder(live_settings.resolved_runs_dir()),
        )
        result = await engine.run(spec)
        assert [s.id for s in result.steps] == ["classify", "happy"]


class TestLiveErrorPaths:
    async def test_missing_model_is_reported_clearly(self, live_settings):
        settings = live_settings.ollama.model_copy()
        settings.model = "definitely-not-a-real-model:0b"
        client = OllamaClient(settings)
        try:
            with pytest.raises(HermesAgentError, match="ollama pull"):
                await client.ensure_ready()
        finally:
            await client.aclose()

    async def test_unreachable_host_is_reported_clearly(self, live_settings):
        settings = live_settings.ollama.model_copy()
        settings.host = "http://127.0.0.1:1"
        settings.connect_timeout_s = 1.0
        settings.max_retries = 1
        client = OllamaClient(settings)
        try:
            with pytest.raises(HermesAgentError, match="ollama serve"):
                await client.list_models()
        finally:
            await client.aclose()
