"""Ollama transport: retries, error classification, streaming assembly."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from hermes_agent.config import OllamaSettings
from hermes_agent.errors import (
    ModelNotPulledError,
    OllamaHTTPError,
    OllamaTimeoutError,
    OllamaUnavailableError,
)
from hermes_agent.llm.client import Message, OllamaClient

HOST = "http://localhost:11434"


def settings(**kwargs) -> OllamaSettings:
    base = {"host": HOST, "model": "hermes3:8b", "max_retries": 3, "backoff_base_s": 0.0}
    base.update(kwargs)
    return OllamaSettings(**base)


def chat_body(content: str = "hi", **extra) -> dict:
    return {
        "model": "hermes3:8b",
        "message": {"role": "assistant", "content": content},
        "done": True,
        "prompt_eval_count": 12,
        "eval_count": 3,
        "total_duration": 1_500_000_000,
        **extra,
    }


@pytest.fixture
async def client():
    c = OllamaClient(settings())
    yield c
    await c.aclose()


class TestChat:
    @respx.mock
    async def test_basic_response(self, client):
        respx.post(f"{HOST}/api/chat").mock(return_value=httpx.Response(200, json=chat_body()))
        response = await client.chat([Message(role="user", content="hello")])
        assert response.content == "hi"
        assert response.prompt_tokens == 12 and response.completion_tokens == 3
        assert response.total_duration_ms == 1500.0

    @respx.mock
    async def test_request_payload_shape(self, client):
        route = respx.post(f"{HOST}/api/chat").mock(
            return_value=httpx.Response(200, json=chat_body())
        )
        tools = [{"type": "function", "function": {"name": "t", "parameters": {}}}]
        await client.chat([Message(role="user", content="x")], tools=tools)
        sent = json.loads(route.calls[0].request.content)
        assert sent["model"] == "hermes3:8b"
        assert sent["stream"] is False
        assert sent["tools"] == tools
        assert sent["options"]["num_ctx"] == 8192
        assert sent["keep_alive"] == "5m"

    @respx.mock
    async def test_tools_omitted_when_empty(self, client):
        route = respx.post(f"{HOST}/api/chat").mock(
            return_value=httpx.Response(200, json=chat_body())
        )
        await client.chat([Message(role="user", content="x")])
        assert "tools" not in json.loads(route.calls[0].request.content)

    @respx.mock
    async def test_native_tool_calls_parsed(self, client):
        body = chat_body("")
        body["message"]["tool_calls"] = [
            {"function": {"name": "calculator", "arguments": {"expression": "1+1"}}}
        ]
        respx.post(f"{HOST}/api/chat").mock(return_value=httpx.Response(200, json=body))
        response = await client.chat([Message(role="user", content="x")])
        assert response.tool_calls[0].name == "calculator"

    @respx.mock
    async def test_tool_message_carries_its_name(self, client):
        route = respx.post(f"{HOST}/api/chat").mock(
            return_value=httpx.Response(200, json=chat_body())
        )
        await client.chat([Message(role="tool", content="42", tool_name="calculator")])
        sent = json.loads(route.calls[0].request.content)
        assert sent["messages"][0]["tool_name"] == "calculator"

    @respx.mock
    async def test_format_schema_is_forwarded(self, client):
        route = respx.post(f"{HOST}/api/chat").mock(
            return_value=httpx.Response(200, json=chat_body('{"a":1}'))
        )
        schema = {"type": "object", "properties": {"a": {"type": "integer"}}}
        await client.chat([Message(role="user", content="x")], response_format=schema)
        assert json.loads(route.calls[0].request.content)["format"] == schema


class TestErrorClassification:
    @respx.mock
    async def test_daemon_down(self, client):
        respx.post(f"{HOST}/api/chat").mock(side_effect=httpx.ConnectError("refused"))
        with pytest.raises(OllamaUnavailableError, match="ollama serve"):
            await client.chat([Message(role="user", content="x")])

    @respx.mock
    async def test_model_not_pulled(self, client):
        respx.post(f"{HOST}/api/chat").mock(
            return_value=httpx.Response(404, text='{"error":"model \'x\' not found, try pulling it"}')
        )
        with pytest.raises(ModelNotPulledError, match="ollama pull"):
            await client.chat([Message(role="user", content="x")])

    @respx.mock
    async def test_timeout(self, client):
        respx.post(f"{HOST}/api/chat").mock(side_effect=httpx.ReadTimeout("slow"))
        with pytest.raises(OllamaTimeoutError, match="request_timeout_s"):
            await client.chat([Message(role="user", content="x")])

    @respx.mock
    async def test_client_error_surfaces_body(self, client):
        respx.post(f"{HOST}/api/chat").mock(return_value=httpx.Response(400, text="bad request"))
        with pytest.raises(OllamaHTTPError, match="bad request"):
            await client.chat([Message(role="user", content="x")])


class TestRetries:
    @respx.mock
    async def test_server_error_is_retried_then_succeeds(self, client):
        route = respx.post(f"{HOST}/api/chat").mock(
            side_effect=[
                httpx.Response(500, text="boom"),
                httpx.Response(200, json=chat_body("recovered")),
            ]
        )
        response = await client.chat([Message(role="user", content="x")])
        assert response.content == "recovered"
        assert route.call_count == 2

    @respx.mock
    async def test_retries_are_bounded(self, client):
        route = respx.post(f"{HOST}/api/chat").mock(return_value=httpx.Response(503, text="down"))
        with pytest.raises(OllamaHTTPError):
            await client.chat([Message(role="user", content="x")])
        assert route.call_count == 3

    @respx.mock
    async def test_client_errors_are_not_retried(self, client):
        """A 400 fails identically on every attempt; retrying just delays the error."""
        route = respx.post(f"{HOST}/api/chat").mock(return_value=httpx.Response(400, text="nope"))
        with pytest.raises(OllamaHTTPError):
            await client.chat([Message(role="user", content="x")])
        assert route.call_count == 1

    @respx.mock
    async def test_429_is_retried(self, client):
        route = respx.post(f"{HOST}/api/chat").mock(
            side_effect=[httpx.Response(429), httpx.Response(200, json=chat_body())]
        )
        await client.chat([Message(role="user", content="x")])
        assert route.call_count == 2


class TestStreaming:
    @respx.mock
    async def test_deltas_then_final(self, client):
        lines = [
            json.dumps({"message": {"content": "Hel"}, "done": False}),
            json.dumps({"message": {"content": "lo"}, "done": False}),
            json.dumps({**chat_body(""), "message": {"content": ""}}),
        ]
        respx.post(f"{HOST}/api/chat").mock(
            return_value=httpx.Response(200, text="\n".join(lines))
        )
        deltas, final = [], None
        async for chunk in client.chat_stream([Message(role="user", content="x")]):
            if chunk.done:
                final = chunk.response
            else:
                deltas.append(chunk.delta)
        assert deltas == ["Hel", "lo"]
        assert final.content == "Hello"
        assert final.prompt_tokens == 12

    @respx.mock
    async def test_malformed_line_is_skipped(self, client):
        lines = [
            json.dumps({"message": {"content": "a"}, "done": False}),
            "{not json",
            json.dumps({**chat_body(""), "message": {"content": ""}}),
        ]
        respx.post(f"{HOST}/api/chat").mock(
            return_value=httpx.Response(200, text="\n".join(lines))
        )
        deltas = [c.delta async for c in client.chat_stream([Message(role="user", content="x")]) if not c.done]
        assert deltas == ["a"]

    @respx.mock
    async def test_duplicate_tool_calls_are_deduped(self, client):
        call = {"function": {"name": "t", "arguments": {"a": 1}}}
        lines = [
            json.dumps({"message": {"content": "", "tool_calls": [call]}, "done": False}),
            json.dumps({**chat_body(""), "message": {"content": "", "tool_calls": [call]}}),
        ]
        respx.post(f"{HOST}/api/chat").mock(
            return_value=httpx.Response(200, text="\n".join(lines))
        )
        final = None
        async for chunk in client.chat_stream([Message(role="user", content="x")]):
            if chunk.done:
                final = chunk.response
        assert len(final.tool_calls) == 1

    @respx.mock
    async def test_stream_connect_error(self, client):
        respx.post(f"{HOST}/api/chat").mock(side_effect=httpx.ConnectError("refused"))
        with pytest.raises(OllamaUnavailableError):
            async for _ in client.chat_stream([Message(role="user", content="x")]):
                pass


class TestModelsAndEmbeddings:
    @respx.mock
    async def test_ensure_ready_accepts_a_tag_variant(self, client):
        respx.get(f"{HOST}/api/tags").mock(
            return_value=httpx.Response(200, json={"models": [{"name": "hermes3:latest"}]})
        )
        await client.ensure_ready("hermes3:8b")  # same base tag: acceptable

    @respx.mock
    async def test_ensure_ready_rejects_missing_model(self, client):
        respx.get(f"{HOST}/api/tags").mock(
            return_value=httpx.Response(200, json={"models": [{"name": "llama3:8b"}]})
        )
        with pytest.raises(ModelNotPulledError):
            await client.ensure_ready("hermes3:8b")

    @respx.mock
    async def test_embed(self, client):
        respx.post(f"{HOST}/api/embed").mock(
            return_value=httpx.Response(200, json={"embeddings": [[0.1, 0.2]]})
        )
        assert await client.embed(["hello"]) == [[pytest.approx(0.1), pytest.approx(0.2)]]

    @respx.mock
    async def test_embed_falls_back_to_legacy_route(self, client):
        """Older Ollama builds only expose the singular /api/embeddings."""
        respx.post(f"{HOST}/api/embed").mock(return_value=httpx.Response(404, text="not found"))
        respx.post(f"{HOST}/api/embeddings").mock(
            return_value=httpx.Response(200, json={"embedding": [0.5]})
        )
        assert await client.embed(["hello"]) == [[pytest.approx(0.5)]]
