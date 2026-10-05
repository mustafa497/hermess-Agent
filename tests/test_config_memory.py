"""Configuration precedence, context-window trimming, and the vector store."""

from __future__ import annotations

import pytest

from hermes_agent.config import Settings, env_overlay, load_settings
from hermes_agent.llm.client import Message, ToolCall
from hermes_agent.memory.short_term import (
    SUMMARY_MARKER,
    ConversationWindow,
    group_into_blocks,
)
from hermes_agent.memory.vector import VectorStore, chunk_text
from hermes_agent.trace import TraceRecorder, read_trace
from tests.conftest import ScriptedClient


class TestConfig:
    def test_defaults(self):
        s = Settings()
        assert s.ollama.model == "hermes3:8b"
        assert s.agent.max_iterations == 10
        assert s.safety.allow_destructive is False

    def test_env_beats_yaml(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text("ollama:\n  model: from-yaml\n  num_ctx: 4096\n")
        s = load_settings(config, environ={"HERMES_OLLAMA__MODEL": "from-env"})
        assert s.ollama.model == "from-env"
        assert s.ollama.num_ctx == 4096  # untouched by env

    def test_overrides_beat_env(self, tmp_path):
        s = load_settings(
            None,
            environ={"HERMES_OLLAMA__MODEL": "from-env"},
            overrides={"ollama.model": "from-cli"},
        )
        assert s.ollama.model == "from-cli"

    def test_flat_aliases(self):
        s = load_settings(None, environ={"HERMES_MODEL": "m", "HERMES_HOST": "http://h:1"})
        assert s.ollama.model == "m" and s.ollama.host == "http://h:1"

    def test_env_values_are_typed(self):
        s = load_settings(
            None,
            environ={
                "HERMES_AGENT__MAX_ITERATIONS": "42",
                "HERMES_SAFETY__ALLOW_DESTRUCTIVE": "true",
                "HERMES_OLLAMA__TEMPERATURE": "0.75",
                "HERMES_OLLAMA__SEED": "7",
            },
        )
        assert s.agent.max_iterations == 42
        assert s.safety.allow_destructive is True
        assert s.ollama.temperature == 0.75
        assert s.ollama.seed == 7

    def test_unknown_flat_var_ignored(self):
        assert env_overlay({}, {"HERMES_NONSENSE": "x"}) == {}

    def test_non_hermes_vars_ignored(self):
        assert env_overlay({}, {"PATH": "/usr/bin"}) == {}

    def test_host_trailing_slash_stripped(self):
        assert Settings.model_validate(
            {"ollama": {"host": "http://localhost:11434/"}}
        ).ollama.host == "http://localhost:11434"

    def test_options_payload(self):
        s = Settings.model_validate({"ollama": {"seed": 5, "num_predict": 100}})
        opts = s.ollama.options()
        assert opts["seed"] == 5 and opts["num_predict"] == 100
        assert Settings().ollama.options().get("seed") is None

    def test_options_overrides(self):
        assert Settings().ollama.options(temperature=0.9)["temperature"] == 0.9

    def test_invalid_value_is_rejected(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Settings.model_validate({"agent": {"max_iterations": 0}})

    def test_missing_config_file_is_fine(self, tmp_path):
        assert load_settings(tmp_path / "nope.yaml", environ={}).ollama.model == "hermes3:8b"


class TestBlocks:
    def test_tool_results_stay_with_their_call(self):
        messages = [
            Message(role="user", content="q"),
            Message(role="assistant", tool_calls=[ToolCall(name="t", arguments={})]),
            Message(role="tool", content="r1", tool_name="t"),
            Message(role="tool", content="r2", tool_name="t"),
            Message(role="assistant", content="done"),
        ]
        blocks = group_into_blocks(messages)
        assert [len(b.messages) for b in blocks] == [1, 3, 1]

    def test_orphan_tool_message(self):
        assert len(group_into_blocks([Message(role="tool", content="r")])) == 1


class TestConversationWindow:
    def make(self, **kwargs) -> ConversationWindow:
        window = ConversationWindow(num_ctx=1000, keep_recent_blocks=2, **kwargs)
        window.set_system("SYSTEM")
        return window

    def test_messages_include_system_first(self):
        window = self.make()
        window.add(Message(role="user", content="hi"))
        assert window.messages()[0].role == "system"

    def test_no_trim_when_small(self):
        window = self.make()
        window.add(Message(role="user", content="short"))
        assert not window.needs_trim()

    async def test_trim_evicts_and_summarizes(self):
        window = self.make()
        for i in range(10):
            window.add(Message(role="user", content=f"message {i} " + "x" * 400))
            window.add(Message(role="assistant", content=f"reply {i} " + "y" * 400))
        assert window.needs_trim()

        client = ScriptedClient([ScriptedClient.answer("Earlier: the user asked things.")])
        evicted = await window.trim(client)

        assert evicted > 0
        rendered = window.messages()
        assert any(SUMMARY_MARKER in m.content for m in rendered)
        assert any("Earlier: the user asked things." in m.content for m in rendered)
        assert len(window.turns) < 20

    async def test_first_user_message_is_preserved(self):
        """Losing the task statement is the fastest way to derail a long run."""
        window = self.make()
        window.add(Message(role="user", content="THE ORIGINAL TASK"))
        for i in range(12):
            window.add(Message(role="assistant", content="z" * 500))
            window.add(Message(role="user", content=f"follow up {i} " + "w" * 400))

        await window.trim(ScriptedClient([ScriptedClient.answer("summary")]))
        assert any("THE ORIGINAL TASK" in m.content for m in window.turns)

    async def test_tool_block_is_not_split(self):
        window = self.make()
        for i in range(8):
            window.add(
                Message(
                    role="assistant",
                    content="x" * 300,
                    tool_calls=[ToolCall(name="t", arguments={"i": i})],
                )
            )
            window.add(Message(role="tool", content="y" * 300, tool_name="t"))

        await window.trim(ScriptedClient([ScriptedClient.answer("summary")]))
        # Every surviving tool message must follow an assistant tool-call message.
        for index, message in enumerate(window.turns):
            if message.role == "tool":
                previous = window.turns[index - 1]
                assert index > 0
                assert previous.role in {"assistant", "tool"}

    async def test_summary_failure_degrades_gracefully(self):
        class Failing(ScriptedClient):
            async def chat(self, *args, **kwargs):
                raise RuntimeError("model down")

        window = self.make()
        for _ in range(10):
            window.add(Message(role="user", content="x" * 500))
        evicted = await window.trim(Failing())
        assert evicted > 0
        assert "no summary was available" in window.messages()[1].content

    async def test_trim_without_a_client(self):
        window = self.make()
        for _ in range(10):
            window.add(Message(role="user", content="x" * 500))
        assert await window.trim(None) > 0

    def test_calibration_tracks_reported_tokens(self):
        window = self.make()
        window.add(Message(role="user", content="x" * 400))
        naive = window.estimated_tokens()
        for _ in range(5):
            window.calibrate(naive * 2)
        assert window.estimated_tokens() > naive


class TestChunking:
    def test_short_text_is_one_chunk(self):
        assert chunk_text("hello") == ["hello"]

    def test_empty(self):
        assert chunk_text("   ") == []

    def test_splits_on_paragraphs(self):
        text = "\n\n".join(f"Paragraph {i}. " + "word " * 80 for i in range(6))
        chunks = chunk_text(text, chunk_chars=600, overlap=50)
        assert len(chunks) > 1
        assert all(len(c) < 1200 for c in chunks)

    def test_oversized_paragraph_is_hard_split(self):
        chunks = chunk_text("x" * 5000, chunk_chars=1000, overlap=100)
        assert len(chunks) > 1


class TestVectorStore:
    async def test_add_search_and_count(self, tmp_path):
        store = VectorStore(tmp_path / "m.sqlite3", ScriptedClient())
        try:
            await store.add(
                ["the cat sat on the mat", "python is a programming language"],
                [{"source": "a.md"}, {"source": "b.md"}],
            )
            assert store.count() == 2
            hits = await store.search("the cat sat on the mat", top_k=1)
            assert len(hits) == 1
            assert hits[0].text == "the cat sat on the mat"
            assert hits[0].metadata["source"] == "a.md"
            assert 0.0 <= hits[0].score <= 1.0001
        finally:
            store.close()

    async def test_empty_store_returns_nothing(self, tmp_path):
        store = VectorStore(tmp_path / "m.sqlite3", ScriptedClient())
        try:
            assert await store.search("anything") == []
        finally:
            store.close()

    async def test_collections_are_isolated(self, tmp_path):
        path = tmp_path / "m.sqlite3"
        client = ScriptedClient()
        a = VectorStore(path, client, collection="a")
        b = VectorStore(path, client, collection="b")
        try:
            await a.add(["only in a"])
            assert a.count() == 1 and b.count() == 0
            assert await b.search("only in a") == []
        finally:
            a.close()
            b.close()

    async def test_persists_across_instances(self, tmp_path):
        path = tmp_path / "m.sqlite3"
        first = VectorStore(path, ScriptedClient())
        await first.add(["remembered"])
        first.close()

        second = VectorStore(path, ScriptedClient())
        try:
            assert second.count() == 1
        finally:
            second.close()

    async def test_clear(self, tmp_path):
        store = VectorStore(tmp_path / "m.sqlite3", ScriptedClient())
        try:
            await store.add(["a", "b"])
            assert store.clear() == 2 and store.count() == 0
        finally:
            store.close()

    async def test_metadata_length_mismatch_rejected(self, tmp_path):
        store = VectorStore(tmp_path / "m.sqlite3", ScriptedClient())
        try:
            with pytest.raises(ValueError, match="same length"):
                await store.add(["a", "b"], [{"x": 1}])
        finally:
            store.close()


class TestTrace:
    def test_jsonl_round_trip(self, tmp_path):
        recorder = TraceRecorder(tmp_path)
        recorder.record("run_start", name="m")
        recorder.record("tool_call", name="t", ok=True, latency_ms=5.0, args={"a": 1})
        records = read_trace(tmp_path, recorder.run_id)
        assert [r["seq"] for r in records] == [1, 2]
        assert records[1]["args"] == {"a": 1}

    def test_timed_records_latency(self, tmp_path):
        recorder = TraceRecorder(tmp_path, enabled=False)
        with recorder.timed("tool_call", name="t") as extra:
            extra["result"] = "done"
        step = recorder.steps[0]
        assert step.ok and step.latency_ms is not None and step.result == "done"

    def test_timed_records_failures_and_reraises(self, tmp_path):
        recorder = TraceRecorder(tmp_path, enabled=False)
        with pytest.raises(ValueError), recorder.timed("tool_call", name="t"):
            raise ValueError("boom")
        assert recorder.steps[0].ok is False
        assert "boom" in recorder.steps[0].error

    def test_totals(self, tmp_path):
        recorder = TraceRecorder(tmp_path, enabled=False)
        recorder.record("llm_call", prompt_tokens=10, completion_tokens=5, latency_ms=100)
        recorder.record("llm_call", prompt_tokens=20, completion_tokens=7, latency_ms=50)
        recorder.record("tool_call", latency_ms=20)
        totals = recorder.totals()
        assert totals["llm_calls"] == 2 and totals["tool_calls"] == 1
        assert totals["prompt_tokens"] == 30 and totals["completion_tokens"] == 12
        assert totals["llm_latency_ms"] == 150

    def test_missing_trace_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_trace(tmp_path, "nope")
