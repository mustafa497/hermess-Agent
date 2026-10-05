"""Agent loop behaviour against a scripted LLM: termination, guards, recovery."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from hermes_agent.agent.loop import Agent, call_signature
from hermes_agent.errors import StructuredOutputError
from hermes_agent.llm.client import ChatResponse, ToolCall
from hermes_agent.tools.registry import ToolRegistry, tool
from tests.conftest import ScriptedClient


@pytest.fixture
def calc_registry() -> ToolRegistry:
    reg = ToolRegistry(allow_destructive=True)

    @tool(registry=reg)
    def calculator(expression: str) -> str:
        """Evaluate arithmetic.

        Args:
            expression: The expression.
        """
        from hermes_agent.tools.builtins import safe_eval_math

        return str(safe_eval_math(expression))

    @tool(registry=reg)
    def echo(text: str) -> str:
        """Echo text back.

        Args:
            text: What to echo.
        """
        return text

    return reg


def make_agent(client, registry, settings, trace, **kwargs) -> Agent:
    return Agent(client, registry, settings, trace=trace, **kwargs)


class TestTermination:
    async def test_answers_without_tools(self, calc_registry, settings, trace):
        client = ScriptedClient([ScriptedClient.answer("The answer is 4.")])
        result = await make_agent(client, calc_registry, settings, trace).run("2+2?")
        assert result.stop_reason == "final_answer"
        assert result.answer == "The answer is 4."
        assert result.iterations == 1

    async def test_tool_then_answer(self, calc_registry, settings, trace):
        client = ScriptedClient(
            [
                ScriptedClient.tool_turn("calculator", {"expression": "2+2"}),
                ScriptedClient.answer("It is 4."),
            ]
        )
        result = await make_agent(client, calc_registry, settings, trace).run("2+2?")
        assert result.stop_reason == "final_answer"
        assert result.iterations == 2
        tool_steps = [s for s in result.steps if s.kind == "tool_call"]
        assert len(tool_steps) == 1 and tool_steps[0].ok

    async def test_tool_result_is_sent_back_with_its_name(self, calc_registry, settings, trace):
        client = ScriptedClient(
            [
                ScriptedClient.tool_turn("calculator", {"expression": "2+2"}),
                ScriptedClient.answer("4"),
            ]
        )
        await make_agent(client, calc_registry, settings, trace).run("2+2?")
        second = client.requests[1]["messages"]
        tool_messages = [m for m in second if m["role"] == "tool"]
        assert len(tool_messages) == 1
        assert tool_messages[0]["tool_name"] == "calculator"
        assert tool_messages[0]["content"] == "4"

    async def test_max_iterations_forces_a_final_answer(self, calc_registry, settings, trace):
        settings.agent.max_iterations = 3
        script = [
            ScriptedClient.tool_turn("calculator", {"expression": f"{i}+1"}) for i in range(5)
        ]
        script.append(ScriptedClient.answer("Ran out of budget, here is what I found."))
        # The forced turn happens after the loop, so put the answer last.
        client = ScriptedClient(script[:3] + [ScriptedClient.answer("Best effort.")])
        result = await make_agent(client, calc_registry, settings, trace).run("go")
        assert result.stop_reason == "max_iterations"
        assert result.iterations == 3
        assert result.answer == "Best effort."

    async def test_forced_turn_withholds_tools(self, calc_registry, settings, trace):
        settings.agent.max_iterations = 1
        client = ScriptedClient(
            [
                ScriptedClient.tool_turn("calculator", {"expression": "1+1"}),
                ScriptedClient.answer("Done."),
            ]
        )
        await make_agent(client, calc_registry, settings, trace).run("go")
        assert client.requests[0]["tools"]      # normal turn has tools
        assert client.requests[-1]["tools"] == []  # forced turn does not


class TestLoopDetection:
    async def test_identical_calls_trip_the_guard(self, calc_registry, settings, trace):
        settings.agent.loop_detection_threshold = 3
        client = ScriptedClient(
            [ScriptedClient.tool_turn("calculator", {"expression": "1+1"})] * 4
            + [ScriptedClient.answer("I was stuck.")]
        )
        result = await make_agent(client, calc_registry, settings, trace).run("go")
        assert result.stop_reason == "loop_detected"
        assert any(s.kind == "loop_detected" for s in result.steps)
        # The third identical call is never executed.
        assert len([s for s in result.steps if s.kind == "tool_call"]) == 2

    async def test_different_arguments_do_not_trip_it(self, calc_registry, settings, trace):
        client = ScriptedClient(
            [
                ScriptedClient.tool_turn("calculator", {"expression": f"{i}+1"})
                for i in range(4)
            ]
            + [ScriptedClient.answer("Done.")]
        )
        result = await make_agent(client, calc_registry, settings, trace).run("go")
        assert result.stop_reason == "final_answer"

    async def test_signature_ignores_key_order(self):
        a = ToolCall(name="t", arguments={"a": 1, "b": 2})
        b = ToolCall(name="t", arguments={"b": 2, "a": 1})
        assert call_signature(a) == call_signature(b)

    async def test_tool_calls_per_turn_are_capped(self, calc_registry, settings, trace):
        settings.agent.max_tool_calls_per_turn = 2
        client = ScriptedClient(
            [
                ChatResponse(
                    content="",
                    tool_calls=[
                        ToolCall(name="echo", arguments={"text": str(i)}) for i in range(6)
                    ],
                ),
                ScriptedClient.answer("Done."),
            ]
        )
        result = await make_agent(client, calc_registry, settings, trace).run("go")
        assert len([s for s in result.steps if s.kind == "tool_call"]) == 2


class TestErrorRecovery:
    async def test_validation_error_is_fed_back(self, calc_registry, settings, trace):
        client = ScriptedClient(
            [
                ScriptedClient.tool_turn("calculator", {"wrong_arg": "2+2"}),
                ScriptedClient.tool_turn("calculator", {"expression": "2+2"}),
                ScriptedClient.answer("4"),
            ]
        )
        result = await make_agent(client, calc_registry, settings, trace).run("go")
        assert result.stop_reason == "final_answer"
        failed = [s for s in result.steps if s.kind == "tool_call" and not s.ok]
        assert len(failed) == 1
        # The error reached the model as a tool message.
        assert any(
            "Invalid arguments" in m.get("content", "")
            for m in client.requests[1]["messages"]
            if m["role"] == "tool"
        )

    async def test_unknown_tool_is_corrected_not_dropped(self, calc_registry, settings, trace):
        client = ScriptedClient(
            [
                ChatResponse(content='<tool_call>{"name":"nonexistent","arguments":{}}</tool_call>'),
                ScriptedClient.answer("Sorry, I cannot."),
            ]
        )
        result = await make_agent(client, calc_registry, settings, trace).run("go")
        assert result.answer == "Sorry, I cannot."
        assert any("Unknown tool" in (s.error or "") for s in result.steps)

    async def test_unparseable_call_prompts_a_retry(self, calc_registry, settings, trace):
        client = ScriptedClient(
            [
                ChatResponse(content="<tool_call>{broken</tool_call>"),
                ScriptedClient.tool_turn("calculator", {"expression": "1+1"}),
                ScriptedClient.answer("2"),
            ]
        )
        result = await make_agent(client, calc_registry, settings, trace).run("go")
        assert result.stop_reason == "final_answer"
        assert any(
            "could not be parsed" in m.get("content", "")
            for m in client.requests[1]["messages"]
        )

    async def test_never_returns_an_empty_answer(self, calc_registry, settings, trace):
        """A model that emits nothing must still produce an explanation."""
        client = ScriptedClient([ChatResponse(content="   ")])
        result = await make_agent(client, calc_registry, settings, trace).run("go")
        assert result.stop_reason == "no_progress"
        assert result.answer.strip()

    async def test_crashing_tool_does_not_kill_the_run(self, settings, trace):
        reg = ToolRegistry(allow_destructive=True)

        @tool(registry=reg)
        def boom() -> str:
            """Always fails."""
            raise RuntimeError("kaboom")

        client = ScriptedClient(
            [ScriptedClient.tool_turn("boom", {}), ScriptedClient.answer("It failed.")]
        )
        result = await make_agent(client, reg, settings, trace).run("go")
        assert result.stop_reason == "final_answer"
        assert any("kaboom" in (s.error or "") for s in result.steps)


class TestTagFallback:
    async def test_recovers_a_tag_emitted_as_text(self, calc_registry, settings, trace):
        client = ScriptedClient(
            [
                ChatResponse(
                    content='Let me compute.\n<tool_call>{"name":"calculator",'
                    '"arguments":{"expression":"6*7"}}</tool_call>'
                ),
                ScriptedClient.answer("42"),
            ]
        )
        result = await make_agent(client, calc_registry, settings, trace).run("go")
        assert result.answer == "42"
        fallback = [s for s in result.steps if s.kind == "parse_fallback"]
        assert fallback and fallback[0].meta["recovered"] == ["calculator"]

    async def test_can_be_disabled(self, calc_registry, settings, trace):
        settings.agent.enable_tag_fallback = False
        text = '<tool_call>{"name":"calculator","arguments":{"expression":"1"}}</tool_call>'
        client = ScriptedClient([ChatResponse(content=text)])
        result = await make_agent(client, calc_registry, settings, trace).run("go")
        assert not [s for s in result.steps if s.kind == "tool_call"]
        assert result.answer == text

    async def test_repair_is_flagged_in_the_trace(self, calc_registry, settings, trace):
        client = ScriptedClient(
            [
                ChatResponse(content="<tool_call>{'name':'echo','arguments':{'text':'hi'}}</tool_call>"),
                ScriptedClient.answer("done"),
            ]
        )
        result = await make_agent(client, calc_registry, settings, trace).run("go")
        assert [s for s in result.steps if s.kind == "parse_fallback"][0].meta["repaired"] == [
            "echo"
        ]


class TestParallelExecution:
    async def test_safe_tools_run_concurrently(self, settings, trace):
        import asyncio

        order: list[str] = []
        reg = ToolRegistry(allow_destructive=True)

        @tool(registry=reg)
        async def slow(label: str) -> str:
            """Sleeps.

            Args:
                label: Marker.
            """
            await asyncio.sleep(0.05)
            order.append(label)
            return label

        @tool(registry=reg, parallel_safe=False)
        async def writer(label: str) -> str:
            """Serialised.

            Args:
                label: Marker.
            """
            order.append(f"serial:{label}")
            return label

        client = ScriptedClient(
            [
                ChatResponse(
                    content="",
                    tool_calls=[
                        ToolCall(name="slow", arguments={"label": "a"}),
                        ToolCall(name="slow", arguments={"label": "b"}),
                        ToolCall(name="writer", arguments={"label": "w"}),
                    ],
                ),
                ScriptedClient.answer("done"),
            ]
        )
        result = await make_agent(client, reg, settings, trace).run("go")
        assert result.stop_reason == "final_answer"
        # The serial tool always runs after the concurrent batch.
        assert order[-1] == "serial:w"

    async def test_results_keep_call_order(self, calc_registry, settings, trace):
        client = ScriptedClient(
            [
                ChatResponse(
                    content="",
                    tool_calls=[
                        ToolCall(name="echo", arguments={"text": "first"}),
                        ToolCall(name="echo", arguments={"text": "second"}),
                    ],
                ),
                ScriptedClient.answer("done"),
            ]
        )
        await make_agent(client, calc_registry, settings, trace).run("go")
        tool_messages = [m for m in client.requests[1]["messages"] if m["role"] == "tool"]
        assert [m["content"] for m in tool_messages] == ["first", "second"]


class TestStructuredOutput:
    async def test_valid_output_is_parsed(self, calc_registry, settings, trace):
        class Answer(BaseModel):
            value: int
            unit: str

        client = ScriptedClient(
            [
                ScriptedClient.answer("It is 4 metres."),
                ChatResponse(content='{"value": 4, "unit": "metres"}'),
            ]
        )
        result = await make_agent(client, calc_registry, settings, trace).run(
            "go", response_model=Answer
        )
        assert result.structured == {"value": 4, "unit": "metres"}
        assert client.requests[-1]["response_format"]["properties"]["value"]["type"] == "integer"

    async def test_one_repair_retry_then_success(self, calc_registry, settings, trace):
        class Answer(BaseModel):
            value: int

        client = ScriptedClient(
            [
                ScriptedClient.answer("four"),
                ChatResponse(content='{"value": "not a number"}'),
                ChatResponse(content='{"value": 4}'),
            ]
        )
        result = await make_agent(client, calc_registry, settings, trace).run(
            "go", response_model=Answer
        )
        assert result.structured == {"value": 4}
        repair_prompt = client.requests[-1]["messages"][-1]["content"]
        assert "did not match the required JSON schema" in repair_prompt

    async def test_gives_up_after_one_retry(self, calc_registry, settings, trace):
        class Answer(BaseModel):
            value: int

        client = ScriptedClient(
            [
                ScriptedClient.answer("hmm"),
                ChatResponse(content="still not json"),
                ChatResponse(content="nope"),
            ]
        )
        with pytest.raises(StructuredOutputError):
            await make_agent(client, calc_registry, settings, trace).run(
                "go", response_model=Answer
            )

    async def test_markdown_wrapped_json_is_repaired(self, calc_registry, settings, trace):
        class Answer(BaseModel):
            value: int

        client = ScriptedClient(
            [
                ScriptedClient.answer("four"),
                ChatResponse(content='```json\n{"value": 4}\n```'),
            ]
        )
        result = await make_agent(client, calc_registry, settings, trace).run(
            "go", response_model=Answer
        )
        assert result.structured == {"value": 4}


class TestTracing:
    async def test_run_boundaries_and_usage(self, calc_registry, settings, trace):
        client = ScriptedClient(
            [
                ScriptedClient.tool_turn("calculator", {"expression": "2+2"}),
                ScriptedClient.answer("4"),
            ]
        )
        result = await make_agent(client, calc_registry, settings, trace).run("go")
        kinds = [s.kind for s in result.steps]
        assert kinds[0] == "run_start" and kinds[-1] == "run_end"
        assert result.usage["llm_calls"] == 2
        assert result.usage["tool_calls"] == 1
        assert result.usage["prompt_tokens"] > 0

    async def test_tool_args_and_latency_recorded(self, calc_registry, settings, trace):
        client = ScriptedClient(
            [
                ScriptedClient.tool_turn("calculator", {"expression": "2+2"}),
                ScriptedClient.answer("4"),
            ]
        )
        result = await make_agent(client, calc_registry, settings, trace).run("go")
        step = next(s for s in result.steps if s.kind == "tool_call")
        assert step.args == {"expression": "2+2"}
        assert step.latency_ms is not None and step.result == "4"

    async def test_jsonl_is_written(self, calc_registry, settings):
        from hermes_agent.trace import TraceRecorder, read_trace

        recorder = TraceRecorder(settings.resolved_runs_dir())
        client = ScriptedClient([ScriptedClient.answer("hi")])
        await make_agent(client, calc_registry, settings, recorder).run("go")
        records = read_trace(settings.resolved_runs_dir(), recorder.run_id)
        assert records[0]["kind"] == "run_start"
        assert all(r["run_id"] == recorder.run_id for r in records)


class TestSystemPrompt:
    async def test_tool_hint_is_included(self, calc_registry, settings, trace):
        client = ScriptedClient([ScriptedClient.answer("hi")])
        await make_agent(client, calc_registry, settings, trace).run("go")
        system = client.requests[0]["messages"][0]["content"]
        assert "calculator(expression: string)" in system
        assert "untrusted data" in system

    async def test_custom_prompt_replaces_the_base(self, calc_registry, settings, trace):
        client = ScriptedClient([ScriptedClient.answer("hi")])
        await make_agent(
            client, calc_registry, settings, trace, system_prompt="You are a pirate."
        ).run("go")
        assert client.requests[0]["messages"][0]["content"].startswith("You are a pirate.")

    async def test_streaming_delivers_deltas(self, calc_registry, settings, trace):
        chunks: list[str] = []
        client = ScriptedClient([ScriptedClient.answer("hello there friend")])
        result = await make_agent(client, calc_registry, settings, trace).run(
            "go", on_delta=lambda c: chunks.append(c)
        )
        assert "".join(chunks).strip() == "hello there friend"
        assert result.answer == "hello there friend"
