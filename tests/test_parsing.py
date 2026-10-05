"""Tool-call parsing: native, tag fallback, and malformed JSON."""

from __future__ import annotations

import pytest

from hermes_agent.llm.client import _parse_native_tool_calls
from hermes_agent.llm.parsing import (
    normalise_call_payload,
    parse_tool_calls,
    repair_json,
    strip_thinking,
)


class TestNativeParsing:
    def test_standard_shape(self):
        calls = _parse_native_tool_calls(
            {"tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "a"}}}]}
        )
        assert len(calls) == 1
        assert calls[0].name == "read_file"
        assert calls[0].arguments == {"path": "a"}

    def test_arguments_as_json_string(self):
        """Some Ollama builds deliver arguments as a string rather than an object."""
        calls = _parse_native_tool_calls(
            {"tool_calls": [{"function": {"name": "x", "arguments": '{"a": 1}'}}]}
        )
        assert calls[0].arguments == {"a": 1}

    def test_unparseable_arguments_become_empty(self):
        calls = _parse_native_tool_calls(
            {"tool_calls": [{"function": {"name": "x", "arguments": "not json"}}]}
        )
        assert calls[0].arguments == {}

    def test_entry_without_name_is_skipped(self):
        assert _parse_native_tool_calls({"tool_calls": [{"function": {"arguments": {}}}]}) == []

    def test_no_tool_calls_key(self):
        assert _parse_native_tool_calls({"content": "hello"}) == []

    def test_each_call_gets_an_id(self):
        calls = _parse_native_tool_calls(
            {"tool_calls": [{"function": {"name": "a"}}, {"function": {"name": "b"}}]}
        )
        assert len({c.id for c in calls}) == 2


class TestTagFallback:
    def test_well_formed_tag(self):
        result = parse_tool_calls(
            '<tool_call>{"name": "read_file", "arguments": {"path": "a.txt"}}</tool_call>'
        )
        assert [c.name for c in result.calls] == ["read_file"]
        assert result.calls[0].arguments == {"path": "a.txt"}
        assert result.calls[0].repaired is False
        assert result.text == ""

    def test_prose_around_the_call_is_preserved(self):
        result = parse_tool_calls(
            'Let me check.\n<tool_call>{"name":"list_dir","arguments":{}}</tool_call>'
        )
        assert result.text == "Let me check."
        assert result.calls[0].arguments == {}

    def test_multiple_calls_in_one_message(self):
        result = parse_tool_calls(
            '<tool_call>{"name":"a","arguments":{}}</tool_call>'
            '<tool_call>{"name":"b","arguments":{"x":1}}</tool_call>'
        )
        assert [c.name for c in result.calls] == ["a", "b"]

    def test_missing_closing_tag(self):
        """Truncated generation: the model ran out of tokens mid-call."""
        result = parse_tool_calls('<tool_call>{"name": "calculator", "arguments": {"expression": "2+2"}}')
        assert result.calls[0].name == "calculator"

    def test_alternative_tag_name(self):
        result = parse_tool_calls('<function_call>{"name":"a","arguments":{}}</function_call>')
        assert result.calls[0].name == "a"

    def test_json_array_of_calls(self):
        result = parse_tool_calls(
            '<tool_call>[{"name":"a","arguments":{}},{"name":"b","arguments":{}}]</tool_call>'
        )
        assert [c.name for c in result.calls] == ["a", "b"]

    def test_openai_style_nesting(self):
        result = parse_tool_calls(
            '<tool_call>{"function": {"name": "w", "arguments": "{\\"p\\": 1}"}}</tool_call>'
        )
        assert result.calls[0].name == "w"
        assert result.calls[0].arguments == {"p": 1}

    def test_zero_argument_call(self):
        result = parse_tool_calls('<tool_call>{"name": "ping"}</tool_call>')
        assert result.calls[0].arguments == {}

    def test_parameters_key_alias(self):
        result = parse_tool_calls('<tool_call>{"name":"a","parameters":{"x":1}}</tool_call>')
        assert result.calls[0].arguments == {"x": 1}


class TestMalformedJson:
    def test_trailing_comma(self):
        result = parse_tool_calls('<tool_call>{"name": "a", "arguments": {"x": 1,}}</tool_call>')
        assert result.calls[0].arguments == {"x": 1}
        assert result.calls[0].repaired is True

    def test_python_literal_syntax(self):
        result = parse_tool_calls("<tool_call>{'name': 'a', 'arguments': {'x': True}}</tool_call>")
        assert result.calls[0].arguments == {"x": True}

    def test_smart_quotes(self):
        result = parse_tool_calls(
            "<tool_call>{“name”: “a”, “arguments”: {}}</tool_call>"
        )
        assert result.calls[0].name == "a"

    def test_unbalanced_braces_are_closed(self):
        result = parse_tool_calls('<tool_call>{"name": "a", "arguments": {"x": "y"')
        assert result.calls[0].arguments == {"x": "y"}

    def test_fenced_json(self):
        result = parse_tool_calls('```json\n{"name":"a","arguments":{"x":1}}\n```')
        assert result.calls[0].name == "a"

    def test_unrecoverable_json_reports_failure(self):
        result = parse_tool_calls("<tool_call>this is not json at all</tool_call>")
        assert not result.calls
        assert result.failures and "decode" in result.failures[0].reason

    def test_missing_name_reports_failure(self):
        result = parse_tool_calls('<tool_call>{"arguments": {"x": 1}}</tool_call>')
        assert not result.calls
        assert "name" in result.failures[0].reason

    def test_non_object_arguments_report_failure(self):
        result = parse_tool_calls('<tool_call>{"name":"a","arguments": 5}</tool_call>')
        assert not result.calls
        assert result.failures

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ('{"a": 1}', {"a": 1}),
            ('{"a": 1,}', {"a": 1}),
            ("{'a': 1}", {"a": 1}),
            ('{"a": [1, 2', {"a": [1, 2]}),
            ('```json\n{"a": 1}\n```', {"a": 1}),
            ('Sure! Here you go: {"a": 1}', {"a": 1}),
        ],
    )
    def test_repair_json_cases(self, raw, expected):
        value, _ = repair_json(raw)
        assert value == expected

    def test_repair_json_gives_up_cleanly(self):
        assert repair_json("absolutely not json") == (None, False)
        assert repair_json("") == (None, False)

    def test_brace_inside_string_does_not_confuse_balancer(self):
        value, _ = repair_json('{"a": "a { brace"')
        assert value == {"a": "a { brace"}


class TestNoFalsePositives:
    def test_plain_prose_is_not_a_tool_call(self):
        result = parse_tool_calls("The answer is 42. I did not need any tools.")
        assert not result.found_any
        assert result.text.startswith("The answer")

    def test_structured_answer_is_not_a_tool_call(self):
        """A JSON answer containing `name` must not be mistaken for a call."""
        result = parse_tool_calls(
            '{"name": "Alice", "age": 30}', known_names={"read_file", "calculator"}
        )
        assert not result.calls
        assert result.text

    def test_bare_json_matching_a_known_tool_is_a_call(self):
        result = parse_tool_calls(
            '{"name": "calculator", "arguments": {"expression": "1+1"}}',
            known_names={"calculator"},
        )
        assert result.calls[0].name == "calculator"

    def test_prose_mentioning_a_tool_is_not_a_call(self):
        result = parse_tool_calls("I could use read_file here, but I already know the answer.")
        assert not result.found_any


class TestThinking:
    def test_think_block_is_separated(self):
        cleaned, thought = strip_thinking("<think>reasoning here</think>Final answer.")
        assert cleaned == "Final answer."
        assert thought == "reasoning here"

    def test_no_thinking_block(self):
        cleaned, thought = strip_thinking("Just an answer.")
        assert cleaned == "Just an answer."
        assert thought is None


class TestNormalise:
    def test_rejects_non_object(self):
        assert hasattr(normalise_call_payload(["a"]), "reason")

    def test_tool_name_alias(self):
        call = normalise_call_payload({"tool_name": "x", "args": {"a": 1}})
        assert call.name == "x" and call.arguments == {"a": 1}
