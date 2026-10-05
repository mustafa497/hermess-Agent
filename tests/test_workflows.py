"""Workflow parsing, expressions, schema validation, and engine execution."""

from __future__ import annotations

import pytest

from hermes_agent.errors import WorkflowError
from hermes_agent.tools.registry import ToolRegistry, tool
from hermes_agent.workflows import jsonschema_mini
from hermes_agent.workflows.engine import WorkflowEngine
from hermes_agent.workflows.expressions import (
    ExpressionError,
    evaluate,
    render,
    render_any,
    resolve_path,
)
from hermes_agent.workflows.patterns import plan_execute_critique
from hermes_agent.workflows.schema import (
    WorkflowSpec,
    find_workflow,
    list_builtin_workflows,
    load_workflow,
)
from tests.conftest import ScriptedClient

CONTEXT = {
    "inputs": {"folder": "docs", "max_fixes": 3, "flag": True},
    "steps": {
        "a": {"answer": "hello", "output": {"count": 0, "items": ["x", "y"], "ok": False}},
        "b": {"answer": "world", "output": {"count": 5, "ok": True}},
    },
    "state": {"visits": {"a": 1, "b": 4}, "step_count": 5},
}


class TestExpressions:
    @pytest.mark.parametrize(
        "path,expected",
        [
            ("inputs.folder", "docs"),
            ("steps.a.output.count", 0),
            ("steps.a.output.items", ["x", "y"]),
            ("state.visits.b", 4),
            ("steps.missing.output.x", None),
            ("steps.a.output.items.0", "x"),
        ],
    )
    def test_resolve_path(self, path, expected):
        assert resolve_path(path, CONTEXT) == expected

    @pytest.mark.parametrize(
        "expression,expected",
        [
            ("steps.a.output.count == 0", True),
            ("steps.a.output.count != 0", False),
            ("steps.b.output.count > inputs.max_fixes", True),
            ("state.visits.b > inputs.max_fixes", True),
            ("state.visits.a > inputs.max_fixes", False),
            ("steps.b.output.ok == true", True),
            ("steps.a.output.ok == false", True),
            ("not steps.a.output.ok", True),
            ("steps.a.output.ok and steps.b.output.ok", False),
            ("steps.a.output.ok or steps.b.output.ok", True),
            ("'x' in steps.a.output.items", True),
            ("'z' in steps.a.output.items", False),
            ("inputs.folder == 'docs'", True),
            ("steps.missing.output.x == 1", False),
            ("steps.missing.output.x == null", True),
            ("", True),
        ],
    )
    def test_evaluate(self, expression, expected):
        assert evaluate(expression, CONTEXT) is expected

    def test_string_number_comparison_is_tolerated(self):
        """Model output often stringifies numbers; a branch should still work."""
        context = {"steps": {"a": {"output": {"n": "5"}}}}
        assert evaluate("steps.a.output.n > 3", context) is True

    @pytest.mark.parametrize(
        "expression",
        [
            "__import__('os').system('ls')",
            "open('/etc/passwd')",
            "1 if True else 2",
            "[x for x in range(3)]",
            "lambda: 1",
            "a := 1",
        ],
    )
    def test_code_execution_is_rejected(self, expression):
        with pytest.raises(ExpressionError):
            evaluate(expression, CONTEXT)

    def test_syntax_error_is_reported(self):
        with pytest.raises(ExpressionError, match="Invalid condition"):
            evaluate("steps.a ==", CONTEXT)


class TestTemplating:
    def test_substitution(self):
        assert render("Folder: {{ inputs.folder }}", CONTEXT) == "Folder: docs"

    def test_missing_path_is_empty(self):
        assert render("[{{ steps.nope.answer }}]", CONTEXT) == "[]"

    def test_structures_render_as_json(self):
        assert '"x"' in render("{{ steps.a.output.items }}", CONTEXT)

    def test_whitespace_tolerance(self):
        assert render("{{inputs.folder}} {{  inputs.folder  }}", CONTEXT) == "docs docs"

    def test_render_any_walks_containers(self):
        out = render_any({"a": ["{{ inputs.folder }}"], "b": 3}, CONTEXT)
        assert out == {"a": ["docs"], "b": 3}

    def test_template_cannot_execute(self):
        """Braces containing code are just an unresolvable path, not an eval."""
        assert render("{{ __import__('os') }}", CONTEXT) == ""


class TestJsonSchemaMini:
    SCHEMA = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "minLength": 1},
            "count": {"type": "integer", "minimum": 0},
            "tags": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "mode": {"enum": ["a", "b"]},
        },
        "required": ["name", "count"],
    }

    def test_valid(self):
        assert jsonschema_mini.validate({"name": "x", "count": 1}, self.SCHEMA) == []

    def test_missing_required(self):
        errors = jsonschema_mini.validate({"name": "x"}, self.SCHEMA)
        assert errors == ["$.count: required property is missing"]

    def test_wrong_types(self):
        errors = jsonschema_mini.validate({"name": 1, "count": "x"}, self.SCHEMA)
        assert len(errors) == 2

    def test_bool_is_not_an_integer(self):
        assert jsonschema_mini.validate({"name": "x", "count": True}, self.SCHEMA)

    def test_nested_array_items(self):
        errors = jsonschema_mini.validate(
            {"name": "x", "count": 1, "tags": ["a", 2]}, self.SCHEMA
        )
        assert errors == ["$.tags[1]: expected type string, got integer"]

    def test_enum(self):
        assert jsonschema_mini.validate({"name": "x", "count": 1, "mode": "z"}, self.SCHEMA)

    def test_bounds(self):
        assert jsonschema_mini.validate({"name": "x", "count": -1}, self.SCHEMA)
        assert jsonschema_mini.validate({"name": "", "count": 1}, self.SCHEMA)
        assert jsonschema_mini.validate({"name": "x", "count": 1, "tags": []}, self.SCHEMA)

    def test_empty_schema_accepts_anything(self):
        assert jsonschema_mini.validate({"whatever": 1}, {}) == []


class TestWorkflowParsing:
    def test_bundled_workflows_load(self):
        names = [n for n, _ in list_builtin_workflows()]
        assert "doc_research" in names and "code_fix_loop" in names

    def test_bundled_graphs_are_valid(self):
        for name, _ in list_builtin_workflows():
            spec = find_workflow(name)
            assert spec.steps

    def test_unknown_goto_is_rejected(self):
        with pytest.raises(WorkflowError, match="unknown target"):
            load_workflow(
                "name: w\nsteps:\n  - id: a\n    prompt: x\n    next:\n      - goto: nowhere\n"
            )

    def test_duplicate_ids_rejected(self):
        with pytest.raises(WorkflowError, match="Duplicate step ids"):
            load_workflow("name: w\nsteps:\n  - id: a\n    prompt: x\n  - id: a\n    prompt: y\n")

    def test_end_is_reserved(self):
        with pytest.raises(WorkflowError, match="reserved"):
            load_workflow("name: w\nsteps:\n  - id: end\n    prompt: x\n")

    def test_missing_required_input(self):
        spec = load_workflow(
            "name: w\ninputs:\n  t:\n    required: true\nsteps:\n  - id: a\n    prompt: x\n"
        )
        with pytest.raises(WorkflowError, match="missing required inputs"):
            spec.resolve_inputs({})
        assert spec.resolve_inputs({"t": "v"})["t"] == "v"

    def test_defaults_applied(self):
        spec = load_workflow(
            "name: w\ninputs:\n  t:\n    default: d\nsteps:\n  - id: a\n    prompt: x\n"
        )
        assert spec.resolve_inputs({})["t"] == "d"
        assert spec.resolve_inputs({"t": "override"})["t"] == "override"

    def test_find_workflow_reports_alternatives(self):
        with pytest.raises(WorkflowError, match="Bundled workflows"):
            find_workflow("no_such_workflow")

    def test_pattern_builder_bounds_the_loop(self):
        spec = plan_execute_critique(max_revisions=2)
        critique = spec.step("critique")
        assert "state.visits.critique < 3" in critique.next[0].when
        assert critique.next[0].goto == "execute"


def simple_spec(**kwargs) -> WorkflowSpec:
    data = {
        "name": "t",
        "inputs": {"topic": {"default": "cats"}},
        "steps": [
            {"id": "one", "prompt": "About {{ inputs.topic }}", "tools": []},
            {"id": "two", "prompt": "Then {{ steps.one.answer }}", "tools": []},
        ],
    }
    data.update(kwargs)
    return WorkflowSpec.model_validate(data)


@pytest.fixture
def empty_registry() -> ToolRegistry:
    return ToolRegistry(allow_destructive=True)


class TestEngine:
    async def test_sequential_execution_threads_context(
        self, empty_registry, settings, trace
    ):
        client = ScriptedClient(
            [ScriptedClient.answer("first result"), ScriptedClient.answer("second result")]
        )
        engine = WorkflowEngine(client, empty_registry, settings, trace=trace)
        result = await engine.run(simple_spec())

        assert [s.id for s in result.steps] == ["one", "two"]
        assert result.final_answer == "second result"
        assert result.completed
        # Step one's prompt saw the input; step two's saw step one's answer.
        assert "About cats" in client.requests[0]["messages"][-1]["content"]
        assert "Then first result" in client.requests[1]["messages"][-1]["content"]

    async def test_conditional_branch_taken(self, empty_registry, settings, trace):
        spec = WorkflowSpec.model_validate(
            {
                "name": "t",
                "steps": [
                    {
                        "id": "check",
                        "prompt": "p",
                        "tools": [],
                        "output": {
                            "schema": {
                                "type": "object",
                                "properties": {"found": {"type": "boolean"}},
                                "required": ["found"],
                            }
                        },
                        "next": [
                            {"when": "steps.check.output.found == false", "goto": "empty"},
                            {"goto": "full"},
                        ],
                    },
                    {"id": "empty", "prompt": "nothing", "tools": [],
                     "next": [{"goto": "end"}]},
                    {"id": "full", "prompt": "something", "tools": []},
                ],
            }
        )
        client = ScriptedClient(
            [
                ScriptedClient.answer("checked"),
                ScriptedClient.answer('{"found": false}'),
                ScriptedClient.answer("nothing found"),
            ]
        )
        engine = WorkflowEngine(client, empty_registry, settings, trace=trace)
        result = await engine.run(spec)
        assert [s.id for s in result.steps] == ["check", "empty"]

    async def test_loop_with_visit_guard_terminates(self, empty_registry, settings, trace):
        spec = WorkflowSpec.model_validate(
            {
                "name": "t",
                "max_steps": 20,
                "steps": [
                    {
                        "id": "work",
                        "prompt": "do",
                        "tools": [],
                        "next": [
                            {"when": "state.visits.work < 3", "goto": "work"},
                            {"goto": "done"},
                        ],
                    },
                    {"id": "done", "prompt": "finish", "tools": []},
                ],
            }
        )
        client = ScriptedClient([ScriptedClient.answer(f"pass {i}") for i in range(10)])
        engine = WorkflowEngine(client, empty_registry, settings, trace=trace)
        result = await engine.run(spec)
        assert [s.id for s in result.steps] == ["work", "work", "work", "done"]
        assert result.completed

    async def test_runaway_loop_hits_the_step_budget(self, empty_registry, settings, trace):
        spec = WorkflowSpec.model_validate(
            {
                "name": "t",
                "max_steps": 4,
                "steps": [
                    {"id": "a", "prompt": "p", "tools": [], "next": [{"goto": "a"}]}
                ],
            }
        )
        client = ScriptedClient([ScriptedClient.answer("again") for _ in range(20)])
        engine = WorkflowEngine(client, empty_registry, settings, trace=trace)
        result = await engine.run(spec)
        assert not result.completed
        assert "step budget exhausted" in result.stopped_reason
        assert len(result.steps) == 4

    async def test_step_tool_subset_is_enforced(self, settings, trace):
        reg = ToolRegistry(allow_destructive=True)

        @tool(registry=reg)
        def allowed() -> str:
            """Allowed."""
            return "ok"

        @tool(registry=reg)
        def forbidden() -> str:
            """Forbidden."""
            return "no"

        spec = WorkflowSpec.model_validate(
            {"name": "t", "steps": [{"id": "a", "prompt": "p", "tools": ["allowed"]}]}
        )
        client = ScriptedClient([ScriptedClient.answer("done")])
        await WorkflowEngine(client, reg, settings, trace=trace).run(spec)
        offered = [t["function"]["name"] for t in client.requests[0]["tools"]]
        assert offered == ["allowed"]

    async def test_destructive_workflow_refused_up_front(self, settings, trace):
        reg = ToolRegistry(allow_destructive=False)
        spec = simple_spec(allow_destructive=True)
        client = ScriptedClient()
        engine = WorkflowEngine(client, reg, settings, trace=trace)
        with pytest.raises(WorkflowError, match="--allow-destructive"):
            await engine.run(spec)
        assert client.requests == []  # refused before any model call

    async def test_required_schema_failure_aborts(self, empty_registry, settings, trace):
        spec = WorkflowSpec.model_validate(
            {
                "name": "t",
                "steps": [
                    {
                        "id": "a",
                        "prompt": "p",
                        "tools": [],
                        "output": {
                            "required": True,
                            "schema": {
                                "type": "object",
                                "properties": {"n": {"type": "integer"}},
                                "required": ["n"],
                            },
                        },
                    }
                ],
            }
        )
        client = ScriptedClient(
            [
                ScriptedClient.answer("prose"),
                ScriptedClient.answer("not json"),
                ScriptedClient.answer("still not json"),
            ]
        )
        engine = WorkflowEngine(client, empty_registry, settings, trace=trace)
        with pytest.raises(WorkflowError, match="valid structured output"):
            await engine.run(spec)

    async def test_optional_schema_failure_degrades_to_prose(
        self, empty_registry, settings, trace
    ):
        spec = WorkflowSpec.model_validate(
            {
                "name": "t",
                "steps": [
                    {
                        "id": "a",
                        "prompt": "p",
                        "tools": [],
                        "output": {
                            "required": False,
                            "schema": {
                                "type": "object",
                                "properties": {"n": {"type": "integer"}},
                                "required": ["n"],
                            },
                        },
                    }
                ],
            }
        )
        client = ScriptedClient(
            [
                ScriptedClient.answer("useful prose"),
                ScriptedClient.answer("nope"),
                ScriptedClient.answer("nope again"),
            ]
        )
        result = await WorkflowEngine(client, empty_registry, settings, trace=trace).run(spec)
        assert result.completed
        assert result.steps[0].answer == "useful prose"
        assert result.steps[0].output is None
        assert result.steps[0].schema_error

    async def test_save_as_renames_the_context_key(self, empty_registry, settings, trace):
        spec = WorkflowSpec.model_validate(
            {
                "name": "t",
                "steps": [
                    {"id": "one", "prompt": "p", "tools": [], "save_as": "renamed"},
                    {"id": "two", "prompt": "Got {{ steps.renamed.answer }}", "tools": []},
                ],
            }
        )
        client = ScriptedClient(
            [ScriptedClient.answer("value"), ScriptedClient.answer("done")]
        )
        await WorkflowEngine(client, empty_registry, settings, trace=trace).run(spec)
        assert "Got value" in client.requests[1]["messages"][-1]["content"]

    async def test_per_step_system_prompt(self, empty_registry, settings, trace):
        spec = WorkflowSpec.model_validate(
            {
                "name": "t",
                "steps": [{"id": "a", "prompt": "p", "tools": [], "system": "Be terse."}],
            }
        )
        client = ScriptedClient([ScriptedClient.answer("ok")])
        await WorkflowEngine(client, empty_registry, settings, trace=trace).run(spec)
        assert client.requests[0]["messages"][0]["content"].startswith("Be terse.")

    async def test_progress_callback(self, empty_registry, settings, trace):
        seen: list[str] = []
        client = ScriptedClient(
            [ScriptedClient.answer("a"), ScriptedClient.answer("b")]
        )
        engine = WorkflowEngine(
            client, empty_registry, settings, trace=trace,
            on_progress=lambda step_id, _desc: seen.append(step_id),
        )
        await engine.run(simple_spec())
        assert seen == ["one", "two"]
