"""Schema generation from type hints, and validation feedback to the model."""

from __future__ import annotations

from typing import Literal

import pytest
from pydantic import BaseModel

from hermes_agent.tools.registry import (
    ToolRegistry,
    parse_docstring,
    stringify_result,
    tool,
)


class Point(BaseModel):
    """Module-level so annotations resolve, as real tool types must."""

    x: int
    y: int


def build(fn, **kwargs) -> ToolRegistry:
    """Register `fn` into a throwaway registry."""
    reg = ToolRegistry(allow_destructive=kwargs.pop("allow_destructive", True))
    tool(registry=reg, **kwargs)(fn)
    return reg


class TestDocstringParsing:
    def test_summary_and_args(self):
        def f(a: str, b: int = 1) -> str:
            """Do a thing.

            Args:
                a: The first thing.
                b: The second thing.

            Returns:
                Something.
            """
            return ""

        summary, params = parse_docstring(f.__doc__)
        assert summary == "Do a thing."
        assert params == {"a": "The first thing.", "b": "The second thing."}

    def test_multiline_arg_description(self):
        def f(a: str) -> str:
            """Summary.

            Args:
                a: A description that
                    wraps onto a second line.
            """
            return ""

        _, params = parse_docstring(f.__doc__)
        assert params["a"] == "A description that wraps onto a second line."

    def test_typed_arg_line(self):
        def f(a: str) -> str:
            """Summary.

            Args:
                a (str): With a type annotation in the docstring.
            """
            return ""

        _, params = parse_docstring(f.__doc__)
        assert params["a"] == "With a type annotation in the docstring."

    def test_no_docstring(self):
        assert parse_docstring(None) == ("", {})


class TestSchemaGeneration:
    def test_types_required_and_descriptions(self):
        def sample(name: str, count: int = 3, ratio: float = 0.5, flag: bool = False) -> str:
            """Do something useful.

            Args:
                name: Who to greet.
                count: How many times.
            """
            return ""

        spec = build(sample).get("sample")
        schema = spec.json_schema()

        assert spec.description == "Do something useful."
        assert schema["properties"]["name"]["type"] == "string"
        assert schema["properties"]["count"]["type"] == "integer"
        assert schema["properties"]["ratio"]["type"] == "number"
        assert schema["properties"]["flag"]["type"] == "boolean"
        assert schema["required"] == ["name"]
        assert schema["properties"]["name"]["description"] == "Who to greet."

    def test_titles_are_stripped(self):
        """Pydantic's auto titles are wasted context for a small model."""

        def sample(max_bytes: int = 1) -> str:
            """S."""
            return ""

        schema = build(sample).get("sample").json_schema()
        assert "title" not in schema
        assert "title" not in schema["properties"]["max_bytes"]

    def test_optional_becomes_nullable_union(self):
        def sample(value: int | None = None) -> str:
            """S."""
            return ""

        prop = build(sample).get("sample").json_schema()["properties"]["value"]
        assert {v["type"] for v in prop["anyOf"]} == {"integer", "null"}

    def test_list_and_literal(self):
        def sample(items: list[str], mode: Literal["fast", "slow"] = "fast") -> str:
            """S."""
            return ""

        schema = build(sample).get("sample").json_schema()
        assert schema["properties"]["items"]["type"] == "array"
        assert schema["properties"]["items"]["items"]["type"] == "string"
        assert schema["properties"]["mode"]["enum"] == ["fast", "slow"]

    def test_nested_model_refs_are_inlined(self):
        """Hermes follows flat schemas far better than $ref indirection."""

        def sample(point: Point) -> str:
            """S."""
            return ""

        schema = build(sample).get("sample").json_schema()
        assert "$defs" not in schema
        assert schema["properties"]["point"]["properties"]["x"]["type"] == "integer"

    def test_unresolvable_annotation_fails_loudly_at_registration(self):
        """A locally-scoped model cannot be resolved; say so now, not at call time."""

        class LocalOnly(BaseModel):
            x: int

        def sample(thing: LocalOnly) -> str:
            """S."""
            return ""

        with pytest.raises(TypeError, match="define any Pydantic models at module level"):
            build(sample)

    def test_untyped_parameter_defaults_to_string(self):
        def sample(thing) -> str:  # noqa: ANN001
            """S."""
            return ""

        assert build(sample).get("sample").json_schema()["properties"]["thing"]["type"] == "string"

    def test_varargs_are_excluded(self):
        def sample(a: str, *args, **kwargs) -> str:
            """S."""
            return ""

        assert list(build(sample).get("sample").json_schema()["properties"]) == ["a"]

    def test_ollama_tool_envelope(self):
        def sample(a: str) -> str:
            """S."""
            return ""

        payload = build(sample).get("sample").to_ollama_tool()
        assert payload["type"] == "function"
        assert payload["function"]["name"] == "sample"
        assert "parameters" in payload["function"]


class TestExecution:
    async def test_success(self):
        def add(a: int, b: int) -> int:
            """Add."""
            return a + b

        result = await build(add).execute("add", {"a": 2, "b": 3})
        assert result.ok and result.content == "5"

    async def test_async_tool(self):
        async def slow(a: int) -> int:
            """Async."""
            return a * 2

        result = await build(slow).execute("slow", {"a": 4})
        assert result.ok and result.content == "8"

    async def test_validation_error_is_returned_not_raised(self):
        def add(a: int, b: int) -> int:
            """Add."""
            return a + b

        result = await build(add).execute("add", {"a": "nope", "b": 3})
        assert result.ok is False
        assert result.meta.get("validation") is True
        assert "valid integer" in result.error
        # The schema is included so the model can self-correct.
        assert "properties" in result.error
        assert "Call the tool again" in result.error

    async def test_missing_required_argument(self):
        def add(a: int, b: int) -> int:
            """Add."""
            return a + b

        result = await build(add).execute("add", {"a": 1})
        assert not result.ok and "b: Field required" in result.error

    async def test_tool_exception_becomes_structured_error(self):
        def boom() -> str:
            """Explodes."""
            raise RuntimeError("kaboom")

        result = await build(boom).execute("boom", {})
        assert not result.ok and result.error == "RuntimeError: kaboom"

    async def test_unknown_tool_lists_alternatives(self):
        def add(a: int) -> int:
            """Add."""
            return a

        result = await build(add).execute("nope", {})
        assert not result.ok
        assert "Unknown tool 'nope'" in result.error and "add" in result.error

    async def test_destructive_tool_blocked_by_default(self):
        def wipe() -> str:
            """Destroys."""
            return "gone"

        reg = ToolRegistry(allow_destructive=False)
        tool(registry=reg, destructive=True)(wipe)
        result = await reg.execute("wipe", {})
        assert not result.ok and "--allow-destructive" in result.error

        reg.allow_destructive = True
        assert (await reg.execute("wipe", {})).ok

    async def test_error_rendering_for_the_model(self):
        def boom() -> str:
            """Explodes."""
            raise ValueError("bad")

        result = await build(boom).execute("boom", {})
        assert result.to_model_text().startswith("ERROR (boom):")


class TestRegistryManagement:
    def test_duplicate_registration_rejected_by_register(self):
        def a() -> str:
            """A."""
            return ""

        reg = build(a)
        with pytest.raises(ValueError, match="already registered"):
            reg.register(reg.get("a"))

    def test_decorator_replaces_and_warns_on_a_name_collision(self, caplog):
        """Re-import must be idempotent, but two different functions is a footgun."""

        def a() -> str:
            """First."""
            return "first"

        def a_other() -> str:
            """Second."""
            return "second"

        reg = build(a)
        tool(registry=reg)(a)  # same function again: silent, no warning
        assert not [r for r in caplog.records if r.levelname == "WARNING"]

        tool(registry=reg, name="a")(a_other)
        assert reg.get("a").description == "Second."
        assert any("re-registered with a different function" in r.message
                   for r in caplog.records)

    def test_subset_selects_and_inherits_permission(self):
        def a() -> str:
            """A."""
            return ""

        def b() -> str:
            """B."""
            return ""

        reg = build(a)
        tool(registry=reg)(b)
        assert reg.subset(["a"]).names() == ["a"]
        assert reg.subset(None).names() == ["a", "b"]
        assert reg.subset(["*"]).names() == ["a", "b"]
        assert reg.subset([]).names() == []

    def test_custom_name_and_description(self):
        def internal() -> str:
            """Ignored."""
            return ""

        reg = build(internal, name="public", description="Shown to the model")
        assert reg.get("public").description == "Shown to the model"

    def test_describe_marks_optional_and_destructive(self):
        def sample(a: str, b: int = 1) -> str:
            """Summary here."""
            return ""

        reg = build(sample, destructive=True)
        text = reg.describe()
        assert "sample(a: string, b: integer?)" in text
        assert "[destructive]" in text


class TestStringify:
    @pytest.mark.parametrize(
        "value,expected",
        [(None, "(no output)"), ("text", "text"), (5, "5"), (True, "true")],
    )
    def test_scalars(self, value, expected):
        assert stringify_result(value) == expected

    def test_dict_becomes_json(self):
        assert '"a": 1' in stringify_result({"a": 1})

    def test_pydantic_model(self):
        class M(BaseModel):
            a: int

        assert '"a": 1' in stringify_result(M(a=1))
