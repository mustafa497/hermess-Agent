"""Declarative workflow definitions.

A workflow is an ordered list of steps. Each step is one agent call with its
own system prompt, tool subset, iteration budget and optional output schema.
Control flow is a per-step `next` list of guarded jumps; the first condition
that matches wins, and falling off the end of the list means "continue to the
next step in order".

That is enough to express sequential pipelines, conditional branching, and
planner -> executor -> critic loops, without inventing a general-purpose
programming language in YAML.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from hermes_agent.errors import WorkflowError

END = "end"


class InputSpec(BaseModel):
    """One declared workflow input."""

    description: str = ""
    default: Any = None
    required: bool = False


class Transition(BaseModel):
    """A guarded jump. `when` omitted means unconditional."""

    when: str | None = None
    goto: str

    @property
    def is_terminal(self) -> bool:
        return self.goto == END


class OutputSpec(BaseModel):
    """Structured output contract for a step."""

    json_schema: dict[str, Any] = Field(default_factory=dict, alias="schema")
    required: bool = True
    """When False, a schema failure logs a warning and keeps the prose answer."""

    model_config = {"populate_by_name": True}


class StepSpec(BaseModel):
    """One agent call in a workflow."""

    id: str
    description: str = ""
    system: str | None = None
    prompt: str
    tools: list[str] | None = None
    """Tool names for this step. `None` means all; `[]` means a pure LLM step."""
    max_iterations: int | None = None
    temperature: float | None = None
    model: str | None = None
    output: OutputSpec | None = None
    save_as: str | None = None
    next: list[Transition] = Field(default_factory=list)
    fresh_context: bool = True
    """True gives the step a clean window; False continues the previous one."""

    @property
    def result_key(self) -> str:
        return self.save_as or self.id


class WorkflowSpec(BaseModel):
    """A complete workflow definition."""

    name: str
    description: str = ""
    inputs: dict[str, InputSpec] = Field(default_factory=dict)
    steps: list[StepSpec]
    allow_destructive: bool = False
    """Declared up front so the CLI can refuse before any model call is made."""
    max_steps: int = 24
    """Hard cap on executed steps; branch loops cannot run away."""
    system: str | None = None
    """Default system prompt inherited by steps that do not set their own."""

    @model_validator(mode="after")
    def _check_graph(self) -> WorkflowSpec:
        if not self.steps:
            raise ValueError("A workflow needs at least one step")

        ids = [s.id for s in self.steps]
        duplicates = {i for i in ids if ids.count(i) > 1}
        if duplicates:
            raise ValueError(f"Duplicate step ids: {sorted(duplicates)}")
        if END in ids:
            raise ValueError(f"{END!r} is reserved and cannot be a step id")

        known = set(ids) | {END}
        for step in self.steps:
            for transition in step.next:
                if transition.goto not in known:
                    raise ValueError(
                        f"Step {step.id!r} jumps to unknown target {transition.goto!r}. "
                        f"Valid targets: {sorted(known)}"
                    )
        return self

    def step(self, step_id: str) -> StepSpec:
        for step in self.steps:
            if step.id == step_id:
                return step
        raise WorkflowError(f"No step with id {step_id!r} in workflow {self.name!r}")

    def resolve_inputs(self, provided: dict[str, Any] | None = None) -> dict[str, Any]:
        """Merge provided inputs with declared defaults, checking required ones."""
        given = dict(provided or {})
        resolved: dict[str, Any] = {}
        missing: list[str] = []
        for name, spec in self.inputs.items():
            if name in given and given[name] is not None:
                resolved[name] = given[name]
            elif spec.default is not None:
                resolved[name] = spec.default
            elif spec.required:
                missing.append(name)
            else:
                resolved[name] = ""
        if missing:
            raise WorkflowError(
                f"Workflow {self.name!r} is missing required inputs: {', '.join(missing)}"
            )
        # Undeclared extras are kept: handy for ad-hoc overrides from the CLI.
        for key, value in given.items():
            resolved.setdefault(key, value)
        return resolved


def load_workflow(source: str | Path) -> WorkflowSpec:
    """Load a workflow from a YAML file path, or from a YAML document string."""
    text = str(source)
    try:
        path = Path(source)
        if path.is_file():
            text = path.read_text(encoding="utf-8")
    except (OSError, ValueError):
        # A YAML document passed as a string can exceed the OS path limit or
        # contain characters that are illegal in a path; treat it as content.
        pass

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise WorkflowError(f"Could not parse workflow YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise WorkflowError("A workflow file must contain a YAML mapping")
    try:
        return WorkflowSpec.model_validate(data)
    except Exception as exc:
        raise WorkflowError(f"Invalid workflow definition: {exc}") from exc


def builtin_workflows_dir() -> Path:
    return Path(__file__).parent / "examples"


def find_workflow(name_or_path: str) -> WorkflowSpec:
    """Resolve a workflow by path, or by name from the bundled examples."""
    candidate = Path(name_or_path)
    if candidate.is_file():
        return load_workflow(candidate)

    directory = builtin_workflows_dir()
    for suffix in (".yaml", ".yml"):
        bundled = directory / f"{name_or_path}{suffix}"
        if bundled.is_file():
            return load_workflow(bundled)

    available = sorted(p.stem for p in directory.glob("*.y*ml"))
    raise WorkflowError(
        f"No workflow named {name_or_path!r}. Bundled workflows: {', '.join(available)}. "
        "You can also pass a path to a YAML file."
    )


def list_builtin_workflows() -> list[tuple[str, str]]:
    """(name, description) for every bundled workflow."""
    out: list[tuple[str, str]] = []
    for path in sorted(builtin_workflows_dir().glob("*.y*ml")):
        try:
            spec = load_workflow(path)
        except WorkflowError:
            continue
        out.append((path.stem, spec.description))
    return out


StepStatus = Literal["ok", "schema_failed", "agent_stopped"]
