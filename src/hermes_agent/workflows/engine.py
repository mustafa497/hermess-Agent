"""Workflow execution engine.

Runs a `WorkflowSpec` step by step, threading each step's result into the
context available to later prompts and branch conditions.

Context shape visible to templates and conditions:

    inputs.<name>              declared workflow inputs
    steps.<id>.answer          the step's prose answer
    steps.<id>.output.<field>  the step's validated structured output
    steps.<id>.stop_reason     how the step's agent loop ended
    state.visits.<id>          how many times a step has executed
    state.step_count           total steps executed so far
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from hermes_agent.agent.loop import Agent, AgentResult
from hermes_agent.config import Settings
from hermes_agent.errors import StructuredOutputError, WorkflowError
from hermes_agent.llm.client import OllamaClient
from hermes_agent.memory.vector import VectorStore
from hermes_agent.tools.registry import ToolRegistry
from hermes_agent.trace import TraceRecorder
from hermes_agent.workflows import jsonschema_mini
from hermes_agent.workflows.expressions import ExpressionError, evaluate, render
from hermes_agent.workflows.schema import END, StepSpec, WorkflowSpec

ProgressCallback = Callable[[str, str], None]


class StepResult(BaseModel):
    """Outcome of one executed step."""

    id: str
    answer: str
    output: dict[str, Any] | list[Any] | None = None
    stop_reason: str = "final_answer"
    schema_error: str | None = None
    iterations: int = 0
    usage: dict[str, Any] = Field(default_factory=dict)

    def as_context(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "output": self.output,
            "stop_reason": self.stop_reason,
            "schema_error": self.schema_error,
        }


class WorkflowResult(BaseModel):
    """Outcome of a whole workflow run."""

    run_id: str
    workflow: str
    steps: list[StepResult] = Field(default_factory=list)
    final_answer: str = ""
    completed: bool = True
    stopped_reason: str = "completed"
    usage: dict[str, Any] = Field(default_factory=dict)

    def step(self, step_id: str) -> StepResult | None:
        for result in reversed(self.steps):
            if result.id == step_id:
                return result
        return None


class WorkflowEngine:
    """Executes workflows against a shared client, registry and trace."""

    def __init__(
        self,
        client: OllamaClient,
        registry: ToolRegistry,
        settings: Settings,
        *,
        trace: TraceRecorder | None = None,
        vector_store: VectorStore | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> None:
        self.client = client
        self.registry = registry
        self.settings = settings
        self.trace = trace or TraceRecorder(settings.resolved_runs_dir())
        self.vector_store = vector_store
        self.on_progress = on_progress

    async def run(
        self, spec: WorkflowSpec, inputs: dict[str, Any] | None = None
    ) -> WorkflowResult:
        """Execute `spec`, following its branch conditions."""
        if spec.allow_destructive and not self.registry.allow_destructive:
            raise WorkflowError(
                f"Workflow {spec.name!r} needs destructive tools (it writes files "
                "and/or runs code). Re-run with --allow-destructive."
            )

        resolved_inputs = spec.resolve_inputs(inputs)
        context: dict[str, Any] = {
            "inputs": resolved_inputs,
            "steps": {},
            "state": {"visits": {}, "step_count": 0},
        }

        self.trace.record(
            "workflow_step",
            name=f"{spec.name}:start",
            meta={"inputs": resolved_inputs, "steps": [s.id for s in spec.steps]},
        )

        results: list[StepResult] = []
        current: str | None = spec.steps[0].id
        executed = 0
        stopped_reason = "completed"
        completed = True

        while current and current != END:
            if executed >= spec.max_steps:
                stopped_reason = (
                    f"step budget exhausted after {executed} steps "
                    f"(max_steps={spec.max_steps}); a branch condition is probably looping"
                )
                completed = False
                self.trace.record(
                    "workflow_step", name="budget_exhausted", ok=False, error=stopped_reason
                )
                break

            step = spec.step(current)
            executed += 1
            visits = context["state"]["visits"]
            visits[step.id] = visits.get(step.id, 0) + 1
            context["state"]["step_count"] = executed

            if self.on_progress:
                self.on_progress(step.id, step.description or step.prompt[:80])

            result = await self._run_step(spec, step, context)
            results.append(result)
            context["steps"][step.result_key] = result.as_context()

            current = self._next_step(spec, step, context)

        final = results[-1].answer if results else ""
        usage = self.trace.totals()
        self.trace.record(
            "workflow_step",
            name=f"{spec.name}:end",
            ok=completed,
            meta={**usage, "steps_executed": executed, "reason": stopped_reason},
        )
        return WorkflowResult(
            run_id=self.trace.run_id,
            workflow=spec.name,
            steps=results,
            final_answer=final,
            completed=completed,
            stopped_reason=stopped_reason,
            usage=usage,
        )

    # -- step execution -------------------------------------------------------

    async def _run_step(
        self, spec: WorkflowSpec, step: StepSpec, context: dict[str, Any]
    ) -> StepResult:
        prompt = render(step.prompt, context)
        system = render(step.system or spec.system or "", context) or None

        step_registry = self.registry.subset(step.tools)
        step_registry.allow_destructive = self.registry.allow_destructive

        settings = self.settings.model_copy(deep=True)
        if step.temperature is not None:
            settings.ollama.temperature = step.temperature
        if step.max_iterations is not None:
            settings.agent.max_iterations = step.max_iterations

        agent = Agent(
            self.client,
            step_registry,
            settings,
            system_prompt=system,
            trace=self.trace,
            vector_store=self.vector_store,
            model=step.model,
        )

        with self.trace.timed(
            "workflow_step", name=step.id, meta={"tools": step_registry.names()}
        ) as extra:
            agent_result: AgentResult = await agent.run(prompt)
            extra["result"] = agent_result.answer[:2000]
            extra["meta"] = {
                "tools": step_registry.names(),
                "stop_reason": agent_result.stop_reason,
                "iterations": agent_result.iterations,
            }

        output: Any = None
        schema_error: str | None = None
        if step.output and step.output.json_schema:
            schema = step.output.json_schema
            try:
                output = await agent.structured_from_schema(
                    schema,
                    answer=agent_result.answer,
                    validator=lambda value: jsonschema_mini.validate(value, schema),
                    label=f"the output schema of step {step.id!r}",
                )
            except StructuredOutputError as exc:
                schema_error = str(exc)
                if step.output.required:
                    # Fail loudly: later steps branch on this output, and
                    # continuing with None silently corrupts the whole run.
                    raise WorkflowError(
                        f"Step {step.id!r} could not produce valid structured output. "
                        f"{exc}"
                    ) from exc
                self.trace.record(
                    "workflow_step", name=f"{step.id}:schema", ok=False, error=schema_error
                )

        return StepResult(
            id=step.id,
            answer=agent_result.answer,
            output=output,
            stop_reason=agent_result.stop_reason,
            schema_error=schema_error,
            iterations=agent_result.iterations,
            usage=agent_result.usage,
        )

    def _next_step(
        self, spec: WorkflowSpec, step: StepSpec, context: dict[str, Any]
    ) -> str | None:
        """First matching transition wins; otherwise fall through in order."""
        for transition in step.next:
            if transition.when is None:
                return transition.goto
            try:
                matched = evaluate(transition.when, context)
            except ExpressionError as exc:
                raise WorkflowError(
                    f"Step {step.id!r} has an invalid condition {transition.when!r}: {exc}"
                ) from exc
            self.trace.record(
                "workflow_step",
                name=f"{step.id}:branch",
                meta={"when": transition.when, "matched": matched, "goto": transition.goto},
            )
            if matched:
                return transition.goto

        index = [s.id for s in spec.steps].index(step.id)
        if index + 1 < len(spec.steps):
            return spec.steps[index + 1].id
        return None
