"""Reusable workflow shapes, built in Python rather than YAML.

`plan_execute_critique` is the pattern worth having in code: the critic loop
needs a visit-count guard to terminate, and getting that wrong in YAML is the
most common way a local-model workflow burns an evening of tokens.
"""

from __future__ import annotations

from hermes_agent.workflows.schema import OutputSpec, StepSpec, Transition, WorkflowSpec

CRITIC_SCHEMA = {
    "type": "object",
    "properties": {
        "approved": {
            "type": "boolean",
            "description": "True only if the work fully satisfies the task.",
        },
        "score": {"type": "integer", "minimum": 0, "maximum": 10},
        "issues": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Concrete, actionable problems. Empty when approved.",
        },
    },
    "required": ["approved", "issues"],
}

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "goal": {"type": "string"},
        "steps": {"type": "array", "items": {"type": "string"}, "minItems": 1},
    },
    "required": ["goal", "steps"],
}


def plan_execute_critique(
    task: str = "{{ inputs.task }}",
    *,
    name: str = "plan_execute_critique",
    tools: list[str] | None = None,
    max_revisions: int = 2,
    allow_destructive: bool = False,
) -> WorkflowSpec:
    """Build a planner -> executor -> critic workflow with a bounded revision loop.

    Args:
        task: Template for the task text; defaults to the `task` workflow input.
        tools: Tool subset for the executor. The planner and critic get none --
            they reason over text, and giving them tools mostly invites
            unnecessary calls.
        max_revisions: How many times the executor may be sent back by the critic.
        allow_destructive: Set when the executor's tools write or run code.
    """
    executor_tools = tools if tools is not None else ["calculator"]

    return WorkflowSpec(
        name=name,
        description="Plan a task, execute the plan, critique the result, revise if needed.",
        allow_destructive=allow_destructive,
        max_steps=4 + 2 * max_revisions,
        inputs={"task": {"description": "What to accomplish", "required": True}},  # type: ignore[dict-item]
        steps=[
            StepSpec(
                id="plan",
                description="Break the task into concrete steps",
                system=(
                    "You are a planner. Produce a short, concrete plan. Do not "
                    "execute anything and do not call tools."
                ),
                prompt=(
                    f"Task:\n{task}\n\n"
                    "Write a plan of at most 6 steps. Each step must be a single "
                    "concrete action."
                ),
                tools=[],
                max_iterations=2,
                output=OutputSpec(schema=PLAN_SCHEMA),
            ),
            StepSpec(
                id="execute",
                description="Carry out the plan",
                system=(
                    "You are an executor. Carry out the plan using the available "
                    "tools. Report exactly what you did and what the results were."
                ),
                prompt=(
                    f"Task:\n{task}\n\n"
                    "Plan:\n{{ steps.plan.output.steps }}\n\n"
                    "Feedback from the previous attempt (empty on the first pass):\n"
                    "{{ steps.critique.output.issues }}\n\n"
                    "Carry out the plan now."
                ),
                tools=executor_tools,
                max_iterations=8,
            ),
            StepSpec(
                id="critique",
                description="Judge the result against the task",
                system=(
                    "You are a strict reviewer. Judge only what was actually "
                    "produced. Do not call tools. Approve only if the task is "
                    "genuinely complete."
                ),
                prompt=(
                    f"Task:\n{task}\n\n"
                    "Work produced:\n{{ steps.execute.answer }}\n\n"
                    "Is this complete and correct? List concrete issues if not."
                ),
                tools=[],
                max_iterations=2,
                output=OutputSpec(schema=CRITIC_SCHEMA),
                next=[
                    Transition(
                        when=(
                            "steps.critique.output.approved == false and "
                            f"state.visits.critique < {max_revisions + 1}"
                        ),
                        goto="execute",
                    ),
                    Transition(goto="finalise"),
                ],
            ),
            StepSpec(
                id="finalise",
                description="Deliver the result",
                system="You write clear final summaries.",
                prompt=(
                    "Summarise the outcome for the user.\n\n"
                    "Work:\n{{ steps.execute.answer }}\n\n"
                    "Reviewer verdict:\n{{ steps.critique.output }}\n\n"
                    "State plainly whether the task was completed, and note any "
                    "issues the reviewer raised that remain unresolved."
                ),
                tools=[],
                max_iterations=2,
            ),
        ],
    )
