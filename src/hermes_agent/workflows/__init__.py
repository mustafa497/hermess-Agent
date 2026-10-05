"""Declarative multi-step workflows."""

from hermes_agent.workflows.engine import (
    StepResult,
    WorkflowEngine,
    WorkflowResult,
)
from hermes_agent.workflows.expressions import evaluate, render, resolve_path
from hermes_agent.workflows.patterns import plan_execute_critique
from hermes_agent.workflows.schema import (
    END,
    InputSpec,
    OutputSpec,
    StepSpec,
    Transition,
    WorkflowSpec,
    find_workflow,
    list_builtin_workflows,
    load_workflow,
)

__all__ = [
    "END",
    "InputSpec",
    "OutputSpec",
    "StepResult",
    "StepSpec",
    "Transition",
    "WorkflowEngine",
    "WorkflowResult",
    "WorkflowSpec",
    "evaluate",
    "find_workflow",
    "list_builtin_workflows",
    "load_workflow",
    "plan_execute_critique",
    "render",
    "resolve_path",
]
