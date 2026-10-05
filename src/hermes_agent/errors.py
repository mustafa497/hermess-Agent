"""Error taxonomy.

The split that matters operationally: `OllamaUnavailableError` and
`ModelNotPulledError` are *operator* problems with actionable messages, while
`ToolExecutionError` / `ToolValidationError` are *model* problems that get fed
back into the conversation so the model can self-correct instead of the run
dying.
"""

from __future__ import annotations


class HermesAgentError(Exception):
    """Base class for everything this package raises."""


# --- transport / infrastructure -------------------------------------------------


class OllamaError(HermesAgentError):
    """Any failure talking to the Ollama HTTP API."""


class OllamaUnavailableError(OllamaError):
    def __init__(self, host: str, detail: str = "") -> None:
        super().__init__(
            f"Cannot reach Ollama at {host}. Start it with `ollama serve` "
            f"(or check HERMES_HOST). {detail}".strip()
        )
        self.host = host


class ModelNotPulledError(OllamaError):
    def __init__(self, model: str, host: str) -> None:
        super().__init__(
            f"Model {model!r} is not available on {host}. Pull it first: "
            f"`ollama pull {model}`."
        )
        self.model = model


class OllamaHTTPError(OllamaError):
    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(f"Ollama returned HTTP {status_code}: {body[:500]}")
        self.status_code = status_code
        self.body = body


class OllamaTimeoutError(OllamaError):
    pass


# --- tools ----------------------------------------------------------------------


class ToolError(HermesAgentError):
    """Base for tool problems. Carries a model-readable message."""


class ToolNotFoundError(ToolError):
    pass


class ToolValidationError(ToolError):
    """Arguments failed schema validation. Returned to the model, not raised out."""


class ToolExecutionError(ToolError):
    """The tool ran and failed. Returned to the model, not raised out."""


class SandboxViolationError(ToolError):
    def __init__(self, path: str, workspace: str) -> None:
        super().__init__(
            f"Path {path!r} resolves outside the workspace {workspace!r}. "
            "File tools may only touch paths inside the workspace."
        )


class DestructiveToolBlockedError(ToolError):
    def __init__(self, name: str) -> None:
        super().__init__(
            f"Tool {name!r} is marked destructive and destructive tools are disabled. "
            "Re-run with --allow-destructive (or safety.allow_destructive: true)."
        )


# --- agent / workflow -------------------------------------------------------------


class AgentError(HermesAgentError):
    pass


class MaxIterationsExceeded(AgentError):
    pass


class LoopDetectedError(AgentError):
    pass


class StructuredOutputError(HermesAgentError):
    """Model could not produce output matching the requested schema, even after repair."""


class WorkflowError(HermesAgentError):
    pass
