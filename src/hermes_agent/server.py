"""Optional FastAPI surface over the same agent. Install with `.[server]`.

    uvicorn hermes_agent.server:app --port 8080

Binds to localhost and has no authentication: it exposes tools that read the
filesystem and (with destructive tools enabled) run code. Do not expose it to a
network you do not control.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from pydantic import BaseModel, Field

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The HTTP server needs the optional extra: pip install -e '.[server]'"
    ) from exc

from hermes_agent import __version__
from hermes_agent.agent.loop import Agent
from hermes_agent.errors import HermesAgentError, WorkflowError
from hermes_agent.runtime import Runtime, build_runtime
from hermes_agent.trace import read_trace
from hermes_agent.workflows.engine import WorkflowEngine
from hermes_agent.workflows.schema import find_workflow, list_builtin_workflows

_runtime: Runtime | None = None


def runtime() -> Runtime:
    if _runtime is None:  # pragma: no cover - lifespan guarantees this
        raise RuntimeError("Runtime is not initialised")
    return _runtime


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    global _runtime
    _runtime = build_runtime()
    try:
        yield
    finally:
        await _runtime.aclose()
        _runtime = None


app = FastAPI(
    title="hermes-agent",
    version=__version__,
    description="Local agentic workflows on Hermes via Ollama.",
    lifespan=lifespan,
)


class ChatRequest(BaseModel):
    message: str
    system: str | None = None
    tools: list[str] | None = Field(
        default=None, description="Tool subset; null means all, [] means none."
    )
    max_iterations: int | None = None
    model: str | None = None


class ChatResponse(BaseModel):
    run_id: str
    answer: str
    stop_reason: str
    iterations: int
    usage: dict[str, Any]


class WorkflowRequest(BaseModel):
    inputs: dict[str, Any] = Field(default_factory=dict)


def _build_agent(request: ChatRequest) -> Agent:
    rt = runtime()
    registry = rt.registry.subset(request.tools)
    registry.allow_destructive = rt.registry.allow_destructive
    settings = rt.settings.model_copy(deep=True)
    if request.max_iterations is not None:
        settings.agent.max_iterations = request.max_iterations
    return Agent(
        rt.client,
        registry,
        settings,
        system_prompt=request.system,
        trace=rt.new_trace(),
        vector_store=rt.vector_store,
        model=request.model,
    )


@app.get("/health")
async def health() -> dict[str, Any]:
    rt = runtime()
    status: dict[str, Any] = {
        "version": __version__,
        "model": rt.settings.ollama.model,
        "host": rt.settings.ollama.host,
        "workspace": str(rt.settings.resolved_workspace()),
        "allow_destructive": rt.registry.allow_destructive,
        "tools": rt.registry.names(),
    }
    try:
        available = await rt.client.list_models()
        base = rt.settings.ollama.model.split(":")[0]
        status["ollama"] = "reachable"
        status["model_pulled"] = any(
            a == rt.settings.ollama.model or a.split(":")[0] == base for a in available
        )
    except HermesAgentError as exc:
        status["ollama"] = "unreachable"
        status["error"] = str(exc)
    return status


@app.get("/tools")
async def tools() -> list[dict[str, Any]]:
    return [
        {
            "name": spec.name,
            "description": spec.description,
            "destructive": spec.destructive,
            "parallel_safe": spec.parallel_safe,
            "tags": list(spec.tags),
            "schema": spec.json_schema(),
        }
        for spec in runtime().registry.specs()
    ]


@app.get("/workflows")
async def workflows() -> list[dict[str, Any]]:
    out = []
    for name, description in list_builtin_workflows():
        spec = find_workflow(name)
        out.append(
            {
                "name": name,
                "description": description,
                "allow_destructive": spec.allow_destructive,
                "inputs": {
                    k: {"description": v.description, "required": v.required, "default": v.default}
                    for k, v in spec.inputs.items()
                },
                "steps": [s.id for s in spec.steps],
            }
        )
    return out


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        result = await _build_agent(request).run(request.message)
    except HermesAgentError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return ChatResponse(
        run_id=result.run_id,
        answer=result.answer,
        stop_reason=result.stop_reason,
        iterations=result.iterations,
        usage=result.usage,
    )


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """Server-sent events: `token` deltas, then one `done` (or `error`) event."""
    agent = _build_agent(request)
    queue: asyncio.Queue[tuple[str, Any] | None] = asyncio.Queue()

    async def on_delta(chunk: str) -> None:
        await queue.put(("token", chunk))

    async def drive() -> None:
        try:
            result = await agent.run(request.message, on_delta=on_delta)
            await queue.put(
                (
                    "done",
                    {
                        "run_id": result.run_id,
                        "answer": result.answer,
                        "stop_reason": result.stop_reason,
                        "iterations": result.iterations,
                        "usage": result.usage,
                    },
                )
            )
        except Exception as exc:
            await queue.put(("error", {"detail": f"{type(exc).__name__}: {exc}"}))
        finally:
            await queue.put(None)

    async def events() -> AsyncIterator[str]:
        task = asyncio.create_task(drive())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                event, data = item
                # Always JSON-encode: a token containing a newline would
                # otherwise break SSE framing. Clients JSON.parse every payload.
                body = json.dumps(data)
                yield f"event: {event}\ndata: {body}\n\n"
        finally:
            # A disconnected client must not leave the agent running.
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/workflows/{name}/run")
async def run_workflow(name: str, request: WorkflowRequest) -> dict[str, Any]:
    rt = runtime()
    try:
        spec = find_workflow(name)
    except WorkflowError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    engine = WorkflowEngine(rt.client, rt.registry, rt.settings, trace=rt.new_trace(),
                            vector_store=rt.vector_store)
    try:
        result = await engine.run(spec, request.inputs)
    except WorkflowError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HermesAgentError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return result.model_dump()


@app.get("/runs/{run_id}")
async def get_run(run_id: str) -> list[dict[str, Any]]:
    try:
        return read_trace(runtime().settings.resolved_runs_dir(), run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
