"""Wiring shared by every entrypoint (CLI, server, notebooks, tests).

One place that turns `Settings` into a live client + configured registry, so
the CLI and the HTTP server cannot drift apart in how they set things up.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from hermes_agent.config import Settings, load_settings
from hermes_agent.llm.client import OllamaClient
from hermes_agent.memory.vector import VectorStore
from hermes_agent.tools import GLOBAL_REGISTRY, configure_builtins, set_workspace
from hermes_agent.tools.registry import ToolRegistry
from hermes_agent.trace import TraceRecorder

logger = logging.getLogger("hermes_agent")


@dataclass(slots=True)
class Runtime:
    """Everything an entrypoint needs to run agents."""

    settings: Settings
    client: OllamaClient
    registry: ToolRegistry
    vector_store: VectorStore | None = None

    def new_trace(self) -> TraceRecorder:
        return TraceRecorder(self.settings.resolved_runs_dir())

    async def aclose(self) -> None:
        if self.vector_store is not None:
            self.vector_store.close()
        await self.client.aclose()


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


def build_runtime(
    settings: Settings | None = None,
    *,
    config_path: str | None = "config.yaml",
    overrides: dict[str, Any] | None = None,
    tools: list[str] | None = None,
) -> Runtime:
    """Build a Runtime from settings, applying workspace and tool configuration."""
    settings = settings or load_settings(config_path, overrides=overrides)
    configure_logging(settings.observability.log_level)

    workspace = settings.resolved_workspace()
    set_workspace(workspace)
    configure_builtins(
        python_timeout_s=settings.safety.python_timeout_s,
        python_max_output_chars=settings.safety.python_max_output_chars,
        max_read_bytes=settings.safety.max_read_bytes,
    )

    registry = GLOBAL_REGISTRY.subset(tools)
    registry.allow_destructive = settings.safety.allow_destructive

    client = OllamaClient(settings.ollama)

    vector_store: VectorStore | None = None
    if settings.memory.long_term_enabled:
        vector_store = VectorStore(
            settings.memory.db_path,
            client,
            collection=settings.memory.collection,
            embed_model=settings.ollama.embed_model,
        )

    logger.debug(
        "runtime ready: model=%s workspace=%s tools=%s destructive=%s",
        settings.ollama.model,
        workspace,
        registry.names(),
        registry.allow_destructive,
    )
    return Runtime(
        settings=settings, client=client, registry=registry, vector_store=vector_store
    )


@asynccontextmanager
async def runtime_context(**kwargs: Any) -> AsyncIterator[Runtime]:
    runtime = build_runtime(**kwargs)
    try:
        yield runtime
    finally:
        await runtime.aclose()
