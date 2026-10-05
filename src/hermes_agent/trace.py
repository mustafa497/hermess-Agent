"""Structured tracing: one JSONL file per run, one record per step.

Everything the agent does emits a record. This is the primary debugging surface
for local models -- when an 8B model mangles a tool call you want the exact
bytes it emitted, not a summary.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

StepKind = Literal[
    "run_start",
    "run_end",
    "llm_call",
    "tool_call",
    "parse_fallback",
    "loop_detected",
    "context_trim",
    "workflow_step",
    "error",
]


def new_run_id() -> str:
    return f"{datetime.now(UTC):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:8]}"


class StepRecord(BaseModel):
    run_id: str
    seq: int
    kind: StepKind
    ts: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    iteration: int | None = None
    name: str | None = None
    latency_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    ok: bool = True
    thought: str | None = None
    args: dict[str, Any] | None = None
    result: str | None = None
    error: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class TraceRecorder:
    """Append-only JSONL writer. Cheap enough to always leave on."""

    def __init__(self, runs_dir: Path, run_id: str | None = None, *, enabled: bool = True) -> None:
        self.run_id = run_id or new_run_id()
        self.runs_dir = Path(runs_dir)
        self.enabled = enabled
        self._seq = 0
        self._steps: list[StepRecord] = []
        if self.enabled:
            self.runs_dir.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self.runs_dir / f"{self.run_id}.jsonl"

    @property
    def steps(self) -> list[StepRecord]:
        return list(self._steps)

    def record(self, kind: StepKind, **fields: Any) -> StepRecord:
        self._seq += 1
        step = StepRecord(run_id=self.run_id, seq=self._seq, kind=kind, **fields)
        self._steps.append(step)
        if self.enabled:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(step.model_dump_json(exclude_none=True) + "\n")
        return step

    @contextmanager
    def timed(self, kind: StepKind, **fields: Any) -> Iterator[dict[str, Any]]:
        """Time a block and emit one record. Mutate the yielded dict to add fields.

        The record is written even when the block raises, with ok=False.
        """
        started = time.perf_counter()
        extra: dict[str, Any] = {}
        try:
            yield extra
        except Exception as exc:
            merged = {**fields, **extra}
            merged["ok"] = False
            merged["error"] = f"{type(exc).__name__}: {exc}"
            merged["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
            self.record(kind, **merged)
            raise
        else:
            merged = {**fields, **extra}
            merged.setdefault("ok", True)
            merged["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
            self.record(kind, **merged)

    def totals(self) -> dict[str, Any]:
        llm = [s for s in self._steps if s.kind == "llm_call"]
        tools = [s for s in self._steps if s.kind == "tool_call"]
        return {
            "run_id": self.run_id,
            "steps": len(self._steps),
            "llm_calls": len(llm),
            "tool_calls": len(tools),
            "prompt_tokens": sum(s.prompt_tokens or 0 for s in llm),
            "completion_tokens": sum(s.completion_tokens or 0 for s in llm),
            "llm_latency_ms": round(sum(s.latency_ms or 0 for s in llm), 2),
            "tool_latency_ms": round(sum(s.latency_ms or 0 for s in tools), 2),
        }


def read_trace(runs_dir: Path, run_id: str) -> list[dict[str, Any]]:
    path = Path(runs_dir) / f"{run_id}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"No trace for run {run_id!r} at {path}")
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def list_runs(runs_dir: Path, limit: int = 20) -> list[str]:
    d = Path(runs_dir)
    if not d.is_dir():
        return []
    files = sorted(d.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [p.stem for p in files[:limit]]
