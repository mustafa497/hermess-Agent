"""Configuration: YAML base + environment overlay, validated by Pydantic.

Precedence (highest first): HERMES_* env vars -> .env file -> config.yaml -> defaults.

Deliberately not using pydantic-settings: the overlay is ~40 lines, and doing it
by hand keeps the precedence rules obvious and the dependency surface smaller.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

ENV_PREFIX = "HERMES_"
NESTED_SEP = "__"

#: Flat convenience aliases so `.env` can stay readable for the common knobs.
FLAT_ALIASES: dict[str, tuple[str, ...]] = {
    "MODEL": ("ollama", "model"),
    "HOST": ("ollama", "host"),
    "EMBED_MODEL": ("ollama", "embed_model"),
    "NUM_CTX": ("ollama", "num_ctx"),
    "WORKSPACE_DIR": ("safety", "workspace_dir"),
    "ALLOW_DESTRUCTIVE": ("safety", "allow_destructive"),
    "RUNS_DIR": ("observability", "runs_dir"),
    "LOG_LEVEL": ("observability", "log_level"),
}


class OllamaSettings(BaseModel):
    host: str = "http://localhost:11434"
    model: str = "hermes3:8b"
    embed_model: str = "nomic-embed-text"
    temperature: float = 0.2
    top_p: float = 0.9
    num_ctx: int = 8192
    num_predict: int = -1
    seed: int | None = None
    keep_alive: str = "5m"
    connect_timeout_s: float = 5.0
    request_timeout_s: float = 180.0
    max_retries: int = 3
    backoff_base_s: float = 0.5
    backoff_max_s: float = 8.0

    @field_validator("host")
    @classmethod
    def _strip_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")

    def options(self, **overrides: Any) -> dict[str, Any]:
        """Build the Ollama `options` payload, dropping unset values."""
        opts: dict[str, Any] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "num_ctx": self.num_ctx,
        }
        if self.num_predict and self.num_predict > 0:
            opts["num_predict"] = self.num_predict
        if self.seed is not None:
            opts["seed"] = self.seed
        opts.update({k: v for k, v in overrides.items() if v is not None})
        return opts


class AgentSettings(BaseModel):
    max_iterations: int = Field(default=10, ge=1, le=100)
    max_tool_calls_per_turn: int = Field(default=5, ge=1, le=32)
    loop_detection_threshold: int = Field(default=3, ge=2)
    context_trim_ratio: float = Field(default=0.75, gt=0.1, le=0.95)
    keep_recent_turns: int = Field(default=6, ge=2)
    enable_tag_fallback: bool = True


class SafetySettings(BaseModel):
    workspace_dir: Path = Path("./workspace")
    allow_destructive: bool = False
    python_timeout_s: float = 20.0
    python_max_output_chars: int = 20_000
    max_read_bytes: int = 200_000


class MemorySettings(BaseModel):
    long_term_enabled: bool = False
    db_path: Path = Path("./runs/memory.sqlite3")
    top_k: int = 5
    collection: str = "default"


class ObservabilitySettings(BaseModel):
    runs_dir: Path = Path("./runs")
    log_level: str = "INFO"


class Settings(BaseModel):
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    safety: SafetySettings = Field(default_factory=SafetySettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)

    def resolved_workspace(self) -> Path:
        p = self.safety.workspace_dir.expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    def resolved_runs_dir(self) -> Path:
        p = self.observability.runs_dir.expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p


def _set_nested(target: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    cursor = target
    for key in path[:-1]:
        nxt = cursor.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            cursor[key] = nxt
        cursor = nxt
    cursor[path[-1]] = value


def _coerce(raw: str) -> Any:
    """Env vars are strings; let YAML scalar rules turn them into bool/int/float/null."""
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def env_overlay(data: dict[str, Any], environ: dict[str, str] | None = None) -> dict[str, Any]:
    """Apply HERMES_* variables on top of `data` (mutates and returns a copy)."""
    env = dict(os.environ if environ is None else environ)
    merged = {k: (dict(v) if isinstance(v, dict) else v) for k, v in data.items()}
    for key, raw in env.items():
        if not key.startswith(ENV_PREFIX):
            continue
        suffix = key[len(ENV_PREFIX) :]
        if NESTED_SEP in suffix:
            path = tuple(part.lower() for part in suffix.split(NESTED_SEP) if part)
        elif suffix in FLAT_ALIASES:
            path = FLAT_ALIASES[suffix]
        else:
            continue  # unknown flat var: ignore rather than guess
        _set_nested(merged, path, _coerce(raw))
    return merged


def load_settings(
    config_path: str | Path | None = "config.yaml",
    *,
    environ: dict[str, str] | None = None,
    overrides: dict[str, Any] | None = None,
) -> Settings:
    """Load config.yaml (if present), overlay env, then explicit CLI overrides."""
    try:
        from dotenv import load_dotenv

        load_dotenv(override=False)
    except ImportError:  # python-dotenv is optional at runtime
        pass

    data: dict[str, Any] = {}
    if config_path is not None:
        path = Path(config_path)
        if path.is_file():
            loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if not isinstance(loaded, dict):
                raise ValueError(f"{path} must contain a YAML mapping at the top level")
            data = loaded

    data = env_overlay(data, environ)

    for path_str, value in (overrides or {}).items():
        if value is None:
            continue
        _set_nested(data, tuple(path_str.split(".")), value)

    return Settings.model_validate(data)
