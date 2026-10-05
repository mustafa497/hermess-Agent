"""Command-line interface.

    hermes doctor                 check Ollama, the model, and the workspace
    hermes chat                   interactive or one-shot agent chat
    hermes run <workflow>         execute a workflow
    hermes tools list             show registered tools and their schemas
    hermes workflows list         show bundled workflows
    hermes trace <run_id>         replay a run's JSONL trace
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.json import JSON
from rich.panel import Panel
from rich.table import Table

from hermes_agent import __version__
from hermes_agent.agent.loop import Agent
from hermes_agent.errors import HermesAgentError
from hermes_agent.memory.vector import chunk_text
from hermes_agent.runtime import Runtime, build_runtime
from hermes_agent.trace import list_runs, read_trace
from hermes_agent.workflows.engine import WorkflowEngine
from hermes_agent.workflows.schema import find_workflow, list_builtin_workflows

app = typer.Typer(
    name="hermes",
    help="Fully local agentic workflows on Hermes via Ollama.",
    no_args_is_help=True,
    add_completion=False,
)
tools_app = typer.Typer(name="tools", help="Inspect registered tools.", no_args_is_help=True)
workflows_app = typer.Typer(
    name="workflows", help="Inspect available workflows.", no_args_is_help=True
)
memory_app = typer.Typer(name="memory", help="Manage long-term memory.", no_args_is_help=True)
app.add_typer(tools_app)
app.add_typer(workflows_app)
app.add_typer(memory_app)

console = Console()
err_console = Console(stderr=True)


def _overrides(
    model: str | None,
    host: str | None,
    workspace: str | None,
    allow_destructive: bool,
    num_ctx: int | None,
    temperature: float | None,
    max_iterations: int | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "ollama.model": model,
        "ollama.host": host,
        "ollama.num_ctx": num_ctx,
        "ollama.temperature": temperature,
        "safety.workspace_dir": workspace,
        "agent.max_iterations": max_iterations,
    }
    if allow_destructive:
        out["safety.allow_destructive"] = True
    return {k: v for k, v in out.items() if v is not None}


def _fail(exc: Exception) -> None:
    err_console.print(f"[bold red]Error:[/bold red] {exc}")
    raise typer.Exit(code=1)


def _print_usage(usage: dict[str, Any]) -> None:
    console.print(
        f"[dim]run {usage.get('run_id')} | {usage.get('llm_calls', 0)} model calls | "
        f"{usage.get('tool_calls', 0)} tool calls | "
        f"{usage.get('prompt_tokens', 0)}+{usage.get('completion_tokens', 0)} tokens | "
        f"{usage.get('llm_latency_ms', 0) / 1000:.1f}s model / "
        f"{usage.get('tool_latency_ms', 0) / 1000:.1f}s tools[/dim]"
    )


# -- doctor ----------------------------------------------------------------------


@app.command()
def doctor(
    model: str = typer.Option(None, "--model", "-m"),
    host: str = typer.Option(None, "--host"),
) -> None:
    """Check that Ollama is running and the configured models are pulled."""

    async def _run() -> int:
        runtime = build_runtime(overrides=_overrides(model, host, None, False, None, None))
        settings = runtime.settings
        table = Table(title="hermes doctor", show_header=False, box=None)
        problems = 0
        try:
            table.add_row("host", settings.ollama.host)
            table.add_row("workspace", str(settings.resolved_workspace()))
            table.add_row("runs dir", str(settings.resolved_runs_dir()))
            table.add_row("tools", ", ".join(runtime.registry.names()))
            table.add_row(
                "destructive tools",
                "[green]enabled[/green]"
                if runtime.registry.allow_destructive
                else "[yellow]disabled[/yellow] (use --allow-destructive)",
            )
            try:
                available = await runtime.client.list_models()
            except HermesAgentError as exc:
                table.add_row("ollama", f"[red]unreachable[/red] - {exc}")
                console.print(table)
                return 1

            table.add_row("ollama", f"[green]reachable[/green] ({len(available)} models)")
            for label, tag in (("chat model", settings.ollama.model),
                               ("embed model", settings.ollama.embed_model)):
                base = tag.split(":")[0]
                if any(a == tag or a.split(":")[0] == base for a in available):
                    table.add_row(label, f"[green]{tag}[/green]")
                else:
                    table.add_row(label, f"[red]{tag} not pulled[/red] - run: ollama pull {tag}")
                    problems += 1
            console.print(table)
        finally:
            await runtime.aclose()
        return problems

    code = asyncio.run(_run())
    if code:
        raise typer.Exit(code=1)
    console.print("[green]All checks passed.[/green]")


# -- chat ------------------------------------------------------------------------


@app.command()
def chat(
    message: str = typer.Argument(None, help="Send one message and exit. Omit for a REPL."),
    model: str = typer.Option(None, "--model", "-m", help="Ollama model tag."),
    host: str = typer.Option(None, "--host"),
    workspace: str = typer.Option(None, "--workspace", "-w"),
    tools: str = typer.Option(None, "--tools", help="Comma-separated tool subset."),
    allow_destructive: bool = typer.Option(
        False, "--allow-destructive", help="Permit write_file and run_python."
    ),
    max_iterations: int = typer.Option(None, "--max-iterations"),
    num_ctx: int = typer.Option(None, "--num-ctx"),
    temperature: float = typer.Option(None, "--temperature"),
    system: str = typer.Option(None, "--system", help="Override the system prompt."),
    no_stream: bool = typer.Option(False, "--no-stream", help="Disable token streaming."),
    show_trace: bool = typer.Option(False, "--show-trace", help="Print each step as it runs."),
) -> None:
    """Chat with the agent, with tools enabled."""
    tool_subset = [t.strip() for t in tools.split(",") if t.strip()] if tools else None

    async def _run() -> None:
        runtime = build_runtime(
            overrides=_overrides(
                model, host, workspace, allow_destructive, num_ctx, temperature,
                max_iterations,
            ),
            tools=tool_subset,
        )
        try:
            await _ensure_model(runtime)
            agent = Agent(
                runtime.client,
                runtime.registry,
                runtime.settings,
                system_prompt=system,
                trace=runtime.new_trace(),
                vector_store=runtime.vector_store,
            )
            if message:
                await _one_turn(agent, message, stream=not no_stream, show_trace=show_trace)
                return

            console.print(
                Panel(
                    f"[bold]hermes[/bold] {__version__} | model "
                    f"[cyan]{runtime.settings.ollama.model}[/cyan] | "
                    f"{len(runtime.registry)} tools\n"
                    "Type your message. /exit to quit, /trace for the run id, "
                    "/tools to list tools.",
                    title="local agent",
                )
            )
            while True:
                try:
                    user = console.input("[bold green]you[/bold green] ").strip()
                except (EOFError, KeyboardInterrupt):
                    console.print()
                    break
                if not user:
                    continue
                if user in {"/exit", "/quit"}:
                    break
                if user == "/tools":
                    console.print(runtime.registry.describe())
                    continue
                if user == "/trace":
                    console.print(f"run id: {agent.trace.run_id} -> {agent.trace.path}")
                    continue
                await _one_turn(agent, user, stream=not no_stream, show_trace=show_trace)
        finally:
            await runtime.aclose()

    try:
        asyncio.run(_run())
    except HermesAgentError as exc:
        _fail(exc)


async def _one_turn(agent: Agent, text: str, *, stream: bool, show_trace: bool) -> None:
    seen = 0
    printed_header = False

    def on_delta(chunk: str) -> None:
        nonlocal printed_header
        if not printed_header:
            console.print("[bold cyan]hermes[/bold cyan] ", end="")
            printed_header = True
        console.print(chunk, end="", highlight=False, markup=False)

    result = await agent.run(text, on_delta=on_delta if stream else None)

    if printed_header:
        console.print()
    else:
        console.print(f"[bold cyan]hermes[/bold cyan] {result.answer}")

    if show_trace:
        for step in result.steps[seen:]:
            if step.kind == "tool_call":
                status = "[green]ok[/green]" if step.ok else "[red]fail[/red]"
                console.print(
                    f"  [dim]tool[/dim] {step.name} {status} "
                    f"{step.latency_ms:.0f}ms {json.dumps(step.args or {})[:90]}"
                )
            elif step.kind in {"loop_detected", "context_trim", "parse_fallback"}:
                console.print(f"  [dim]{step.kind}[/dim] {step.meta}")

    if result.stop_reason != "final_answer":
        console.print(f"[yellow]stopped: {result.stop_reason}[/yellow]")
    _print_usage(result.usage)


# -- run -------------------------------------------------------------------------


@app.command()
def run(
    workflow: str = typer.Argument(..., help="Bundled workflow name, or a path to a YAML file."),
    input_pairs: list[str] = typer.Option(
        None, "--input", "-i", help="Workflow input as key=value. Repeatable."
    ),
    model: str = typer.Option(None, "--model", "-m"),
    host: str = typer.Option(None, "--host"),
    workspace: str = typer.Option(None, "--workspace", "-w"),
    allow_destructive: bool = typer.Option(False, "--allow-destructive"),
    num_ctx: int = typer.Option(None, "--num-ctx"),
    temperature: float = typer.Option(None, "--temperature"),
    json_out: bool = typer.Option(False, "--json", help="Print the result as JSON."),
) -> None:
    """Execute a declarative workflow."""
    inputs: dict[str, Any] = {}
    for pair in input_pairs or []:
        if "=" not in pair:
            _fail(ValueError(f"--input expects key=value, got {pair!r}"))
        key, value = pair.split("=", 1)
        inputs[key.strip()] = value

    async def _run() -> None:
        runtime = build_runtime(
            overrides=_overrides(
                model, host, workspace, allow_destructive, num_ctx, temperature
            )
        )
        try:
            spec = find_workflow(workflow)
            # Validate what needs no network first: a missing --allow-destructive
            # or a missing input is the user's to fix, and telling them that is
            # more useful than "Ollama is unreachable".
            if spec.allow_destructive and not runtime.registry.allow_destructive:
                _fail(
                    ValueError(
                        f"Workflow {spec.name!r} writes files and/or runs code. "
                        "Re-run with --allow-destructive."
                    )
                )
            spec.resolve_inputs(inputs)
            await _ensure_model(runtime)

            def on_progress(step_id: str, description: str) -> None:
                console.print(f"[bold blue]> {step_id}[/bold blue] [dim]{description}[/dim]")

            engine = WorkflowEngine(
                runtime.client,
                runtime.registry,
                runtime.settings,
                trace=runtime.new_trace(),
                vector_store=runtime.vector_store,
                on_progress=None if json_out else on_progress,
            )
            result = await engine.run(spec, inputs)

            if json_out:
                console.print_json(result.model_dump_json())
                return

            console.print(Panel(result.final_answer or "(no output)", title=spec.name))
            if not result.completed:
                console.print(f"[yellow]incomplete: {result.stopped_reason}[/yellow]")
            for step in result.steps:
                flag = "" if step.stop_reason == "final_answer" else f" [{step.stop_reason}]"
                console.print(f"  [dim]{step.id}: {step.iterations} iterations{flag}[/dim]")
            _print_usage(result.usage)
        finally:
            await runtime.aclose()

    try:
        asyncio.run(_run())
    except HermesAgentError as exc:
        _fail(exc)


# -- tools / workflows / trace -----------------------------------------------------


@tools_app.command("list")
def tools_list(
    schemas: bool = typer.Option(False, "--schemas", help="Print full JSON schemas."),
) -> None:
    """List registered tools."""
    runtime = build_runtime()
    table = Table(title="registered tools")
    table.add_column("name", style="cyan")
    table.add_column("args")
    table.add_column("flags")
    table.add_column("description")
    for spec in runtime.registry.specs():
        schema = spec.json_schema()
        required = set(schema.get("required", []))
        args = ", ".join(
            f"{k}{'' if k in required else '?'}" for k in schema.get("properties", {})
        )
        flags = []
        if spec.destructive:
            flags.append("[red]destructive[/red]")
        if not spec.parallel_safe:
            flags.append("[yellow]serial[/yellow]")
        table.add_row(spec.name, args, " ".join(flags), spec.description)
    console.print(table)

    if schemas:
        for spec in runtime.registry.specs():
            console.print(f"\n[bold cyan]{spec.name}[/bold cyan]")
            console.print(JSON(json.dumps(spec.to_ollama_tool())))


@workflows_app.command("list")
def workflows_list() -> None:
    """List bundled workflows."""
    table = Table(title="bundled workflows")
    table.add_column("name", style="cyan")
    table.add_column("inputs")
    table.add_column("description")
    for name, description in list_builtin_workflows():
        spec = find_workflow(name)
        inputs = ", ".join(
            f"{k}{'*' if v.required else ''}" for k, v in spec.inputs.items()
        )
        table.add_row(name, inputs, description)
    console.print(table)
    console.print("[dim]* = required. Pass with -i key=value.[/dim]")


@workflows_app.command("show")
def workflows_show(name: str = typer.Argument(...)) -> None:
    """Print a workflow's steps and branch conditions."""
    try:
        spec = find_workflow(name)
    except HermesAgentError as exc:
        _fail(exc)
        return
    console.print(f"[bold]{spec.name}[/bold] - {spec.description}")
    console.print(
        f"[dim]destructive={spec.allow_destructive} max_steps={spec.max_steps}[/dim]\n"
    )
    for step in spec.steps:
        tools = "all" if step.tools is None else (", ".join(step.tools) or "none")
        console.print(f"[cyan]{step.id}[/cyan] - {step.description}")
        console.print(f"  tools: {tools}   iterations: {step.max_iterations or 'default'}")
        if step.output and step.output.json_schema:
            required = "required" if step.output.required else "optional"
            console.print(f"  output: structured ({required})")
        for transition in step.next:
            when = transition.when or "always"
            console.print(f"  -> {transition.goto} when {when}")
        console.print()


@app.command()
def trace(
    run_id: str = typer.Argument(None, help="Run id. Omit to list recent runs."),
    runs_dir: str = typer.Option(None, "--runs-dir"),
    kind: str = typer.Option(None, "--kind", help="Filter by step kind."),
    full: bool = typer.Option(False, "--full", help="Print whole records as JSON."),
) -> None:
    """Replay a recorded run trace."""
    settings = build_runtime().settings
    directory = Path(runs_dir) if runs_dir else settings.resolved_runs_dir()

    if not run_id:
        runs = list_runs(directory)
        if not runs:
            console.print(f"No traces in {directory}")
            return
        console.print(f"[bold]recent runs in {directory}[/bold]")
        for name in runs:
            console.print(f"  {name}")
        return

    try:
        records = read_trace(directory, run_id)
    except FileNotFoundError as exc:
        _fail(exc)
        return

    if full:
        for record in records:
            if kind and record.get("kind") != kind:
                continue
            console.print(JSON(json.dumps(record)))
        return

    table = Table(title=f"trace {run_id}")
    table.add_column("#", justify="right")
    table.add_column("kind")
    table.add_column("name")
    table.add_column("ms", justify="right")
    table.add_column("tokens", justify="right")
    table.add_column("detail")
    for record in records:
        if kind and record.get("kind") != kind:
            continue
        tokens = ""
        if record.get("prompt_tokens") or record.get("completion_tokens"):
            tokens = f"{record.get('prompt_tokens', 0)}+{record.get('completion_tokens', 0)}"
        detail = record.get("error") or record.get("result") or json.dumps(record.get("meta", {}))
        style = "red" if not record.get("ok", True) else ""
        table.add_row(
            str(record.get("seq")),
            record.get("kind", ""),
            record.get("name") or "",
            f"{record.get('latency_ms', 0):.0f}" if record.get("latency_ms") else "",
            tokens,
            str(detail)[:80].replace("\n", " "),
            style=style,
        )
    console.print(table)


# -- memory ---------------------------------------------------------------------------


@memory_app.command("index")
def memory_index(
    path: str = typer.Argument(..., help="File or folder inside the workspace to index."),
    collection: str = typer.Option(None, "--collection"),
    glob: str = typer.Option("**/*.md", "--glob", help="Glob used when path is a folder."),
) -> None:
    """Embed workspace documents into long-term memory."""

    async def _run() -> None:
        overrides: dict[str, Any] = {"memory.long_term_enabled": True}
        if collection:
            overrides["memory.collection"] = collection
        runtime = build_runtime(overrides=overrides)
        try:
            assert runtime.vector_store is not None
            root = runtime.settings.resolved_workspace() / path
            files = [root] if root.is_file() else sorted(root.glob(glob))
            if not files:
                console.print(f"[yellow]No files matched {path}/{glob}[/yellow]")
                return
            total = 0
            for file in files:
                if not file.is_file():
                    continue
                chunks = chunk_text(file.read_text(encoding="utf-8", errors="replace"))
                if not chunks:
                    continue
                await runtime.vector_store.add(
                    chunks, [{"source": file.name} for _ in chunks]
                )
                total += len(chunks)
                console.print(f"  indexed {file.name}: {len(chunks)} chunks")
            console.print(
                f"[green]{total} chunks stored[/green] "
                f"(collection {runtime.vector_store.collection!r}, "
                f"{runtime.vector_store.count()} total)"
            )
        finally:
            await runtime.aclose()

    try:
        asyncio.run(_run())
    except HermesAgentError as exc:
        _fail(exc)


@memory_app.command("search")
def memory_search(
    query: str = typer.Argument(...),
    top_k: int = typer.Option(5, "--top-k", "-k"),
    collection: str = typer.Option(None, "--collection"),
) -> None:
    """Query long-term memory."""

    async def _run() -> None:
        overrides: dict[str, Any] = {"memory.long_term_enabled": True}
        if collection:
            overrides["memory.collection"] = collection
        runtime = build_runtime(overrides=overrides)
        try:
            assert runtime.vector_store is not None
            hits = await runtime.vector_store.search(query, top_k=top_k)
            if not hits:
                console.print("[yellow]No matches.[/yellow]")
                return
            for hit in hits:
                console.print(f"[cyan]{hit.score:.3f}[/cyan] {hit.metadata.get('source', '?')}")
                console.print(f"  {hit.text[:300]}\n")
        finally:
            await runtime.aclose()

    try:
        asyncio.run(_run())
    except HermesAgentError as exc:
        _fail(exc)


# -- shared helpers ------------------------------------------------------------------


async def _ensure_model(runtime: Runtime) -> None:
    """Fail before the first token rather than after a confusing timeout."""
    await runtime.client.ensure_ready()


@app.command()
def version() -> None:
    """Print the version."""
    console.print(f"hermes-agent {__version__}")


def main() -> None:
    try:
        app()
    except HermesAgentError as exc:  # pragma: no cover - top-level safety net
        err_console.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
