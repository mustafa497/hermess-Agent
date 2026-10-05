# hermes-agent

A fully local agentic workflow system built on a [Nous Research Hermes](https://nousresearch.com/)
model served by [Ollama](https://ollama.com). No cloud APIs, no telemetry, no
network after the initial model pull.

The agent loop is written by hand — no LangChain, no LlamaIndex — so that every
decision (when to stop, what gets sent, how a failure is reported back to the
model) is readable in one file: `src/hermes_agent/agent/loop.py`.

---

## Quick start

```bash
# 1. Ollama
#    https://ollama.com/download   (or: winget install Ollama.Ollama)
ollama serve                      # leave running in its own terminal

# 2. Models
ollama pull hermes3:8b            # the agent
ollama pull nomic-embed-text      # only needed for long-term memory

# 3. This package
python -m pip install -e ".[dev]"      # add ",server" for the HTTP API
cp .env.example .env                   # optional

# 4. Verify the whole chain
hermes doctor
```

`hermes doctor` checks that Ollama is reachable, that both model tags are
actually pulled, and prints the resolved workspace and tool list. Run it first
whenever something behaves oddly.

```bash
hermes chat "What is 17.5 squared, to three decimal places?"
hermes chat                                    # interactive REPL
hermes tools list --schemas
hermes workflows list
hermes run doc_research -i folder=docs --allow-destructive
hermes trace                                   # list recent runs
hermes trace 20260930T142201-9f3ab0c1          # replay one
```

---

## Hardware guidance

Pick the largest quantization that leaves ~2 GB of headroom. Ollama will spill
to CPU when a model does not fit in VRAM; it still works, just several times
slower.

| VRAM / RAM | Model tag | Notes |
|---|---|---|
| 8 GB VRAM | `hermes3:8b` (Q4_K_M, the default) | ~4.7 GB. Comfortable with `num_ctx: 8192`. |
| 12 GB VRAM | `hermes3:8b-q8_0` | ~8.5 GB. Noticeably better tool-call accuracy than Q4. |
| 16 GB VRAM | `hermes3:8b-fp16` or `hermes3:70b-q2_K` | Prefer 8B fp16: a heavily quantized 70B is usually worse at tool calls. |
| 24 GB VRAM | `hermes3:70b-q4_K_M` (partial offload) | ~40 GB total; expect CPU spill and ~3–8 tok/s. |
| CPU only, 16 GB RAM | `hermes3:8b` | Works. Raise `ollama.request_timeout_s` to 600. |
| CPU only, 32 GB RAM | `hermes3:8b-q8_0` | Better quality; still slow. Keep `num_ctx` at 8192. |

Context length costs memory quadratically in KV cache. On 8 GB, `num_ctx: 8192`
is roughly the ceiling for the 8B at Q4. If you see the model silently
forgetting early instructions, check the `context_trim` records in the trace
before raising `num_ctx`.

Changing model is config only — no code changes:

```bash
hermes chat -m hermes3:70b --num-ctx 16384 "..."
# or
HERMES_OLLAMA__MODEL=hermes3:70b hermes chat "..."
```

---

## Configuration

Precedence, highest first: **`HERMES_*` env vars → `.env` → `config.yaml` → defaults.**

Nested keys use a double underscore; a few flat aliases exist for convenience:

```bash
HERMES_MODEL=hermes3:70b                # alias for ollama.model
HERMES_OLLAMA__NUM_CTX=16384
HERMES_AGENT__MAX_ITERATIONS=15
HERMES_SAFETY__ALLOW_DESTRUCTIVE=true
```

CLI flags override everything for a single invocation.

---

## Architecture

```
llm/client.py       async Ollama wrapper: streaming + non-streaming, retries,
                    error classification (daemon down vs. model not pulled)
llm/parsing.py      fallback parser for <tool_call>{...}</tool_call> + JSON repair
tools/registry.py   @tool decorator -> JSON schema from type hints + docstring
tools/sandbox.py    workspace confinement (resolve-then-compare)
tools/builtins.py   read_file, write_file, list_dir, run_python, calculator, web_search
agent/loop.py       the ReAct loop, guards, tracing
memory/short_term.py rolling window + summarization of evicted turns
memory/vector.py    optional SQLite + nomic-embed-text long-term store
workflows/          declarative multi-step workflows, branching, structured output
runtime.py          shared wiring for every entrypoint
cli.py / server.py  Typer CLI and optional FastAPI + SSE
```

One agent iteration:

```
messages -> model -> tool_calls?
                      |-- yes: execute (parallel where safe) -> append results -> loop
                      `-- no:  that is the final answer
```

---

## Reliability: where small local models actually break

This is the part that matters. Each risk below has a specific, tested
mitigation in the code.

### 1. Tool calls that are *almost* valid

Hermes is trained on the ChatML `<tool_call>` convention, and Ollama's native
`tools` parameter usually populates `message.tool_calls` correctly. But an 8B
model regularly emits the raw tags as plain text instead, truncates the closing
tag, writes a Python dict with single quotes, or leaves a trailing comma.

**Mitigation** — `llm/parsing.py` runs as a backstop whenever native parsing
yields nothing. It handles tagged calls, fenced JSON, and bare JSON objects,
and performs *bounded* repairs: smart quotes, trailing commas, bracket
balancing for truncated output, and Python-literal syntax.

```python
parse_tool_calls('<tool_call>{"name": "calculator", "arguments": {"expression": "2+2",}')
# -> ParsedCall(name='calculator', arguments={'expression': '2+2'}, repaired=True)
```

Every fallback parse is recorded as a `parse_fallback` trace step with a
`repaired` flag, so you can measure how often your model needs it. Unrepairable
output is fed back to the model as an explicit parse error rather than
crashing.

The bare-JSON path is gated on known tool names — otherwise a structured JSON
*answer* containing a `name` field gets misread as a tool call.

### 2. Invalid arguments

**Mitigation** — `ToolRegistry.execute` validates against the generated
Pydantic model and, on failure, returns the error *plus the full schema* to the
model as an ordinary tool result. Validation problems never raise out of the
loop:

```
ERROR (read_file): Invalid arguments for read_file: max_bytes: Input should be
a valid integer... Expected schema: {...}. Call the tool again with corrected
arguments.
```

Unknown tool names get the same treatment, listing what *is* available. In
practice Hermes recovers from both on the next turn.

### 3. Infinite loops

Small models retry identical failing calls indefinitely.

**Mitigation** — three independent guards: a signature counter on
`(tool, arguments)` that trips after `loop_detection_threshold` (default 3),
`max_tool_calls_per_turn`, and `max_iterations`. When a guard trips the agent
makes one final call *with tools withheld* so the user still gets an answer.
If even that produces nothing usable, `_degraded_answer()` returns the last
successful tool result with an honest explanation — the loop never returns an
empty string.

### 4. JSON that does not match the schema

**Mitigation** — three layers: Ollama's `format` parameter constrains
generation to the JSON schema; the result is validated; on failure exactly one
repair retry is sent with the precise validation error and the schema. One
retry, not a loop — a model that fails twice with the error in front of it
essentially never succeeds on the third try.

Steps can set `output.required: false` to degrade to prose instead of failing.

### 5. Context overflow

Exceeding `num_ctx` makes Ollama silently drop the *start* of the prompt —
usually the system prompt and the task.

**Mitigation** — `memory/short_term.py` estimates usage (chars/4, then
calibrated against the `prompt_eval_count` Ollama reports after every call) and
summarizes evicted turns once usage passes `context_trim_ratio` of `num_ctx`.
Eviction operates on *blocks*: an assistant message with `tool_calls` and its
`tool` replies move together, because splitting them produces a malformed
prompt. The first user message is always preserved.

### 6. Fabricated tool results

Models sometimes report a file as written without calling `write_file`.

**Mitigation** — partly prompt ("Never claim to have done something a tool did
not actually report doing"), but mainly the trace: every `tool_call` record has
the real arguments, result and latency. If it is not in the JSONL, it did not
happen.

---

## Safety

- **Filesystem.** Every file tool routes through `resolve_in_workspace()`,
  which resolves to an absolute real path and requires it to sit under
  `WORKSPACE_DIR`. Resolving *before* comparing defeats `../` traversal,
  absolute paths, and symlinks pointing outside.
- **Destructive tools.** `write_file` and `run_python` are marked
  `destructive=True` and refuse to run without `--allow-destructive`
  (or `safety.allow_destructive: true`). A workflow declaring
  `allow_destructive: true` is rejected up front, before any model call.
- **Code execution.** `run_python` runs in a subprocess under `python -I`, with
  the workspace as cwd, a scrubbed environment, a wall-clock timeout, `socket`
  neutered in-process (which blocks every stdlib HTTP client), and POSIX
  rlimits on memory, CPU, file size and process count.

  **This is defence in depth against a confused model, not a security boundary
  against hostile code.** Windows has no rlimit equivalent, so only the network
  block and the timeout apply there. A determined snippet can still escape via
  `ctypes` or by spawning a process. Run genuinely untrusted code in a
  container or VM.
- **Prompt injection.** Tool results are wrapped with an explicit instruction
  that they are untrusted data. Keep `--allow-destructive` off when processing
  documents you did not write. The real boundary is the workspace and the
  destructive flag, not the prompt.

---

## Adding a tool

Write a typed function with a Google-style docstring and decorate it. The JSON
schema is derived from the signature, so it cannot drift from the code.

```python
# src/hermes_agent/tools/builtins.py  (or your own module, imported at startup)
from hermes_agent.tools.registry import tool
from hermes_agent.tools.sandbox import resolve_in_workspace

@tool(tags=["fs"], destructive=False, parallel_safe=True)
def count_lines(path: str, skip_blank: bool = False) -> str:
    """Count the lines in a workspace text file.

    Args:
        path: Path relative to the workspace root.
        skip_blank: Ignore blank lines in the count.
    """
    text = resolve_in_workspace(path, must_exist=True).read_text(encoding="utf-8")
    lines = text.splitlines()
    if skip_blank:
        lines = [ln for ln in lines if ln.strip()]
    return f"{len(lines)} lines"
```

That is the whole contract:

- **Parameter types** become the JSON schema. Prefer plain `str`, `int`,
  `float`, `bool`, `list[str]`, and Pydantic models. Avoid `Any` — the model
  cannot guess what to send.
- **The docstring summary** becomes the tool description, and the `Args:`
  entries become per-parameter descriptions. Both go into the prompt, so write
  them for the model.
- **Return value** is stringified (`dict`/`list` as indented JSON).
- **Raise on failure.** The registry catches it and returns a structured error
  to the model; do not return error strings pretending to be success.
- `destructive=True` puts it behind `--allow-destructive`.
- `parallel_safe=False` makes it run alone, after the concurrent batch — use it
  for anything that mutates shared state.

Async functions work too; sync ones are run in a thread automatically.

Check it: `hermes tools list --schemas`.

---

## Writing a workflow

A workflow is YAML: an ordered list of steps, each one agent call with its own
system prompt, tool subset, iteration budget and optional output schema.

```yaml
name: my_workflow
description: What this does.
allow_destructive: false
max_steps: 12                 # hard cap; stops runaway branch loops

inputs:
  topic:
    description: What to work on
    required: true
  depth:
    default: "brief"

steps:
  - id: gather
    system: You are precise and cite filenames.
    prompt: Find everything about {{ inputs.topic }} in the workspace.
    tools: [list_dir, read_file]      # [] = no tools; omit = all tools
    max_iterations: 6
    output:
      schema:                          # JSON Schema; enforced via Ollama `format`
        type: object
        properties:
          found: { type: boolean }
          items: { type: array, items: { type: string } }
        required: [found, items]

  - id: report
    prompt: |
      Summarise at {{ inputs.depth }} depth:
      {{ steps.gather.output.items }}
    tools: []
    next:
      - when: "steps.gather.output.found == false"
        goto: end
      - goto: wrap_up
```

**Templates** (`{{ }}`) and **conditions** (`when:`) are two different
mini-languages, and the distinction trips people up:

- `prompt:` and `system:` are **templates** — `{{ dotted.path }}` substitution only.
- `when:` is an **expression** — no braces. Write `inputs.max_fixes`, not
  `{{ inputs.max_fixes }}`. Supports `== != < <= > >=`, `and or not`, `in`,
  and literals `true/false/null`. Neither language can execute code.

Context available to both:

| Path | Meaning |
|---|---|
| `inputs.<name>` | Declared workflow inputs |
| `steps.<id>.answer` | That step's prose answer |
| `steps.<id>.output.<field>` | That step's validated structured output |
| `steps.<id>.stop_reason` | How that step's agent loop ended |
| `state.visits.<id>` | Times a step has executed — **use this to bound loops** |
| `state.step_count` | Total steps executed |

Control flow: the first matching `next` transition wins; no match falls through
to the next step in the list; `goto: end` terminates. Missing paths resolve to
`None` rather than raising, so a branch referencing a step that has not run
evaluates falsy.

Drop the file in `src/hermes_agent/workflows/examples/` to get it by name, or
pass any path: `hermes run ./my_workflow.yaml`.

For the planner → executor → critic shape, `workflows/patterns.py` builds it in
Python with the visit-count guard already wired:

```python
from hermes_agent.workflows import plan_execute_critique
spec = plan_execute_critique(tools=["read_file", "calculator"], max_revisions=2)
```

### Bundled examples

**`doc_research`** — read a folder of documents and write a summary report.
Demonstrates conditional branching (empty folder → early exit), structured
output, and an optional schema that degrades to prose.

```bash
mkdir -p workspace/docs   # add some .md files
hermes run doc_research -i folder=docs -i report_path=summary.md --allow-destructive
```

**`code_fix_loop`** — plan → write → test → fix, looping until the tests pass
or the repair budget runs out. Demonstrates a bounded loop via
`state.visits.run_tests > inputs.max_fixes`.

```bash
hermes run code_fix_loop \
  -i task="a function that parses 'HH:MM:SS' into total seconds, raising ValueError on bad input" \
  --allow-destructive
```

---

## Long-term memory (optional)

Off by default. SQLite + `nomic-embed-text`, chosen over ChromaDB because
SQLite ships with Python and needs no native wheel. Brute-force cosine, which
is fine at single-machine scale (numpy is used when present, pure Python
otherwise).

```bash
ollama pull nomic-embed-text
hermes memory index docs --glob "**/*.md"
hermes memory search "retry policy"
HERMES_MEMORY__LONG_TERM_ENABLED=true hermes chat "What did the notes say about retries?"
```

When enabled, the agent retrieves `memory.top_k` chunks before the first model
call and injects them as system context, labelled as untrusted data. Swapping
in Chroma or `sqlite-vec` means reimplementing `add`/`search` on `VectorStore`
and nothing else.

---

## Observability

One JSONL file per run in `./runs/`, one record per step: model calls (with
token counts and latency), tool calls (with arguments, result and latency),
fallback parses, loop detections, context trims, and workflow transitions.

```bash
hermes trace                                    # recent runs
hermes trace <run_id>                           # table view
hermes trace <run_id> --kind tool_call --full   # raw JSON
```

```bash
# Slowest tool calls across every run
cat runs/*.jsonl | jq -r 'select(.kind=="tool_call") | [.latency_ms, .name] | @tsv' | sort -rn | head

# How often did the fallback parser have to rescue a tool call?
cat runs/*.jsonl | jq -r 'select(.kind=="parse_fallback") | .meta.recovered[]' | sort | uniq -c
```

That second query is the single most useful number for judging whether a given
model/quantization is reliable enough for your workload.

---

## HTTP API (optional)

```bash
python -m pip install -e ".[server]"
uvicorn hermes_agent.server:app --port 8080
```

| Endpoint | Purpose |
|---|---|
| `GET /health` | Ollama reachability and model status |
| `GET /tools` | Registered tools and schemas |
| `GET /workflows` | Bundled workflows |
| `POST /chat` | Run the agent, returns JSON |
| `POST /chat/stream` | Same, streamed as SSE |
| `POST /workflows/{name}/run` | Execute a workflow |
| `GET /runs/{run_id}` | Fetch a trace |

It binds to localhost and has no authentication. Do not expose it.

---

## Testing

```bash
pytest                      # unit tests, no Ollama needed
pytest -m ollama            # integration test against a live local model
pytest -m "not ollama"      # explicitly skip it
```

The integration test skips itself automatically when Ollama is unreachable or
the model is not pulled. Everything else mocks the LLM client, so the suite is
deterministic and runs in a couple of seconds.

---

## Known limitations of small local models

Honest expectations for `hermes3:8b` at Q4, from the failure modes this system
is built to absorb:

- **Tool-call accuracy is the bottleneck**, not reasoning. Expect the fallback
  parser to fire on a meaningful fraction of turns. Check
  `grep parse_fallback runs/*.jsonl` before blaming the loop.
- **Temperature above ~0.4 degrades tool calling sharply.** The default is 0.2.
  Raise it for prose, not for tool-using steps.
- **More than ~8 tools in one prompt** measurably hurts selection accuracy. Use
  per-step `tools:` subsets in workflows rather than exposing everything.
- **Multi-step plans drift.** Around 5–6 tool calls, the model starts losing the
  original goal. Prefer several narrow workflow steps over one long agent run —
  this is the main reason the workflow layer exists.
- **Structured output is much more reliable with `format`** than with prompting
  alone, but deeply nested schemas still fail. Keep them shallow and flat.
- **Self-assessment is unreliable.** A critic step will approve broken work. The
  `code_fix_loop` workflow deliberately gates on *actual test output*, not on
  the model's opinion.
- **No true parallel reasoning.** Concurrent tool execution is declared via
  `parallel_safe`, not inferred; the model requests calls sequentially in
  practice.
- **70B variants are substantially better at tool use** but, once they spill to
  CPU, often too slow for interactive loops. An 8B at `q8_0` is usually the
  better trade on 12–16 GB.

---

## License

MIT.
