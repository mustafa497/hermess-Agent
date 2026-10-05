"""Rolling conversation window with summarization of evicted turns.

Two things make this non-trivial and worth its own module:

1. **Atomic blocks.** An assistant message carrying `tool_calls` and the `tool`
   messages answering it must be evicted together. Splitting them produces a
   malformed ChatML prompt and Hermes will either error or hallucinate results.
2. **Token estimation.** We cannot tokenize locally without shipping a
   tokenizer, so we start from chars/4 and then calibrate against the
   `prompt_eval_count` Ollama reports after every call. Within a few turns the
   estimate tracks the real number closely.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from hermes_agent.llm.client import Message, OllamaClient

SUMMARY_MARKER = "[conversation summary]"

SUMMARIZER_PROMPT = (
    "You compress conversation history. Rewrite the exchange below as terse notes "
    "that preserve: the user's goals and constraints, decisions made, facts "
    "discovered, file paths and identifiers, and anything still outstanding. "
    "Drop pleasantries and repetition. Do not invent information. "
    "Answer with the notes only, under 200 words."
)


def estimate_tokens(text: str) -> int:
    """Chars/4 heuristic. Deliberately crude; calibrated at runtime."""
    return max(1, len(text) // 4)


def message_tokens(message: Message) -> int:
    total = estimate_tokens(message.content or "")
    for call in message.tool_calls:
        total += estimate_tokens(call.name) + estimate_tokens(str(call.arguments))
    return total + 4  # role tags and delimiters


@dataclass(slots=True)
class Block:
    """One atomically-evictable unit of history."""

    messages: list[Message]

    @property
    def tokens(self) -> int:
        return sum(message_tokens(m) for m in self.messages)


def group_into_blocks(messages: Sequence[Message]) -> list[Block]:
    """Group messages so assistant tool-call turns stay attached to their results."""
    blocks: list[Block] = []
    pending: list[Message] = []

    for message in messages:
        if message.role == "tool" and pending:
            pending.append(message)
            continue
        if pending:
            blocks.append(Block(pending))
        pending = [message]
    if pending:
        blocks.append(Block(pending))
    return blocks


@dataclass
class ConversationWindow:
    """Holds the live message list and trims it to fit `num_ctx`.

    Args:
        num_ctx: The model's context window.
        trim_ratio: Start trimming once the estimate exceeds this fraction.
        keep_recent_blocks: Never evict this many trailing blocks.
        reserve_tokens: Headroom left for the model's own completion.
    """

    num_ctx: int = 8192
    trim_ratio: float = 0.75
    keep_recent_blocks: int = 6
    reserve_tokens: int = 512

    system: list[Message] = field(default_factory=list)
    turns: list[Message] = field(default_factory=list)
    summaries: list[str] = field(default_factory=list)
    _calibration: float = 1.0

    # -- construction -------------------------------------------------------

    def set_system(self, content: str) -> None:
        self.system = [Message(role="system", content=content)] if content else []

    def add(self, message: Message) -> None:
        self.turns.append(message)

    def extend(self, messages: Iterable[Message]) -> None:
        self.turns.extend(messages)

    # -- budgeting ----------------------------------------------------------

    @property
    def budget(self) -> int:
        return max(512, int(self.num_ctx * self.trim_ratio) - self.reserve_tokens)

    def estimated_tokens(self) -> int:
        raw = sum(message_tokens(m) for m in self.messages())
        return int(raw * self._calibration)

    def calibrate(self, reported_prompt_tokens: int | None) -> None:
        """Correct the chars/4 estimate using Ollama's reported prompt_eval_count.

        Exponential smoothing, clamped, so one odd measurement cannot make the
        window either wildly over-eager or blind to overflow.
        """
        if not reported_prompt_tokens or reported_prompt_tokens <= 0:
            return
        raw = sum(message_tokens(m) for m in self.messages())
        if raw <= 0:
            return
        observed = reported_prompt_tokens / raw
        self._calibration = max(0.5, min(3.0, 0.7 * self._calibration + 0.3 * observed))

    def messages(self) -> list[Message]:
        """The full prompt: system, then any summary note, then live turns."""
        out = list(self.system)
        if self.summaries:
            joined = "\n\n".join(self.summaries)
            out.append(Message(role="system", content=f"{SUMMARY_MARKER}\n{joined}"))
        out.extend(self.turns)
        return out

    def needs_trim(self) -> bool:
        return self.estimated_tokens() > self.budget

    # -- trimming -----------------------------------------------------------

    def _select_evictable(self) -> tuple[list[Message], list[Message]]:
        """Split turns into (to_evict, to_keep) honouring block atomicity.

        The first user message is preserved: it states the task, and losing it
        is the single fastest way to make a long agent run drift off target.
        """
        blocks = group_into_blocks(self.turns)
        if len(blocks) <= self.keep_recent_blocks:
            return [], list(self.turns)

        head, tail = blocks[: -self.keep_recent_blocks], blocks[-self.keep_recent_blocks :]

        anchor: list[Message] = []
        if head and head[0].messages and head[0].messages[0].role == "user":
            anchor = [head[0].messages[0]]

        target = self.budget
        kept_tail = [m for b in tail for m in b.messages]
        running = sum(message_tokens(m) for m in [*self.system, *anchor, *kept_tail])
        running = int(running * self._calibration)

        evict: list[Message] = []
        keep_from_head: list[Message] = []
        # Walk the head newest-first, keeping what still fits. Once a block no
        # longer fits we stop keeping entirely: history must stay contiguous,
        # and a hole in the middle reads worse to the model than a clean cut.
        overflowed = False
        for block in reversed(head):
            block_cost = int(block.tokens * self._calibration)
            if not overflowed and running + block_cost <= target:
                running += block_cost
                keep_from_head[0:0] = block.messages
            else:
                overflowed = True
                evict[0:0] = block.messages

        if anchor and anchor[0] in evict:
            evict.remove(anchor[0])
            keep_from_head.insert(0, anchor[0])

        return evict, [*keep_from_head, *kept_tail]

    async def trim(self, client: OllamaClient | None = None, *, model: str | None = None) -> int:
        """Evict old blocks, summarizing them when a client is available.

        Returns the number of messages evicted. Summarization failure is not
        fatal -- we fall back to a truncation note, because losing the summary
        is much better than losing the run.
        """
        if not self.needs_trim():
            return 0

        evicted, kept = self._select_evictable()
        if not evicted:
            return 0
        self.turns = kept

        if client is not None:
            try:
                self.summaries.append(await self._summarize(evicted, client, model))
                return len(evicted)
            except Exception:
                pass  # fall through to the cheap note

        self.summaries.append(
            f"({len(evicted)} earlier messages were dropped to fit the context window; "
            "no summary was available.)"
        )
        return len(evicted)

    async def _summarize(
        self, evicted: Sequence[Message], client: OllamaClient, model: str | None
    ) -> str:
        transcript_parts: list[str] = []
        for m in evicted:
            label = m.tool_name if m.role == "tool" and m.tool_name else m.role
            body = m.content or ""
            if m.tool_calls:
                calls = ", ".join(f"{c.name}({c.arguments})" for c in m.tool_calls)
                body = f"{body}\n[called: {calls}]".strip()
            transcript_parts.append(f"{label}: {body[:2000]}")
        transcript = "\n\n".join(transcript_parts)

        response = await client.chat(
            [
                Message(role="system", content=SUMMARIZER_PROMPT),
                Message(role="user", content=transcript),
            ],
            model=model,
            options={"temperature": 0.0},
        )
        text = response.content.strip()
        if not text:
            raise ValueError("summarizer returned nothing")
        return text
