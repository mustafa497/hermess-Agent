"""Optional long-term memory: SQLite + Ollama embeddings.

Behind `memory.long_term_enabled`. Chosen over ChromaDB because SQLite ships
with Python, survives being copied around, and needs no native wheel -- for the
thousands-of-chunks scale a single-machine agent actually reaches, brute-force
cosine is fast enough. The `VectorStore` surface is small on purpose: swapping
in Chroma or sqlite-vec later means reimplementing `add`/`search`, nothing else.
"""

from __future__ import annotations

import json
import math
import sqlite3
import struct
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hermes_agent.llm.client import OllamaClient

try:  # numpy makes search ~50x faster; the pure-Python path keeps it optional
    import numpy as _np
except ImportError:  # pragma: no cover - exercised only when numpy is absent
    _np = None  # type: ignore[assignment]

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id          TEXT PRIMARY KEY,
    collection  TEXT NOT NULL,
    text        TEXT NOT NULL,
    metadata    TEXT NOT NULL DEFAULT '{}',
    embedding   BLOB NOT NULL,
    dim         INTEGER NOT NULL,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memories_collection ON memories(collection);
"""


@dataclass(slots=True)
class MemoryHit:
    id: str
    text: str
    score: float
    metadata: dict[str, Any]

    def render(self) -> str:
        source = self.metadata.get("source")
        tag = f" (source: {source})" if source else ""
        return f"[{self.score:.3f}]{tag} {self.text}"


def _pack(vector: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


def _unpack(blob: bytes, dim: int) -> tuple[float, ...]:
    return struct.unpack(f"<{dim}f", blob)


def _normalise(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0:
        return list(vector)
    return [x / norm for x in vector]


class VectorStore:
    """Cosine-similarity store over Ollama embeddings.

    Vectors are L2-normalised on write, so similarity is a plain dot product.
    """

    def __init__(
        self,
        db_path: str | Path,
        client: OllamaClient,
        *,
        collection: str = "default",
        embed_model: str | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.client = client
        self.collection = collection
        self.embed_model = embed_model
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> VectorStore:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    async def add(
        self,
        texts: Sequence[str],
        metadatas: Sequence[dict[str, Any]] | None = None,
    ) -> list[str]:
        """Embed and store texts. Returns the new ids."""
        texts = [t for t in texts if t and t.strip()]
        if not texts:
            return []
        metas = list(metadatas or [{} for _ in texts])
        if len(metas) != len(texts):
            raise ValueError("metadatas must be the same length as texts")

        vectors = await self.client.embed(texts, model=self.embed_model)
        if len(vectors) != len(texts):
            raise ValueError(
                f"Embedding model returned {len(vectors)} vectors for {len(texts)} inputs"
            )

        ids: list[str] = []
        now = time.time()
        rows = []
        for text, meta, vector in zip(texts, metas, vectors, strict=True):
            unit = _normalise(vector)
            row_id = uuid.uuid4().hex
            ids.append(row_id)
            rows.append(
                (
                    row_id,
                    self.collection,
                    text,
                    json.dumps(meta, default=str),
                    _pack(unit),
                    len(unit),
                    now,
                )
            )
        self._conn.executemany(
            "INSERT INTO memories (id, collection, text, metadata, embedding, dim, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()
        return ids

    async def search(self, query: str, top_k: int = 5) -> list[MemoryHit]:
        """Return the `top_k` most similar memories in this collection."""
        if not query.strip():
            return []
        rows = self._conn.execute(
            "SELECT id, text, metadata, embedding, dim FROM memories WHERE collection = ?",
            (self.collection,),
        ).fetchall()
        if not rows:
            return []

        query_vec = _normalise((await self.client.embed([query], model=self.embed_model))[0])
        dim = len(query_vec)
        usable = [r for r in rows if r[4] == dim]
        if not usable:
            # Dimension mismatch means the embed model changed under the store.
            raise ValueError(
                f"No stored vectors of dimension {dim}: the embedding model changed. "
                f"Delete {self.db_path} and re-index."
            )

        if _np is not None:
            matrix = _np.frombuffer(b"".join(r[3] for r in usable), dtype=_np.float32)
            matrix = matrix.reshape(len(usable), dim)
            scores = matrix @ _np.asarray(query_vec, dtype=_np.float32)
            order = _np.argsort(-scores)[:top_k]
            picked = [(float(scores[i]), usable[i]) for i in order]
        else:
            scored = [
                (sum(a * b for a, b in zip(_unpack(r[3], dim), query_vec, strict=True)), r)
                for r in usable
            ]
            scored.sort(key=lambda pair: pair[0], reverse=True)
            picked = scored[:top_k]

        return [
            MemoryHit(
                id=row[0],
                text=row[1],
                score=score,
                metadata=json.loads(row[2] or "{}"),
            )
            for score, row in picked
        ]

    def count(self) -> int:
        cur = self._conn.execute(
            "SELECT COUNT(*) FROM memories WHERE collection = ?", (self.collection,)
        )
        return int(cur.fetchone()[0])

    def clear(self) -> int:
        cur = self._conn.execute(
            "DELETE FROM memories WHERE collection = ?", (self.collection,)
        )
        self._conn.commit()
        return cur.rowcount


def chunk_text(text: str, *, chunk_chars: int = 1200, overlap: int = 150) -> list[str]:
    """Split text on paragraph boundaries, falling back to hard slices.

    Paragraph-aware because arbitrary character cuts routinely bisect a fact and
    make both halves useless at retrieval time.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_chars:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        if len(para) > chunk_chars:
            if current:
                chunks.append("\n\n".join(current))
                current, size = [], 0
            step = chunk_chars - overlap
            for start in range(0, len(para), step):
                chunks.append(para[start : start + chunk_chars])
            continue
        if size + len(para) > chunk_chars and current:
            chunks.append("\n\n".join(current))
            tail = current[-1] if len(current[-1]) <= overlap else current[-1][-overlap:]
            current, size = [tail], len(tail)
        current.append(para)
        size += len(para) + 2
    if current:
        chunks.append("\n\n".join(current))
    return [c for c in chunks if c.strip()]
