import sqlite3
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import math


# ─────────────────────────────────────────────
# 1.  Memory dataclass
# ─────────────────────────────────────────────


@dataclass
class Memory:
    id: str  # SHA-256 of text (deterministic & unique)
    text: str  # Raw text content
    timestamp: str  # ISO-8601 string
    category: str  # e.g. "fact", "event", "todo" …
    embedding: list[float]  # Dense vector from embedding model


# ─────────────────────────────────────────────
# 2.  MemoryStore
# ─────────────────────────────────────────────


class MemoryStore:
    def __init__(
        self,
        db_path: str = "data/memories.db",
        md_dir: str = "memories",
        model_name: str = "all-MiniLM-L6-v2",  # used when real model is available
    ):
        # ── Embedding model ──────────────────────────────────────────────
        self.model = self._load_embedding_model(model_name)

        # ── SQLite ───────────────────────────────────────────────────────
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_table()

        # ── Markdown output directory ─────────────────────────────────────
        self.md_dir = Path(md_dir)
        self.md_dir.mkdir(parents=True, exist_ok=True)

        print(f"MemoryStore ready — db: '{db_path}'  |  md dir: '{md_dir}/'")

    # ── private helpers ───────────────────────────────────────────────────

    def _load_embedding_model(self, model_name: str):
        """
        Try to load a real sentence-transformer.
        Falls back to a hash-based mock so the rest of the system still works.
        Swap this out once you have `pip install sentence-transformers`.
        """
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(model_name)
            print(f"Loaded embedding model: {model_name}")
            return model
        except ImportError:
            print("sentence-transformers not installed — using mock embedder.")
            return None

    def _embed(self, text: str) -> list[float]:
        """Return a 384-dim embedding vector (or mock if no model)."""
        if self.model is not None:
            return self.model.encode(text).tolist()

        # ── Mock: deterministic 384-dim float vector from SHA-256 ─────────
        digest = hashlib.sha256(text.encode()).digest()  # 32 bytes
        # Tile the digest bytes to reach 384 values, normalise to [-1, 1]
        tiled = (digest * 12)[:384]  # 384 bytes
        return [(b - 127.5) / 127.5 for b in tiled]

    def _create_table(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id        TEXT PRIMARY KEY,
                text      TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                category  TEXT NOT NULL DEFAULT 'general',
                embedding TEXT NOT NULL          -- stored as JSON array
            )
        """)
        self.conn.commit()

    # ── public API ────────────────────────────────────────────────────────

    def add(self, text: str, category: str = "general") -> Memory:
        """
        Embed text → save to SQLite → write .md file.
        Returns the stored Memory object.
        """
        # Build Memory object
        mem = Memory(
            id=hashlib.sha256(text.encode()).hexdigest()[:16],
            text=text,
            timestamp=datetime.utcnow().isoformat(timespec="seconds") + "Z",
            category=category,
            embedding=self._embed(text),
        )

        # ── SQLite insert (ignore duplicates) ─────────────────────────────
        self.conn.execute(
            """
            INSERT OR IGNORE INTO memories (id, text, timestamp, category, embedding)
            VALUES (?, ?, ?, ?, ?)
            """,
            (mem.id, mem.text, mem.timestamp, mem.category, json.dumps(mem.embedding)),
        )
        self.conn.commit()

        # ── Markdown file ─────────────────────────────────────────────────
        md_path = self.md_dir / f"{mem.id}.md"
        md_path.write_text(
            f"---\n"
            f"id: {mem.id}\n"
            f"timestamp: {mem.timestamp}\n"
            f"category: {mem.category}\n"
            f"---\n\n"
            f"# Memory\n\n"
            f"{mem.text}\n"
        )

        print(f"Saved  [{mem.category}]  id={mem.id}  →  {md_path.name}")
        return mem

    def count(self) -> int:
        """Return total number of rows in the DB."""
        row = self.conn.execute("SELECT COUNT(*) FROM memories").fetchone()
        return row[0]

    def all(self) -> list[Memory]:
        """Fetch every stored memory (without re-loading embeddings into vector)."""
        rows = self.conn.execute(
            "SELECT id, text, timestamp, category, embedding FROM memories"
        ).fetchall()
        return [
            Memory(
                id=r[0],
                text=r[1],
                timestamp=r[2],
                category=r[3],
                embedding=json.loads(r[4]),
            )
            for r in rows
        ]

    # ── Retrieval ─────────────────────────────────────────────────
    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        """Cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def search(self, query: str, top_k: int = 3) -> list[tuple[Memory, float]]:
        """
        Embed query → cosine-score every memory → return top-K (memory, score) pairs.
        """
        query_vec = self._embed(query)
        all_memories = self.load_all()

        scored = [(mem, self._cosine(query_vec, mem.embedding)) for mem in all_memories]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def load_all(self) -> list[Memory]:
        """SELECT all rows ordered by timestamp ascending."""
        rows = self.conn.execute(
            """
            SELECT id, text, timestamp, category, embedding
            FROM memories
            ORDER BY timestamp ASC
            """
        ).fetchall()
        return [
            Memory(
                id=r[0],
                text=r[1],
                timestamp=r[2],
                category=r[3],
                embedding=json.loads(r[4]),
            )
            for r in rows
        ]

    def delete(self, memory_id: str) -> bool:
        """
        Remove from SQLite + delete the matching .md file.
        Returns True if something was deleted, False if id not found.
        """
        row = self.conn.execute(
            "SELECT id FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()

        if row is None:
            print(f"id={memory_id} not found — nothing deleted.")
            return False

        # ── SQLite ────────────────────────────────────────────────────────
        self.conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self.conn.commit()

        # ── Markdown file ─────────────────────────────────────────────────
        md_path = self.md_dir / f"{memory_id}.md"
        if md_path.exists():
            md_path.unlink()
            print(f"Deleted  id={memory_id}  +  {md_path.name}")
        else:
            print(f"Deleted  id={memory_id}  (no .md file found)")

        return True
