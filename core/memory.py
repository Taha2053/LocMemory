import sqlite3
import hashlib
import json
import uuid
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import math

# ─────────────────────────────────────────────
# 1.  Memory dataclass + Category detection
# ─────────────────────────────────────────────


CATEGORY_KEYWORDS = {
    "work": [
        "work",
        "job",
        "office",
        "project",
        "coding",
        "developer",
        "meeting",
        "deadline",
        "client",
        "task",
        "bug",
        "feature",
        "deploy",
        "team",
    ],
    "learning": [
        "learn",
        "study",
        "course",
        "tutorial",
        "book",
        "research",
        "practice",
        "training",
        "education",
        "knowledge",
        "skill",
    ],
    "personal": [
        "family",
        "friend",
        "home",
        "house",
        "weekend",
        "vacation",
        "hobby",
        "sport",
        "exercise",
        "gym",
        "cook",
        "game",
    ],
    "fact": [
        "fact",
        "remember",
        "know",
        "is",
        "are",
        "was",
        "were",
        "lives",
        "born",
        "name",
    ],
    "todo": [
        "buy",
        "need",
        "should",
        "must",
        "remember to",
        "todo",
        "task",
        "appointment",
        "schedule",
    ],
}


def category_detect(text: str) -> str:
    """Detect category based on keyword matching."""
    text_lower = text.lower()
    best_category = "general"
    best_score = 0

    for category, keywords in CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > best_score:
            best_score = score
            best_category = category

    return best_category


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

    def add(self, text: str, category: str = None) -> Memory:
        """
        Embed text → save to SQLite → write .md file.
        Auto-detects category if not provided.
        Returns the stored Memory object.
        """
        detected_category = category if category else category_detect(text)

        # Build Memory object
        mem = Memory(
            id=str(uuid.uuid4()),
            text=text,
            timestamp=datetime.utcnow().isoformat(timespec="seconds") + "Z",
            category=detected_category,
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

    def _recency_score(self, timestamp: str) -> float:
        """Calculate recency score: 1/(1+days_old). Newer = higher score."""
        try:
            mem_time = datetime.fromisoformat(timestamp.rstrip("Z"))
            days_old = (datetime.utcnow() - mem_time).total_seconds() / 86400
            return 1 / (1 + days_old)
        except ValueError:
            return 0.0

    def search(
        self, query: str, top_k: int = 3, category: str = None
    ) -> list[tuple[Memory, dict]]:
        """
        Embed query → score every memory using hybrid scorer.
        Returns list of (memory, score_breakdown_dict) tuples.
        Score breakdown: {'cosine': x, 'recency': y, 'category_bonus': z, 'total': t}
        Filter by category if provided.
        """
        query_vec = self._embed(query)
        all_memories = self.load_all()

        if category:
            all_memories = [m for m in all_memories if m.category == category]

        scored = []
        for mem in all_memories:
            cosine = self._cosine(query_vec, mem.embedding)
            recency = self._recency_score(mem.timestamp)
            category_bonus = 0.1 if category and mem.category == category else 0.0
            total = 0.6 * cosine + 0.3 * recency + 0.1 * category_bonus
            breakdown = {
                "cosine": cosine,
                "recency": recency,
                "category_bonus": category_bonus,
                "total": total,
            }
            scored.append((mem, breakdown))

        scored.sort(key=lambda x: x[1]["total"], reverse=True)
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
    
    def close(self):
        self.conn.close()

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
