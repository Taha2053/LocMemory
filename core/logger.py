"""
core/logger.py
──────────────────────────────────────────────────────────────────
Retrieval quality logger for LocMemory.

Records every retrieval event to a CSV file so the metrics panel
(WK8) and the RL reward function (WK11) can read live analytics.

CSV columns:
  id                – UUID for this log entry
  timestamp         – ISO-8601 UTC
  query             – raw query text
  query_domain      – classified domain of the query
  result_count      – number of memories returned
  avg_score         – mean score across returned memories
  top_score         – highest individual score
  keyword_overlap   – fraction of query keywords found in top result text
  response_length   – character count of the LLM response (0 if not logged)
  user_rating       – 1–5 rating from user (null until rated)
  latency_ms        – wall-clock retrieval time in milliseconds (0 if unknown)
"""

from __future__ import annotations

import csv
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Optional

# ────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────

DEFAULT_LOG_PATH = Path("data/retrieval_log.csv")

_STOPWORDS = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would could should may might must can i you he she it we they "
    "me him her us them my your his its our their what which who when "
    "where why how and or but not of in on at to for with as by from".split()
)

CSV_FIELDS = [
    "id",
    "timestamp",
    "query",
    "query_domain",
    "result_count",
    "avg_score",
    "top_score",
    "keyword_overlap",
    "response_length",
    "user_rating",
    "latency_ms",
]


# ────────────────────────────────────────────────────────────────
# Public helpers
# ────────────────────────────────────────────────────────────────

def compute_keyword_overlap(query: str, text: str) -> float:
    """
    Fraction of meaningful query keywords that appear in *text*.

    Returns a float in [0.0, 1.0].  Returns 0.0 when the query has no
    meaningful keywords (all stop-words or empty).
    """
    query_words = set(re.findall(r"\b\w+\b", query.lower())) - _STOPWORDS
    if not query_words:
        return 0.0
    text_lower = text.lower()
    hits = sum(1 for w in query_words if w in text_lower)
    return round(hits / len(query_words), 4)


def compute_precision_at_k(
    logs: list[dict],
    k: int = 5,
    rating_threshold: int = 3,
) -> float:
    """
    Precision@K: fraction of the most-recent K rated entries with
    user_rating >= rating_threshold.

    Ignores entries with no user_rating.  Returns 0.0 when there are
    fewer than one rated entry.
    """
    rated = [e for e in logs if e.get("user_rating") not in (None, "", "null")]
    recent_k = rated[-k:]
    if not recent_k:
        return 0.0
    hits = sum(
        1 for e in recent_k if int(e["user_rating"]) >= rating_threshold
    )
    return round(hits / len(recent_k), 4)


# ────────────────────────────────────────────────────────────────
# RetrievalLogger
# ────────────────────────────────────────────────────────────────

class RetrievalLogger:
    """
    Thread-safe CSV logger for retrieval quality metrics.

    Usage:
        logger = RetrievalLogger()                 # default path
        entry_id = logger.log_retrieval(...)
        logger.log_rating(entry_id, rating=4)
        rows = logger.get_recent_logs(50)
    """

    def __init__(self, log_path: str | Path = DEFAULT_LOG_PATH) -> None:
        self._path = Path(log_path)
        self._lock = Lock()
        self._ensure_file()

    # ── public api ────────────────────────────────────────────────

    def log_retrieval(
        self,
        query: str,
        results: list[dict],
        query_domain: str = "",
        response_text: str = "",
        latency_ms: float = 0.0,
    ) -> str:
        """
        Append one retrieval event.  Returns the entry UUID so the
        caller can later attach a user rating.

        Args:
            query        : raw user query
            results      : list of memory dicts from GraphRetriever.retrieve()
            query_domain : domain classified for the query
            response_text: LLM response text (empty when not generated)
            latency_ms   : wall-clock retrieval time

        Returns:
            entry_id (str UUID)
        """
        entry_id = str(uuid.uuid4())
        scores = [r.get("score", 0.0) for r in results if "score" in r]
        top_text = results[0].get("text", "") if results else ""

        row = {
            "id": entry_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "query_domain": query_domain,
            "result_count": len(results),
            "avg_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
            "top_score": round(max(scores), 4) if scores else 0.0,
            "keyword_overlap": compute_keyword_overlap(query, top_text),
            "response_length": len(response_text),
            "user_rating": "",
            "latency_ms": round(latency_ms, 1),
        }

        with self._lock:
            with self._path.open("a", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
                writer.writerow(row)

        return entry_id

    def log_rating(self, entry_id: str, rating: int) -> bool:
        """
        Attach a user rating (1–5) to an existing log entry.

        Rewrites the CSV in-place.  Returns True when the entry was found
        and updated, False otherwise.
        """
        if not (1 <= rating <= 5):
            raise ValueError(f"rating must be 1–5, got {rating}")

        with self._lock:
            rows = self._read_rows()
            found = False
            for row in rows:
                if row["id"] == entry_id:
                    row["user_rating"] = rating
                    found = True
                    break
            if found:
                self._write_rows(rows)

        return found

    def get_recent_logs(self, n: int = 100) -> list[dict]:
        """Return the most-recent *n* log entries (newest last)."""
        with self._lock:
            rows = self._read_rows()
        return rows[-n:]

    def get_summary(self, n: int = 100) -> dict:
        """
        Aggregate metrics over the last *n* entries, ready for the
        /api/metrics endpoint.
        """
        rows = self.get_recent_logs(n)
        if not rows:
            return {
                "total_retrievals": 0,
                "avg_result_count": 0.0,
                "avg_score": 0.0,
                "avg_keyword_overlap": 0.0,
                "avg_latency_ms": 0.0,
                "precision_at_5": 0.0,
                "rated_count": 0,
                "domain_distribution": {},
                "recent": [],
            }

        def _f(key: str) -> float:
            vals = [float(r[key]) for r in rows if r.get(key) not in ("", None)]
            return round(sum(vals) / len(vals), 4) if vals else 0.0

        domain_dist: dict[str, int] = {}
        for r in rows:
            d = r.get("query_domain") or "(none)"
            domain_dist[d] = domain_dist.get(d, 0) + 1

        rated = [r for r in rows if r.get("user_rating") not in ("", None)]

        return {
            "total_retrievals": len(rows),
            "avg_result_count": _f("result_count"),
            "avg_score": _f("avg_score"),
            "avg_keyword_overlap": _f("keyword_overlap"),
            "avg_latency_ms": _f("latency_ms"),
            "precision_at_5": compute_precision_at_k(rows, k=5),
            "rated_count": len(rated),
            "domain_distribution": domain_dist,
            "recent": [
                {
                    "id": r["id"],
                    "timestamp": r["timestamp"],
                    "query": r["query"],
                    "query_domain": r.get("query_domain", ""),
                    "result_count": int(r["result_count"] or 0),
                    "avg_score": float(r["avg_score"] or 0),
                    "keyword_overlap": float(r["keyword_overlap"] or 0),
                    "latency_ms": float(r["latency_ms"] or 0),
                    "user_rating": int(r["user_rating"]) if r.get("user_rating") else None,
                }
                for r in rows[-20:]
            ],
        }

    # ── private helpers ───────────────────────────────────────────

    def _ensure_file(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            with self._path.open("w", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, fieldnames=CSV_FIELDS).writeheader()

    def _read_rows(self) -> list[dict]:
        with self._path.open("r", newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))

    def _write_rows(self, rows: list[dict]) -> None:
        with self._path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)


# ────────────────────────────────────────────────────────────────
# Module-level singleton (shared by backend and chat.py)
# ────────────────────────────────────────────────────────────────

_default_logger: Optional[RetrievalLogger] = None


def get_logger(log_path: str | Path = DEFAULT_LOG_PATH) -> RetrievalLogger:
    """Return (and lazily create) the module-level singleton logger."""
    global _default_logger
    if _default_logger is None:
        _default_logger = RetrievalLogger(log_path)
    return _default_logger
