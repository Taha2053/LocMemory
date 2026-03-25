import csv
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
RETRIEVAL_LOG = LOG_DIR / "retrieval_log.csv"


@dataclass
class RetrievalLog:
    timestamp: str
    query: str
    top_k: int
    category_filter: Optional[str]
    num_results: int
    top1_score: float
    top3_scores: list[float]
    latency_ms: float
    precision_at_k: Optional[float] = None


class RetrievalLogger:
    def __init__(self, log_file: Path = RETRIEVAL_LOG):
        self.log_file = log_file
        self._ensure_header()

    def _ensure_header(self):
        if not self.log_file.exists():
            self.log_file.write_text(
                "timestamp,query,top_k,category_filter,num_results,top1_score,top3_score1,top3_score2,top3_score3,latency_ms,precision_at_k\n"
            )

    def log(
        self,
        query: str,
        results: list,
        latency_ms: float,
        category: Optional[str] = None,
        relevant_ids: Optional[list] = None,
    ):
        top_k = len(results)
        top1 = results[0][1]["total"] if results else 0.0
        top3_scores = [results[i][1]["total"] for i in range(min(3, len(results)))]
        while len(top3_scores) < 3:
            top3_scores.append(0.0)

        precision_at_k = None
        if relevant_ids:
            retrieved_ids = [r[0].id for r in results]
            relevant_retrieved = len(set(retrieved_ids) & set(relevant_ids))
            precision_at_k = relevant_retrieved / top_k if top_k > 0 else 0.0

        log_entry = RetrievalLog(
            timestamp=datetime.utcnow().isoformat(timespec="seconds") + "Z",
            query=query,
            top_k=top_k,
            category_filter=category,
            num_results=len(results),
            top1_score=top1,
            top3_scores=top3_scores,
            latency_ms=latency_ms,
            precision_at_k=precision_at_k,
        )

        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    log_entry.timestamp,
                    log_entry.query,
                    log_entry.top_k,
                    log_entry.category_filter or "",
                    log_entry.num_results,
                    f"{log_entry.top1_score:.4f}",
                    f"{log_entry.top3_scores[0]:.4f}",
                    f"{log_entry.top3_scores[1]:.4f}",
                    f"{log_entry.top3_scores[2]:.4f}",
                    f"{log_entry.latency_ms:.2f}",
                    f"{log_entry.precision_at_k:.4f}"
                    if log_entry.precision_at_k
                    else "",
                ]
            )

    def export_csv(self, output_path: Path = None) -> Path:
        if output_path is None:
            output_path = (
                LOG_DIR
                / f"retrieval_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
        if self.log_file.exists():
            output_path.write_text(self.log_file.read_text())
        return output_path

    def compute_precision_at_k(self, test_queries: list[tuple[str, list[str]]]) -> dict:
        total_precision = 0.0
        for query, relevant_ids in test_queries:
            from core.memory import MemoryStore

            store = MemoryStore()
            start = time.time()
            results = store.search(query, top_k=len(relevant_ids))
            latency = (time.time() - start) * 1000

            retrieved_ids = [r[0].id for r in results]
            relevant_retrieved = len(set(retrieved_ids) & set(relevant_ids))
            precision = relevant_retrieved / len(relevant_ids) if relevant_ids else 0.0
            total_precision += precision

            self.log(query, results, latency, relevant_ids=relevant_ids)

        return {
            "precision_at_k": total_precision / len(test_queries)
            if test_queries
            else 0.0,
            "num_queries": len(test_queries),
        }
