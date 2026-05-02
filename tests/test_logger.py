"""
Tests for core/logger.py — retrieval quality logger.
"""

import csv
import time
from pathlib import Path

import pytest

from core.logger import (
    RetrievalLogger,
    compute_keyword_overlap,
    compute_precision_at_k,
    CSV_FIELDS,
)


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def log_path(tmp_path: Path) -> Path:
    return tmp_path / "test_log.csv"


@pytest.fixture
def logger(log_path: Path) -> RetrievalLogger:
    return RetrievalLogger(log_path)


def _make_results(n: int = 3, score: float = 0.8) -> list[dict]:
    return [{"node_id": f"n{i}", "text": f"fact {i}", "score": score} for i in range(n)]


# ─────────────────────────────────────────────
# compute_keyword_overlap
# ─────────────────────────────────────────────

class TestKeywordOverlap:
    def test_full_overlap(self):
        assert compute_keyword_overlap("python programming", "python is a programming language") == 1.0

    def test_partial_overlap(self):
        score = compute_keyword_overlap("python java rust", "python is great")
        assert 0.0 < score < 1.0

    def test_no_overlap(self):
        assert compute_keyword_overlap("python programming", "cats are friendly") == 0.0

    def test_only_stopwords_returns_zero(self):
        assert compute_keyword_overlap("the is a", "some text here") == 0.0

    def test_empty_query_returns_zero(self):
        assert compute_keyword_overlap("", "some text here") == 0.0

    def test_case_insensitive(self):
        assert compute_keyword_overlap("Python", "python language") == 1.0


# ─────────────────────────────────────────────
# compute_precision_at_k
# ─────────────────────────────────────────────

class TestPrecisionAtK:
    def test_all_high_ratings(self):
        logs = [{"user_rating": "5"}, {"user_rating": "4"}, {"user_rating": "3"}]
        assert compute_precision_at_k(logs, k=3) == 1.0

    def test_all_low_ratings(self):
        logs = [{"user_rating": "1"}, {"user_rating": "2"}]
        assert compute_precision_at_k(logs, k=5) == 0.0

    def test_mixed_ratings(self):
        logs = [{"user_rating": "5"}, {"user_rating": "1"}, {"user_rating": "5"}, {"user_rating": "1"}]
        assert compute_precision_at_k(logs, k=4) == 0.5

    def test_ignores_unrated_entries(self):
        logs = [{"user_rating": ""}, {"user_rating": None}, {"user_rating": "5"}]
        assert compute_precision_at_k(logs, k=5) == 1.0

    def test_empty_logs_returns_zero(self):
        assert compute_precision_at_k([]) == 0.0

    def test_all_unrated_returns_zero(self):
        logs = [{"user_rating": ""}, {"user_rating": ""}]
        assert compute_precision_at_k(logs) == 0.0


# ─────────────────────────────────────────────
# RetrievalLogger — file creation
# ─────────────────────────────────────────────

class TestLoggerInit:
    def test_creates_csv_with_header(self, log_path: Path, logger: RetrievalLogger):
        assert log_path.exists()
        with log_path.open() as f:
            header = f.readline().strip().split(",")
        assert header == CSV_FIELDS

    def test_creates_parent_dirs(self, tmp_path: Path):
        nested = tmp_path / "a" / "b" / "log.csv"
        RetrievalLogger(nested)
        assert nested.exists()


# ─────────────────────────────────────────────
# RetrievalLogger — log_retrieval
# ─────────────────────────────────────────────

class TestLogRetrieval:
    def test_returns_uuid_string(self, logger: RetrievalLogger):
        eid = logger.log_retrieval("test query", _make_results())
        assert isinstance(eid, str) and len(eid) == 36

    def test_row_written_to_csv(self, logger: RetrievalLogger, log_path: Path):
        logger.log_retrieval("hello world", _make_results(2, 0.9))
        rows = list(csv.DictReader(log_path.open()))
        assert len(rows) == 1
        assert rows[0]["query"] == "hello world"
        assert rows[0]["result_count"] == "2"

    def test_scores_computed_correctly(self, logger: RetrievalLogger, log_path: Path):
        logger.log_retrieval("q", _make_results(3, 0.6))
        rows = list(csv.DictReader(log_path.open()))
        assert float(rows[0]["avg_score"]) == pytest.approx(0.6, abs=1e-3)
        assert float(rows[0]["top_score"]) == pytest.approx(0.6, abs=1e-3)

    def test_empty_results_logged_cleanly(self, logger: RetrievalLogger, log_path: Path):
        logger.log_retrieval("nothing", [])
        rows = list(csv.DictReader(log_path.open()))
        assert rows[0]["result_count"] == "0"
        assert rows[0]["avg_score"] == "0.0"

    def test_query_domain_stored(self, logger: RetrievalLogger, log_path: Path):
        logger.log_retrieval("rust ownership", _make_results(), query_domain="programming")
        rows = list(csv.DictReader(log_path.open()))
        assert rows[0]["query_domain"] == "programming"

    def test_response_length_stored(self, logger: RetrievalLogger, log_path: Path):
        logger.log_retrieval("q", [], response_text="hello world")
        rows = list(csv.DictReader(log_path.open()))
        assert rows[0]["response_length"] == "11"

    def test_latency_stored(self, logger: RetrievalLogger, log_path: Path):
        logger.log_retrieval("q", [], latency_ms=123.4)
        rows = list(csv.DictReader(log_path.open()))
        assert float(rows[0]["latency_ms"]) == pytest.approx(123.4, abs=0.5)

    def test_user_rating_empty_by_default(self, logger: RetrievalLogger, log_path: Path):
        logger.log_retrieval("q", [])
        rows = list(csv.DictReader(log_path.open()))
        assert rows[0]["user_rating"] == ""

    def test_multiple_entries_appended(self, logger: RetrievalLogger, log_path: Path):
        for i in range(5):
            logger.log_retrieval(f"query {i}", [])
        rows = list(csv.DictReader(log_path.open()))
        assert len(rows) == 5

    def test_keyword_overlap_populated(self, logger: RetrievalLogger, log_path: Path):
        results = [{"node_id": "x", "text": "python programming language", "score": 0.9}]
        logger.log_retrieval("python programming", results)
        rows = list(csv.DictReader(log_path.open()))
        assert float(rows[0]["keyword_overlap"]) > 0


# ─────────────────────────────────────────────
# RetrievalLogger — log_rating
# ─────────────────────────────────────────────

class TestLogRating:
    def test_updates_existing_entry(self, logger: RetrievalLogger, log_path: Path):
        eid = logger.log_retrieval("q", [])
        result = logger.log_rating(eid, 4)
        assert result is True
        rows = list(csv.DictReader(log_path.open()))
        assert rows[0]["user_rating"] == "4"

    def test_returns_false_for_unknown_id(self, logger: RetrievalLogger):
        result = logger.log_rating("nonexistent-id", 3)
        assert result is False

    def test_rejects_rating_below_1(self, logger: RetrievalLogger):
        eid = logger.log_retrieval("q", [])
        with pytest.raises(ValueError):
            logger.log_rating(eid, 0)

    def test_rejects_rating_above_5(self, logger: RetrievalLogger):
        eid = logger.log_retrieval("q", [])
        with pytest.raises(ValueError):
            logger.log_rating(eid, 6)

    def test_only_target_entry_modified(self, logger: RetrievalLogger, log_path: Path):
        eid1 = logger.log_retrieval("first", [])
        eid2 = logger.log_retrieval("second", [])
        logger.log_rating(eid1, 5)
        rows = list(csv.DictReader(log_path.open()))
        assert rows[0]["user_rating"] == "5"
        assert rows[1]["user_rating"] == ""


# ─────────────────────────────────────────────
# RetrievalLogger — get_recent_logs
# ─────────────────────────────────────────────

class TestGetRecentLogs:
    def test_returns_list(self, logger: RetrievalLogger):
        assert isinstance(logger.get_recent_logs(), list)

    def test_empty_when_no_entries(self, logger: RetrievalLogger):
        assert logger.get_recent_logs() == []

    def test_respects_n_limit(self, logger: RetrievalLogger):
        for i in range(10):
            logger.log_retrieval(f"q{i}", [])
        assert len(logger.get_recent_logs(5)) == 5

    def test_returns_all_when_fewer_than_n(self, logger: RetrievalLogger):
        for i in range(3):
            logger.log_retrieval(f"q{i}", [])
        assert len(logger.get_recent_logs(100)) == 3


# ─────────────────────────────────────────────
# RetrievalLogger — get_summary
# ─────────────────────────────────────────────

class TestGetSummary:
    def test_empty_summary_structure(self, logger: RetrievalLogger):
        s = logger.get_summary()
        assert s["total_retrievals"] == 0
        assert s["precision_at_5"] == 0.0
        assert s["recent"] == []

    def test_counts_total_retrievals(self, logger: RetrievalLogger):
        for i in range(7):
            logger.log_retrieval(f"q{i}", [])
        s = logger.get_summary()
        assert s["total_retrievals"] == 7

    def test_domain_distribution_populated(self, logger: RetrievalLogger):
        logger.log_retrieval("q", [], query_domain="programming")
        logger.log_retrieval("q", [], query_domain="health")
        logger.log_retrieval("q", [], query_domain="programming")
        s = logger.get_summary()
        assert s["domain_distribution"]["programming"] == 2
        assert s["domain_distribution"]["health"] == 1

    def test_avg_latency_computed(self, logger: RetrievalLogger):
        logger.log_retrieval("q", [], latency_ms=100.0)
        logger.log_retrieval("q", [], latency_ms=200.0)
        s = logger.get_summary()
        assert s["avg_latency_ms"] == pytest.approx(150.0, abs=1.0)

    def test_precision_at_5_with_ratings(self, logger: RetrievalLogger):
        for _ in range(5):
            eid = logger.log_retrieval("q", [])
            logger.log_rating(eid, 5)
        s = logger.get_summary()
        assert s["precision_at_5"] == 1.0

    def test_recent_capped_at_20(self, logger: RetrievalLogger):
        for i in range(25):
            logger.log_retrieval(f"q{i}", [])
        s = logger.get_summary()
        assert len(s["recent"]) == 20

    def test_rated_count(self, logger: RetrievalLogger):
        eid1 = logger.log_retrieval("q1", [])
        logger.log_retrieval("q2", [])
        logger.log_rating(eid1, 3)
        s = logger.get_summary()
        assert s["rated_count"] == 1
