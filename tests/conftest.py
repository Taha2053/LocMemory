"""
Test fixtures for LocMemory.

Tests run against the current graph-based memory API
(GraphManager + GraphRetriever + MemoryExtractor + MemoryClassifier).
"""

from unittest.mock import MagicMock, patch

import pytest

from core.memory.graph import GraphManager
from core.memory.classifier import MemoryClassifier


@pytest.fixture
def temp_db(tmp_path):
    """Path to a throwaway SQLite file for one test."""
    return str(tmp_path / "test.db")


@pytest.fixture
def gm(temp_db):
    """
    Fresh GraphManager per test.

    The context manager handles initialize_db/load_graph/save/close.
    """
    with GraphManager(temp_db) as manager:
        yield manager


@pytest.fixture(scope="session")
def classifier():
    """
    Real MemoryClassifier shared across the session.

    Loading SentenceTransformer is the slow part (~10s), so we pay it once.
    We disable the Ollama fallback so tests don't hang on a missing server.
    """
    return MemoryClassifier(use_fallback=False)


@pytest.fixture
def mock_ollama_ok():
    """Patch requests.post to return a configurable fake Ollama response."""
    with patch("requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "[]"}
        mock_post.return_value = mock_response
        yield mock_post


def make_ollama_response(text: str):
    """Build a mock response object shaped like requests.Response."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"response": text}
    return resp
