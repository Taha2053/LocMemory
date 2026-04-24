"""
Test fixtures for LocMemory test suite.
"""

import json
import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from core.memory.graph import GraphManager, TIER_LEAF, TIER_CONTEXT, TIER_ANCHOR
from core.memory.classifier import MemoryClassifier


@pytest.fixture
def tmp_path(tmp_path):
    """Provide a temporary path for test files."""
    return tmp_path


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary SQLite database path."""
    db_path = tmp_path / "test.db"
    yield str(db_path)
    # Cleanup handled by tmp_path fixture


@pytest.fixture
def gm(temp_db):
    """Create a GraphManager with temp database."""
    with GraphManager(temp_db) as manager:
        manager.initialize_db()
        manager.seed_domains()
        manager.load_graph()
        yield manager


@pytest.fixture
def sample_graph(gm):
    """Populate store with 20 leaf nodes across 4 domains."""
    domain_nodes = {
        "health": gm.add_node("health context", TIER_CONTEXT, "health"),
        "work": gm.add_node("work context", TIER_CONTEXT, "work"),
        "personal": gm.add_node("personal context", TIER_CONTEXT, "personal"),
        "programming": gm.add_node("programming context", TIER_CONTEXT, "programming"),
    }

    # Add leaf nodes across domains
    nodes = {
        "health": [
            "Went to the gym this morning",
            "Feeling tired, need more sleep",
            "Diet includes more vegetables now",
            "Running 5km three times a week",
            "Yoga session on Sunday",
        ],
        "work": [
            "Meeting with team tomorrow",
            "Project deadline is Friday",
            "Need to prepare presentation",
            "Client call at 3pm",
            "Updating documentation",
        ],
        "personal": [
            "Weekend plans with family",
            "Movie night on Saturday",
            "Reading a new novel",
            "Cooking experiment tonight",
            "Calling parents this week",
        ],
        "programming": [
            "Writing Python code",
            "Debugging a memory leak",
            "Using Git for version control",
            "Reading API documentation",
            "Code review scheduled",
        ],
    }

    added_ids = []
    for domain, texts in nodes.items():
        anchor_id = gm.add_node(f"{domain} anchor", TIER_ANCHOR, domain)
        for text in texts:
            node_id = gm.add_node(text, TIER_LEAF, domain)
            gm.add_edge(domain_nodes[domain], node_id, "has_memory", 0.8)
            gm.add_edge(anchor_id, node_id, "contains", 0.7)
            added_ids.append(node_id)

    return {
        "ids": added_ids,
        "domains": list(nodes.keys()),
        "domain_node": domain_nodes,
    }


@pytest.fixture
def sample_leaves(gm):
    """Populate with 20 leaf nodes across 4 domains for quick access."""
    domains = ["health", "work", "personal", "programming"]
    ids = []

    for domain in domains:
        ctx = gm.add_node(f"{domain} context", TIER_CONTEXT, domain)
        for i in range(5):
            id_ = gm.add_node(f"{domain} leaf {i}", TIER_LEAF, domain)
            ids.append(id_)
            gm.add_edge(ctx, id_, "has_memory", 0.5)

    return ids


@pytest.fixture
def mock_ollama():
    """Patch requests.post to return fake LLM responses."""
    with patch("requests.post") as mock_post:
        # Default mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": '```json\n[{"fact": "Test fact", "confidence": 0.9, "domain": "general"}]\n```'
        }
        mock_post.return_value = mock_response
        yield mock_post


@pytest.fixture
def sample_embedding():
    """Return a fixed 384-dim numpy vector for testing."""
    np.random.seed(42)
    emb = np.random.randn(384).astype(np.float32)
    emb = emb / np.linalg.norm(emb)  # Normalize
    return emb


@pytest.fixture
def mock_embedding():
    """Mock the embedding model to return fixed vectors."""
    np.random.seed(42)

    def mock_encode(texts, convert_to_numpy=True):
        """Return deterministic embeddings."""
        if isinstance(texts, str):
            texts = [texts]
        embeddings = []
        for text in texts:
            # Simple deterministic hash-based embedding
            h = hash(text) % (2**31)
            np.random.seed(h)
            emb = np.random.randn(384).astype(np.float32)
            emb = emb / np.linalg.norm(emb)
            embeddings.append(emb)
        return np.array(embeddings) if convert_to_numpy else embeddings

    with patch("sentence_transformers.SentenceTransformer.encode", mock_encode):
        yield mock_encode


@pytest.fixture
def classifier_with_mock():
    """MemoryClassifier with mocked embedding model."""
    return MemoryClassifier()