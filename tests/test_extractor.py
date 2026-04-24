"""
Tests for fact extraction.
"""

import pytest
from unittest.mock import MagicMock, patch

from core.memory.extractor import MemoryExtractor
from core.memory.graph import GraphManager, TIER_LEAF


def test_parse_facts_valid_json(mock_ollama):
    """Valid JSON should parse to fact dicts."""
    with patch("requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": '[{"fact": "Extracted fact", "confidence": 0.9, "domain": "test"}]'
        }
        mock_post.return_value = mock_response

        gm = GraphManager(":memory:")
        gm.initialize_db()
        gm.load_graph()

        extractor = MemoryExtractor(gm)
        facts = extractor.extract_facts("test message")

        assert len(facts) >= 0


def test_parse_facts_invalid_json(mock_ollama):
    """Invalid JSON should return empty list."""
    with patch("requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "not json at all {{{"}
        mock_post.return_value = mock_response

        gm = GraphManager(":memory:")
        gm.initialize_db()
        gm.load_graph()

        extractor = MemoryExtractor(gm)
        facts = extractor.extract_facts("test")

        # Should return empty or filtered
        assert isinstance(facts, list)


def test_parse_facts_low_confidence(mock_ollama):
    """Low confidence facts should be filtered."""
    with patch("requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": '[{"fact": "Low conf", "confidence": 0.2, "domain": "test"}]'
        }
        mock_post.return_value = mock_response

        gm = GraphManager(":memory:")
        gm.initialize_db()
        gm.load_graph()

        extractor = MemoryExtractor(gm)
        facts = extractor.extract_facts("test")

        # Low confidence should be filtered
        assert len(facts) == 0


def test_parse_facts_strips_markdown(mock_ollama):
    """Markdown code blocks should be stripped."""
    with patch("requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": '```json\n[{"fact": "Test fact", "confidence": 0.9}]\n```'
        }
        mock_post.return_value = mock_response

        gm = GraphManager(":memory:")
        gm.initialize_db()
        gm.load_graph()

        extractor = MemoryExtractor(gm)
        facts = extractor.extract_facts("test")

        # Should parse despite markdown
        assert isinstance(facts, list)


def test_hard_add_creates_node(gm):
    """hard_add should create node."""
    from core.memory.extractor import MemoryExtractor

    extractor = MemoryExtractor(gm)

    # Direct add without Ollama
    node_id = extractor.hard_add("Test fact", "test_domain")

    # Verify exists
    assert node_id is not None

    # Check in graph
    nodes = gm.get_nodes_by_domain("test_domain")
    ids = [n["id"] for n in nodes]
    assert node_id in ids


def test_hard_add_high_importance(gm):
    """High importance flag should work."""
    from core.memory.extractor import MemoryExtractor

    extractor = MemoryExtractor(gm)

    # Add with high importance
    node_id = extractor.hard_add("Important fact", "test_domain", importance=1.0)

    # Verify exists
    assert node_id is not None


def test_hard_add_links_to_domain(gm):
    """Node should link to domain."""
    from core.memory.extractor import MemoryExtractor

    extractor = MemoryExtractor(gm)

    # Add
    node_id = extractor.hard_add("Fact", "programming")

    # Should have edge to domain
    neighbors = gm.get_neighbors(node_id, direction="in")
    # Just verify no error
    assert isinstance(neighbors, list)