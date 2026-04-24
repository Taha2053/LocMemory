"""Tests for MemoryExtractor: JSON parsing + graph writes (Ollama mocked)."""

from unittest.mock import patch

from core.memory.extractor import MemoryExtractor
from core.memory.graph import TIER_LEAF

from tests.conftest import make_ollama_response


def test_parse_facts_plain_json_array(gm, classifier):
    ext = MemoryExtractor(gm, classifier=classifier)
    out = ext._parse_facts('[{"fact": "User likes coffee", "domain": "personal"}]')
    assert len(out) == 1
    assert out[0]["fact"] == "User likes coffee"
    assert out[0]["domain"] == "personal"


def test_parse_facts_strips_markdown_fences(gm, classifier):
    ext = MemoryExtractor(gm, classifier=classifier)
    raw = '```json\n[{"fact": "Fact from markdown", "domain": "work"}]\n```'
    out = ext._parse_facts(raw)
    assert len(out) == 1
    assert out[0]["fact"] == "Fact from markdown"


def test_parse_facts_invalid_returns_empty(gm, classifier):
    ext = MemoryExtractor(gm, classifier=classifier)
    assert ext._parse_facts("not json at all {{{") == []


def test_parse_facts_drops_too_short(gm, classifier):
    ext = MemoryExtractor(gm, classifier=classifier)
    out = ext._parse_facts('[{"fact": "ab", "domain": "x"}, {"fact": "long enough", "domain": "x"}]')
    assert len(out) == 1
    assert out[0]["fact"] == "long enough"


def test_parse_facts_defaults_domain_to_general(gm, classifier):
    ext = MemoryExtractor(gm, classifier=classifier)
    out = ext._parse_facts('[{"fact": "No domain provided here"}]')
    assert out[0]["domain"] == "general"


def test_extract_facts_mocked_ollama_end_to_end(gm, classifier):
    ext = MemoryExtractor(gm, classifier=classifier)
    with patch(
        "core.memory.extractor.requests.post",
        return_value=make_ollama_response(
            '[{"fact": "User loves hiking", "domain": "personal"}]'
        ),
    ):
        facts = ext.extract_facts("I went hiking this weekend")
    assert facts == [{"fact": "User loves hiking", "domain": "personal"}]


def test_extract_facts_handles_connection_error(gm, classifier):
    import requests

    ext = MemoryExtractor(gm, classifier=classifier)
    with patch(
        "core.memory.extractor.requests.post",
        side_effect=requests.exceptions.ConnectionError(),
    ):
        assert ext.extract_facts("anything") == []


def test_process_message_writes_leaf_nodes(gm, classifier):
    ext = MemoryExtractor(gm, classifier=classifier)
    with patch(
        "core.memory.extractor.requests.post",
        return_value=make_ollama_response(
            '[{"fact": "User is learning Rust", "domain": "learning"}]'
        ),
    ):
        ids = ext.process_message("I started learning Rust")

    assert len(ids) == 1
    node = gm.graph.nodes[ids[0]]
    assert node["tier"] == TIER_LEAF
    assert node["domain"] == "learning"
    assert "Rust" in node["text"]


def test_process_message_reclassifies_when_domain_missing(gm, classifier):
    ext = MemoryExtractor(gm, classifier=classifier)
    with patch(
        "core.memory.extractor.requests.post",
        return_value=make_ollama_response(
            '[{"fact": "User debugs python memory leaks regularly", "domain": "general"}]'
        ),
    ):
        ids = ext.process_message("I debugged a python memory leak")

    assert len(ids) == 1
    domain = gm.graph.nodes[ids[0]]["domain"]
    assert domain != "general"
    assert domain in classifier.list_domains()
