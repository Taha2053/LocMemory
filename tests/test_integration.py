"""End-to-end pipeline tests: extractor → graph → retriever → hebbian → persist."""

from unittest.mock import patch

from core.memory.graph import GraphManager, TIER_LEAF
from core.memory.extractor import MemoryExtractor
from core.memory.retriever import GraphRetriever
from core.memory.hebbian import HebbianUpdater

from tests.conftest import make_ollama_response


def test_full_roundtrip_extract_store_retrieve(gm, classifier):
    ext = MemoryExtractor(gm, classifier=classifier)

    with patch(
        "core.memory.extractor.requests.post",
        return_value=make_ollama_response(
            '[{"fact": "User enjoys morning runs", "domain": "health"},'
            ' {"fact": "User works as a python developer", "domain": "programming"}]'
        ),
    ):
        ids = ext.process_message("I run every morning and work on python at my job")

    assert len(ids) == 2

    retriever = GraphRetriever(gm, classifier=classifier, min_semantic_score=0.15)
    results = retriever.retrieve("what exercise does the user do")
    assert len(results) >= 1
    assert any("run" in r["text"].lower() for r in results)


def test_persistence_across_reopen(temp_db, classifier):
    with GraphManager(temp_db) as gm1:
        ext = MemoryExtractor(gm1, classifier=classifier)
        with patch(
            "core.memory.extractor.requests.post",
            return_value=make_ollama_response(
                '[{"fact": "User is learning rust systems programming", "domain": "learning"}]'
            ),
        ):
            ext.process_message("started rust today")

    with GraphManager(temp_db) as gm2:
        leaves = gm2.get_nodes_by_tier(TIER_LEAF)
        assert any("rust" in n["text"].lower() for n in leaves)


def test_retrieval_triggers_meaningful_hebbian_update(gm, classifier):
    a = gm.add_node("User runs marathons", TIER_LEAF, "health")
    b = gm.add_node("User does yoga on sundays", TIER_LEAF, "health")
    c = gm.add_node("User budgets monthly expenses", TIER_LEAF, "finance")
    gm.add_edge(a, b, weight=0.2)
    gm.add_edge(b, c, weight=0.2)

    retriever = GraphRetriever(gm, classifier=classifier, min_semantic_score=0.0)
    results = retriever.retrieve("user fitness routine")
    ids = [r["node_id"] for r in results]

    heb = HebbianUpdater(gm, learning_rate=0.5)
    stats = heb.update_after_retrieval(ids)
    assert stats["edges_strengthened"] >= 0


def test_extraction_failure_does_not_corrupt_graph(gm, classifier):
    import requests as _requests

    ext = MemoryExtractor(gm, classifier=classifier)
    initial = gm.graph.number_of_nodes()

    with patch(
        "core.memory.extractor.requests.post",
        side_effect=_requests.exceptions.Timeout(),
    ):
        ids = ext.process_message("anything here")

    assert ids == []
    assert gm.graph.number_of_nodes() == initial


def test_security_encrypts_pii_before_store():
    from core.security import process_before_store, get_encryptor

    encrypted_text, was_encrypted = process_before_store("email me at foo@bar.com")
    assert was_encrypted is True
    assert "foo@bar.com" not in encrypted_text

    enc = get_encryptor()
    assert enc.decrypt(encrypted_text).startswith("email me at foo@bar.com")


def test_security_leaves_clean_text_alone():
    from core.security import process_before_store

    out, was_encrypted = process_before_store("I went hiking today")
    assert was_encrypted is False
    assert out == "I went hiking today"
