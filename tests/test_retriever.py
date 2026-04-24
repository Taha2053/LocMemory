"""Tests for GraphRetriever."""

import time

from core.memory.graph import TIER_CONTEXT, TIER_LEAF
from core.memory.retriever import GraphRetriever


def test_retrieve_returns_list_of_dicts(gm, classifier):
    gm.add_node("I love morning runs", TIER_LEAF, "health")
    retriever = GraphRetriever(gm, classifier=classifier)

    results = retriever.retrieve("running and fitness")
    assert isinstance(results, list)
    for r in results:
        assert {"node_id", "text", "domain", "tier", "score", "depth"} <= r.keys()


def test_retrieve_surfaces_relevant_memories(gm, classifier):
    gm.add_node("User loves hiking in the mountains", TIER_LEAF, "personal")
    gm.add_node("Quarterly tax filing deadline is April 15", TIER_LEAF, "finance")
    retriever = GraphRetriever(gm, classifier=classifier, min_semantic_score=0.15)

    results = retriever.retrieve("tell me about outdoor activities and hiking")
    assert len(results) >= 1
    top_text = results[0]["text"].lower()
    assert "hiking" in top_text


def test_retrieve_filters_out_irrelevant_by_threshold(gm, classifier):
    gm.add_node("Quarterly tax filing deadline is April 15", TIER_LEAF, "finance")
    retriever = GraphRetriever(gm, classifier=classifier, min_semantic_score=0.6)

    results = retriever.retrieve("user loves hiking in the mountains")
    assert results == []


def test_retrieve_results_sorted_by_score_desc(gm, classifier):
    for i in range(6):
        gm.add_node(f"Hiking adventure number {i}", TIER_LEAF, "personal")
    retriever = GraphRetriever(gm, classifier=classifier, min_semantic_score=0.0)

    results = retriever.retrieve("hiking")
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_retrieve_respects_max_candidates(gm, classifier):
    for i in range(30):
        gm.add_node(f"Running session {i}", TIER_LEAF, "health")
    retriever = GraphRetriever(
        gm, classifier=classifier, max_candidates=5, min_semantic_score=0.0
    )

    results = retriever.retrieve("running")
    assert len(results) <= 5


def test_retrieve_empty_graph(gm, classifier):
    retriever = GraphRetriever(gm, classifier=classifier)
    assert retriever.retrieve("anything") == []


def test_retrieve_latency_reasonable(gm, classifier):
    for i in range(40):
        gm.add_node(f"Memory node {i}", TIER_LEAF, "personal")
    retriever = GraphRetriever(gm, classifier=classifier)

    start = time.time()
    retriever.retrieve("test query")
    elapsed_ms = (time.time() - start) * 1000
    assert elapsed_ms < 5000


def test_retrieve_traverses_through_edges(gm, classifier):
    ctx = gm.add_node("personal hub", TIER_CONTEXT, "personal")
    leaf = gm.add_node("User prefers green tea over coffee", TIER_LEAF, "personal")
    gm.add_edge(ctx, leaf, weight=0.9)

    retriever = GraphRetriever(gm, classifier=classifier, min_semantic_score=0.15)
    results = retriever.retrieve("what beverage does the user like")
    texts = [r["text"] for r in results]
    assert any("tea" in t.lower() for t in texts)
