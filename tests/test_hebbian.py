"""Tests for Hebbian edge updates."""

from datetime import datetime, timedelta, timezone

from core.memory.graph import TIER_LEAF
from core.memory.hebbian import HebbianUpdater


def test_strengthen_edges_updates_existing_edge(gm):
    a = gm.add_node("n1", TIER_LEAF, "d")
    b = gm.add_node("n2", TIER_LEAF, "d")
    gm.add_edge(a, b, weight=0.2)

    heb = HebbianUpdater(gm, learning_rate=0.5)
    count = heb.strengthen_edges([a, b])
    assert count == 1
    assert gm.graph.edges[a, b]["weight"] > 0.2


def test_strengthen_edges_is_bounded_by_max_weight(gm):
    a = gm.add_node("n1", TIER_LEAF, "d")
    b = gm.add_node("n2", TIER_LEAF, "d")
    gm.add_edge(a, b, weight=0.5)

    heb = HebbianUpdater(gm, learning_rate=0.9, max_weight=1.0)
    for _ in range(50):
        heb.strengthen_edges([a, b])
    assert gm.graph.edges[a, b]["weight"] <= 1.0


def test_strengthen_edges_skips_missing_node(gm):
    a = gm.add_node("only", TIER_LEAF, "d")
    heb = HebbianUpdater(gm)
    assert heb.strengthen_edges([a, "ghost"]) == 0


def test_strengthen_edges_requires_two_nodes(gm):
    a = gm.add_node("solo", TIER_LEAF, "d")
    heb = HebbianUpdater(gm)
    assert heb.strengthen_edges([a]) == 0
    assert heb.strengthen_edges([]) == 0


def test_apply_decay_reduces_old_weights(gm):
    a = gm.add_node("n1", TIER_LEAF, "d")
    b = gm.add_node("n2", TIER_LEAF, "d")
    gm.add_edge(a, b, weight=0.8)

    old = (datetime.now(timezone.utc) - timedelta(hours=100)).isoformat().replace("+00:00", "Z")
    gm.graph.edges[a, b]["last_accessed"] = old

    heb = HebbianUpdater(gm, decay_lambda=0.1, min_weight=0.01)
    updated = heb.apply_decay()
    assert updated == 1
    assert gm.graph.edges[a, b]["weight"] < 0.8


def test_apply_decay_respects_min_weight_floor(gm):
    a = gm.add_node("n1", TIER_LEAF, "d")
    b = gm.add_node("n2", TIER_LEAF, "d")
    gm.add_edge(a, b, weight=0.5)

    very_old = (
        datetime.now(timezone.utc) - timedelta(days=365 * 5)
    ).isoformat().replace("+00:00", "Z")
    gm.graph.edges[a, b]["last_accessed"] = very_old

    heb = HebbianUpdater(gm, decay_lambda=1.0, min_weight=0.05)
    heb.apply_decay()
    assert gm.graph.edges[a, b]["weight"] >= 0.05


def test_update_after_retrieval_returns_stats(gm):
    a = gm.add_node("n1", TIER_LEAF, "d")
    b = gm.add_node("n2", TIER_LEAF, "d")
    c = gm.add_node("n3", TIER_LEAF, "d")
    gm.add_edge(a, b, weight=0.3)
    gm.add_edge(b, c, weight=0.3)

    heb = HebbianUpdater(gm)
    stats = heb.update_after_retrieval([a, b, c])
    assert "edges_decayed" in stats
    assert "edges_strengthened" in stats
    assert stats["edges_strengthened"] >= 1


def test_get_edge_stats_empty_graph(gm):
    heb = HebbianUpdater(gm)
    assert heb.get_edge_stats() == {"count": 0}


def test_get_edge_stats_populated(gm):
    a = gm.add_node("n1", TIER_LEAF, "d")
    b = gm.add_node("n2", TIER_LEAF, "d")
    c = gm.add_node("n3", TIER_LEAF, "d")
    gm.add_edge(a, b, weight=0.2)
    gm.add_edge(b, c, weight=0.8)

    heb = HebbianUpdater(gm)
    stats = heb.get_edge_stats()
    assert stats["count"] == 2
    assert stats["min_weight"] == 0.2
    assert stats["max_weight"] == 0.8
    assert 0.2 < stats["avg_weight"] < 0.8


def test_reset_edge_weights(gm):
    a = gm.add_node("n1", TIER_LEAF, "d")
    b = gm.add_node("n2", TIER_LEAF, "d")
    gm.add_edge(a, b, weight=0.9)

    heb = HebbianUpdater(gm)
    heb.reset_edge_weights(weight=0.15)
    assert gm.graph.edges[a, b]["weight"] == 0.15
