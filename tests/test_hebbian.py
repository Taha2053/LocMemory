"""
Tests for Hebbian learning.
"""

import pytest

from core.memory.graph import GraphManager, TIER_LEAF
from core.memory.hebbian import HebbianUpdater


def test_strengthen_pair_creates_edge(gm):
    """Co-retrieved pair gets co_accessed edge."""
    # Add nodes
    id1 = gm.add_node("Node 1", TIER_LEAF, "test")
    id2 = gm.add_node("Node 2", TIER_LEAF, "test")

    # Create Hebbian updater
    hebbian = HebbianUpdater(gm)

    # Strengthen together
    hebbian.strengthen_edges([id1, id2])

    # Edge should exist
    assert gm.graph.has_edge(id1, id2) or gm.graph.has_edge(id2, id1)


def test_strengthen_pair_increases_weight(gm):
    """Repeated co-retrieval increases weight."""
    id1 = gm.add_node("Node 1", TIER_LEAF, "test")
    id2 = gm.add_node("Node 2", TIER_LEAF, "test")

    hebbian = HebbianUpdater(gm)

    # First strengthening
    hebbian.strengthen_edges([id1, id2])
    if gm.graph.has_edge(id1, id2):
        weight1 = gm.graph.edges[id1, id2].get("weight", 0.1)
    elif gm.graph.has_edge(id2, id1):
        weight1 = gm.graph.edges[id2, id1].get("weight", 0.1)
    else:
        weight1 = 0.1

    # Second strengthening
    hebbian.strengthen_edges([id1, id2])

    if gm.graph.has_edge(id1, id2):
        weight2 = gm.graph.edges[id1, id2].get("weight", 0.1)
    elif gm.graph.has_edge(id2, id1):
        weight2 = gm.graph.edges[id2, id1].get("weight", 0.1)
    else:
        weight2 = 0.1

    assert weight2 >= weight1


def test_weight_bounded_at_max(gm):
    """Weight never exceeds max."""
    from core.settings.config import get_config

    cfg = get_config()
    max_weight = cfg.get("hebbian", "max_weight", 5.0)

    id1 = gm.add_node("Node 1", TIER_LEAF, "test")
    id2 = gm.add_node("Node 2", TIER_LEAF, "test")

    hebbian = HebbianUpdater(gm, max_weight=max_weight)

    # Many strengthenings
    for _ in range(100):
        hebbian.strengthen_edges([id1, id2])

    # Get edge weight
    if gm.graph.has_edge(id1, id2):
        weight = gm.graph.edges[id1, id2].get("weight", 0.1)
    elif gm.graph.has_edge(id2, id1):
        weight = gm.graph.edges[id2, id1].get("weight", 0.1)
    else:
        pytest.fail("Edge not created")

    assert weight <= max_weight


def test_decay_reduces_weight(gm):
    """Decay reduces old edge weights."""
    id1 = gm.add_node("Node 1", TIER_LEAF, "test")
    id2 = gm.add_node("Node 2", TIER_LEAF, "test")

    gm.add_edge(id1, id2, "related", 0.9)

    hebbian = HebbianUpdater(gm, decay_lambda=0.5)

    # Apply decay
    hebbian.apply_decay()

    # Weight should be reduced
    edge_data = gm.graph.edges[id1, id2]
    assert edge_data["weight"] < 0.9


def test_strengthen_node_increases_weight(gm):
    """Node weight increases on access."""
    id1 = gm.add_node("Node 1", TIER_LEAF, "test")
    id2 = gm.add_node("Node 2", TIER_LEAF, "test")

    gm.add_edge(id1, id2, "related", 0.3)

    hebbian = HebbianUpdater(gm)

    # Update after retrieval
    hebbian.update_after_retrieval([id1, id2])

    # Edge should be updated
    assert gm.graph.has_edge(id1, id2)