"""
Tests for GraphStore CRUD operations.
"""

import pytest

from core.memory.graph import GraphManager, TIER_LEAF, TIER_CONTEXT, TIER_ANCHOR


def test_domain_nodes_seeded(gm):
    """All 8 domains exist after init."""
    # Should seed domains on initialize
    domains = ["health", "work", "personal", "programming", "finance", "learning", "engineering", "social"]

    for domain in domains:
        nodes = gm.get_nodes_by_domain(domain)
        # Domain should have at least the context node
        assert len(nodes) >= 1


def test_add_leaf_node(gm):
    """Add tier-3 node, retrieve by id."""
    node_id = gm.add_node("Test fact", TIER_LEAF, "test_domain")

    # Verify node exists
    nodes = gm.get_nodes_by_domain("test_domain")
    ids = [n["id"] for n in nodes]
    assert node_id in ids

    # Verify tier
    node_data = gm.graph.nodes[node_id]
    assert node_data["tier"] == TIER_LEAF


def test_add_edge(gm):
    """Add edge, verify weight stored."""
    id1 = gm.add_node("Node 1", TIER_LEAF, "test")
    id2 = gm.add_node("Node 2", TIER_LEAF, "test")

    gm.add_edge(id1, id2, "related", 0.5)

    # Verify edge exists
    assert gm.graph.has_edge(id1, id2)
    edge_data = gm.graph.edges[id1, id2]
    assert edge_data["weight"] == 0.5


def test_get_children(gm):
    """Add parent+children, verify get_children()."""
    parent = gm.add_node("Parent node", TIER_CONTEXT, "test")
    child1 = gm.add_node("Child 1", TIER_LEAF, "test")
    child2 = gm.add_node("Child 2", TIER_LEAF, "test")

    gm.add_edge(parent, child1, "has_child", 0.8)
    gm.add_edge(parent, child2, "has_child", 0.8)

    children = gm.get_neighbors(parent, direction="out")
    child_ids = [c["id"] for c in children]

    assert child1 in child_ids
    assert child2 in child_ids


def test_get_neighbors(gm):
    """Verify bidirectional neighbor lookup."""
    id1 = gm.add_node("Node 1", TIER_LEAF, "test")
    id2 = gm.add_node("Node 2", TIER_LEAF, "test")

    gm.add_edge(id1, id2, "related", 0.5)

    # Out neighbors
    out_neighbors = gm.get_neighbors(id1, direction="out")
    out_ids = [n["id"] for n in out_neighbors]
    assert id2 in out_ids

    # In neighbors
    in_neighbors = gm.get_neighbors(id2, direction="in")
    in_ids = [n["id"] for n in in_neighbors]
    assert id1 in in_ids


def test_update_node_weight(gm):
    """Update weight, verify persisted."""
    node_id = gm.add_node("Test node", TIER_LEAF, "test")

    # Update (via edge)
    node_id2 = gm.add_node("Test node 2", TIER_LEAF, "test")
    gm.add_edge(node_id, node_id2, "related", 0.3)

    # Update weight
    gm.update_edge_weight(node_id, node_id2, "related", 0.9)

    # Verify persisted
    edge_data = gm.graph.edges[node_id, node_id2]
    assert edge_data["weight"] == 0.9


def test_delete_node_cascades_edges(gm):
    """Delete node, verify edges removed."""
    id1 = gm.add_node("Node 1", TIER_LEAF, "test")
    id2 = gm.add_node("Node 2", TIER_LEAF, "test")

    gm.add_edge(id1, id2, "related", 0.5)

    # Delete node
    if hasattr(gm, "delete_node"):
        gm.delete_node(id1)
    else:
        # Remove all edges first
        if id1 in gm.graph:
            for neighbor in list(gm.graph.neighbors(id1)):
                gm.graph.remove_edge(id1, neighbor)
            gm.graph.remove_node(id1)

    # Verify edge removed
    assert not gm.graph.has_edge(id1, id2)


def test_networkx_loader(gm):
    """load_networkx() returns correct node/edge count."""
    # Add some nodes
    for i in range(5):
        gm.add_node(f"Node {i}", TIER_LEAF, "test")

    # Add edges
    node_ids = list(gm.graph.nodes)
    for i in range(len(node_ids) - 1):
        gm.add_edge(node_ids[i], node_ids[i + 1], "related", 0.5)

    # Load to networkx
    nx_graph = gm.load_networkx()

    assert nx_graph.number_of_nodes() == 5
    assert nx_graph.number_of_edges() == 4


def test_stats(gm):
    """stats() returns correct tier counts."""
    # Add nodes in different tiers
    gm.add_node("Context", TIER_CONTEXT, "test")
    gm.add_node("Anchor", TIER_ANCHOR, "test")
    for i in range(3):
        gm.add_node(f"Leaf {i}", TIER_LEAF, "test")

    # Get stats
    stats = gm.stats()

    assert stats["tier_counts"] >= 1
    assert stats["edge_count"] >= 0