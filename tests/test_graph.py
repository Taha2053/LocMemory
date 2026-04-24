"""Tests for GraphManager CRUD and persistence."""

from core.memory.graph import (
    GraphManager,
    TIER_CONTEXT,
    TIER_ANCHOR,
    TIER_LEAF,
    TIER_PROCEDURAL,
    TIER_NAMES,
)


def test_add_node_returns_id_and_appears_in_graph(gm):
    node_id = gm.add_node("A fact", TIER_LEAF, "personal")
    assert isinstance(node_id, str) and len(node_id) > 0
    assert node_id in gm.graph
    assert gm.graph.nodes[node_id]["tier"] == TIER_LEAF
    assert gm.graph.nodes[node_id]["domain"] == "personal"
    assert gm.graph.nodes[node_id]["text"] == "A fact"


def test_add_node_deduplicates_identical_text(gm):
    a = gm.add_node("Same fact", TIER_LEAF, "work")
    b = gm.add_node("Same fact", TIER_LEAF, "work")
    assert a == b
    assert gm.graph.number_of_nodes() == 1


def test_add_node_does_not_dedupe_across_domains(gm):
    a = gm.add_node("Shared text", TIER_LEAF, "work")
    b = gm.add_node("Shared text", TIER_LEAF, "personal")
    assert a != b


def test_add_edge_stores_weight_and_relation(gm):
    a = gm.add_node("n1", TIER_LEAF, "d")
    b = gm.add_node("n2", TIER_LEAF, "d")
    assert gm.add_edge(a, b, relation="links", weight=0.7) is True
    assert gm.graph.has_edge(a, b)
    data = gm.graph.edges[a, b]
    assert data["relation"] == "links"
    assert data["weight"] == 0.7


def test_add_edge_rejects_missing_nodes(gm):
    a = gm.add_node("exists", TIER_LEAF, "d")
    assert gm.add_edge(a, "nonexistent-id") is False


def test_update_edge_weight_persists_in_memory_and_db(gm):
    a = gm.add_node("n1", TIER_LEAF, "d")
    b = gm.add_node("n2", TIER_LEAF, "d")
    gm.add_edge(a, b, weight=0.1)

    assert gm.update_edge_weight(a, b, 0.9) is True
    assert gm.graph.edges[a, b]["weight"] == 0.9

    row = gm.conn.execute(
        "SELECT weight FROM edges WHERE source_id=? AND target_id=?", (a, b)
    ).fetchone()
    assert row["weight"] == 0.9


def test_update_edge_weight_missing_edge_returns_false(gm):
    a = gm.add_node("n1", TIER_LEAF, "d")
    b = gm.add_node("n2", TIER_LEAF, "d")
    assert gm.update_edge_weight(a, b, 0.5) is False


def test_get_nodes_by_tier(gm):
    gm.add_node("ctx", TIER_CONTEXT, "d")
    gm.add_node("leaf1", TIER_LEAF, "d")
    gm.add_node("leaf2", TIER_LEAF, "d")

    leaves = gm.get_nodes_by_tier(TIER_LEAF)
    texts = {n["text"] for n in leaves}
    assert texts == {"leaf1", "leaf2"}


def test_get_nodes_by_domain(gm):
    gm.add_node("x", TIER_LEAF, "work")
    gm.add_node("y", TIER_LEAF, "work")
    gm.add_node("z", TIER_LEAF, "personal")

    work = gm.get_nodes_by_domain("work")
    assert len(work) == 2
    assert all(n["domain"] == "work" for n in work)


def test_get_neighbors_direction(gm):
    a = gm.add_node("a", TIER_LEAF, "d")
    b = gm.add_node("b", TIER_LEAF, "d")
    c = gm.add_node("c", TIER_LEAF, "d")
    gm.add_edge(a, b)
    gm.add_edge(c, a)

    succ_ids = {n["id"] for n in gm.get_neighbors(a, direction="successors")}
    pred_ids = {n["id"] for n in gm.get_neighbors(a, direction="predecessors")}
    both_ids = {n["id"] for n in gm.get_neighbors(a, direction="both")}

    assert succ_ids == {b}
    assert pred_ids == {c}
    assert both_ids == {b, c}


def test_persistence_reload_recovers_graph(temp_db):
    with GraphManager(temp_db) as gm:
        a = gm.add_node("persist me", TIER_LEAF, "work")
        b = gm.add_node("and me", TIER_LEAF, "work")
        gm.add_edge(a, b, weight=0.42)

    with GraphManager(temp_db) as gm2:
        assert gm2.graph.number_of_nodes() == 2
        assert gm2.graph.number_of_edges() == 1
        ids_by_text = {d["text"]: n for n, d in gm2.graph.nodes(data=True)}
        assert "persist me" in ids_by_text and "and me" in ids_by_text
        edge = gm2.graph.edges[ids_by_text["persist me"], ids_by_text["and me"]]
        assert edge["weight"] == 0.42


def test_tier_constants_mapped():
    assert TIER_NAMES[TIER_CONTEXT] == "context"
    assert TIER_NAMES[TIER_ANCHOR] == "anchor"
    assert TIER_NAMES[TIER_LEAF] == "leaf"
    assert TIER_NAMES[TIER_PROCEDURAL] == "procedural"
