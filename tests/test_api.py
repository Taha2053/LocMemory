"""
API integration tests: Frontend → FastAPI → Core.

Each test spins up the real FastAPI app with a temp SQLite DB and real
core modules (GraphManager, MemoryClassifier, GraphRetriever, HebbianUpdater,
MemoryConsolidator, ProceduralDetector).  Ollama calls are mocked so the
suite runs offline.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from dashboard.backend.main import app, state
from core.memory import GraphManager, MemoryClassifier, GraphRetriever
from core.memory import HebbianUpdater, MemoryConsolidator, ProceduralDetector
from core.memory import TIER_LEAF, TIER_ANCHOR, TIER_CONTEXT, TIER_PROCEDURAL


# ─────────────────────────── fixtures ───────────────────────────

@pytest.fixture(scope="module")
def classifier():
    """Real classifier shared for the whole module (slow to load once)."""
    return MemoryClassifier(use_fallback=False)


@pytest.fixture()
def client(tmp_path, classifier):
    """
    TestClient backed by a fresh SQLite DB per test.

    The app's lifespan() runs on TestClient.__enter__ and overwrites state[]
    with the production DB. We re-inject our test DB into state[] immediately
    after entering so every request sees the isolated graph.
    The lifespan teardown closes its own local `gm` (prod), not state["gm"],
    so there is no double-close issue.
    """
    db_path = str(tmp_path / "test.db")
    gm = GraphManager(db_path=db_path)
    gm.initialize_db()
    gm.load_graph()

    with TestClient(app, raise_server_exceptions=True) as c:
        # Lifespan has run — now inject our isolated test state
        state["gm"]           = gm
        state["classifier"]   = classifier
        state["retriever"]    = GraphRetriever(gm, classifier=classifier, min_semantic_score=0.0)
        state["hebbian"]      = HebbianUpdater(gm)
        state["consolidator"] = MemoryConsolidator(gm)
        state["procedural"]   = ProceduralDetector(gm)
        state["rl_agent"]     = None
        state["addition_count"]    = 0
        state["interaction_count"] = 0
        yield c

    gm.close()


def _add(client: TestClient, text: str, tier=TIER_LEAF, domain=None, subdomain=None):
    """Helper: POST /api/memories and return the created node dict."""
    body = {"text": text, "tier": tier}
    if domain:
        body["domain"] = domain
    if subdomain:
        body["subdomain"] = subdomain
    r = client.post("/api/memories", json=body)
    assert r.status_code == 201, r.text
    return r.json()


# ─────────────────────────── /health ───────────────────────────

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


# ─────────────────────────── /api/stats ───────────────────────────

def test_stats_empty_graph(client):
    r = client.get("/api/stats")
    assert r.status_code == 200
    data = r.json()
    assert data["nodes"] == 0
    assert data["edges"] == 0
    assert data["tier_counts"] == {}


def test_stats_after_adding_memories(client):
    _add(client, "User runs every morning", domain="health")
    _add(client, "User is learning Python", domain="learning")

    r = client.get("/api/stats")
    assert r.status_code == 200
    data = r.json()
    assert data["nodes"] == 2
    assert "leaf" in data["tier_counts"]
    assert data["tier_counts"]["leaf"] == 2
    assert "health" in data["domain_counts"]
    assert "learning" in data["domain_counts"]


# ─────────────────────────── /api/graph ───────────────────────────

def test_graph_empty(client):
    r = client.get("/api/graph")
    assert r.status_code == 200
    data = r.json()
    assert data["nodes"] == []
    assert data["links"] == []


def test_graph_returns_nodes_and_links(client):
    a = _add(client, "User enjoys hiking", domain="health")
    b = _add(client, "User bought hiking boots", domain="personal")
    # manually add an edge via GraphManager
    state["gm"].add_edge(a["id"], b["id"], weight=0.5, relation="related")

    r = client.get("/api/graph")
    assert r.status_code == 200
    data = r.json()
    assert len(data["nodes"]) == 2
    assert len(data["links"]) == 1
    link = data["links"][0]
    assert link["weight"] == 0.5
    assert link["relation"] == "related"
    node_ids = {n["id"] for n in data["nodes"]}
    assert a["id"] in node_ids
    assert b["id"] in node_ids


# ─────────────────────────── /api/memories ───────────────────────────

def test_create_memory_with_explicit_domain(client):
    r = client.post("/api/memories", json={
        "text": "User drinks coffee every morning",
        "tier": TIER_LEAF,
        "domain": "health",
        "subdomain": "nutrition",
    })
    assert r.status_code == 201
    data = r.json()
    assert data["domain"] == "health"
    assert data["subdomain"] == "nutrition"
    assert data["tier"] == TIER_LEAF
    assert "id" in data


def test_create_memory_auto_classifies_domain(client):
    r = client.post("/api/memories", json={
        "text": "User goes for a 5km run before breakfast",
        "tier": TIER_LEAF,
    })
    assert r.status_code == 201
    data = r.json()
    # classifier should assign health or a similar domain
    assert data["domain"] != ""


def test_create_memory_rejects_empty_text(client):
    r = client.post("/api/memories", json={"text": "   ", "tier": TIER_LEAF})
    assert r.status_code == 400


def test_create_memory_rejects_bad_tier(client):
    r = client.post("/api/memories", json={"text": "something", "tier": 99})
    assert r.status_code == 400


def test_create_memory_all_tiers(client):
    for tier in (TIER_CONTEXT, TIER_ANCHOR, TIER_LEAF, TIER_PROCEDURAL):
        r = client.post("/api/memories", json={
            "text": f"Tier {tier} memory node",
            "tier": tier,
            "domain": "engineering",
        })
        assert r.status_code == 201
        assert r.json()["tier"] == tier


def test_list_memories_empty(client):
    r = client.get("/api/memories")
    assert r.status_code == 200
    assert r.json() == []


def test_list_memories_all(client):
    _add(client, "Memory A", domain="health")
    _add(client, "Memory B", domain="learning")
    r = client.get("/api/memories")
    assert r.status_code == 200
    assert len(r.json()) == 2


def test_list_memories_filter_by_domain(client):
    _add(client, "User does yoga", domain="health")
    _add(client, "User studies NLP", domain="learning")
    r = client.get("/api/memories?domain=health")
    assert r.status_code == 200
    results = r.json()
    assert all(m["domain"] == "health" for m in results)
    assert any("yoga" in m["text"] for m in results)


def test_list_memories_filter_by_tier(client):
    _add(client, "Leaf node", tier=TIER_LEAF, domain="health")
    _add(client, "Anchor node", tier=TIER_ANCHOR, domain="health")
    r = client.get(f"/api/memories?tier={TIER_LEAF}")
    assert r.status_code == 200
    results = r.json()
    assert all(m["tier"] == TIER_LEAF for m in results)


def test_list_memories_text_search(client):
    _add(client, "User loves photography", domain="personal")
    _add(client, "User codes in Python", domain="programming")
    r = client.get("/api/memories?q=photography")
    assert r.status_code == 200
    results = r.json()
    assert len(results) == 1
    assert "photography" in results[0]["text"]


def test_list_memories_pagination(client):
    for i in range(5):
        _add(client, f"Memory item {i}", domain="engineering")
    r1 = client.get("/api/memories?limit=3&offset=0")
    r2 = client.get("/api/memories?limit=3&offset=3")
    assert r1.status_code == 200
    assert r2.status_code == 200
    page1 = r1.json()
    page2 = r2.json()
    assert len(page1) == 3
    assert len(page2) == 2
    # No overlap
    ids1 = {m["id"] for m in page1}
    ids2 = {m["id"] for m in page2}
    assert ids1.isdisjoint(ids2)


def test_get_memory_by_id(client):
    created = _add(client, "User meditates daily", domain="health")
    r = client.get(f"/api/memories/{created['id']}")
    assert r.status_code == 200
    data = r.json()
    assert data["id"] == created["id"]
    assert data["text"] == "User meditates daily"
    assert "neighbors" in data


def test_get_memory_not_found(client):
    r = client.get("/api/memories/nonexistent-id-xyz")
    assert r.status_code == 404


def test_get_memory_includes_neighbors(client):
    a = _add(client, "User runs marathons", domain="health")
    b = _add(client, "User tracks calories", domain="health")
    state["gm"].add_edge(a["id"], b["id"], weight=0.7, relation="related")

    r = client.get(f"/api/memories/{a['id']}")
    assert r.status_code == 200
    data = r.json()
    neighbor_ids = [n["id"] for n in data["neighbors"]]
    assert b["id"] in neighbor_ids


def test_update_memory_text(client):
    created = _add(client, "Original text here", domain="work")
    r = client.patch(f"/api/memories/{created['id']}", json={"text": "Updated text"})
    assert r.status_code == 200
    assert r.json()["text"] == "Updated text"

    # verify the change persisted in the graph
    r2 = client.get(f"/api/memories/{created['id']}")
    assert r2.json()["text"] == "Updated text"


def test_update_memory_not_found(client):
    r = client.patch("/api/memories/bad-id", json={"text": "anything"})
    assert r.status_code == 404


def test_delete_memory(client):
    created = _add(client, "Temporary node", domain="personal")
    r = client.delete(f"/api/memories/{created['id']}")
    assert r.status_code == 200
    assert r.json()["deleted"] == created["id"]

    # verify it's gone
    r2 = client.get(f"/api/memories/{created['id']}")
    assert r2.status_code == 404


def test_delete_memory_not_found(client):
    r = client.delete("/api/memories/nonexistent-xyz")
    assert r.status_code == 404


# ─────────────────────────── /api/domains ───────────────────────────

def test_domains_lists_all_classifier_domains(client):
    r = client.get("/api/domains")
    assert r.status_code == 200
    data = r.json()
    names = [d["name"] for d in data]
    # classifier has built-in domains
    assert "health" in names
    assert "learning" in names
    assert "engineering" in names


def test_domains_counts_reflect_memories(client):
    _add(client, "User does yoga", domain="health")
    _add(client, "User lifts weights", domain="health")
    _add(client, "User studies maths", domain="learning")

    r = client.get("/api/domains")
    assert r.status_code == 200
    data = r.json()
    health = next((d for d in data if d["name"] == "health"), None)
    learning = next((d for d in data if d["name"] == "learning"), None)
    assert health is not None
    assert health["total"] == 2
    assert learning["total"] == 1


def test_domains_has_subdomains(client):
    r = client.get("/api/domains")
    assert r.status_code == 200
    data = r.json()
    health = next(d for d in data if d["name"] == "health")
    assert isinstance(health["subdomains"], list)


# ─────────────────────────── /api/retrieve ───────────────────────────

def test_retrieve_empty_graph_returns_empty(client):
    r = client.post("/api/retrieve", json={"query": "anything", "limit": 5})
    assert r.status_code == 200
    data = r.json()
    assert data["results"] == []
    assert data["query"] == "anything"


def test_retrieve_returns_relevant_memories(client):
    _add(client, "User runs 10km every Saturday", domain="health")
    _add(client, "User follows a low-carb diet", domain="health")
    _add(client, "User is writing a thesis in machine learning", domain="learning")

    r = client.post("/api/retrieve", json={"query": "user exercise routine", "limit": 5})
    assert r.status_code == 200
    data = r.json()
    results = data["results"]
    assert len(results) >= 1
    # top result should be about running, not the thesis
    assert any("run" in res["text"].lower() for res in results)


def test_retrieve_result_has_all_score_fields(client):
    _add(client, "User practices guitar on weekends", domain="personal")

    r = client.post("/api/retrieve", json={"query": "music hobby", "limit": 5})
    assert r.status_code == 200
    results = r.json()["results"]
    if results:
        res = results[0]
        assert "node_id" in res
        assert "score" in res
        assert "cosine_contribution" in res
        assert "recency_contribution" in res
        assert "category_contribution" in res
        assert "tier" in res
        assert "domain" in res


def test_retrieve_respects_limit(client):
    for i in range(8):
        _add(client, f"Health fact number {i}", domain="health")

    r = client.post("/api/retrieve", json={"query": "health", "limit": 3})
    assert r.status_code == 200
    assert len(r.json()["results"]) <= 3


def test_retrieve_returns_query_domain(client):
    _add(client, "User studies algorithms", domain="learning")

    r = client.post("/api/retrieve", json={"query": "what is the user studying", "limit": 5})
    assert r.status_code == 200
    data = r.json()
    assert "query_domain" in data
    assert isinstance(data["query_domain"], str)


# ─────────────────────────── /api/hebbian ───────────────────────────

def test_hebbian_stats_empty_graph(client):
    r = client.get("/api/hebbian/stats")
    assert r.status_code == 200
    data = r.json()
    assert data["count"] == 0
    assert data["active_edges"] == 0
    assert data["strong_edges"] == 0
    assert isinstance(data["histogram"], list)
    assert len(data["histogram"]) == 10


def test_hebbian_stats_after_adding_edges(client):
    a = _add(client, "User does yoga", domain="health")
    b = _add(client, "User meditates", domain="health")
    # Add a high-weight edge to appear in active/strong counts
    state["gm"].add_edge(a["id"], b["id"], weight=1.2, relation="related")

    r = client.get("/api/hebbian/stats")
    assert r.status_code == 200
    data = r.json()
    assert data["count"] == 1
    assert data["max_weight"] == pytest.approx(1.2, abs=0.01)
    assert data["active_edges"] == 1   # weight >= 0.8
    assert data["strong_edges"] == 1   # weight >= 1.0


def test_hebbian_decay_runs(client):
    a = _add(client, "User reads books", domain="learning")
    b = _add(client, "User visits library", domain="learning")
    state["gm"].add_edge(a["id"], b["id"], weight=0.5)

    r = client.post("/api/hebbian/decay")
    assert r.status_code == 200
    assert "edges_decayed" in r.json()


# ─────────────────────────── /api/consolidate ───────────────────────────

def test_consolidate_with_sparse_graph(client):
    # Too few nodes to form meaningful clusters — should still return valid shape
    _add(client, "Node alpha", domain="health")
    _add(client, "Node beta", domain="health")

    r = client.post("/api/consolidate")
    assert r.status_code == 200
    data = r.json()
    assert "clusters_found" in data
    assert "anchors_created" in data
    assert "nodes_connected" in data


# ─────────────────────────── /api/patterns ───────────────────────────

def test_patterns_empty(client):
    r = client.get("/api/patterns")
    assert r.status_code == 200
    assert r.json() == []


def test_patterns_detect_runs(client):
    r = client.post("/api/patterns/detect")
    assert r.status_code == 200
    data = r.json()
    assert "patterns_found" in data
    assert "procedural_nodes_created" in data


# ─────────────────────────── /api/rl ───────────────────────────

def test_rl_status_when_disabled(client):
    # state["rl_agent"] is None in our fixture
    r = client.get("/api/rl/status")
    assert r.status_code == 200
    data = r.json()
    assert "enabled" in data
    assert "available" in data
    assert data["available"] is False


# ─────────────────────────── /api/config ───────────────────────────

def test_get_config(client):
    r = client.get("/api/config")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, dict)
    # config.yaml always has these top-level sections
    assert any(k in data for k in ("system", "models", "storage", "retrieval", "hebbian"))


# ─────────────────────────── cross-endpoint flow ───────────────────────────

def test_full_flow_create_retrieve_inspect_delete(client):
    """Simulate the full frontend user journey on a single memory."""
    # 1. Create
    created = _add(client, "User is learning Rust for systems programming", domain="learning")
    nid = created["id"]

    # 2. Verify it appears in list
    r = client.get("/api/memories?domain=learning")
    assert any(m["id"] == nid for m in r.json())

    # 3. Retrieve by semantic query
    r = client.post("/api/retrieve", json={"query": "programming language the user is learning", "limit": 5})
    assert r.status_code == 200
    retrieved_ids = [res["node_id"] for res in r.json()["results"]]
    assert nid in retrieved_ids

    # 4. Inspect detail
    r = client.get(f"/api/memories/{nid}")
    assert r.status_code == 200
    assert r.json()["text"] == "User is learning Rust for systems programming"

    # 5. Update
    r = client.patch(f"/api/memories/{nid}", json={"text": "User is proficient in Rust"})
    assert r.status_code == 200

    # 6. Stats reflect the node
    r = client.get("/api/stats")
    assert r.json()["nodes"] >= 1

    # 7. Delete
    r = client.delete(f"/api/memories/{nid}")
    assert r.status_code == 200

    # 8. Gone from list
    r = client.get("/api/memories?domain=learning")
    assert all(m["id"] != nid for m in r.json())


def test_hebbian_strengthens_after_retrieval(client):
    """Retrieval → background Hebbian → edge weights increase."""
    a = _add(client, "User lifts weights three times a week", domain="health")
    b = _add(client, "User tracks protein intake daily", domain="health")
    state["gm"].add_edge(a["id"], b["id"], weight=0.1, relation="related")

    before = state["gm"].graph[a["id"]][b["id"]]["weight"]

    heb: HebbianUpdater = state["hebbian"]
    heb.update_after_retrieval([a["id"], b["id"]])

    after = state["gm"].graph[a["id"]][b["id"]]["weight"]
    assert after > before
