"""
FastAPI backend for the LocMemory dashboard.

Exposes the cognitive memory graph + retriever over HTTP so the
React frontend can visualize nodes, edges, and retrieval scores.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.memory import GraphManager, GraphRetriever, MemoryClassifier
from core.memory.classifier import DEFAULT_SUBDOMAINS
from core.memory.graph import TIER_NAMES
from core.settings.config import get_config


state: dict = {"gm": None, "retriever": None, "classifier": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = get_config()
    db_path = config.get("storage", "sqlite_db_path", "data/memory.db")

    gm = GraphManager(db_path=db_path)
    gm.initialize_db()
    gm.load_graph()

    classifier = MemoryClassifier()
    retriever = GraphRetriever(gm, classifier=classifier)

    state["gm"] = gm
    state["classifier"] = classifier
    state["retriever"] = retriever
    print(f"[dashboard] loaded graph: {gm.graph.number_of_nodes()} nodes")

    yield

    try:
        gm.save_graph()
    finally:
        gm.close()


app = FastAPI(title="LocMemory Dashboard API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ───────────────────────── models ─────────────────────────

class MemoryPatch(BaseModel):
    text: str


class RetrieveRequest(BaseModel):
    query: str
    limit: int = 10


class ConfigUpdate(BaseModel):
    data: dict


# ───────────────────────── health ─────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


# ───────────────────────── /api/stats ─────────────────────────

@app.get("/api/stats")
def stats():
    gm: GraphManager = state["gm"]
    g = gm.graph
    tier_counts: dict[str, int] = {}
    domain_counts: dict[str, int] = {}
    for _, data in g.nodes(data=True):
        tier_name = TIER_NAMES.get(data.get("tier", 0), "?")
        tier_counts[tier_name] = tier_counts.get(tier_name, 0) + 1
        dom = data.get("domain", "") or "(none)"
        domain_counts[dom] = domain_counts.get(dom, 0) + 1
    return {
        "nodes": g.number_of_nodes(),
        "edges": g.number_of_edges(),
        "tier_counts": tier_counts,
        "domain_counts": domain_counts,
    }


# ───────────────────────── /api/graph ─────────────────────────

@app.get("/api/graph")
def get_graph():
    """Return nodes + edges suitable for a force-directed view."""
    gm: GraphManager = state["gm"]
    g = gm.graph
    nodes = [
        {
            "id": nid,
            "text": data.get("text", ""),
            "tier": data.get("tier", 3),
            "tier_name": TIER_NAMES.get(data.get("tier", 0), "?"),
            "domain": data.get("domain", ""),
            "subdomain": data.get("subdomain", ""),
            "created_at": data.get("created_at", ""),
        }
        for nid, data in g.nodes(data=True)
    ]
    links = [
        {
            "source": s,
            "target": t,
            "relation": data.get("relation", "related"),
            "weight": data.get("weight", 0.1),
        }
        for s, t, data in g.edges(data=True)
    ]
    return {"nodes": nodes, "links": links}


# ───────────────────────── /api/memories ─────────────────────────

@app.get("/api/memories")
def list_memories(
    domain: Optional[str] = None,
    subdomain: Optional[str] = None,
    tier: Optional[int] = None,
    q: Optional[str] = None,
    limit: int = Query(200, le=1000),
):
    gm: GraphManager = state["gm"]
    results = []
    for nid, data in gm.graph.nodes(data=True):
        if domain and data.get("domain") != domain:
            continue
        if subdomain and data.get("subdomain") != subdomain:
            continue
        if tier is not None and data.get("tier") != tier:
            continue
        if q and q.lower() not in str(data.get("text", "")).lower():
            continue
        results.append({
            "id": nid,
            "text": data.get("text", ""),
            "tier": data.get("tier", 3),
            "tier_name": TIER_NAMES.get(data.get("tier", 0), "?"),
            "domain": data.get("domain", ""),
            "subdomain": data.get("subdomain", ""),
            "created_at": data.get("created_at", ""),
        })
    results.sort(key=lambda r: r["created_at"], reverse=True)
    return results[:limit]


@app.get("/api/memories/{node_id}")
def get_memory(node_id: str):
    gm: GraphManager = state["gm"]
    if node_id not in gm.graph:
        raise HTTPException(404, "memory not found")
    data = gm.graph.nodes[node_id]
    neighbors = gm.get_neighbors(node_id, direction="both")
    return {
        "id": node_id,
        "text": data.get("text", ""),
        "tier": data.get("tier", 3),
        "tier_name": TIER_NAMES.get(data.get("tier", 0), "?"),
        "domain": data.get("domain", ""),
        "subdomain": data.get("subdomain", ""),
        "created_at": data.get("created_at", ""),
        "neighbors": [
            {
                "id": n["id"],
                "text": n.get("text", ""),
                "relation": n.get("edge_relation"),
                "weight": n.get("edge_weight"),
            }
            for n in neighbors
        ],
    }


@app.patch("/api/memories/{node_id}")
def update_memory(node_id: str, patch: MemoryPatch):
    gm: GraphManager = state["gm"]
    if not gm.update_node_text(node_id, patch.text):
        raise HTTPException(404, "memory not found")
    return {"id": node_id, "text": patch.text}


@app.delete("/api/memories/{node_id}")
def delete_memory(node_id: str):
    gm: GraphManager = state["gm"]
    if not gm.delete_node(node_id):
        raise HTTPException(404, "memory not found")
    return {"deleted": node_id}


# ───────────────────────── /api/domains ─────────────────────────

@app.get("/api/domains")
def list_domains():
    gm: GraphManager = state["gm"]
    classifier: MemoryClassifier = state["classifier"]

    counts: dict[str, dict[str, int]] = {}
    for _, data in gm.graph.nodes(data=True):
        d = data.get("domain") or "(none)"
        sd = data.get("subdomain") or ""
        counts.setdefault(d, {"_total": 0})
        counts[d]["_total"] += 1
        if sd:
            counts[d][sd] = counts[d].get(sd, 0) + 1

    domains = []
    for d in classifier.list_domains():
        subs = classifier.list_subdomains(d) or DEFAULT_SUBDOMAINS.get(d, [])
        domain_counts = counts.get(d, {"_total": 0})
        domains.append({
            "name": d,
            "total": domain_counts.get("_total", 0),
            "subdomains": [
                {"name": sd, "count": domain_counts.get(sd, 0)}
                for sd in subs
            ],
        })

    for d, cmap in counts.items():
        if d not in [x["name"] for x in domains] and d != "(none)":
            domains.append({
                "name": d,
                "total": cmap.get("_total", 0),
                "subdomains": [],
            })

    return domains


# ───────────────────────── /api/retrieve ─────────────────────────

@app.post("/api/retrieve")
def retrieve(req: RetrieveRequest):
    retriever: GraphRetriever = state["retriever"]
    results = retriever.retrieve(req.query)
    return {
        "query": req.query,
        "query_domain": retriever._query_domain,
        "results": results[: req.limit],
    }


# ───────────────────────── /api/config ─────────────────────────

@app.get("/api/config")
def get_config_api():
    return get_config().as_dict()


@app.put("/api/config")
def update_config_api(body: ConfigUpdate):
    cfg = get_config()
    cfg.update(body.data)
    cfg.save()
    return {"ok": True, "data": cfg.as_dict()}
