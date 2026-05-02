"""
FastAPI backend for the LocMemory dashboard.

Exposes the cognitive memory graph + retriever over HTTP so the
React frontend can visualize nodes, edges, and retrieval scores.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import networkx as nx

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.memory import (
    GraphManager,
    GraphRetriever,
    MemoryClassifier,
    HebbianUpdater,
    MemoryConsolidator,
    ProceduralDetector,
    TIER_CONTEXT,
    TIER_ANCHOR,
    TIER_LEAF,
    TIER_PROCEDURAL,
)
from core.memory.classifier import DEFAULT_SUBDOMAINS
from core.memory.graph import TIER_NAMES
from core.settings.config import get_config
from core.logger import RetrievalLogger, get_logger
from core.security.security import get_encryptor


PROJECT_ROOT = Path(__file__).resolve().parents[2]

state: dict = {
    "gm": None,
    "retriever": None,
    "classifier": None,
    "hebbian": None,
    "consolidator": None,
    "procedural": None,
    "rl_agent": None,
    "logger": None,
    "encryptor": None,
    "addition_count": 0,      # triggers consolidation every 30
    "interaction_count": 0,   # triggers procedural detection every 50
}


def _memory_dict(nid, data, encryptor):
    text = data.get("text", "")
    encrypted = encryptor.is_encrypted(text) if encryptor else False
    return {
        "id": nid,
        "text": text,
        "tier": data.get("tier", 3),
        "tier_name": TIER_NAMES.get(data.get("tier", 0), "?"),
        "domain": data.get("domain", ""),
        "subdomain": data.get("subdomain", ""),
        "created_at": data.get("created_at", ""),
        "is_encrypted": encrypted,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = get_config()
    db_path = config.get("storage", "sqlite_db_path", "data/memory.db")
    if not Path(db_path).is_absolute():
        db_path = str(PROJECT_ROOT / db_path)
    print(f"[dashboard] using db: {db_path}")

    gm = GraphManager(db_path=db_path)
    gm.initialize_db()
    gm.load_graph()

    classifier = MemoryClassifier()
    retriever  = GraphRetriever(gm, classifier=classifier)
    hebbian    = HebbianUpdater(gm)
    consolidator = MemoryConsolidator(gm)
    procedural = ProceduralDetector(gm)

    # Initialize RL agent if enabled
    rl_agent = None
    config = get_config()
    if config.get("rl", "enabled", False):
        try:
            from core.rl.agent import RLAgent
            model_path = config.get("rl", "model_path", "data/rl_agent.zip")
            rl_agent = RLAgent(model_path)
            print(f"[rl] agent initialized: available={rl_agent.is_available()}")
        except Exception as e:
            print(f"[rl] agent init failed: {e}")

    log_path = PROJECT_ROOT / "data" / "retrieval_log.csv"
    retrieval_logger = get_logger(log_path)

    state["gm"]          = gm
    state["classifier"]  = classifier
    state["retriever"]   = retriever
    state["hebbian"]     = hebbian
    state["consolidator"] = consolidator
    state["procedural"]  = procedural
    state["rl_agent"]    = rl_agent
    state["logger"]      = retrieval_logger
    state["encryptor"]   = get_encryptor()

    print(f"[dashboard] loaded graph: {gm.graph.number_of_nodes()} nodes, "
          f"{gm.graph.number_of_edges()} edges")

    yield

    try:
        gm.save_graph()
    finally:
        gm.close()


app = FastAPI(title="LocMemory Dashboard API", version="0.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────── models ───────────────────────────

class MemoryCreate(BaseModel):
    text: str
    tier: int = TIER_LEAF
    domain: Optional[str] = None    # auto-classified when omitted
    subdomain: Optional[str] = None


class MemoryPatch(BaseModel):
    text: str


class RetrieveRequest(BaseModel):
    query: str
    limit: int = 10
    include_rejected: bool = False


class RateRequest(BaseModel):
    rating: int   # 1–5


class ConfigUpdate(BaseModel):
    data: dict


# ─────────────────────────── background helpers ───────────────────────────

def _hebbian_bg(node_ids: list[str]) -> None:
    hebbian: HebbianUpdater = state["hebbian"]
    if hebbian and node_ids:
        try:
            hebbian.update_after_retrieval(node_ids)
        except Exception as e:
            print(f"[hebbian] background update error: {e}")


def _maybe_consolidate() -> None:
    state["addition_count"] += 1
    consolidator: MemoryConsolidator = state["consolidator"]
    if consolidator and consolidator.should_run(state["addition_count"]):
        try:
            consolidator.run()
            print(f"[consolidator] ran at addition #{state['addition_count']}")
        except Exception as e:
            print(f"[consolidator] error: {e}")


def _maybe_detect_patterns() -> None:
    state["interaction_count"] += 1
    procedural: ProceduralDetector = state["procedural"]
    if procedural and procedural.increment_interaction():
        try:
            procedural.run_detection()
            print(f"[procedural] ran at interaction #{state['interaction_count']}")
        except Exception as e:
            print(f"[procedural] error: {e}")


# ─────────────────────────── health ───────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


# ─────────────────────────── /api/rl ───────────────────────────

@app.get("/api/rl/status")
def rl_status():
    """Return RL agent status and configuration."""
    rl_agent = state.get("rl_agent")
    config = get_config()

    if rl_agent is None:
        return {
            "enabled": config.get("rl", "enabled", False),
            "available": False,
            "message": "RL agent not initialized (disabled in config or init failed)",
        }

    return {
        "enabled": config.get("rl", "enabled", False),
        "available": rl_agent.is_available(),
        "model_path": config.get("rl", "model_path", "data/rl_agent.zip"),
        "candidate_pool_size": config.get("rl", "candidate_pool_size", 25),
        "top_k": config.get("rl", "top_k", 5),
        "token_budget": config.get("rl", "token_budget", 512),
    }


# ─────────────────────────── /api/stats ───────────────────────────

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

    n = g.number_of_nodes()
    e = g.number_of_edges()
    density = round(nx.density(g), 6) if n > 1 else 0.0
    avg_degree = round((2 * e / n), 2) if n > 0 else 0.0
    communities = nx.number_weakly_connected_components(g) if n > 0 else 0

    return {
        "nodes": n,
        "edges": e,
        "density": density,
        "avg_degree": avg_degree,
        "communities": communities,
        "tier_counts": tier_counts,
        "domain_counts": domain_counts,
    }


# ─────────────────────────── /api/graph ───────────────────────────

@app.get("/api/graph")
def get_graph():
    """Return all nodes + edges for the force-directed visualization."""
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


# ─────────────────────────── /api/memories ───────────────────────────

@app.get("/api/memories")
def list_memories(
    domain: Optional[str] = None,
    subdomain: Optional[str] = None,
    tier: Optional[int] = None,
    q: Optional[str] = None,
    limit: int = Query(200, le=1000),
    offset: int = Query(0, ge=0),
):
    gm: GraphManager = state["gm"]
    encryptor = state.get("encryptor")
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
        results.append(_memory_dict(nid, data, encryptor))
    results.sort(key=lambda r: r["created_at"], reverse=True)
    return results[offset : offset + limit]


@app.post("/api/memories", status_code=201)
def create_memory(body: MemoryCreate, background_tasks: BackgroundTasks):
    """
    Manually add a new memory node.
    Domain/subdomain are auto-classified when omitted.
    Triggers Hebbian decay + consolidation check in background.
    """
    gm: GraphManager = state["gm"]
    classifier: MemoryClassifier = state["classifier"]

    text = body.text.strip()
    if not text:
        raise HTTPException(400, "text must not be empty")

    tier = body.tier
    if tier not in (TIER_CONTEXT, TIER_ANCHOR, TIER_LEAF, TIER_PROCEDURAL):
        raise HTTPException(400, f"tier must be 1–4, got {tier}")

    domain    = body.domain or ""
    subdomain = body.subdomain or ""

    if not domain:
        try:
            result    = classifier.classify(text)
            domain    = result.get("domain", "")
            subdomain = result.get("subdomain", "")
        except Exception:
            pass

    embedding = None
    try:
        embedding = classifier._embed([text])[0]
    except Exception:
        pass

    node_id = gm.add_node(
        text=text,
        tier=tier,
        domain=domain,
        subdomain=subdomain,
        embedding=embedding,
    )

    background_tasks.add_task(_hebbian_bg, [node_id])
    background_tasks.add_task(_maybe_consolidate)

    encryptor = state.get("encryptor")
    data = gm.graph.nodes[node_id]
    return _memory_dict(node_id, data, encryptor)


@app.get("/api/memories/{node_id}")
def get_memory(node_id: str):
    gm: GraphManager = state["gm"]
    if node_id not in gm.graph:
        raise HTTPException(404, "memory not found")
    data = gm.graph.nodes[node_id]
    encryptor = state.get("encryptor")
    neighbors = gm.get_neighbors(node_id, direction="both")
    result = _memory_dict(node_id, data, encryptor)
    result["neighbors"] = [
        {
            "id": n["id"],
            "text": n.get("text", ""),
            "relation": n.get("edge_relation"),
            "weight": n.get("edge_weight"),
        }
        for n in neighbors
    ]
    return result


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


@app.post("/api/memories/{node_id}/decrypt")
def decrypt_memory(node_id: str):
    """Decrypt a memory's text for display-only (does not modify stored data)."""
    gm: GraphManager = state["gm"]
    encryptor = state.get("encryptor")
    if node_id not in gm.graph:
        raise HTTPException(404, "memory not found")
    text = gm.graph.nodes[node_id].get("text", "")
    if encryptor and encryptor.is_encrypted(text):
        decrypted = encryptor.decrypt(text)
        return {"text": decrypted}
    return {"text": text}


# ─────────────────────────── /api/domains ───────────────────────────

@app.get("/api/domains")
def list_domains():
    gm: GraphManager = state["gm"]
    classifier: MemoryClassifier = state["classifier"]

    counts: dict[str, dict[str, int]] = {}
    for _, data in gm.graph.nodes(data=True):
        d  = data.get("domain") or "(none)"
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


# ─────────────────────────── /api/retrieve ───────────────────────────

@app.post("/api/retrieve")
def retrieve(req: RetrieveRequest, background_tasks: BackgroundTasks):
    """
    Retrieve memories for a query.
    Fires Hebbian update + procedural detection check in background.
    Logs the retrieval event for quality metrics.
    Optionally returns rejected candidates when RL agent is enabled.
    """
    import time
    retriever: GraphRetriever = state["retriever"]
    logger: RetrievalLogger = state["logger"]
    rl_agent = state.get("rl_agent")
    encryptor = state.get("encryptor")

    t0 = time.monotonic()
    results = retriever.retrieve(req.query)
    latency_ms = (time.monotonic() - t0) * 1000

    def add_encrypted(item):
        text = item.get("text", "")
        item["is_encrypted"] = encryptor.is_encrypted(text) if encryptor else False
        return item

    top = [add_encrypted(r) for r in results[: req.limit]]
    query_domain = retriever._query_domain

    rejected = []
    if req.include_rejected and rl_agent and rl_agent.is_available():
        rejected = [add_encrypted(r) for r in results[req.limit: min(len(results), 25)]]

    entry_id = None
    if logger:
        try:
            entry_id = logger.log_retrieval(
                query=req.query,
                results=top,
                query_domain=query_domain,
                latency_ms=latency_ms,
            )
        except Exception as e:
            print(f"[logger] log_retrieval error: {e}")

    retrieved_ids = [r["node_id"] for r in top if "node_id" in r]
    if retrieved_ids:
        background_tasks.add_task(_hebbian_bg, retrieved_ids)
        background_tasks.add_task(_maybe_detect_patterns)

    return {
        "query": req.query,
        "query_domain": query_domain,
        "entry_id": entry_id,
        "results": top,
        "rejected": rejected,
    }


class CompareRequest(BaseModel):
    query: str
    limit: int = 5


@app.post("/api/retrieve/compare")
def retrieve_compare(req: CompareRequest):
    """
    Run retrieval in both Hybrid and RL modes for side-by-side comparison.
    """
    retriever: GraphRetriever = state["retriever"]
    rl_agent = state.get("rl_agent")

    rl_available = rl_agent is not None and rl_agent.is_available()

    if not rl_available:
        results = retriever.retrieve(req.query)
        top = results[: req.limit]
        return {
            "query": req.query,
            "query_domain": retriever._query_domain,
            "rl_available": False,
            "hybrid": top,
            "rl": top,
            "overlap_count": len(top),
        }

    original_rl = retriever._rl_agent

    try:
        retriever._rl_agent = None
        hybrid_results = retriever.retrieve(req.query)
        hybrid = hybrid_results[: req.limit]
    finally:
        retriever._rl_agent = original_rl

    retriever._rl_agent = original_rl
    rl_results = retriever.retrieve(req.query)
    rl = rl_results[: req.limit]

    hybrid_ids = set(r.get("node_id") for r in hybrid if r.get("node_id"))
    rl_ids = set(r.get("node_id") for r in rl if r.get("node_id"))
    overlap_count = len(hybrid_ids & rl_ids)

    return {
        "query": req.query,
        "query_domain": retriever._query_domain,
        "rl_available": True,
        "hybrid": hybrid,
        "rl": rl,
        "overlap_count": overlap_count,
    }


@app.post("/api/retrieve/{entry_id}/rate")
def rate_retrieval(entry_id: str, body: RateRequest):
    """Attach a 1–5 user rating to a logged retrieval entry."""
    logger: RetrievalLogger = state["logger"]
    if not logger:
        raise HTTPException(503, "Logger not available")
    if not (1 <= body.rating <= 5):
        raise HTTPException(422, "rating must be between 1 and 5")
    found = logger.log_rating(entry_id, body.rating)
    if not found:
        raise HTTPException(404, f"entry '{entry_id}' not found in log")
    return {"ok": True, "entry_id": entry_id, "rating": body.rating}


@app.get("/api/metrics")
def get_metrics(n: int = Query(100, ge=1, le=1000)):
    """
    Aggregated retrieval quality metrics for the last *n* log entries.
    Powers the WK8 metrics panel.
    """
    logger: RetrievalLogger = state["logger"]
    if not logger:
        raise HTTPException(503, "Logger not available")
    return logger.get_summary(n)


# ─────────────────────────── /api/hebbian ───────────────────────────

@app.get("/api/hebbian/stats")
def hebbian_stats():
    """Edge weight distribution from the Hebbian updater."""
    hebbian: HebbianUpdater = state["hebbian"]
    if not hebbian:
        raise HTTPException(503, "Hebbian updater not available")

    edge_stats = hebbian.get_edge_stats()

    gm: GraphManager = state["gm"]
    weights = [data.get("weight", 0.1) for _, _, data in gm.graph.edges(data=True)]
    buckets = [0] * 10
    active_edges = 0  # "neurons that fire together" - high weight edges
    strong_edges = 0   # weight > 1.0
    for w in weights:
        buckets[min(int(w / 0.5), 9)] += 1
        if w >= 0.8:  # Co-activated edges
            active_edges += 1
        if w >= 1.0:  # Strong connections
            strong_edges += 1

    return {
        **edge_stats,
        "active_edges": active_edges,  # edges with weight >= 0.8 (co-activated)
        "strong_edges": strong_edges,   # edges with weight >= 1.0
        "histogram": [
            {"range": f"{i * 0.5:.1f}–{(i + 1) * 0.5:.1f}", "count": buckets[i]}
            for i in range(10)
        ],
    }


@app.post("/api/hebbian/decay")
def run_hebbian_decay():
    """Manually trigger time-based decay on all edges."""
    hebbian: HebbianUpdater = state["hebbian"]
    if not hebbian:
        raise HTTPException(503, "Hebbian updater not available")
    updated = hebbian.apply_decay()
    return {"edges_decayed": updated}


# ─────────────────────────── /api/consolidate ───────────────────────────

@app.post("/api/consolidate")
def run_consolidation():
    """
    Manually trigger Louvain clustering → create Tier 2 Anchor nodes.
    May take several seconds if Ollama is called for summarization.
    """
    consolidator: MemoryConsolidator = state["consolidator"]
    if not consolidator:
        raise HTTPException(503, "Consolidator not available")
    try:
        return consolidator.run()
    except ImportError as e:
        raise HTTPException(503, f"Missing dependency: {e}")
    except Exception as e:
        raise HTTPException(500, f"Consolidation failed: {e}")


# ─────────────────────────── /api/patterns ───────────────────────────

@app.get("/api/patterns")
def get_patterns():
    """Return all existing Tier 4 procedural pattern nodes."""
    procedural: ProceduralDetector = state["procedural"]
    if not procedural:
        raise HTTPException(503, "Procedural detector not available")
    return [
        {
            "id": n["id"],
            "text": n.get("text", ""),
            "domain": n.get("domain", ""),
            "created_at": n.get("created_at", ""),
        }
        for n in procedural.get_procedural_nodes()
    ]


@app.post("/api/patterns/detect")
def detect_patterns():
    """
    Manually trigger procedural pattern detection.
    Finds cross-domain coactivation patterns and creates Tier 4 nodes.
    """
    procedural: ProceduralDetector = state["procedural"]
    if not procedural:
        raise HTTPException(503, "Procedural detector not available")
    try:
        return procedural.run_detection()
    except Exception as e:
        raise HTTPException(500, f"Pattern detection failed: {e}")


# ─────────────────────────── /api/config ───────────────────────────

@app.get("/api/config")
def get_config_api():
    return get_config().as_dict()


@app.put("/api/config")
def update_config_api(body: ConfigUpdate):
    cfg = get_config()
    cfg.update(body.data)
    cfg.save()
    return {"ok": True, "data": cfg.as_dict()}
