"""
Graph-based retrieval system for cognitive memory architecture.

Retrieves relevant memories from a multi-layer cognitive graph using
semantic embeddings and graph traversal.
"""

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from core.memory.graph import GraphManager
from core.memory.classifier import MemoryClassifier
from core.settings.config import get_config


@dataclass
class RetrievedMemory:
    """A retrieved memory node with its score."""
    node_id: str
    text: str
    domain: str
    tier: int
    score: float
    graph_score: float
    semantic_score: float
    depth: int
    edge_weight: float


class GraphRetriever:
    """Retrieves relevant memories from the cognitive graph."""

    def __init__(
        self,
        graph_manager: GraphManager,
        classifier: Optional[MemoryClassifier] = None,
        max_candidates: int = 20,
        traversal_depth: int = 2,
        cross_domain_threshold: float = 0.5,
        semantic_weight: float = 0.6,
        graph_weight: float = 0.4,
    ):
        self.graph_manager = graph_manager
        self.classifier = classifier or MemoryClassifier()
        self.max_candidates = max_candidates
        self.traversal_depth = traversal_depth
        self.cross_domain_threshold = cross_domain_threshold
        self.semantic_weight = semantic_weight
        self.graph_weight = graph_weight

        self._config = get_config()
        self._embedding_cache: dict[str, list[float]] = {}
        self._query_embedding: Optional[list[float]] = None

        # RL agent for intelligent selection
        self._rl_agent = None
        self._rl_enabled = self._config.get("rl", "enabled", False)
        if self._rl_enabled:
            try:
                from core.rl.agent import RLAgent

                model_path = self._config.get("rl", "model_path", "data/rl_agent.zip")
                self._rl_agent = RLAgent(model_path)
                if not self._rl_agent.is_available():
                    self._rl_agent = None
            except Exception:
                self._rl_agent = None

    def retrieve(self, query: str) -> list[dict]:
        """
        Retrieve relevant memories for the query.

        Returns list of dicts with node_id, text, score.
        """
        start_time = time.time()
        self._query_embedding = None
        self._embedding_cache = {}

        query_domain = self.classifier.detect_domain(query)[0]

        candidates = self._traverse_graph(query_domain)

        scored = self._score_candidates(query, candidates)

        scored.sort(key=lambda x: x.score, reverse=True)

        results = scored[:self.max_candidates]

        # RL-based selection if agent available
        if self._rl_agent is not None and self._query_embedding is not None:
            token_budget = self._config.get("rl", "token_budget", 512)
            results = self._rl_select(results, token_budget)

        elapsed_ms = (time.time() - start_time) * 1000
        if elapsed_ms > 80:
            print(f"Warning: Retrieval took {elapsed_ms:.1f}ms (target: <80ms)")

        return [
            {
                "node_id": r.node_id,
                "text": r.text,
                "domain": r.domain,
                "tier": r.tier,
                "score": round(r.score, 4),
                "depth": r.depth,
            }
            for r in results
        ]

    def _traverse_graph(self, query_domain: str) -> list[tuple[str, int, float]]:
        """
        Traverse the graph starting from domain-matching nodes.

        Returns list of (node_id, depth, edge_weight) tuples.
        """
        graph = self.graph_manager.graph
        if graph is None:
            return []

        entry_nodes = [
            n for n in graph.nodes
            if graph.nodes[n].get("domain") == query_domain
        ]

        if not entry_nodes:
            entry_nodes = list(graph.nodes)[:5]

        candidates = []
        seen = set()

        for entry_id in entry_nodes:
            self._collect_neighbors(
                entry_id, depth=0, incoming_weight=1.0,
                query_domain=query_domain, candidates=candidates, seen=seen
            )

        if len(candidates) < self.max_candidates:
            remaining = [
                (n, 0, 1.0) for n in graph.nodes
                if n not in seen and n not in [e[0] for e in candidates]
            ]
            candidates.extend(remaining[:self.max_candidates - len(candidates)])

        return candidates[:self.max_candidates]

    def _collect_neighbors(
        self,
        node_id: str,
        depth: int,
        incoming_weight: float,
        query_domain: str,
        candidates: list,
        seen: set,
    ):
        """Recursively collect neighboring nodes up to traversal depth."""
        if depth > self.traversal_depth:
            return

        if node_id in seen:
            return
        seen.add(node_id)

        node_data = self.graph_manager.graph.nodes[node_id]
        current_domain = node_data.get("domain", "")

        domain_match = (current_domain == query_domain)
        high_weight = (incoming_weight >= self.cross_domain_threshold)
        can_traverse_cross_domain = domain_match or high_weight

        candidates.append((node_id, depth, incoming_weight))

        if len(candidates) >= self.max_candidates * 2:
            return

        if not can_traverse_cross_domain and depth > 0:
            return

        for neighbor in self.graph_manager.graph.neighbors(node_id):
            edge_data = self.graph_manager.graph.edges[node_id, neighbor]
            edge_weight = edge_data.get("weight", 0.1)
            self._collect_neighbors(
                neighbor, depth + 1, edge_weight,
                query_domain, candidates, seen
            )

        for predecessor in self.graph_manager.graph.predecessors(node_id):
            edge_data = self.graph_manager.graph.edges[predecessor, node_id]
            edge_weight = edge_data.get("weight", 0.1)
            self._collect_neighbors(
                predecessor, depth + 1, edge_weight,
                query_domain, candidates, seen
            )

    def _score_candidates(
        self,
        query: str,
        candidates: list[tuple[str, int, float]],
    ) -> list[RetrievedMemory]:
        """Score candidates combining semantic and graph proximity."""
        if not self._query_embedding:
            self._query_embedding = self.classifier._embed([query])[0]

        scored = []
        depth_weights = {0: 1.0, 1: 0.7, 2: 0.4}

        for node_id, depth, edge_weight in candidates:
            node_data = self.graph_manager.graph.nodes[node_id]

            text = node_data.get("text", "")
            domain = node_data.get("domain", "")
            tier = node_data.get("tier", 3)

            semantic_score = self._compute_similarity(text)

            graph_score = self._compute_graph_score(
                depth=depth,
                edge_weight=edge_weight,
                domain_match=(domain == self.classifier.detect_domain(query)[0]),
                tier=tier,
            )

            combined_score = (
                self.semantic_weight * semantic_score +
                self.graph_weight * graph_score
            )

            scored.append(RetrievedMemory(
                node_id=node_id,
                text=text,
                domain=domain,
                tier=tier,
                score=combined_score,
                graph_score=graph_score,
                semantic_score=semantic_score,
                depth=depth,
                edge_weight=edge_weight,
            ))

        return scored

    def _compute_similarity(self, text: str) -> float:
        """Compute cosine similarity between query and text."""
        if text in self._embedding_cache:
            text_embedding = self._embedding_cache[text]
        else:
            text_embedding = self.classifier._embed([text])[0]
            self._embedding_cache[text] = text_embedding

        return self._cosine_similarity(self._query_embedding, text_embedding)

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _compute_graph_score(
        self,
        depth: int,
        edge_weight: float,
        domain_match: bool,
        tier: int,
    ) -> float:
        """Compute graph proximity score."""
        depth_weight = {0: 1.0, 1: 0.7, 2: 0.4}.get(depth, 0.2)

        tier_bonus = {1: 1.2, 2: 1.0, 3: 0.8, 4: 0.6}.get(tier, 0.5)

        domain_bonus = 1.2 if domain_match else 1.0

        graph_score = edge_weight * depth_weight * tier_bonus * domain_bonus
        return min(graph_score, 1.0)

    def _rl_select(
        self,
        candidates: list,
        token_budget: int,
    ) -> list:
        """
        Use RL agent for intelligent candidate selection.

        Falls back to hybrid scoring if RL fails.
        """
        if self._rl_agent is None:
            return candidates[: self._config.get("rl", "top_k", 5)]

        try:
            from core.rl.agent import RetrievalResult

            # Build RetrievalResult
            result = RetrievalResult(
                candidates=[
                    {
                        "node_id": c.node_id,
                        "text": c.text,
                        "domain": c.domain,
                        "tier": c.tier,
                        "score": c.score,
                        "hebbian": c.edge_weight,
                        "last_accessed": "",
                    }
                    for c in candidates
                ],
                context_str="",
            )

            # Get query embedding
            query_emb = (
                np.array(self._query_embedding, dtype=np.float32)
                if self._query_embedding
                else np.zeros(384, dtype=np.float32)
            )

            # Use RL agent to select
            selected = self._rl_agent.select(result, query_emb, token_budget)

            return selected if selected else candidates

        except Exception:
            # Silent fallback
            return candidates[: self._config.get("rl", "top_k", 5)]

    def retrieve_with_context(
        self,
        query: str,
        context_limit: int = 3,
    ) -> dict:
        """
        Retrieve memories with surrounding context.

        Returns {'memories': [...], 'context_nodes': [...]}
        """
        results = self.retrieve(query)

        context_nodes = []
        for result in results[:context_limit]:
            neighbors = self.graph_manager.get_neighbors(result["node_id"], direction="both")
            context_nodes.extend(neighbors[:3])

        unique_context = {
            n["id"]: n for n in context_nodes
            if n["id"] not in [r["node_id"] for r in results]
        }

        return {
            "memories": results,
            "context_nodes": list(unique_context.values())[:10],
        }


if __name__ == "__main__":
    from core.memory.graph import GraphManager, TIER_CONTEXT, TIER_ANCHOR, TIER_LEAF

    print("Building test graph...")
    with GraphManager("data/test_retriever.db") as gm:
        id_ctx = gm.add_node("I am a Python developer", TIER_CONTEXT, "programming")
        id_anchor1 = gm.add_node("Python is great for AI", TIER_ANCHOR, "programming")
        id_anchor2 = gm.add_node("Machine learning fundamentals", TIER_ANCHOR, "learning")
        id_leaf1 = gm.add_node("Studied neural networks today", TIER_LEAF, "learning")
        id_leaf2 = gm.add_node("Using PyTorch for deep learning", TIER_LEAF, "programming")
        id_leaf3 = gm.add_node("Deployed model to production", TIER_LEAF, "work")
        id_leaf4 = gm.add_node("Reading about transformers", TIER_LEAF, "learning")

        gm.add_edge(id_ctx, id_anchor1, "specializes_in", 0.9)
        gm.add_edge(id_ctx, id_anchor2, "related_to", 0.7)
        gm.add_edge(id_anchor1, id_leaf2, "used_for", 0.8)
        gm.add_edge(id_anchor1, id_leaf3, "applied_to", 0.6)
        gm.add_edge(id_anchor2, id_leaf1, "covers", 0.9)
        gm.add_edge(id_anchor2, id_leaf4, "includes", 0.7)
        gm.add_edge(id_leaf2, id_leaf4, "related", 0.5)
        gm.add_edge(id_leaf1, id_leaf4, "connected", 0.6)

        print("\n" + "=" * 60)
        print("Testing GraphRetriever")
        print("=" * 60)

        retriever = GraphRetriever(gm)

        test_queries = [
            "What am I learning about?",
            "Tell me about my work with AI",
            "What programming skills do I have?",
        ]

        for query in test_queries:
            print(f"\nQuery: {query}")
            start = time.time()
            results = retriever.retrieve(query)
            elapsed = (time.time() - start) * 1000

            print(f"  Retrieved {len(results)} memories ({elapsed:.1f}ms)")
            for r in results:
                print(f"    [{r['tier']}] {r['text'][:50]}... score={r['score']}")

        print("\n" + "=" * 60)
        print("Context retrieval test:")
        print("=" * 60)
        context_result = retriever.retrieve_with_context("learning about neural networks")
        print(f"Main memories: {len(context_result['memories'])}")
        print(f"Context nodes: {len(context_result['context_nodes'])}")