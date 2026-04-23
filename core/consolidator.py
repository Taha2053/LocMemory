"""
Memory consolidation module for LocMemory.

Automatically creates concept anchor nodes by clustering leaf memories using Louvain community detection.
"""

import requests
from typing import Optional

try:
    import community as community_louvain
except ImportError:
    community_louvain = None

import networkx as nx

from core.graph import GraphManager, TIER_LEAF, TIER_ANCHOR
from core.config import get_config


OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "mistral:7b-instruct"


class MemoryConsolidator:
    """Consolidates leaf memories into concept anchor nodes."""

    def __init__(
        self,
        graph_manager: GraphManager,
        ollama_url: str = OLLAMA_URL,
        ollama_model: str = DEFAULT_MODEL,
        min_cluster_size: int = 10,
    ):
        self.graph_manager = graph_manager
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model
        self.min_cluster_size = min_cluster_size

        self._config = get_config()
        self._created_anchors: set[str] = set()

    def _ensure_louvain(self):
        if community_louvain is None:
            raise ImportError(
                "python-louvain not installed. Run: pip install python-louvain"
            )

    def detect_clusters(self) -> dict[int, list[str]]:
        """
        Detect communities among Tier 3 nodes using Louvain algorithm.

        Returns dict of cluster_id -> list of node_ids.
        """
        self._ensure_louvain()

        graph = self.graph_manager.graph
        if graph is None:
            raise RuntimeError("Graph not loaded")

        leaf_nodes = [
            n for n in graph.nodes
            if graph.nodes[n].get("tier") == TIER_LEAF
        ]

        if len(leaf_nodes) < self.min_cluster_size:
            return {}

        subgraph = graph.subgraph(leaf_nodes).copy()

        undirected_subgraph = subgraph.to_undirected()

        if undirected_subgraph.number_of_edges() < 1:
            return self._fallback_clustering(leaf_nodes)

        partition = community_louvain.best_partition(undirected_subgraph)

        clusters: dict[int, list[str]] = {}
        for node_id, cluster_id in partition.items():
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(node_id)

        significant_clusters = {
            cid: nodes for cid, nodes in clusters.items()
            if len(nodes) >= self.min_cluster_size
        }

        return significant_clusters

    def _fallback_clustering(self, leaf_nodes: list[str]) -> dict[int, list[str]]:
        """
        Fallback clustering when graph has no edges.

        Uses semantic similarity to group nodes.
        """
        if len(leaf_nodes) < self.min_cluster_size:
            return {}

        texts = [
            self.graph_manager.graph.nodes[n].get("text", "")
            for n in leaf_nodes
        ]

        try:
            from core.classifier import MemoryClassifier
            classifier = MemoryClassifier()
            query_embeddings = classifier._embed(texts)

            similarity_matrix = []
            for i, emb_i in enumerate(query_embeddings):
                row = []
                for j, emb_j in enumerate(query_embeddings):
                    sim = classifier._cosine_similarity(emb_i, emb_j)
                    row.append(sim)
                similarity_matrix.append(row)

            import numpy as np
            sim_matrix = np.array(similarity_matrix)

            from scipy.spatial.distance import squareform
            dist_matrix = 1 - (sim_matrix + sim_matrix.T) / 2

            adj_matrix = np.where(dist_matrix < 0.5, 1, 0)
            np.fill_diagonal(adj_matrix, 0)

            import community as com
            temp_graph = nx.from_numpy_array(adj_matrix)
            partition = com.best_partition(temp_graph)

            clusters = {}
            for node_idx, cluster_id in partition.items():
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(leaf_nodes[node_idx])

        except Exception:
            clusters = {0: leaf_nodes}

        return {cid: nodes for cid, nodes in clusters.items() if len(nodes) >= self.min_cluster_size}

    def summarize_cluster(self, texts: list[str]) -> Optional[str]:
        """
        Use Ollama to summarize the shared concept of memory texts.

        Returns one-sentence summary or None if failed.
        """
        if not texts:
            return None

        if len(texts) > 20:
            texts = texts[:20]

        memories_text = "\n".join(f"- {t}" for t in texts)

        prompt = f"""Summarize the shared concept of these memories in one concise sentence.
Focus on the common theme or recurring topic.

Memories:
{memories_text}

Summary:"""

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.3,
                    "max_tokens": 100,
                },
                timeout=30,
            )

            if response.status_code == 200:
                summary = response.json().get("response", "").strip()
                if summary:
                    return summary

        except Exception as e:
            print(f"Ollama summarization failed: {e}")

        return self._fallback_summary(texts)

    def _fallback_summary(self, texts: list[str]) -> str:
        """Fallback summary using keyword extraction."""
        from collections import Counter
        import re

        all_words = []
        for text in texts:
            words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
            all_words.extend(words)

        common = Counter(all_words).most_common(5)
        keywords = [w for w, _ in common]

        return f"Shared theme: {', '.join(keywords)}"

    def create_anchor_node(
        self,
        summary: str,
        cluster_nodes: list[str],
        domain: str = "",
    ) -> Optional[str]:
        """
        Create a Tier 2 anchor node for a cluster.

        Returns node_id or None if anchor already exists.
        """
        summary_lower = summary.lower()

        for anchor_id in self._created_anchors:
            existing_text = self.graph_manager.graph.nodes[anchor_id].get("text", "").lower()
            if existing_text == summary_lower:
                return anchor_id

        all_texts = [
            self.graph_manager.graph.nodes[n].get("text", "")
            for n in cluster_nodes
        ]
        combined_text = " ".join(all_texts)

        embedding = None
        try:
            from core.classifier import MemoryClassifier
            classifier = MemoryClassifier()
            embedding = classifier._embed([combined_text])[0]
        except Exception:
            pass

        domains = [
            self.graph_manager.graph.nodes[n].get("domain", "")
            for n in cluster_nodes
        ]
        if not domain and domains:
            domain = max(set(domains), key=domains.count)

        anchor_id = self.graph_manager.add_node(
            text=summary,
            tier=TIER_ANCHOR,
            domain=domain,
            embedding=embedding,
        )

        self._created_anchors.add(anchor_id)

        for leaf_id in cluster_nodes:
            try:
                self.graph_manager.add_edge(
                    leaf_id,
                    anchor_id,
                    relation="summarized_as",
                    weight=0.5,
                )
            except Exception:
                pass

        self.graph_manager.add_edge(
            anchor_id,
            cluster_nodes[0] if cluster_nodes else "",
            relation="represents",
            weight=0.9,
        )

        return anchor_id

    def run(self) -> dict:
        """
        Run full consolidation process.

        Returns dict with stats: clusters_found, anchors_created, nodes_connected.
        """
        clusters = self.detect_clusters()

        stats = {
            "clusters_found": len(clusters),
            "anchors_created": 0,
            "nodes_connected": 0,
        }

        print(f"\n{'='*60}")
        print("Memory Consolidation Report")
        print(f"{'='*60}")
        print(f"Clusters detected: {len(clusters)}")

        for cluster_id, node_ids in clusters.items():
            print(f"\nCluster {cluster_id}: {len(node_ids)} nodes")

            texts = [
                self.graph_manager.graph.nodes[n].get("text", "")
                for n in node_ids
            ]

            summary = self.summarize_cluster(texts)
            if not summary:
                print("  Failed to generate summary, skipping")
                continue

            print(f"  Summary: {summary[:80]}...")

            anchor_id = self.create_anchor_node(summary, node_ids)
            if anchor_id:
                stats["anchors_created"] += 1
                stats["nodes_connected"] += len(node_ids)
                print(f"  Created anchor: {anchor_id[:8]}...")

        print(f"\n{'='*60}")
        print(f"Consolidation complete: {stats['anchors_created']} anchors created")
        print(f"{'='*60}\n")

        return stats

    def should_run(self, addition_count: int, run_every_n: int = 30) -> bool:
        """
        Check if consolidation should run based on memory additions.

        Returns True if addition_count % run_every_n == 0.
        """
        return addition_count > 0 and addition_count % run_every_n == 0


if __name__ == "__main__":
    from core.graph import GraphManager, TIER_LEAF, TIER_ANCHOR

    print("Building test graph with leaf nodes...")
    with GraphManager("data/test_consolidate.db") as gm:
        programming_nodes = []
        learning_nodes = []

        for i in range(15):
            n = gm.add_node(
                f"Learned about Python {['lists', 'dictionaries', 'functions', 'classes', 'decorators', 'generators', 'context managers', 'testing', 'async', 'type hints', ' dataclasses', 'logging', 'regex', 'file handling', 'comprehensions'][i]}",
                TIER_LEAF,
                "programming",
            )
            programming_nodes.append(n)

        for i in range(12):
            n = gm.add_node(
                f"Studied {['machine learning', 'deep learning', 'neural networks', 'transformers', 'attention mechanism', 'reinforcement learning', 'natural language processing', 'computer vision', 'generative models', 'optimization', 'backpropagation', 'embeddings'][i]}",
                TIER_LEAF,
                "learning",
            )
            learning_nodes.append(n)

        for i in range(len(programming_nodes) - 1):
            gm.add_edge(programming_nodes[i], programming_nodes[i + 1], "related", 0.7)

        for i in range(len(learning_nodes) - 1):
            gm.add_edge(learning_nodes[i], learning_nodes[i + 1], "related", 0.7)

        print("\n" + "=" * 60)
        print("Testing MemoryConsolidator")
        print("=" * 60)

        try:
            consolidator = MemoryConsolidator(gm, min_cluster_size=3)
            stats = consolidator.run()

            print(f"\nFinal graph stats:")
            print(f"  Total nodes: {gm.graph.number_of_nodes()}")
            print(f"  Total edges: {gm.graph.number_of_edges()}")

            anchors = gm.get_nodes_by_tier(TIER_ANCHOR)
            print(f"  Anchor nodes: {len(anchors)}")
            for anchor in anchors:
                print(f"    - {anchor['text'][:60]}...")

        except ImportError as e:
            print(f"\nError: {e}")
            print("Install with: pip install python-louvain")