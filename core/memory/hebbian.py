"""
Hebbian learning module for LocMemory.

Implements "neurons that fire together, wire together" learning
to update relationships between memory nodes.
"""

import math
from datetime import datetime, timezone

from core.memory.graph import GraphManager


class HebbianUpdater:
    """Implements Hebbian learning for memory graph edge weights."""

    def __init__(
        self,
        graph_manager: GraphManager,
        decay_lambda: float = 0.01,
        learning_rate: float = 0.2,
        min_weight: float = 0.01,
        max_weight: float = 5.0,
    ):
        self.graph_manager = graph_manager
        self.decay_lambda = decay_lambda
        self.learning_rate = learning_rate
        self.min_weight = min_weight
        self.max_weight = max_weight

    def apply_decay(self) -> int:
        """
        Decay edge weights based on time since last_accessed.

        Returns number of edges updated.
        """
        graph = self.graph_manager.graph
        if graph is None:
            return 0

        now = datetime.now(timezone.utc)
        updated_count = 0

        for u, v, data in graph.edges(data=True):
            last_accessed_str = data.get("last_accessed", "")

            if not last_accessed_str:
                continue

            try:
                last_accessed = datetime.fromisoformat(
                    last_accessed_str.replace("Z", "+00:00")
                )
            except ValueError:
                continue

            if last_accessed.tzinfo is None:
                last_accessed = last_accessed.replace(tzinfo=timezone.utc)

            delta_t = (now - last_accessed).total_seconds() / 3600.0

            current_weight = data.get("weight", 0.1)

            decay_factor = math.exp(-self.decay_lambda * delta_t)
            new_weight = current_weight * decay_factor
            new_weight = max(self.min_weight, new_weight)

            if abs(new_weight - current_weight) > 0.001:
                data["weight"] = new_weight
                data["last_accessed"] = now.isoformat().replace("+00:00", "Z")
                updated_count += 1

        if updated_count > 0:
            self.graph_manager.save_graph()

        return updated_count

    def strengthen_edges(self, node_ids: list[str]) -> int:
        """
        Increase edge weights for nodes that were retrieved together.

        For each pair of nodes in node_ids, create or strengthen
        the edge between them using Hebbian update rule.

        Returns number of edges strengthened/created.
        """
        graph = self.graph_manager.graph
        if graph is None or len(node_ids) < 2:
            return 0

        now = datetime.now(timezone.utc)
        now_str = now.isoformat().replace("+00:00", "Z")
        updated_count = 0

        for i, node_a in enumerate(node_ids):
            for node_b in node_ids[i + 1:]:
                if node_a not in graph or node_b not in graph:
                    continue

                if graph.has_edge(node_a, node_b):
                    data = graph.edges[node_a, node_b]
                    current_weight = data.get("weight", self.min_weight)

                    coactivation = 1.0

                    new_weight = (
                        current_weight +
                        self.learning_rate * coactivation * (1.0 - current_weight)
                    )

                    new_weight = min(self.max_weight, max(self.min_weight, new_weight))

                    data["weight"] = new_weight
                    data["last_accessed"] = now_str
                    updated_count += 1

                elif graph.has_edge(node_b, node_a):
                    data = graph.edges[node_b, node_a]
                    current_weight = data.get("weight", self.min_weight)

                    coactivation = 1.0

                    new_weight = (
                        current_weight +
                        self.learning_rate * coactivation * (1.0 - current_weight)
                    )

                    new_weight = min(self.max_weight, max(self.min_weight, new_weight))

                    data["weight"] = new_weight
                    data["last_accessed"] = now_str
                    updated_count += 1

        if updated_count > 0:
            self.graph_manager.save_graph()

        return updated_count

    def update_after_retrieval(self, node_ids: list[str]) -> dict:
        """
        Called after every retrieval event.

        1. Apply time-based decay to all edges
        2. Strengthen edges between retrieved nodes
        3. Update timestamps for retrieved nodes

        Returns dict with stats: edges_decayed, edges_strengthened.
        """
        decay_count = self.apply_decay()

        strengthen_count = self.strengthen_edges(node_ids)

        graph = self.graph_manager.graph
        if graph is not None and node_ids:
            now = datetime.now(timezone.utc)
            now_str = now.isoformat().replace("+00:00", "Z")
            for node_id in node_ids:
                if node_id in graph:
                    for _, _, data in graph.edges(node_id, data=True):
                        data["last_accessed"] = now_str

        return {
            "edges_decayed": decay_count,
            "edges_strengthened": strengthen_count,
        }

    def get_edge_stats(self) -> dict:
        """Get statistics about edge weights in the graph."""
        graph = self.graph_manager.graph
        if graph is None:
            return {}

        weights = [data.get("weight", 0.1) for _, _, data in graph.edges(data=True)]

        if not weights:
            return {"count": 0}

        return {
            "count": len(weights),
            "min_weight": min(weights),
            "max_weight": max(weights),
            "avg_weight": sum(weights) / len(weights),
        }

    def reset_edge_weights(self, weight: float = 0.1) -> int:
        """
        Reset all edge weights to a uniform value.

        Returns number of edges reset.
        """
        graph = self.graph_manager.graph
        if graph is None:
            return 0

        now = datetime.now(timezone.utc)
        now_str = now.isoformat().replace("+00:00", "Z")

        reset_count = 0
        for u, v, data in graph.edges(data=True):
            data["weight"] = weight
            data["last_accessed"] = now_str
            reset_count += 1

        if reset_count > 0:
            self.graph_manager.save_graph()

        return reset_count


if __name__ == "__main__":
    from core.memory.graph import GraphManager, TIER_LEAF

    print("Building test graph...")
    with GraphManager("data/test_hebbian.db") as gm:
        nodes = [
            gm.add_node("Python is great for AI", TIER_LEAF, "programming"),
            gm.add_node("I use PyTorch for deep learning", TIER_LEAF, "programming"),
            gm.add_node("Neural networks are fascinating", TIER_LEAF, "learning"),
            gm.add_node("I deployed a model to production", TIER_LEAF, "work"),
            gm.add_node("Reading about transformers", TIER_LEAF, "learning"),
        ]

        gm.add_edge(nodes[0], nodes[1], "related", 0.5)
        gm.add_edge(nodes[1], nodes[2], "related", 0.3)
        gm.add_edge(nodes[2], nodes[3], "related", 0.2)
        gm.add_edge(nodes[0], nodes[3], "related", 0.1)
        gm.add_edge(nodes[1], nodes[4], "related", 0.4)

        print("\n" + "=" * 60)
        print("Initial edge weights:")
        for u, v, data in gm.graph.edges(data=True):
            print(f"  {u[:8]} <-> {v[:8]}: weight={data.get('weight', 0):.3f}")

        print("\n" + "=" * 60)
        print("Testing HebbianUpdater")
        print("=" * 60)

        hebbian = HebbianUpdater(gm, decay_lambda=0.1, learning_rate=0.3)

        print("\n1. Simulate retrieval of [nodes 0, 1, 2] together:")
        hebbian.update_after_retrieval([nodes[0], nodes[1], nodes[2]])

        print("\nEdge weights after first retrieval:")
        for u, v, data in gm.graph.edges(data=True):
            print(f"  {u[:8]} <-> {v[:8]}: weight={data.get('weight', 0):.3f}")

        print("\n2. Simulate retrieval of [nodes 1, 3, 4] together:")
        hebbian.update_after_retrieval([nodes[1], nodes[3], nodes[4]])

        print("\nEdge weights after second retrieval:")
        for u, v, data in gm.graph.edges(data=True):
            print(f"  {u[:8]} <-> {v[:8]}: weight={data.get('weight', 0):.3f}")

        print("\n3. Edge statistics:")
        stats = hebbian.get_edge_stats()
        print(f"  Count: {stats['count']}")
        print(f"  Min: {stats['min_weight']:.3f}")
        print(f"  Max: {stats['max_weight']:.3f}")
        print(f"  Avg: {stats['avg_weight']:.3f}")

        print("\n4. Apply decay (simulate time passing):")
        decay_count = hebbian.apply_decay()
        print(f"  Decayed {decay_count} edges")

        print("\nEdge weights after decay:")
        for u, v, data in gm.graph.edges(data=True):
            print(f"  {u[:8]} <-> {v[:8]}: weight={data.get('weight', 0):.3f}")

        print("\n" + "=" * 60)
        print("Hebbian learning test complete")