"""
Procedural pattern detection module for LocMemory.

Detects recurring cross-domain patterns in the memory graph
and converts them into procedural (Tier 4) nodes.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from core.memory.graph import GraphManager, TIER_PROCEDURAL
from core.settings.config import get_config


@dataclass
class Pattern:
    """Represents a detected procedural pattern."""
    source_node: str
    target_node: str
    source_domain: str
    target_domain: str
    coactivation_count: int
    avg_edge_weight: float
    pattern_text: str


class ProceduralDetector:
    """Detects recurring cross-domain patterns and creates procedural nodes."""

    def __init__(
        self,
        graph_manager: GraphManager,
        min_pattern_support: int = 3,
        min_confidence: float = 0.5,
        cross_domain_threshold: float = 0.6,
    ):
        self.graph_manager = graph_manager
        self.min_pattern_support = min_pattern_support
        self.min_confidence = min_confidence
        self.cross_domain_threshold = cross_domain_threshold

        self._config = get_config()
        self._interaction_count = 0
        self._detected_patterns: dict[str, str] = {}

    def find_cross_domain_edges(self) -> list[tuple[str, str, str, str, float]]:
        """
        Find edges connecting nodes from different domains.

        Returns list of (source_id, target_id, source_domain, target_domain, weight).
        """
        graph = self.graph_manager.graph
        if graph is None:
            return []

        cross_domain_edges = []

        for u, v, data in graph.edges(data=True):
            u_domain = graph.nodes[u].get("domain", "")
            v_domain = graph.nodes[v].get("domain", "")

            if u_domain and v_domain and u_domain != v_domain:
                weight = data.get("weight", 0.1)
                cross_domain_edges.append((u, v, u_domain, v_domain, weight))

        return cross_domain_edges

    def detect_patterns(self) -> list[Pattern]:
        """
        Detect recurring patterns from cross-domain edges.

        Groups edges by domain pairs and calculates coactivation scores.
        """
        cross_edges = self.find_cross_domain_edges()

        domain_pair_edges: dict[tuple[str, str], list[tuple]] = defaultdict(list)

        for source_id, target_id, source_domain, target_domain, weight in cross_edges:
            pair_key = tuple(sorted([source_domain, target_domain]))
            domain_pair_edges[pair_key].append((source_id, target_id, weight))

        patterns = []

        for (domain_a, domain_b), edges in domain_pair_edges.items():
            if len(edges) < self.min_pattern_support:
                continue

            weights = [w for _, _, w in edges]
            avg_weight = sum(weights) / len(weights)

            node_pairs = [(s, t) for s, t, _ in edges]
            all_nodes = set()
            for s, t in node_pairs:
                all_nodes.add(s)
                all_nodes.add(t)

            pattern_text = self._generate_pattern_text(
                domain_a, domain_b, len(edges), avg_weight
            )

            source_ids = [s for s, _, _ in edges]
            target_ids = [t for _, t, _ in edges]

            most_frequent_source = max(set(source_ids), key=source_ids.count)
            most_frequent_target = max(set(target_ids), key=target_ids.count)

            pattern = Pattern(
                source_node=most_frequent_source,
                target_node=most_frequent_target,
                source_domain=domain_a,
                target_domain=domain_b,
                coactivation_count=len(edges),
                avg_edge_weight=avg_weight,
                pattern_text=pattern_text,
            )

            patterns.append(pattern)

        patterns.sort(key=lambda p: (p.coactivation_count * p.avg_edge_weight), reverse=True)

        return patterns

    def _generate_pattern_text(
        self,
        domain_a: str,
        domain_b: str,
        coactivation_count: int,
        avg_weight: float,
    ) -> str:
        """Generate a human-readable pattern description."""
        confidence = min(coactivation_count / 10.0, 1.0) * avg_weight

        if domain_a == "health" and domain_b == "programming":
            return f"Physical health activities correlate with improved {domain_b} performance (confidence: {confidence:.2f})"
        elif domain_a == "learning" and domain_b == "work":
            return f"Learning activities enhance work productivity (confidence: {confidence:.2f})"
        elif domain_a == "personal" and domain_b in ("work", "learning"):
            return f"Personal activities influence {domain_b} outcomes (confidence: {confidence:.2f})"
        elif domain_a == domain_b:
            return f"Strong internal connections within {domain_a} domain ({coactivation_count} links, weight: {avg_weight:.2f})"
        else:
            return f"Cross-domain correlation between {domain_a} and {domain_b}: {coactivation_count} coactivations at avg weight {avg_weight:.2f}"

    def create_procedural_node(self, pattern: Pattern) -> Optional[str]:
        """
        Create a Tier 4 procedural node for a detected pattern.

        Returns node_id or None if pattern already exists.
        """
        pattern_key = f"{pattern.source_domain}:{pattern.target_domain}"

        if pattern_key in self._detected_patterns:
            return self._detected_patterns[pattern_key]

        source_text = self.graph_manager.graph.nodes[pattern.source_node].get("text", "")
        target_text = self.graph_manager.graph.nodes[pattern.target_node].get("text", "")

        combined_text = f"{pattern.pattern_text}. Related: {source_text[:50]}... and {target_text[:50]}..."

        embedding = None
        try:
            from core.memory.classifier import MemoryClassifier
            classifier = MemoryClassifier()
            embedding = classifier._embed([combined_text])[0]
        except Exception:
            pass

        node_id = self.graph_manager.add_node(
            text=pattern.pattern_text,
            tier=TIER_PROCEDURAL,
            domain="pattern",
            embedding=embedding,
        )

        self._detected_patterns[pattern_key] = node_id

        self.graph_manager.add_edge(
            pattern.source_node,
            node_id,
            relation="pattern_source",
            weight=pattern.avg_edge_weight,
        )

        self.graph_manager.add_edge(
            node_id,
            pattern.target_node,
            relation="pattern_target",
            weight=pattern.avg_edge_weight,
        )

        return node_id

    def increment_interaction(self) -> bool:
        """
        Increment interaction counter and check if detection should run.

        Returns True if detection should run (every 50 interactions).
        """
        self._interaction_count += 1
        return self.should_run_detection()

    def should_run_detection(self) -> bool:
        """Check if detection should run based on interaction count."""
        run_every = self._config.get("procedural", "run_every_n_interactions", 50)
        return self._interaction_count > 0 and self._interaction_count % run_every == 0

    def run_detection(self) -> dict:
        """
        Run the full pattern detection pipeline.

        Returns dict with stats: patterns_found, procedural_nodes_created.
        """
        patterns = self.detect_patterns()

        significant_patterns = [
            p for p in patterns
            if p.coactivation_count >= self.min_pattern_support
        ]

        stats = {
            "patterns_found": len(significant_patterns),
            "procedural_nodes_created": 0,
        }

        print(f"\n{'='*60}")
        print("Procedural Pattern Detection Report")
        print(f"{'='*60}")
        print(f"Cross-domain edges found: {len(self.find_cross_domain_edges())}")
        print(f"Significant patterns: {len(significant_patterns)}")

        for pattern in significant_patterns:
            confidence = pattern.avg_edge_weight * (pattern.coactivation_count / 10.0)

            if confidence < self.min_confidence:
                print(f"\nSkipping low-confidence pattern: {pattern.pattern_text[:50]}...")
                continue

            print(f"\nPattern: {pattern.pattern_text[:60]}...")
            print(f"  Domains: {pattern.source_domain} <-> {pattern.target_domain}")
            print(f"  Coactivations: {pattern.coactivation_count}")
            print(f"  Avg weight: {pattern.avg_edge_weight:.3f}")
            print(f"  Confidence: {confidence:.3f}")

            node_id = self.create_procedural_node(pattern)
            if node_id:
                stats["procedural_nodes_created"] += 1
                print(f"  Created procedural node: {node_id[:8]}...")

        print(f"\n{'='*60}")
        print(f"Detection complete: {stats['procedural_nodes_created']} procedural nodes")
        print(f"{'='*60}\n")

        return stats

    def get_procedural_nodes(self) -> list[dict]:
        """Get all existing procedural nodes from the graph."""
        graph = self.graph_manager.graph
        if graph is None:
            return []

        return [
            {"id": node, **graph.nodes[node]}
            for node in graph.nodes
            if graph.nodes[node].get("tier") == TIER_PROCEDURAL
        ]


if __name__ == "__main__":
    from core.memory.graph import GraphManager, TIER_LEAF, TIER_ANCHOR

    print("Building test graph with cross-domain patterns...")
    with GraphManager("data/test_procedural.db") as gm:
        health_nodes = []
        programming_nodes = []
        work_nodes = []
        personal_nodes = []

        for i in range(5):
            n = gm.add_node(
                f"Went to the gym and {['did cardio', 'ran 5k', 'lifted weights', 'did yoga', 'swam'][i]}",
                TIER_LEAF, "health"
            )
            health_nodes.append(n)

        for i in range(5):
            n = gm.add_node(
                f"Programmed for {['3 hours', '5 hours', '2 hours', '4 hours', '6 hours'][i]} today",
                TIER_LEAF, "programming"
            )
            programming_nodes.append(n)

        for i in range(4):
            n = gm.add_node(
                f"Completed {['sprint planning', 'code review', 'deployment', 'client meeting'][i]}",
                TIER_LEAF, "work"
            )
            work_nodes.append(n)

        for i in range(3):
            n = gm.add_node(
                f"Spent weekend {['hiking', 'with family', 'reading'][i]}",
                TIER_LEAF, "personal"
            )
            personal_nodes.append(n)

        for i in range(len(health_nodes)):
            for j in range(len(programming_nodes)):
                weight = 0.5 + (i * 0.05) + (j * 0.02)
                gm.add_edge(health_nodes[i], programming_nodes[j], "boosts", weight)

        for i in range(len(work_nodes)):
            for j in range(len(programming_nodes)):
                weight = 0.4 + (i * 0.1)
                gm.add_edge(work_nodes[i], programming_nodes[j], "requires", weight)

        for i in range(len(personal_nodes)):
            for j in range(len(work_nodes)):
                gm.add_edge(personal_nodes[i], work_nodes[j], "enables", 0.3)

        print("\n" + "=" * 60)
        print("Testing ProceduralDetector")
        print("=" * 60)

        detector = ProceduralDetector(gm, min_pattern_support=3)

        for i in range(55):
            detector.increment_interaction()

        stats = detector.run_detection()

        print(f"\nFinal graph stats:")
        print(f"  Total nodes: {gm.graph.number_of_nodes()}")
        print(f"  Total edges: {gm.graph.number_of_edges()}")

        procedural = detector.get_procedural_nodes()
        print(f"  Procedural nodes: {len(procedural)}")
        for p in procedural:
            print(f"    - {p['text'][:60]}...")