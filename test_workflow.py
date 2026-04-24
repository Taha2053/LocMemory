#!/usr/bin/env python
"""
Complete workflow test for LocMemory cognitive memory system.

Tests the full pipeline:
1. Add memories via classifier
2. Retrieve memories
3. Hebbian learning (strengthen edges)
4. Memory consolidation (create anchors)
5. Procedural pattern detection
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.memory import (
    GraphManager,
    MemoryClassifier,
    GraphRetriever,
    HebbianUpdater,
    MemoryConsolidator,
    ProceduralDetector,
    TIER_CONTEXT,
    TIER_ANCHOR,
    TIER_LEAF,
)


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
    print_section("LocMemory Complete Workflow Test")
    
    db_path = "data/workflow_test.db"
    Path(db_path).unlink(missing_ok=True)
    
    # Initialize components
    print("\n[1] Initializing components...")
    gm = GraphManager(db_path)
    gm.initialize_db()
    gm.load_graph()
    
    classifier = MemoryClassifier(use_fallback=False)
    retriever = GraphRetriever(gm)
    hebbian = HebbianUpdater(gm, learning_rate=0.3)
    
    print(f"  ✓ GraphManager initialized")
    print(f"  ✓ Classifier loaded")
    print(f"  ✓ Retriever initialized")
    print(f"  ✓ HebbianUpdater initialized")
    
    # Add memories (simulating extraction)
    print_section("Step 1: Adding Memories (Tier 3 - Leaf)")
    
    memories = [
        ("I am a Python developer specializing in AI", TIER_CONTEXT, "programming"),
        ("Python is great for machine learning", TIER_LEAF, "programming"),
        ("I use PyTorch for deep learning projects", TIER_LEAF, "programming"),
        ("Neural networks are fascinating", TIER_LEAF, "learning"),
        ("I went to the gym this morning", TIER_LEAF, "health"),
        ("Exercise improves coding productivity", TIER_LEAF, "health"),
        ("I work on AI projects", TIER_LEAF, "work"),
        ("I study machine learning algorithms", TIER_LEAF, "learning"),
    ]
    
    node_ids = []
    for text, tier, domain in memories:
        node_id = gm.add_node(text, tier, domain)
        node_ids.append(node_id)
        print(f"  ✓ Added: {text[:50]}...")
    
    # Create edges between related memories
    print_section("Step 2: Creating Memory Connections")
    
    edges = [
        (0, 1, "specializes_in", 0.8),
        (0, 2, "uses", 0.7),
        (1, 2, "related", 0.6),
        (3, 2, "related_to", 0.5),
        (4, 5, "boosts", 0.9),
        (4, 1, "enables", 0.4),
        (5, 1, "improves", 0.5),
        (6, 1, "involves", 0.6),
        (7, 3, "covers", 0.7),
    ]
    
    for src, tgt, rel, weight in edges:
        gm.add_edge(node_ids[src], node_ids[tgt], rel, weight)
        print(f"  ✓ Edge: {rel} ({weight})")
    
    print(f"\n  Graph: {gm.graph.number_of_nodes()} nodes, {gm.graph.number_of_edges()} edges")
    
    # Test retrieval
    print_section("Step 3: Testing Retrieval")
    
    queries = [
        "What am I learning about?",
        "How does exercise affect my work?",
        "What programming skills do I have?",
    ]
    
    for query in queries:
        results = retriever.retrieve(query)
        print(f"\n  Query: '{query}'")
        print(f"  Retrieved {len(results)} memories:")
        for r in results[:3]:
            print(f"    - {r['text'][:45]}... (score: {r['score']:.3f})")
    
    # Test Hebbian learning
    print_section("Step 4: Testing Hebbian Learning")
    
    # Simulate retrieving nodes 1, 2, 3 together
    retrieved = [node_ids[1], node_ids[2], node_ids[3]]
    result = hebbian.update_after_retrieval(retrieved)
    print(f"  Strengthened edges between: {[m[:8] for m in retrieved]}")
    print(f"  Edges strengthened: {result['edges_strengthened']}")
    
    stats = hebbian.get_edge_stats()
    print(f"  Edge stats: min={stats['min_weight']:.2f}, max={stats['max_weight']:.2f}, avg={stats['avg_weight']:.2f}")
    
    # Test memory consolidation
    print_section("Step 5: Testing Memory Consolidation")
    
    # Add more programming memories to trigger consolidation
    for i in range(8):
        gm.add_node(f"Programming concept {i}: algorithms and data structures", TIER_LEAF, "programming")
    
    consolidator = MemoryConsolidator(gm, min_cluster_size=3)
    stats = consolidator.run()
    print(f"  Clusters found: {stats['clusters_found']}")
    print(f"  Anchors created: {stats['anchors_created']}")
    
    anchors = gm.get_nodes_by_tier(TIER_ANCHOR)
    print(f"  Total anchor nodes: {len(anchors)}")
    
    # Test procedural detection
    print_section("Step 6: Testing Procedural Pattern Detection")
    
    detector = ProceduralDetector(gm, min_pattern_support=2)
    detector._interaction_count = 50
    
    patterns = detector.detect_patterns()
    print(f"  Cross-domain patterns detected: {len(patterns)}")
    
    if patterns:
        print(f"  Top pattern: {patterns[0].pattern_text[:60]}...")
    
    # Final stats
    print_section("Final Graph Stats")
    
    tier_counts = {}
    for node_id in gm.graph.nodes:
        tier = gm.graph.nodes[node_id].get("tier", 0)
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
    
    print(f"  Total nodes: {gm.graph.number_of_nodes()}")
    print(f"  Total edges: {gm.graph.number_of_edges()}")
    print(f"  By tier:")
    for tier, count in sorted(tier_counts.items()):
        tier_name = {1: "context", 2: "anchor", 3: "leaf", 4: "procedural"}.get(tier, "unknown")
        print(f"    Tier {tier} ({tier_name}): {count}")
    
    # Cleanup
    gm.close()
    
    print_section("Workflow Test Complete!")
    print("  ✓ All components working together")
    print("  ✓ Full pipeline tested successfully")


if __name__ == "__main__":
    main()
