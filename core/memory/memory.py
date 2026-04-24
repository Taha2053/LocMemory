"""
Compatibility layer for legacy MemoryStore API.

Wraps the new GraphManager to maintain backward compatibility
with existing code (chat.py, context.py).
"""

from dataclasses import dataclass
from typing import Optional

from core.memory.graph import GraphManager, TIER_CONTEXT, TIER_ANCHOR, TIER_LEAF, TIER_PROCEDURAL
from core.memory.classifier import MemoryClassifier
from core.memory.retriever import GraphRetriever


TIER_NAMES = {
    "context": TIER_CONTEXT,
    "anchor": TIER_ANCHOR,
    "leaf": TIER_LEAF,
    "procedural": TIER_PROCEDURAL,
}


@dataclass
class Memory:
    """Legacy memory object for compatibility."""
    id: str
    text: str
    category: str
    timestamp: str = ""
    embedding: Optional[list] = None


class MemoryStore:
    """
    Legacy MemoryStore API - wraps GraphManager for backward compatibility.
    """

    def __init__(self, db_path: str = "data/memories.db", md_dir: str = "memories"):
        self.db_path = db_path
        self.md_dir = md_dir
        self.gm = GraphManager(db_path)
        self.gm.initialize_db()
        self.gm.load_graph()
        self.classifier = MemoryClassifier(use_fallback=False)
        self.retriever = GraphRetriever(self.gm)

    def add(self, text: str, category: Optional[str] = None) -> Memory:
        """Add a memory to the store."""
        if category is None:
            result = self.classifier.classify(text)
            category = result.get("domain", "general")

        # Process for PII and encrypt if needed
        from core.security.security import process_before_store, get_encryptor
        processed_text, was_encrypted = process_before_store(text, get_encryptor())

        tier = TIER_LEAF
        node_id = self.gm.add_node(
            processed_text,
            tier,
            category,
            metadata={"encrypted": was_encrypted} if was_encrypted else None,
        )

        return Memory(
            id=node_id,
            text=text,  # Return original text
            category=category,
            timestamp="",
        )

    def search(
        self,
        query: str,
        top_k: int = 3,
        category: Optional[str] = None
    ):
        """
        Search for memories matching the query.
        Returns list of (Memory, score_dict) tuples.
        """
        results = self.retriever.retrieve(query)

        from core.security.security import decrypt_for_retrieval, get_encryptor
        encryptor = get_encryptor()

        output = []
        for r in results[:top_k]:
            text = r.get("text", "")
            # Decrypt if encrypted
            if r.get("encrypted") or encryptor.is_encrypted(text):
                text = decrypt_for_retrieval(text, encryptor)

            mem = Memory(
                id=r.get("node_id", ""),
                text=text,
                category=r.get("domain", ""),
            )
            score = {"total": r.get("score", 0)}
            output.append((mem, score))

        return output

    def load_all(self) -> list[Memory]:
        """Load all memories."""
        memories = []
        for node_id in self.gm.graph.nodes:
            node_data = self.gm.graph.nodes[node_id]
            memories.append(Memory(
                id=node_id,
                text=node_data.get("text", ""),
                category=node_data.get("domain", ""),
                timestamp=node_data.get("created_at", ""),
            ))
        return memories

    def count(self) -> int:
        """Count total memories."""
        return self.gm.graph.number_of_nodes() if self.gm.graph else 0

    def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        if self.gm.graph and memory_id in self.gm.graph:
            self.gm.graph.remove_node(memory_id)
            return True
        return False

    def get_nodes_by_tier(self, tier: int) -> list[dict]:
        """Get nodes by tier."""
        return self.gm.get_nodes_by_tier(tier)

    def close(self):
        """Close the memory store."""
        self.gm.close()