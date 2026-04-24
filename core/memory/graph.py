"""
Graph-based memory storage system.

SQLite for persistence, NetworkX for in-memory graph operations.
"""

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import networkx as nx


TIER_CONTEXT = 1
TIER_ANCHOR = 2
TIER_LEAF = 3
TIER_PROCEDURAL = 4

TIER_NAMES = {
    1: "context",
    2: "anchor",
    3: "leaf",
    4: "procedural",
}


class GraphManager:
    """Manages a graph-based memory system with SQLite persistence and NetworkX in-memory operations."""

    def __init__(self, db_path: str = "data/graph.db"):
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        self.graph: Optional[nx.DiGraph] = None

    def _ensure_db_dir(self):
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    def initialize_db(self):
        """Create tables if they do not exist."""
        self._ensure_db_dir()
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                tier INTEGER NOT NULL,
                domain TEXT NOT NULL DEFAULT '',
                subdomain TEXT NOT NULL DEFAULT '',
                embedding BLOB,
                created_at TEXT NOT NULL,
                metadata TEXT
            )
        """)

        existing_cols = {row["name"] for row in self.conn.execute("PRAGMA table_info(nodes)").fetchall()}
        if "subdomain" not in existing_cols:
            self.conn.execute("ALTER TABLE nodes ADD COLUMN subdomain TEXT NOT NULL DEFAULT ''")

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS edges (
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relation TEXT NOT NULL DEFAULT 'related',
                weight REAL NOT NULL DEFAULT 0.1,
                last_accessed TEXT NOT NULL,
                PRIMARY KEY (source_id, target_id),
                FOREIGN KEY (source_id) REFERENCES nodes(id) ON DELETE CASCADE,
                FOREIGN KEY (target_id) REFERENCES nodes(id) ON DELETE CASCADE
            )
        """)

        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_nodes_tier ON nodes(tier)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_nodes_domain ON nodes(domain)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_nodes_subdomain ON nodes(subdomain)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id)
        """)

        self.conn.commit()
        print(f"Database initialized: {self.db_path}")

    def seed_domains(self):
        """Create domain context nodes if they don't exist."""
        domains = [
            "health", "engineering", "programming", "work",
            "personal", "finance", "learning", "social"
        ]
        for domain in domains:
            existing = self.get_nodes_by_domain(domain)
            if not existing:
                self.add_node(f"{domain} context", TIER_CONTEXT, domain)

    def load_graph(self):
        """Load nodes and edges from SQLite into a NetworkX graph in RAM."""
        if self.conn is None:
            raise RuntimeError("Database not initialized. Call initialize_db() first.")

        self.graph = nx.DiGraph()

        nodes = self.conn.execute("SELECT * FROM nodes").fetchall()
        for row in nodes:
            embedding = json.loads(row["embedding"]) if row["embedding"] else None
            self.graph.add_node(
                row["id"],
                text=row["text"],
                tier=row["tier"],
                domain=row["domain"],
                subdomain=row["subdomain"] if "subdomain" in row.keys() else "",
                embedding=embedding,
                created_at=row["created_at"],
            )

        edges = self.conn.execute("SELECT * FROM edges").fetchall()
        for row in edges:
            self.graph.add_edge(
                row["source_id"],
                row["target_id"],
                relation=row["relation"],
                weight=row["weight"],
                last_accessed=row["last_accessed"],
            )

        print(f"Graph loaded: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")

    def add_node(
        self,
        text: str,
        tier: int,
        domain: str = "",
        subdomain: str = "",
        embedding: list[float] | None = None,
        metadata: dict | None = None,
    ) -> str:
        """
        Add a node to the graph and persist to SQLite.

        Returns the node ID.
        """
        if self.graph is None:
            raise RuntimeError("Graph not loaded. Call load_graph() first.")

        normalized = text.strip().lower()
        for existing_id, data in self.graph.nodes(data=True):
            if (
                data.get("tier") == tier
                and data.get("domain", "") == domain
                and str(data.get("text", "")).strip().lower() == normalized
            ):
                print(f"Skipped duplicate node: {existing_id[:8]}... {text[:50]}")
                return existing_id

        node_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        embedding_json = json.dumps(embedding) if embedding else None
        metadata_json = json.dumps(metadata) if metadata else None

        self.conn.execute(
            """
            INSERT INTO nodes (id, text, tier, domain, subdomain, embedding, created_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (node_id, text, tier, domain, subdomain, embedding_json, timestamp, metadata_json),
        )
        self.conn.commit()

        self.graph.add_node(
            node_id,
            text=text,
            tier=tier,
            domain=domain,
            subdomain=subdomain,
            embedding=embedding,
            created_at=timestamp,
        )

        print(f"Added node: {node_id[:8]}... [{TIER_NAMES.get(tier, tier)}] {text[:50]}")
        return node_id

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation: str = "related",
        weight: float = 0.1,
    ) -> bool:
        """
        Add an edge between two nodes and persist to SQLite.

        Returns True if successful, False if nodes don't exist.
        """
        if self.graph is None:
            raise RuntimeError("Graph not loaded. Call load_graph() first.")

        if source_id not in self.graph or target_id not in self.graph:
            return False

        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

        self.conn.execute(
            """
            INSERT OR REPLACE INTO edges (source_id, target_id, relation, weight, last_accessed)
            VALUES (?, ?, ?, ?, ?)
            """,
            (source_id, target_id, relation, weight, timestamp),
        )
        self.conn.commit()

        self.graph.add_edge(
            source_id,
            target_id,
            relation=relation,
            weight=weight,
            last_accessed=timestamp,
        )

        print(f"Added edge: {source_id[:8]}... -> {target_id[:8]}... ({relation}: {weight})")
        return True

    def get_nodes_by_tier(self, tier: int) -> list[dict]:
        """Retrieve all nodes of a specific tier from in-memory graph."""
        if self.graph is None:
            raise RuntimeError("Graph not loaded. Call load_graph() first.")

        return [
            {"id": node, **self.graph.nodes[node]}
            for node in self.graph
            if self.graph.nodes[node].get("tier") == tier
        ]

    def get_nodes_by_domain(self, domain: str) -> list[dict]:
        """Retrieve all nodes in a specific domain from in-memory graph."""
        if self.graph is None:
            raise RuntimeError("Graph not loaded. Call load_graph() first.")

        return [
            {"id": node, **self.graph.nodes[node]}
            for node in self.graph
            if self.graph.nodes[node].get("domain") == domain
        ]

    def get_nodes_by_subdomain(self, subdomain: str) -> list[dict]:
        """Retrieve all nodes in a specific subdomain from in-memory graph."""
        if self.graph is None:
            raise RuntimeError("Graph not loaded. Call load_graph() first.")

        return [
            {"id": node, **self.graph.nodes[node]}
            for node in self.graph
            if self.graph.nodes[node].get("subdomain") == subdomain
        ]

    def update_node_text(self, node_id: str, new_text: str) -> bool:
        if self.graph is None:
            raise RuntimeError("Graph not loaded. Call load_graph() first.")
        if node_id not in self.graph:
            return False
        self.conn.execute("UPDATE nodes SET text = ? WHERE id = ?", (new_text, node_id))
        self.conn.commit()
        self.graph.nodes[node_id]["text"] = new_text
        return True

    def delete_node(self, node_id: str) -> bool:
        if self.graph is None:
            raise RuntimeError("Graph not loaded. Call load_graph() first.")
        if node_id not in self.graph:
            return False
        self.conn.execute("DELETE FROM nodes WHERE id = ?", (node_id,))
        self.conn.commit()
        self.graph.remove_node(node_id)
        return True

    def update_edge_weight(self, source_id: str, target_id: str, new_weight: float) -> bool:
        """
        Update the weight of an edge and persist to SQLite.

        Returns True if successful, False if edge doesn't exist.
        """
        if self.graph is None:
            raise RuntimeError("Graph not loaded. Call load_graph() first.")

        if not self.graph.has_edge(source_id, target_id):
            return False

        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

        self.conn.execute(
            """
            UPDATE edges SET weight = ?, last_accessed = ?
            WHERE source_id = ? AND target_id = ?
            """,
            (new_weight, timestamp, source_id, target_id),
        )
        self.conn.commit()

        self.graph.edges[source_id, target_id]["weight"] = new_weight
        self.graph.edges[source_id, target_id]["last_accessed"] = timestamp

        print(f"Updated edge weight: {source_id[:8]}... -> {target_id[:8]}... = {new_weight}")
        return True

    def stats(self) -> dict:
        """Get graph statistics."""
        if self.graph is None:
            return {"tier_counts": 0, "edge_count": 0}

        tier_counts = {}
        for node_id in self.graph.nodes:
            tier = self.graph.nodes[node_id].get("tier", 3)
            tier_counts[tier] = tier_counts.get(tier, 0) + 1

        return {
            "tier_counts": tier_counts,
            "edge_count": self.graph.number_of_edges(),
            "node_count": self.graph.number_of_nodes(),
        }

    def load_networkx(self):
        """Return the NetworkX graph (for compatibility)."""
        return self.graph

    def get_neighbors(self, node_id: str, direction: str = "both") -> list[dict]:
        """
        Get neighboring nodes.

        direction: 'predecessors', 'successors', or 'both'
        """
        if self.graph is None:
            raise RuntimeError("Graph not loaded. Call load_graph() first.")

        if node_id not in self.graph:
            return []

        neighbors = []
        if direction == "predecessors":
            neighbor_ids = self.graph.predecessors(node_id)
        elif direction == "successors":
            neighbor_ids = self.graph.successors(node_id)
        else:
            neighbor_ids = list(self.graph.predecessors(node_id)) + list(self.graph.successors(node_id))

        seen = set()
        for nid in neighbor_ids:
            if nid in seen:
                continue
            seen.add(nid)

            edge_data = self.graph.edges[nid, node_id] if self.graph.has_edge(nid, node_id) else self.graph.edges[node_id, nid]
            neighbors.append({
                "id": nid,
                **self.graph.nodes[nid],
                "edge_relation": edge_data.get("relation"),
                "edge_weight": edge_data.get("weight"),
            })

        return neighbors

    def save_graph(self):
        """Explicit save - commits any pending changes to SQLite."""
        if self.conn is None:
            raise RuntimeError("Database not initialized.")

        self.conn.commit()
        print("Graph saved to database.")

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
        self.graph = None
        print("Graph manager closed.")

    def __enter__(self):
        self.initialize_db()
        self.load_graph()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.save_graph()
        self.close()


if __name__ == "__main__":
    with GraphManager("data/graph.db") as gm:
        id1 = gm.add_node(
            "I am a software developer specializing in Python",
            tier=TIER_CONTEXT,
            domain="work",
            embedding=[0.1, 0.2, 0.3],
        )
        id2 = gm.add_node(
            "I learned about graph databases today",
            tier=TIER_LEAF,
            domain="learning",
            embedding=[0.4, 0.5, 0.6],
        )
        id3 = gm.add_node(
            "Stable-Baselines3 is great for RL",
            tier=TIER_LEAF,
            domain="learning",
            embedding=[0.7, 0.8, 0.9],
        )
        id4 = gm.add_node(
            "NetworkX is useful for graph operations",
            tier=TIER_ANCHOR,
            domain="technical",
            embedding=[0.2, 0.4, 0.6],
        )

        gm.add_edge(id1, id2, relation="relates_to", weight=0.8)
        gm.add_edge(id1, id3, relation="relates_to", weight=0.6)
        gm.add_edge(id2, id4, relation="inspired_by", weight=0.9)
        gm.add_edge(id3, id4, relation="related_to", weight=0.7)

        print("\n--- Nodes by domain 'learning' ---")
        for node in gm.get_nodes_by_domain("learning"):
            print(f"  {node['text']}")

        print("\n--- Nodes by tier (anchor) ---")
        for node in gm.get_nodes_by_tier(TIER_ANCHOR):
            print(f"  {node['text']}")

        print("\n--- Neighbors of context node ---")
        for neighbor in gm.get_neighbors(id1, direction="successors"):
            print(f"  -> {neighbor['text']} ({neighbor['edge_relation']}: {neighbor['edge_weight']})")

        gm.update_edge_weight(id1, id2, 0.95)

        print("\n--- Graph stats ---")
        print(f"  Nodes: {gm.graph.number_of_nodes()}")
        print(f"  Edges: {gm.graph.number_of_edges()}")