import pytest
from pathlib import Path
import tempfile
import shutil

from core.memory.graph import GraphManager, TIER_CONTEXT, TIER_ANCHOR, TIER_LEAF, TIER_PROCEDURAL
from core.memory.classifier import MemoryClassifier
from core.memory.retriever import GraphRetriever
from core.memory.hebbian import HebbianUpdater


@pytest.fixture
def temp_db():
    db_path = tempfile.mktemp(suffix=".db")
    yield db_path
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def gm(temp_db):
    with GraphManager(temp_db) as manager:
        yield manager


class TestGraphManager:
    def test_initialize_db_creates_tables(self, temp_db):
        gm = GraphManager(temp_db)
        gm.initialize_db()
        assert Path(temp_db).exists()
        gm.close()

    def test_add_node_returns_id(self, gm):
        node_id = gm.add_node("Test memory", TIER_LEAF, "test")
        assert node_id is not None
        assert len(node_id) == 36

    def test_add_edge_creates_connection(self, gm):
        id1 = gm.add_node("Node 1", TIER_LEAF, "domain1")
        id2 = gm.add_node("Node 2", TIER_LEAF, "domain2")
        result = gm.add_edge(id1, id2, "related", 0.5)
        assert result is True

    def test_get_nodes_by_tier(self, gm):
        gm.add_node("Context 1", TIER_CONTEXT, "test")
        gm.add_node("Anchor 1", TIER_ANCHOR, "test")
        gm.add_node("Leaf 1", TIER_LEAF, "test")

        anchors = gm.get_nodes_by_tier(TIER_ANCHOR)
        assert len(anchors) == 1
        assert anchors[0]["text"] == "Anchor 1"

    def test_get_nodes_by_domain(self, gm):
        gm.add_node("Work task 1", TIER_LEAF, "work")
        gm.add_node("Work task 2", TIER_LEAF, "work")
        gm.add_node("Health fact", TIER_LEAF, "health")

        work_nodes = gm.get_nodes_by_domain("work")
        assert len(work_nodes) == 2

    def test_update_edge_weight(self, gm):
        id1 = gm.add_node("Node 1", TIER_LEAF, "test")
        id2 = gm.add_node("Node 2", TIER_LEAF, "test")
        gm.add_edge(id1, id2, "related", 0.5)

        result = gm.update_edge_weight(id1, id2, 0.9)
        assert result is True
        assert gm.graph.edges[id1, id2]["weight"] == 0.9


class TestMemoryClassifier:
    @pytest.fixture
    def classifier(self):
        return MemoryClassifier(use_fallback=False)

    def test_detect_domain(self, classifier):
        domain, confidence = classifier.detect_domain("I love writing Python code")
        assert domain == "programming"
        assert 0 <= confidence <= 1

    def test_extract_concepts(self, classifier):
        concepts = classifier.extract_concepts("I learned about neural networks today")
        assert len(concepts) > 0
        assert isinstance(concepts, list)

    def test_classify_returns_dict(self, classifier):
        result = classifier.classify("Went to the gym today")
        assert "domain" in result
        assert "concepts" in result
        assert "confidence" in result

    def test_list_domains(self, classifier):
        domains = classifier.list_domains()
        assert "programming" in domains
        assert "health" in domains


class TestGraphRetriever:
    @pytest.fixture
    def retriever(self, gm):
        return GraphRetriever(gm)

    def test_retrieve_returns_list(self, retriever, gm):
        gm.add_node("Test memory about Python", TIER_LEAF, "programming")
        results = retriever.retrieve("What about Python?")
        assert isinstance(results, list)

    def test_retrieve_includes_scores(self, retriever, gm):
        gm.add_node("Health and fitness facts", TIER_LEAF, "health")
        results = retriever.retrieve("exercise and wellness")
        if results:
            assert "score" in results[0]
            assert "node_id" in results[0]


class TestHebbianUpdater:
    @pytest.fixture
    def hebbian(self, gm):
        return HebbianUpdater(gm, learning_rate=0.3)

    def test_strengthen_edges(self, hebbian, gm):
        id1 = gm.add_node("Memory 1", TIER_LEAF, "test")
        id2 = gm.add_node("Memory 2", TIER_LEAF, "test")
        gm.add_edge(id1, id2, "related", 0.5)

        count = hebbian.strengthen_edges([id1, id2])
        assert count >= 1

    def test_get_edge_stats(self, hebbian, gm):
        id1 = gm.add_node("A", TIER_LEAF, "test")
        id2 = gm.add_node("B", TIER_LEAF, "test")
        gm.add_edge(id1, id2, "related", 0.5)

        stats = hebbian.get_edge_stats()
        assert stats["count"] >= 1
        assert "min_weight" in stats
        assert "max_weight" in stats

    def test_update_after_retrieval(self, hebbian, gm):
        ids = [
            gm.add_node("Memory 1", TIER_LEAF, "test"),
            gm.add_node("Memory 2", TIER_LEAF, "test"),
        ]

        result = hebbian.update_after_retrieval(ids)
        assert "edges_decayed" in result
        assert "edges_strengthened" in result


class TestIntegration:
    def test_full_pipeline(self, temp_db):
        with GraphManager(temp_db) as gm:
            classifier = MemoryClassifier(use_fallback=False)

            gm.add_node("I am a Python developer", TIER_CONTEXT, "programming")
            gm.add_node("Learning about ML", TIER_LEAF, "learning")

            retriever = GraphRetriever(gm)
            results = retriever.retrieve("What am I learning?")

            assert isinstance(results, list)

            hebbian = HebbianUpdater(gm)
            if results:
                node_ids = [r["node_id"] for r in results[:2]]
                hebbian.update_after_retrieval(node_ids)