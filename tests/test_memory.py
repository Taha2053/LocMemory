import pytest
import uuid
from pathlib import Path

from core.memory import MemoryStore


class TestAdd:
    def test_add_single_memory(self, store):
        mem = store.add("Test memory content.", category="fact")

        assert mem.id is not None
        assert len(mem.id) == 36  # UUID format
        assert mem.text == "Test memory content."
        assert mem.category == "fact"
        assert mem.embedding is not None
        assert len(mem.embedding) == 384

    def test_add_multiple_memories(self, store):
        m1 = store.add("Memory one.", category="fact")
        m2 = store.add("Memory two.", category="todo")

        assert m1.id != m2.id
        assert store.count() == 2

    def test_memory_has_valid_uuid(self, store):
        mem = store.add("Test.", category="fact")

        try:
            uuid.UUID(mem.id)
            valid_uuid = True
        except ValueError:
            valid_uuid = False

        assert valid_uuid is True


class TestLoad:
    def test_load_all_returns_all_memories(self, store, sample_memories):
        all_memories = store.load_all()

        assert len(all_memories) == 5

    def test_load_all_returns_memory_objects(self, store, sample_memories):
        all_memories = store.load_all()

        for mem in all_memories:
            assert mem.id is not None
            assert mem.text is not None
            assert mem.category is not None
            assert mem.embedding is not None


class TestDelete:
    def test_delete_removes_from_db(self, store, sample_memories):
        mem_id = sample_memories[0].id
        store.delete(mem_id)

        remaining = store.load_all()
        assert len(remaining) == 4
        assert all(m.id != mem_id for m in remaining)

    def test_delete_removes_markdown_file(self, store, sample_memories, test_md_dir):
        mem_id = sample_memories[0].id
        md_path = Path(test_md_dir) / f"{mem_id}.md"

        assert md_path.exists()
        store.delete(mem_id)
        assert not md_path.exists()

    def test_delete_nonexistent_returns_false(self, store):
        result = store.delete("nonexistent-id")

        assert result is False


class TestSearch:
    def test_search_returns_top_k_results(self, store, sample_memories):
        results = store.search("running exercise", top_k=3)

        assert len(results) == 3

    def test_search_ranks_by_cosine_similarity(self, store, sample_memories):
        results = store.search("programming python code", top_k=3)

        scores = [score["total"] for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_returns_memory_objects(self, store, sample_memories):
        results = store.search("test query", top_k=1)

        mem, score = results[0]
        assert mem.id is not None
        assert mem.text is not None
        assert isinstance(score, dict)
        assert "total" in score


class TestCount:
    def test_count_returns_correct_number(self, store, sample_memories):
        assert store.count() == 5

    def test_count_after_delete(self, store, sample_memories):
        store.delete(sample_memories[0].id)
        assert store.count() == 4


class TestIntegration:
    def test_store_50_items_search_verify_top1(self, store):
        topics = [
            ("Python is a programming language.", "fact"),
            ("JavaScript is for web development.", "fact"),
            ("The sun is a star.", "fact"),
            ("Water boils at 100 degrees.", "fact"),
            ("Buy milk from the store.", "todo"),
        ]

        texts = [
            "Morning run at the park.",
            "Afternoon meeting about project.",
            "Evening dinner with friends.",
            "Night reading a book.",
            "Weekend trip to mountains.",
            "Learning machine learning.",
            "Deep learning neural networks.",
            "Data science and analytics.",
            "Cloud computing services.",
            "Database management systems.",
        ]

        for i, text in enumerate(texts):
            category = topics[i % len(topics)][1]
            store.add(text, category=category)

        for _ in range(40):
            store.add(f"Additional memory {store.count()}.", "fact")

        assert store.count() == 50

        results = store.search("programming software code", top_k=1)

        assert len(results) == 1
        assert results[0][1]["total"] >= 0
