import pytest
import sqlite3
import shutil
import uuid
from pathlib import Path

from core.memory import MemoryStore


@pytest.fixture
def test_db_path(tmp_path):
    return str(tmp_path / "test_memories.db")


@pytest.fixture
def test_md_dir(tmp_path):
    return str(tmp_path / "memories")


@pytest.fixture
def store(test_db_path, test_md_dir):
    return MemoryStore(db_path=test_db_path, md_dir=test_md_dir)


@pytest.fixture
def sample_memories(store):
    memories = [
        ("The Eiffel Tower is 330 metres tall.", "fact"),
        ("Buy oat milk and sourdough bread.", "todo"),
        ("Finished reading Dune on a rainy Tuesday.", "event"),
        ("Python uses indentation to define code blocks.", "fact"),
        ("I went for a morning run along the beach.", "event"),
    ]
    added = []
    for text, category in memories:
        added.append(store.add(text, category=category))
    return added
