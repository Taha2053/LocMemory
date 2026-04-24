import pytest
import tempfile
from pathlib import Path

from core.memory.graph import GraphManager, TIER_LEAF


@pytest.fixture
def temp_db():
    db_path = tempfile.mktemp(suffix=".db")
    yield db_path
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def gm(temp_db):
    with GraphManager(temp_db) as manager:
        yield manager


@pytest.fixture
def sample_graph(gm):
    ids = [
        gm.add_node("The Eiffel Tower is 330 metres tall.", TIER_LEAF, "fact"),
        gm.add_node("Buy oat milk and sourdough bread.", TIER_LEAF, "todo"),
        gm.add_node("Finished reading Dune on a rainy Tuesday.", TIER_LEAF, "personal"),
        gm.add_node("Python uses indentation to define code blocks.", TIER_LEAF, "programming"),
        gm.add_node("I went for a morning run along the beach.", TIER_LEAF, "health"),
    ]
    return ids