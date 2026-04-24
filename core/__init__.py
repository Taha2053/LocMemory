"""
LocMemory - Local Memory System

A local-first memory system using graph-based storage, semantic embeddings,
and LLM-powered retrieval.
"""

from core.settings.config import Config, get_config
from core.memory.graph import GraphManager, TIER_CONTEXT, TIER_ANCHOR, TIER_LEAF, TIER_PROCEDURAL
from core.memory.classifier import MemoryClassifier
from core.memory.extractor import MemoryExtractor
from core.memory.retriever import GraphRetriever, RetrievedMemory
from core.memory.consolidator import MemoryConsolidator
from core.memory.hebbian import HebbianUpdater
from core.memory.procedural import ProceduralDetector, Pattern

__all__ = [
    "Config",
    "get_config",
    "GraphManager",
    "MemoryClassifier",
    "MemoryExtractor",
    "GraphRetriever",
    "RetrievedMemory",
    "MemoryConsolidator",
    "HebbianUpdater",
    "ProceduralDetector",
    "Pattern",
    "TIER_CONTEXT",
    "TIER_ANCHOR",
    "TIER_LEAF",
    "TIER_PROCEDURAL",
    "MemoryStore",
    "Memory",
]

__version__ = "1.0.0"