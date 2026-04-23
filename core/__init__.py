"""
LocMemory - Cognitive Memory System

A local-first memory system using graph-based storage, semantic embeddings,
and LLM-powered retrieval.
"""

from core.config import Config, get_config
from core.graph import GraphManager, TIER_CONTEXT, TIER_ANCHOR, TIER_LEAF, TIER_PROCEDURAL
from core.classifier import MemoryClassifier
from core.extractor import MemoryExtractor
from core.retriever import GraphRetriever, RetrievedMemory
from core.consolidator import MemoryConsolidator
from core.hebbian import HebbianUpdater
from core.procedural import ProceduralDetector, Pattern

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
]

__version__ = "1.0.0"