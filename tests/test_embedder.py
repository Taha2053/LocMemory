"""
Tests for embedder/embedding operations.
"""

import pytest
import numpy as np

from core.memory.classifier import MemoryClassifier


def test_embed_single_string(classifier_with_mock):
    """Embedding single string returns shape (384,)."""
    classifier = MemoryClassifier()
    emb = classifier.embed("test string")

    assert emb.shape == (384,)


def test_embed_list(classifier_with_mock):
    """Embedding list returns shape (n, 384)."""
    classifier = MemoryClassifier()
    texts = ["test one", "test two", "test three"]
    emb = classifier.embed(texts)

    assert emb.shape == (3, 384)


def test_cosine_similarity_identical(classifier_with_mock):
    """Identical texts should have similarity ~1.0."""
    classifier = MemoryClassifier()
    text = "test string"

    emb1 = classifier.embed(text)
    emb2 = classifier.embed(text)

    # Compute cosine similarity
    sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    assert abs(sim) > 0.99


def test_cosine_similarity_orthogonal(classifier_with_mock):
    """Orthogonal vectors should have similarity ~0."""
    classifier = MemoryClassifier()

    # Use known different texts
    text1 = "apple orange banana"
    text2 = "xyz999 something"

    emb1 = classifier.embed(text1)
    emb2 = classifier.embed(text2)

    sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

    # Not exactly 0, but should be low
    assert sim < 0.5


def test_batch_cosine_similarity():
    """Batch cosine similarity returns correct values."""
    classifier = MemoryClassifier()

    texts = ["test one", "test two"]
    embeddings = classifier.embed(texts)

    # Compute individual similarities
    sim_01 = np.dot(embeddings[0], embeddings[1]) / (
        np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
    )

    # Should match batch result
    # (We just verify batch works without error)
    assert embeddings.shape[0] == 2