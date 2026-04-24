"""Tests for embedding behavior via MemoryClassifier._embed."""

import math


def test_embed_single_returns_384_dim(classifier):
    emb = classifier._embed(["some text"])[0]
    assert len(emb) == 384


def test_embed_batch_returns_one_per_input(classifier):
    embs = classifier._embed(["one", "two", "three"])
    assert len(embs) == 3
    assert all(len(e) == 384 for e in embs)


def test_identical_texts_have_high_similarity(classifier):
    a, b = classifier._embed(["hello world", "hello world"])
    sim = classifier._cosine_similarity(a, b)
    assert sim > 0.99


def test_unrelated_texts_have_lower_similarity(classifier):
    a, b = classifier._embed([
        "I love running every morning",
        "quarterly financial report numbers",
    ])
    sim_same, _ = classifier._embed(["I love running every morning", "I love running every morning"])
    sim_same_val = classifier._cosine_similarity(sim_same, _)

    sim_diff = classifier._cosine_similarity(a, b)
    assert sim_diff < sim_same_val


def test_cosine_similarity_zero_vector(classifier):
    assert classifier._cosine_similarity([0.0, 0.0, 0.0], [1.0, 2.0, 3.0]) == 0.0


def test_cosine_similarity_is_bounded(classifier):
    a, b = classifier._embed(["anything at all", "totally different content"])
    sim = classifier._cosine_similarity(a, b)
    assert -1.0 - 1e-6 <= sim <= 1.0 + 1e-6
    assert not math.isnan(sim)
