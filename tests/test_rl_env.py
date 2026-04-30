"""
Tests for the RL retrieval environment (core/rl/env.py).
"""

import numpy as np
import pytest

from core.rl.env import RetrievalEnv, RetrievalResult


def _make_result(n_candidates: int = 10) -> RetrievalResult:
    """Build a synthetic RetrievalResult with `n_candidates` entries."""
    candidates = []
    for i in range(n_candidates):
        candidates.append({
            "node_id": f"node_{i}",
            "text": f"Memory about topic {i}",
            "domain": "programming",
            "tier": 3,
            "score": 0.5 + i * 0.05,
            "hebbian": 0.1 + i * 0.02,
            "last_accessed": "2026-01-01T00:00:00+00:00",
        })
    return RetrievalResult(candidates=candidates, selected=[], context_str="")


def _make_embeddings(n: int, dim: int = 384, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    arr = rng.randn(n, dim).astype(np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


# ── Initialization ────────────────────────────────────────────


class TestInitialization:
    def test_default_observation_space_shape(self):
        env = RetrievalEnv()
        assert env.observation_space.shape == (485,)

    def test_default_action_space(self):
        env = RetrievalEnv()
        assert env.action_space.shape == (25,)

    def test_custom_pool_size(self):
        env = RetrievalEnv(candidate_pool_size=10)
        assert env.action_space.shape == (10,)
        state_dim = 384 + 10 * 4 + 1
        assert env.observation_space.shape == (state_dim,)

    def test_custom_token_budget(self):
        env = RetrievalEnv(token_budget=256)
        assert env.token_budget == 256

    def test_custom_top_k(self):
        env = RetrievalEnv(top_k=3)
        assert env.top_k == 3


# ── Reset ─────────────────────────────────────────────────────


class TestReset:
    def test_reset_returns_observation_and_info(self):
        env = RetrievalEnv(candidate_pool_size=10)
        result = _make_result(n_candidates=10)
        q_emb = _make_embeddings(1, 384)[0]
        c_emb = _make_embeddings(10, 384)

        obs, info = env.reset(
            seed=42,
            options={
                "retrieval_result": result,
                "query_embedding": q_emb,
                "candidate_embeddings": c_emb,
                "token_budget": 512,
            },
        )

        assert isinstance(obs, np.ndarray)
        assert obs.dtype == np.float32
        assert obs.shape == env.observation_space.shape
        assert isinstance(info, dict)

    def test_reset_with_empty_candidates(self):
        env = RetrievalEnv(candidate_pool_size=5)
        result = RetrievalResult(candidates=[], selected=[], context_str="")
        q_emb = np.zeros(384, dtype=np.float32)
        c_emb = np.zeros((5, 384), dtype=np.float32)

        obs, info = env.reset(options={
            "retrieval_result": result,
            "query_embedding": q_emb,
            "candidate_embeddings": c_emb,
        })

        assert obs.shape == env.observation_space.shape
        assert not np.any(np.isnan(obs))

    def test_reset_without_options(self):
        env = RetrievalEnv()
        obs, info = env.reset()
        assert obs.shape == env.observation_space.shape
        assert not np.any(np.isnan(obs))


# ── Step ──────────────────────────────────────────────────────


class TestStep:
    def _setup_env(self, n=10, pool=25):
        env = RetrievalEnv(candidate_pool_size=pool)
        result = _make_result(n_candidates=n)
        q_emb = _make_embeddings(1, 384)[0]
        c_emb = _make_embeddings(n, 384)
        env.reset(options={
            "retrieval_result": result,
            "query_embedding": q_emb,
            "candidate_embeddings": c_emb,
            "token_budget": 512,
        })
        return env

    def test_step_returns_correct_tuple(self):
        env = self._setup_env()
        action = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0], dtype=np.int32)
        obs, reward, terminated, truncated, info = env.step(action)

        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (float, np.floating))
        assert terminated is True
        assert truncated is False
        assert "selected_count" in info

    def test_step_with_select_all(self):
        env = self._setup_env(n=5, pool=10)
        action = np.ones(10, dtype=np.int32)
        _, reward, _, _, info = env.step(action)
        assert info["selected_count"] == 10

    def test_step_with_select_none(self):
        env = self._setup_env()
        action = np.zeros(25, dtype=np.int32)
        _, reward, _, _, info = env.step(action)
        assert info["selected_count"] == 0
        assert info["reward_components"]["empty_penalty"] == 1.0

    def test_step_with_single_selection(self):
        env = self._setup_env()
        action = np.zeros(25, dtype=np.int32)
        action[0] = 1
        _, reward, _, _, info = env.step(action)
        assert info["selected_count"] == 1

    def test_reward_is_reasonable_range(self):
        env = self._setup_env()
        action = np.zeros(25, dtype=np.int32)
        action[:3] = 1
        _, reward, _, _, _ = env.step(action)
        assert -1.0 <= reward <= 1.0

    def test_select_none_gives_lower_reward_than_select_some(self):
        env = self._setup_env()
        action_none = np.zeros(25, dtype=np.int32)
        _, reward_none, _, _, _ = env.step(action_none)

        env2 = self._setup_env()
        action_some = np.zeros(25, dtype=np.int32)
        action_some[:2] = 1
        _, reward_some, _, _, _ = env2.step(action_some)

        assert reward_some > reward_none

    def test_info_contains_selected_indices(self):
        env = self._setup_env()
        action = np.zeros(25, dtype=np.int32)
        action[1] = 1
        action[3] = 1
        _, _, _, _, info = env.step(action)
        assert info["selected_indices"] == [1, 3]

    def test_reward_components_keys(self):
        env = self._setup_env()
        action = np.zeros(25, dtype=np.int32)
        action[0] = 1
        _, _, _, _, info = env.step(action)
        comps = info["reward_components"]
        for key in ["semantic_overlap", "diversity", "token_efficiency", "empty_penalty"]:
            assert key in comps


# ── Reward calculation ────────────────────────────────────────


class TestRewardCalculation:
    def test_empty_selection_has_full_penalty(self):
        env = RetrievalEnv(candidate_pool_size=10)
        result = _make_result(n_candidates=10)
        q_emb = _make_embeddings(1, 384)[0]
        c_emb = _make_embeddings(10, 384)
        env.reset(options={
            "retrieval_result": result,
            "query_embedding": q_emb,
            "candidate_embeddings": c_emb,
        })
        action = np.zeros(10, dtype=np.int32)
        _, _, _, _, info = env.step(action)
        assert info["reward_components"]["empty_penalty"] == 1.0
        assert info["reward_components"]["semantic_overlap"] == 0.0

    def test_diversity_with_identical_candidates(self):
        env = RetrievalEnv(candidate_pool_size=5)
        candidates = [
            {"node_id": "a", "text": "same", "domain": "x", "tier": 3,
             "score": 0.5, "hebbian": 0.1, "last_accessed": ""},
            {"node_id": "b", "text": "same", "domain": "x", "tier": 3,
             "score": 0.5, "hebbian": 0.1, "last_accessed": ""},
        ]
        embs = np.array([[1, 0, 0], [1, 0, 0]], dtype=np.float32)
        q_emb = np.array([1, 0, 0], dtype=np.float32)
        result = RetrievalResult(candidates=candidates, selected=[], context_str="")
        env.reset(options={
            "retrieval_result": result,
            "query_embedding": q_emb,
            "candidate_embeddings": embs,
        })
        action = np.array([1, 1, 0, 0, 0], dtype=np.int32)
        _, _, _, _, info = env.step(action)
        assert info["reward_components"]["diversity"] == pytest.approx(0.0, abs=0.01)

    def test_token_efficiency_with_large_text(self):
        env = RetrievalEnv(candidate_pool_size=5, token_budget=100)
        candidates = [
            {"node_id": "a", "text": "x" * 200, "domain": "x", "tier": 3,
             "score": 0.5, "hebbian": 0.1, "last_accessed": ""},
        ]
        embs = np.ones((1, 384), dtype=np.float32)
        q_emb = np.ones(384, dtype=np.float32)
        result = RetrievalResult(candidates=candidates, selected=[], context_str="")
        env.reset(options={
            "retrieval_result": result,
            "query_embedding": q_emb,
            "candidate_embeddings": embs,
        })
        action = np.array([1, 0, 0, 0, 0], dtype=np.int32)
        _, _, _, _, info = env.step(action)
        assert info["reward_components"]["token_efficiency"] == pytest.approx(1.0, abs=0.01)


# ── State building ────────────────────────────────────────────


class TestStateBuilding:
    def test_state_dimension(self):
        env = RetrievalEnv(candidate_pool_size=10, embedding_dim=384)
        result = _make_result(n_candidates=10)
        q_emb = np.zeros(384, dtype=np.float32)
        c_emb = np.zeros((10, 384), dtype=np.float32)
        obs, _ = env.reset(options={
            "retrieval_result": result,
            "query_embedding": q_emb,
            "candidate_embeddings": c_emb,
        })
        expected = 384 + 10 * 4 + 1
        assert obs.shape == (expected,)

    def test_state_has_no_nans(self):
        env = RetrievalEnv()
        result = _make_result()
        q_emb = _make_embeddings(1, 384)[0]
        c_emb = _make_embeddings(10, 384)
        obs, _ = env.reset(options={
            "retrieval_result": result,
            "query_embedding": q_emb,
            "candidate_embeddings": c_emb,
        })
        assert not np.any(np.isnan(obs))
        assert not np.any(np.isinf(obs))


# ── Cosine similarity utility ─────────────────────────────────


class TestCosineSimilarity:
    def test_identical_vectors(self):
        from core.rl.env import cosine_similarity_batch
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        sims = cosine_similarity_batch(v, v.reshape(1, -1))
        assert sims[0] == pytest.approx(1.0, abs=1e-5)

    def test_orthogonal_vectors(self):
        from core.rl.env import cosine_similarity_batch
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        sims = cosine_similarity_batch(a, b.reshape(1, -1))
        assert sims[0] == pytest.approx(0.0, abs=1e-5)

    def test_batch_returns_array(self):
        from core.rl.env import cosine_similarity_batch
        q = np.array([1.0, 0.0], dtype=np.float32)
        cs = np.array([[1.0, 0.0], [0.0, 1.0], [0.707, 0.707]], dtype=np.float32)
        sims = cosine_similarity_batch(q, cs)
        assert sims.shape == (3,)

    def test_none_inputs(self):
        from core.rl.env import cosine_similarity_batch
        result = cosine_similarity_batch(None, np.zeros((5, 384), dtype=np.float32))
        assert len(result) == 5


# ── Render ────────────────────────────────────────────────────


class TestRender:
    def test_render_returns_none(self):
        env = RetrievalEnv()
        result = _make_result(n_candidates=5)
        q_emb = np.zeros(384, dtype=np.float32)
        c_emb = np.zeros((5, 384), dtype=np.float32)
        env.reset(options={
            "retrieval_result": result,
            "query_embedding": q_emb,
            "candidate_embeddings": c_emb,
        })
        output = env.render(mode="human")
        assert output is None

    def test_render_with_no_result(self):
        env = RetrievalEnv()
        output = env.render(mode="human")
        assert output is None
