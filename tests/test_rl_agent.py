"""
Tests for the RL agent (core/rl/agent.py).
"""

import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from core.rl.agent import RLAgent, RetrievalResult


def _make_result(n_candidates: int = 10) -> RetrievalResult:
    candidates = []
    for i in range(n_candidates):
        candidates.append({
            "node_id": f"node_{i}",
            "text": f"Memory about topic {i}",
            "domain": "programming",
            "tier": 3,
            "score": 0.5 + i * 0.05,
            "hebbian": 0.1,
            "last_accessed": "2026-01-01T00:00:00+00:00",
        })
    return RetrievalResult(candidates=candidates, selected=[], context_str="")


# ── Initialization ────────────────────────────────────────────


class TestRLAgentInit:
    def test_agent_initializes_without_model(self, tmp_path):
        agent = RLAgent(model_path=str(tmp_path / "nonexistent.zip"))
        assert agent.is_available() is False

    def test_agent_loads_config_defaults(self, tmp_path):
        agent = RLAgent(model_path=str(tmp_path / "nonexistent.zip"))
        stats = agent.get_stats()
        assert stats["candidate_pool_size"] == 25
        assert stats["top_k"] == 5
        assert stats["token_budget"] == 512

    def test_agent_with_mock_model(self, tmp_path):
        model_path = str(tmp_path / "fake.zip")
        agent = RLAgent(model_path=model_path)

        with patch("stable_baselines3.PPO") as mock_ppo:
            mock_model = MagicMock()
            mock_ppo.load.return_value = mock_model

            agent2 = RLAgent(model_path=model_path)

            import os
            Path(model_path).touch()
            agent3 = RLAgent(model_path=model_path)


# ── Fallback hybrid select ───────────────────────────────────


class TestHybridSelect:
    def test_fallback_returns_top_k_within_budget(self):
        agent = RLAgent(model_path="/no/model.zip")
        result = _make_result(n_candidates=10)
        q_emb = np.zeros(384, dtype=np.float32)

        selected = agent.select(result, q_emb, token_budget=200)
        assert len(selected) > 0

    def test_fallback_empty_candidates(self):
        agent = RLAgent(model_path="/no/model.zip")
        result = RetrievalResult(candidates=[], selected=[], context_str="")
        q_emb = np.zeros(384, dtype=np.float32)

        selected = agent.select(result, q_emb, token_budget=200)
        assert selected == []

    def test_fallback_respects_tiny_budget(self):
        agent = RLAgent(model_path="/no/model.zip")
        result = RetrievalResult(
            candidates=[{"node_id": "x", "text": "A" * 1000, "tier": 3, "score": 0.9, "hebbian": 0.1, "last_accessed": ""}],
            selected=[],
            context_str="",
        )
        q_emb = np.zeros(384, dtype=np.float32)

        selected = agent.select(result, q_emb, token_budget=10)
        assert selected == []

    def test_fallback_sorts_by_score(self):
        agent = RLAgent(model_path="/no/model.zip")
        result = RetrievalResult(
            candidates=[
                {"node_id": "low", "text": "low score memory", "tier": 3, "score": 0.1, "hebbian": 0.1, "last_accessed": ""},
                {"node_id": "high", "text": "high score memory", "tier": 3, "score": 0.9, "hebbian": 0.1, "last_accessed": ""},
                {"node_id": "mid", "text": "mid score memory", "tier": 3, "score": 0.5, "hebbian": 0.1, "last_accessed": ""},
            ],
            selected=[],
            context_str="",
        )
        q_emb = np.zeros(384, dtype=np.float32)
        selected = agent.select(result, q_emb, token_budget=1000)
        assert selected[0]["node_id"] == "high"


# ── RL-based select (mocked model) ────────────────────────────


class TestRLSelect:
    def _make_agent_with_model(self, tmp_path):
        model_path = str(tmp_path / "model.zip")
        model_file = Path(model_path)
        model_file.touch()

        with patch("stable_baselines3.PPO") as mock_ppo:
            mock_model = MagicMock()
            mock_ppo.load.return_value = mock_model
            agent = RLAgent(model_path=model_path)

        return agent, mock_model

    def test_rl_select_calls_predict(self, tmp_path):
        agent, mock_model = self._make_agent_with_model(tmp_path)
        mock_model.predict.return_value = (
            np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0]),
            None,
        )
        result = _make_result(n_candidates=10)
        q_emb = np.zeros(384, dtype=np.float32)

        selected = agent.select(result, q_emb, token_budget=500)
        mock_model.predict.assert_called_once()

    def test_rl_select_falls_back_on_predict_error(self, tmp_path):
        agent, mock_model = self._make_agent_with_model(tmp_path)
        mock_model.predict.side_effect = RuntimeError("predict failed")

        result = _make_result(n_candidates=10)
        q_emb = np.zeros(384, dtype=np.float32)

        selected = agent.select(result, q_emb, token_budget=500)
        assert len(selected) > 0


# ── State building ────────────────────────────────────────────


class TestStateBuilding:
    def test_build_state_shape(self):
        agent = RLAgent(model_path="/no/model.zip")
        result = _make_result(n_candidates=25)
        q_emb = np.zeros(384, dtype=np.float32)

        state = agent._build_state(result, q_emb, token_budget=512)
        expected_dim = 384 + 25 * 4 + 1
        assert state.shape == (expected_dim,)
        assert state.dtype == np.float32

    def test_build_state_no_nans(self):
        agent = RLAgent(model_path="/no/model.zip")
        result = _make_result(n_candidates=10)
        q_emb = np.ones(384, dtype=np.float32)

        state = agent._build_state(result, q_emb, token_budget=256)
        assert not np.any(np.isnan(state))

    def test_build_state_with_truncated_embedding(self):
        agent = RLAgent(model_path="/no/model.zip")
        result = _make_result(n_candidates=5)
        q_emb = np.zeros(500, dtype=np.float32)

        state = agent._build_state(result, q_emb, token_budget=512)
        assert state.shape[0] == 384 + 25 * 4 + 1

    def test_build_state_with_small_embedding(self):
        agent = RLAgent(model_path="/no/model.zip")
        result = _make_result(n_candidates=5)
        q_emb = np.zeros(100, dtype=np.float32)

        state = agent._build_state(result, q_emb, token_budget=512)
        assert state.shape[0] == 384 + 25 * 4 + 1

    def test_parse_recency_defaults(self):
        agent = RLAgent(model_path="/no/model.zip")
        assert agent._parse_recency("") == 0.5
        assert agent._parse_recency("invalid") == 0.5

    def test_parse_recency_recent(self):
        from datetime import datetime, timezone, timedelta
        agent = RLAgent(model_path="/no/model.zip")
        recent = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        score = agent._parse_recency(recent)
        assert score > 0.0 and score < 0.01


# ── Stats ─────────────────────────────────────────────────────


class TestStats:
    def test_get_stats_returns_dict(self):
        agent = RLAgent(model_path="/no/model.zip")
        stats = agent.get_stats()
        assert isinstance(stats, dict)
        for key in ["available", "model_path", "candidate_pool_size", "top_k", "token_budget"]:
            assert key in stats

    def test_stats_available_false_without_model(self):
        agent = RLAgent(model_path="/no/model.zip")
        stats = agent.get_stats()
        assert stats["available"] is False


# ── RetrievalResult ──────────────────────────────────────────


class TestRetrievalResult:
    def test_default_values(self):
        r = RetrievalResult(candidates=[{"id": 1}])
        assert r.selected == []
        assert r.context_str == ""

    def test_with_all_fields(self):
        r = RetrievalResult(
            candidates=[{"id": 1}],
            selected=[{"id": 1}],
            context_str="hello",
        )
        assert len(r.candidates) == 1
        assert len(r.selected) == 1
        assert r.context_str == "hello"
