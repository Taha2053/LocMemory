"""
RL Agent for intelligent memory selection.

Integrates a trained PPO agent into the retrieval pipeline
for intelligent candidate selection.
"""

from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from core.settings.config import get_config


class RLAgent:
    """
    RL agent for memory candidate selection.

    Loads a trained PPO model and uses it for selection.
    Falls back to hybrid scoring if model unavailable.
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the RL agent.

        Args:
            model_path: Path to trained model. Defaults to config path.
        """
        self._config = get_config()
        self._model_path = model_path or self._config.get(
            "rl", "model_path", "data/rl_agent.zip"
        )

        # Load config values
        self._candidate_pool_size = self._config.get(
            "rl", "candidate_pool_size", 25
        )
        self._top_k = self._config.get("rl", "top_k", 5)
        self._token_budget = self._config.get("rl", "token_budget", 512)
        self._embedding_dim = 384  # all-MiniLM-L6-v2

        self._model = None
        self._available = False
        self._load_model()

    def _load_model(self) -> None:
        """Load the trained PPO model."""
        model_file = Path(self._model_path)

        if not model_file.exists():
            self._available = False
            return

        try:
            from stable_baselines3 import PPO

            self._model = PPO.load(str(model_file))
            self._available = True
        except Exception as e:
            print(f"Warning: Failed to load RL model: {e}")
            self._available = False

    def is_available(self) -> bool:
        """Check if trained model is available."""
        return self._available and self._model is not None

    def select(
        self,
        retrieval_result: "RetrievalResult",
        query_embedding: NDArray[np.float32],
        token_budget: int,
    ) -> list[dict]:
        """
        Select the best candidates using the RL agent.

        Args:
            retrieval_result: The retrieval result with candidate pool
            query_embedding: The embedded query (384-dim)
            token_budget: Token budget for selection

        Returns:
            List of selected candidate dicts
        """
        # Fallback if model not available
        if not self.is_available():
            return self._hybrid_select(retrieval_result, token_budget)

        # Build state for the agent
        state = self._build_state(
            retrieval_result, query_embedding, token_budget
        )

        # Get action from model
        try:
            action, _ = self._model.predict(state, deterministic=True)

            # Convert action to selected indices
            selected_indices = np.where(action == 1)[0]

            # Map back to candidates
            selected = []
            candidates = retrieval_result.candidates
            for idx in selected_indices:
                if idx < len(candidates):
                    selected.append(candidates[idx])

            # Ensure we have results (fallback if empty)
            if not selected:
                return self._hybrid_select(retrieval_result, token_budget)

            return selected

        except Exception as e:
            # Fallback on any error
            return self._hybrid_select(retrieval_result, token_budget)

    def _build_state(
        self,
        retrieval_result: "RetrievalResult",
        query_embedding: NDArray[np.float32],
        token_budget: int,
    ) -> NDArray[np.float32]:
        """Build the state vector for the agent."""
        candidates = retrieval_result.candidates
        n_candidates = len(candidates)

        # Pad arrays
        cosines = np.zeros(self._candidate_pool_size, dtype=np.float32)
        weights = np.zeros(self._candidate_pool_size, dtype=np.float32)
        recency = np.zeros(self._candidate_pool_size, dtype=np.float32)
        tiers = np.zeros(self._candidate_pool_size, dtype=np.float32)

        # Fill candidate data
        for i, cand in enumerate(candidates[: self._candidate_pool_size]):
            cosines[i] = cand.get("score", 0.0)
            weights[i] = cand.get("hebbian", 0.1)
            recency[i] = self._parse_recency(cand.get("last_accessed", ""))
            tiers[i] = cand.get("tier", 3) / 4.0

        # Pad query embedding
        query_emb = query_embedding.astype(np.float32)
        if len(query_emb) < self._embedding_dim:
            query_emb = np.pad(
                query_emb,
                (0, self._embedding_dim - len(query_emb)),
                mode="constant",
            )
        elif len(query_emb) > self._embedding_dim:
            query_emb = query_emb[: self._embedding_dim]

        # Token budget normalized
        remaining = token_budget / self._token_budget

        # Build state
        state = np.concatenate([
            query_emb,
            cosines,
            weights,
            recency,
            tiers,
            [remaining],
        ])

        return state.astype(np.float32)

    def _parse_recency(self, last_accessed: str) -> float:
        """Parse last_accessed timestamp."""
        if not last_accessed:
            return 0.5

        try:
            from datetime import datetime, timezone

            dt = datetime.fromisoformat(last_accessed.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)

            days_ago = (datetime.now(timezone.utc) - dt).total_seconds() / 86400
            return min(days_ago / 365.0, 1.0)
        except Exception:
            return 0.5

    def _hybrid_select(
        self,
        retrieval_result: "RetrievalResult",
        token_budget: int,
    ) -> list[dict]:
        """
        Fallback hybrid selection using bandit-style reward signal.

        Uses multiple signals: semantic score, recency, hebbian weight, domain match.
        """
        candidates = retrieval_result.candidates
        if not candidates:
            return []

        scored_candidates = []
        for cand in candidates:
            base_score = cand.get("score", 0.0)
            
            hebbian_score = cand.get("hebbian", 0.5)
            domain_match = 1.0 if cand.get("domain") else 0.0
            tier_bonus = {1: 1.2, 2: 1.0, 3: 0.8, 4: 0.6}.get(cand.get("tier", 3), 0.5)
            
            reward_signal = (
                0.4 * base_score +
                0.2 * hebbian_score +
                0.2 * domain_match +
                0.2 * tier_bonus
            )
            
            scored_candidates.append((reward_signal, cand))
        
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        selected = []
        total_chars = 0

        for _, cand in scored_candidates:
            text_len = len(cand.get("text", ""))
            if total_chars + text_len <= token_budget:
                selected.append(cand)
                total_chars += text_len

            if len(selected) >= self._top_k:
                break

        return selected

    def update_with_feedback(
        self,
        selected_candidates: list[dict],
        response_quality: float,
    ) -> None:
        """
        Update internal Q-values based on response quality feedback.

        This is a simple bandit-style update for future selections.
        """
        for cand in selected_candidates:
            node_id = cand.get("node_id")
            if node_id:
                key = f"q_{node_id}"
                current_q = getattr(self, key, 0.5)
                new_q = current_q + 0.1 * (response_quality - current_q)
                setattr(self, key, max(0.0, min(1.0, new_q)))

    def get_stats(self) -> dict:
        """Get agent statistics."""
        return {
            "available": self._available,
            "model_path": self._model_path,
            "candidate_pool_size": self._candidate_pool_size,
            "top_k": self._top_k,
            "token_budget": self._token_budget,
        }


# Type alias for RetrievalResult (imported from env.py to avoid circular import)
class RetrievalResult:
    """Result from the retriever with candidate pool."""

    def __init__(
        self,
        candidates: list[dict],
        selected: Optional[list[dict]] = None,
        context_str: str = "",
    ):
        self.candidates = candidates
        self.selected = selected or []
        self.context_str = context_str