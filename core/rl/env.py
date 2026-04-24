"""
Gymnasium environment for RL-based memory retrieval.

The RL agent learns to select the best subset of candidate memories
to include in the context window.
"""

from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray


@dataclass
class RetrievalResult:
    """Result from the retriever with candidate pool."""
    candidates: list[dict]  # List of candidate dicts with id, tier, domain, text, score, etc.
    selected: list[dict]  # Current naive selection
    context_str: str  # Formatted context for LLM


class RetrievalEnv(gym.Env):
    """
    Gymnasium environment for memory candidate selection.

    State: 485-dim vector
    Action: MultiBinary(25) - which candidates to select

    Reward components:
    - semantic_overlap: cosine sim between query and selected
    - diversity: 1 - avg pairwise similarity among selected
    - token_efficiency: how well we use budget
    - empty_penalty: penalize selecting nothing
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        candidate_pool_size: int = 25,
        top_k: int = 5,
        token_budget: int = 512,
        embedding_dim: int = 384,
    ):
        super().__init__()

        self.candidate_pool_size = candidate_pool_size
        self.top_k = top_k
        self.token_budget = token_budget
        self.embedding_dim = embedding_dim

        # State dimension: query_emb(384) + cosines(25) + weights(25) + recency(25) + tiers(25) + budget(1) = 485
        self.state_dim = (
            embedding_dim
            + candidate_pool_size * 4
            + 1
        )

        # Action space: which candidates to include (binary for each of 25)
        self.action_space = gym.spaces.MultiBinary(candidate_pool_size)

        # Observation space
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32,
        )

        # Internal state
        self._retrieval_result: Optional[RetrievalResult] = None
        self._query_embedding: Optional[NDArray[np.float32]] = None
        self._candidate_embeddings: Optional[NDArray[np.float32]] = None
        self._token_budget: int = token_budget

        self.observation: Optional[NDArray[np.float32]] = None
        self.episode_reward: float = 0.0

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[NDArray[np.float32], dict]:
        """Reset the environment with a new retrieval result."""
        super().reset(seed=seed)

        # Get retrieval result and query embedding from options
        self._retrieval_result = options.get("retrieval_result") if options else None
        self._query_embedding = (
            options.get("query_embedding").astype(np.float32)
            if options and options.get("query_embedding") is not None
            else None
        )
        self._candidate_embeddings = (
            options.get("candidate_embeddings").astype(np.float32)
            if options and options.get("candidate_embeddings") is not None
            else None
        )
        self._token_budget = options.get("token_budget", self.token_budget)

        # Build initial state
        self.observation = self._build_state()
        self.episode_reward = 0.0

        return self.observation, {}

    def _build_state(self) -> NDArray[np.float32]:
        """Build the state vector from retrieval result."""
        candidates = self._retrieval_result.candidates if self._retrieval_result else []
        n_candidates = len(candidates)

        # Pad arrays to candidate_pool_size
        cosines = np.zeros(self.candidate_pool_size, dtype=np.float32)
        weights = np.zeros(self.candidate_pool_size, dtype=np.float32)
        recency = np.zeros(self.candidate_pool_size, dtype=np.float32)
        tiers = np.zeros(self.candidate_pool_size, dtype=np.float32)

        # Fill in candidate data
        for i, cand in enumerate(candidates[: self.candidate_pool_size]):
            cosines[i] = cand.get("score", 0.0)  # Use score as proxy for cosine
            weights[i] = cand.get("hebbian", 0.1)
            # Normalize recency: days since last accessed (max 365 days)
            recency[i] = self._parse_recency(cand.get("last_accessed", ""))
            # Normalize tier: /4
            tiers[i] = cand.get("tier", 3) / 4.0

        # Query embedding
        query_emb = (
            self._query_embedding
            if self._query_embedding is not None
            else np.zeros(self.embedding_dim, dtype=np.float32)
        )

        # Remaining token budget normalized
        remaining = self._token_budget / self.token_budget

        # Concatenate: query_emb + cosines + weights + recency + tiers + remaining
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
        """Parse last_accessed timestamp and return normalized recency (0-1)."""
        if not last_accessed:
            return 0.5  # Default middle value

        try:
            from datetime import datetime, timezone

            # Parse ISO format
            dt = datetime.fromisoformat(last_accessed.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)

            # Days since last access
            days_ago = (datetime.now(timezone.utc) - dt).total_seconds() / 86400
            return min(days_ago / 365.0, 1.0)  # Normalize to 0-1
        except Exception:
            return 0.5

    def step(
        self,
        action: NDArray[np.int32],
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict]:
        """Execute action and return (observation, reward, terminated, truncated, info)."""
        selected_indices = np.where(action == 1)[0]

        # Calculate reward
        reward = self._calculate_reward(selected_indices)
        self.episode_reward += reward

        # Build new state (in this simple env, state doesn't change after action)
        self.observation = self._build_state()

        # Terminal: always single step
        terminated = True
        truncated = False

        info = {
            "selected_count": len(selected_indices),
            "selected_indices": selected_indices.tolist(),
            "reward_components": self._get_reward_components(selected_indices),
        }

        return self.observation, reward, terminated, truncated, info

    def _calculate_reward(self, selected_indices: list[int]) -> float:
        """Calculate the reward for the selected candidates."""
        components = self._get_reward_components(selected_indices)

        return (
            0.5 * components["semantic_overlap"]
            + 0.3 * components["diversity"]
            + 0.2 * components["token_efficiency"]
            - 0.1 * components["empty_penalty"]
        )

    def _get_reward_components(self, selected_indices: list[int]) -> dict[str, float]:
        """Get individual reward components."""
        candidates = self._retrieval_result.candidates if self._retrieval_result else []

        # Empty penalty
        if len(selected_indices) == 0:
            return {
                "semantic_overlap": 0.0,
                "diversity": 0.0,
                "token_efficiency": 0.0,
                "empty_penalty": 1.0,
            }

        # Get texts and embeddings of selected
        selected_texts = [
            candidates[i]["text"]
            for i in selected_indices
            if i < len(candidates)
        ]
        selected_embs = (
            self._candidate_embeddings[selected_indices]
            if self._candidate_embeddings is not None
            else None
        )

        # Semantic overlap: cosine between query and selected
        semantic_overlap = 0.0
        if self._query_embedding is not None and selected_embs is not None:
            # Average cosine between query and each selected
            if len(selected_embs) > 0:
                cosines = self._cosine_similarity(
                    self._query_embedding,
                    selected_embs,
                )
                semantic_overlap = float(np.mean(cosines))

        # Diversity: 1 - avg pairwise similarity
        diversity = 0.0
        if selected_embs is not None and len(selected_embs) > 1:
            # Compute all pairwise similarities
            n = len(selected_embs)
            sims = []
            for i in range(n):
                for j in range(i + 1, n):
                    sim = self._cosine_similarity(
                        selected_embs[i],
                        selected_embs[j],
                    )
                    sims.append(sim)
            if sims:
                avg_sim = float(np.mean(sims))
                diversity = 1.0 - avg_sim  # Higher diversity = higher reward

        # Token efficiency: use budget well
        total_chars = sum(len(t) for t in selected_texts)
        token_efficiency = min(total_chars, self._token_budget) / self._token_budget

        return {
            "semantic_overlap": semantic_overlap,
            "diversity": diversity,
            "token_efficiency": token_efficiency,
            "empty_penalty": 0.0,
        }

    def _cosine_similarity(
        self,
        a: NDArray[np.float32],
        b: NDArray[np.float32],
    ) -> float:
        """Compute cosine similarity between two vectors."""
        if a is None or b is None:
            return 0.0

        a = a.flatten()
        if b.ndim > 1:
            # b is multiple vectors, return array of similarities
            norms_a = np.linalg.norm(a)
            norms_b = np.linalg.norm(b, axis=1)
            dots = np.dot(b, a)
            with np.errstate(divide="ignore", invalid="ignore"):
                sims = dots / (norms_a * norms_b)
                sims = np.where(np.isfinite(sims), sims, 0.0)
            return sims
        else:
            b = b.flatten()
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return float(np.dot(a, b) / (norm_a * norm_b))

    def render(self, mode: str = "human") -> Optional[NDArray]:
        """Render the selected memories."""
        if self._retrieval_result is None:
            return None

        candidates = self._retrieval_result.candidates

        # Get current action (from observation, infer selected)
        # For visualization, show all candidates
        print("\n" + "=" * 60)
        print("Retrieval Environment - Candidate Pool")
        print("=" * 60)

        for i, cand in enumerate(candidates[: self.candidate_pool_size]):
            text = cand.get("text", "")[:60]
            tier = cand.get("tier", 3)
            score = cand.get("score", 0.0)
            print(f"  [{i}] [T{tier}] score={score:.3f} | {text}...")

        print(f"\nToken Budget: {self._token_budget}")
        print(f"Query Embedding Dim: {self.embedding_dim}")

        return None


def cosine_similarity_batch(
    query: NDArray[np.float32],
    candidates: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Compute cosine similarity between query and all candidates."""
    if query is None or candidates is None:
        return np.zeros(len(candidates) if candidates is not None else np.array([]))

    query = query.flatten()
    norms_query = np.linalg.norm(query)
    norms_cand = np.linalg.norm(candidates, axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        sims = np.dot(candidates, query) / (norms_query * norms_cand)
        sims = np.where(np.isfinite(sims), sims, 0.0)

    return sims.astype(np.float32)