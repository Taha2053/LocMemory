"""
Training script for RL memory selection agent.

Generates synthetic episodes using the existing graph store
and trains a PPO agent for intelligent candidate selection.
"""

import csv
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from core.memory.graph import GraphManager
from core.memory.classifier import MemoryClassifier
from core.memory.retriever import GraphRetriever
from core.settings.config import get_config


# Training queries covering all domains
TRAINING_QUERIES = [
    "how is my health lately",
    "what am I working on",
    "what did I learn recently",
    "how is my project going",
    "what are my habits",
    "how is my finances",
    "what are my goals",
    "tell me about my family",
    "what should I focus on today",
    "how have I been feeling",
]

# Domain keywords for query generation
DOMAIN_KEYWORDS = {
    "health": ["health", "fitness", "exercise", "gym", "workout", "diet", "sleep"],
    "engineering": ["engine", "design", "build", "system", "architecture", "infrastructure"],
    "programming": ["code", "program", "python", "software", "debug", "algorithm"],
    "work": ["work", "job", "meeting", "deadline", "project", "task", "office"],
    "personal": ["family", "friend", "home", "hobby", "weekend", "vacation"],
    "finance": ["money", "invest", "budget", "savings", "stock", " expense"],
    "learning": ["learn", "study", "course", "book", "research", "practice"],
    "social": ["social", "friend", "party", "event", "community"],
}


class RetrievalTrainer:
    """Trainer for the retrieval RL agent."""

    def __init__(self):
        self._config = get_config()
        self._graph_manager = None
        self._retriever = None
        self._classifier = None
        self._embedding_model = None

        # Config values
        self._candidate_pool_size = self._config.get(
            "rl", "candidate_pool_size", 25
        )
        self._top_k = self._config.get("rl", "top_k", 5)
        self._token_budget = self._config.get("rl", "token_budget", 512)
        self._training_steps = self._config.get(
            "rl", "training_timesteps", 10000
        )
        self._embedding_dim = 384

        # CSV logging
        self._log_path = Path("data/rl_training_log.csv")
        self._curve_path = Path("data/rl_curve.png")

        self._setup()

    def _setup(self) -> None:
        """Initialize the graph and retriever."""
        print("Setting up graph and retriever...")

        # Initialize graph manager
        db_path = self._config.get(
            "storage", "sqlite_db_path", "data/memory.db"
        )
        self._graph_manager = GraphManager(db_path)

        # Initialize classifier (provides embeddings)
        self._classifier = MemoryClassifier()
        self._embedding_model = self._classifier._model

        # Initialize retriever
        self._retriever = GraphRetriever(
            self._graph_manager,
            self._classifier,
            max_candidates=self._candidate_pool_size,
        )

    def _generate_variations(self, base_query: str) -> list[str]:
        """Generate query variations from a base query."""
        words = base_query.lower().split()

        variations = [base_query]

        # Add some variations with different patterns
        patterns = [
            lambda: "tell me about " + " ".join(words[1:]),
            lambda: "what " + " ".join(words[1:]) if len(words) > 1 else base_query,
            lambda: base_query + "?",
            lambda: base_query + " recently",
        ]

        for _ in range(5):
            pattern = random.choice(patterns)()
            if pattern and pattern != base_query:
                variations.append(pattern)

        return variations

    def _get_query_embedding(self, query: str) -> np.ndarray:
        """Get embedding for a query."""
        emb = self._embedding_model.encode([query], convert_to_numpy=True)[0]
        return emb.astype(np.float32)

    def _create_episode_data(self) -> list[dict]:
        """Create a batch of training episodes."""
        episodes = []

        # Select random queries
        queries = random.choices(TRAINING_QUERIES, k=5)

        for query in queries:
            # Get retrieval result
            try:
                results = self._retriever.retrieve(query)
            except Exception:
                results = []

            if not results:
                # Add dummy data if no results
                results = [
                    {
                        "node_id": f"dummy_{i}",
                        "text": f"Dummy memory {i} about {query}",
                        "domain": "personal",
                        "tier": 3,
                        "score": 0.5,
                        "hebbian": 0.1,
                        "last_accessed": datetime.now(timezone.utc).isoformat(),
                    }
                    for i in range(min(self._candidate_pool_size, 10))
                ]

            # Pad to candidate_pool_size
            while len(results) < self._candidate_pool_size:
                idx = len(results)
                results.append({
                    "node_id": f"pad_{idx}",
                    "text": "",
                    "domain": "",
                    "tier": 3,
                    "score": 0.0,
                    "hebbian": 0.0,
                    "last_accessed": "",
                })

            # Get query embedding
            query_emb = self._get_query_embedding(query)

            # Get candidate embeddings
            candidate_texts = [r.get("text", "") for r in results]
            candidate_embs = self._embedding_model.encode(
                candidate_texts,
                convert_to_numpy=True,
            ).astype(np.float32)

            episodes.append({
                "query": query,
                "query_emb": query_emb,
                "candidates": results,
                "candidate_embs": candidate_embs,
                "token_budget": self._token_budget,
            })

        return episodes

    def _build_env_fn(self):
        """Build the environment for training."""
        from core.rl.env import RetrievalEnv

        def make_env():
            # Create episodes
            episodes = self._create_episode_data()

            class EpisodicEnv(RetrievalEnv):
                def __init__(self):
                    super().__init__(
                        candidate_pool_size=self._candidate_pool_size,
                        top_k=self._top_k,
                        token_budget=self._token_budget,
                        embedding_dim=self._embedding_dim,
                    )

                def reset(self, seed=None, options=None):
                    # Use random episode
                    ep = random.choice(episodes) if episodes else None
                    if ep:
                        opts = {
                            "retrieval_result": type("obj", (object,), {
                                "candidates": ep["candidates"],
                            })(),
                            "query_embedding": ep["query_emb"],
                            "candidate_embeddings": ep["candidate_embs"],
                            "token_budget": ep["token_budget"],
                        }
                        return super().reset(seed=seed, options=opts)
                    return super().reset(seed=seed, options=options)

            return make_env() if episodes else None

    def train(self) -> None:
        """Train the RL agent."""
        print("\n" + "=" * 60)
        print("Training RL Agent")
        print("=" * 60)

        # Create environment
        from core.rl.env import RetrievalEnv

        env = RetrievalEnv(
            candidate_pool_size=self._candidate_pool_size,
            top_k=self._top_k,
            token_budget=self._token_budget,
            embedding_dim=self._embedding_dim,
        )

        # Wrap in vectorized env for SB3
        env = make_vec_env(lambda: env, n_envs=1)

        # Create PPO agent
        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            tensorboard_log="data/tensorboard",
        )

        # Create log file
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(self._log_path, "w", newline="")
        csv_writer = csv.writer(log_file)
        csv_writer.writerow(["timestep", "mean_reward", "ep_length"])

        # Training loop
        print(f"Training for {self._training_steps} timesteps...")
        print(f"Logging to {self._log_path}")

        rewards_history = []
        start_time = time.time()

        try:
            for step in range(0, self._training_steps, 1000):
                # Generate new episodes for each batch
                episodes = self._create_episode_data()

                # Create fresh env with new episodes
                ep_env = RetrievalEnv(
                    candidate_pool_size=self._candidate_pool_size,
                    top_k=self._top_k,
                    token_budget=self._token_budget,
                    embedding_dim=self._embedding_dim,
                )

                # Reset with random episode
                ep = random.choice(episodes)
                obs, _ = ep_env.reset(options={
                    "retrieval_result": type("obj", (object,), {
                        "candidates": ep["candidates"],
                    })(),
                    "query_embedding": ep["query_emb"],
                    "candidate_embeddings": ep["candidate_embs"],
                    "token_budget": ep["token_budget"],
                })

                # Run a few steps
                episode_rewards = []
                for _ in range(min(100, self._training_steps - step)):
                    # Random action (exploration)
                    action = env.action_space.sample()
                    obs, reward, terminated, truncated, info = env.step(action)

                    episode_rewards.append(reward)

                    if terminated or truncated:
                        # Reset with new episode
                        ep = random.choice(episodes)
                        obs, _ = ep_env.reset(options={
                            "retrieval_result": type("obj", (object,), {
                                "candidates": ep["candidates"],
                            })(),
                            "query_embedding": ep["query_emb"],
                            "candidate_embeddings": ep["candidate_embs"],
                            "token_budget": ep["token_budget"],
                        })

                # Actually train with PPO
                model.learn(
                    total_timesteps=1000,
                    progress_bar=False,
                    reset_num_timesteps=False,
                )

                # Calculate mean reward
                mean_reward = np.mean(episode_rewards) if episode_rewards else 0.0
                rewards_history.append(mean_reward)

                # Log to CSV
                csv_writer.writerow([
                    step + 1000,
                    mean_reward,
                    len(episode_rewards),
                ])
                log_file.flush()

                # Print progress
                elapsed = time.time() - start_time
                print(f"  Step {step + 1000}/{self._training_steps} | "
                      f"Mean reward: {mean_reward:.4f} | "
                      f"Elapsed: {elapsed:.1f}s")

        except KeyboardInterrupt:
            print("\nTraining interrupted by user")

        finally:
            log_file.close()

        # Save model
        model_path = self._config.get("rl", "model_path", "data/rl_agent.zip")
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        model.save(model_path)
        print(f"\nModel saved to {model_path}")

        # Plot learning curve
        self._plot_curve(rewards_history)

        print("\nTraining complete!")

    def _plot_curve(self, rewards: list[float]) -> None:
        """Plot and save the learning curve."""
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 6))
            plt.plot(rewards)
            plt.xlabel("Training Step (x1000)")
            plt.ylabel("Mean Reward")
            plt.title("RL Training Learning Curve")
            plt.grid(True)
            plt.savefig(self._curve_path)
            plt.close()

            print(f"Learning curve saved to {self._curve_path}")
        except ImportError:
            print("Warning: matplotlib not available for plotting")


def main():
    """Main entry point."""
    print("=" * 60)
    print("RL Agent Training")
    print("=" * 60)

    trainer = RetrievalTrainer()
    trainer.train()


if __name__ == "__main__":
    main()