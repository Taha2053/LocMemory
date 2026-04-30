"""
Training script for RL memory selection agent.

Generates synthetic episodes and trains a PPO agent for intelligent
candidate selection.  Uses a self-contained environment so that
Stable-Baselines3's learn() loop works correctly without manual stepping.
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
from core.rl.env import RetrievalEnv, RetrievalResult
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

SAMPLE_MEMORIES = [
    "I started exercising three times a week",
    "Working on a Python machine learning project",
    "Reading a book about neural networks",
    "Had a meeting with the team about the deadline",
    "Saving 20% of my monthly income",
    "My cousin is visiting next weekend",
    "Finished the online course on deep learning",
    "Bought a new GPU for training models",
    "Went for a run this morning",
    "The budget review is scheduled for Friday",
    "Practicing piano every evening",
    "Deployed the new API endpoint to production",
    "Studying for the certification exam",
    "Investing in index funds for retirement",
    "Family dinner planned for Sunday",
    "Fixed a critical bug in the auth module",
    "Learning Spanish with an app",
    "Completed the marathon training plan",
    "Refactored the database layer",
    "Reading research papers on transformers",
]

DOMAINS = ["health", "engineering", "programming", "work", "personal", "finance", "learning"]


class SyntheticRetrievalEnv(RetrievalEnv):
    """Self-contained RetrievalEnv that generates synthetic episodes on reset.

    Works with SB3's learn() without needing external episode injection.
    """

    def __init__(
        self,
        candidate_pool_size: int = 25,
        top_k: int = 5,
        token_budget: int = 512,
        embedding_dim: int = 384,
        embedding_model=None,
    ):
        super().__init__(
            candidate_pool_size=candidate_pool_size,
            top_k=top_k,
            token_budget=token_budget,
            embedding_dim=embedding_dim,
        )
        self._embedding_model = embedding_model

    def _generate_synthetic_episode(self) -> dict:
        """Create a random training episode."""
        query = random.choice(TRAINING_QUERIES)

        # Pick a random subset of sample memories as candidates
        n_candidates = random.randint(5, self.candidate_pool_size)
        chosen = random.sample(SAMPLE_MEMORIES, min(n_candidates, len(SAMPLE_MEMORIES)))

        candidates = []
        for i, text in enumerate(chosen):
            candidates.append({
                "node_id": f"syn_{i}",
                "text": text,
                "domain": random.choice(DOMAINS),
                "tier": random.choice([1, 2, 3, 3, 3]),
                "score": round(random.uniform(0.1, 0.95), 3),
                "hebbian": round(random.uniform(0.05, 0.5), 3),
                "last_accessed": datetime.now(timezone.utc).isoformat(),
            })

        # Pad to candidate_pool_size
        while len(candidates) < self.candidate_pool_size:
            idx = len(candidates)
            candidates.append({
                "node_id": f"pad_{idx}",
                "text": "",
                "domain": "",
                "tier": 3,
                "score": 0.0,
                "hebbian": 0.0,
                "last_accessed": "",
            })

        # Generate embeddings
        if self._embedding_model is not None:
            query_emb = self._embedding_model.encode(
                [query], convert_to_numpy=True
            )[0].astype(np.float32)
            candidate_texts = [c["text"] for c in candidates]
            candidate_embs = self._embedding_model.encode(
                candidate_texts, convert_to_numpy=True
            ).astype(np.float32)
        else:
            # Fallback: random embeddings (still trains, just less meaningful)
            query_emb = np.random.randn(self.embedding_dim).astype(np.float32)
            candidate_embs = np.random.randn(
                self.candidate_pool_size, self.embedding_dim
            ).astype(np.float32)

        return {
            "query": query,
            "query_emb": query_emb,
            "candidates": candidates,
            "candidate_embs": candidate_embs,
            "token_budget": self.token_budget,
        }

    def reset(self, seed=None, options=None):
        """Reset with a fresh synthetic episode."""
        super().reset(seed=seed)

        episode = self._generate_synthetic_episode()

        retrieval_result = RetrievalResult(
            candidates=episode["candidates"],
            selected=[],
            context_str="",
        )

        opts = {
            "retrieval_result": retrieval_result,
            "query_embedding": episode["query_emb"],
            "candidate_embeddings": episode["candidate_embs"],
            "token_budget": episode["token_budget"],
        }

        self.observation = self._build_state()
        self.episode_reward = 0.0

        return self.observation, {}


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
        """Initialize the graph, classifier, and embedding model."""
        print("Setting up embedding model...")

        # Initialize classifier (provides embeddings)
        self._classifier = MemoryClassifier()
        self._embedding_model = self._classifier._model

        # Optionally populate graph for richer episodes
        db_path = self._config.get(
            "storage", "sqlite_db_path", "data/memory.db"
        )
        self._graph_manager = GraphManager(db_path)
        try:
            self._graph_manager.initialize_db()
            self._graph_manager.load_graph()
            if self._graph_manager.graph.number_of_nodes() > 0:
                self._retriever = GraphRetriever(
                    self._graph_manager,
                    self._classifier,
                    max_candidates=self._candidate_pool_size,
                )
                print(f"  Graph loaded: {self._graph_manager.graph.number_of_nodes()} nodes")
            else:
                print("  Graph is empty, will use synthetic episodes only")
        except Exception as e:
            print(f"  Could not load graph ({e}), using synthetic episodes only")

    def train(self) -> None:
        """Train the RL agent using synthetic episodes."""
        print("\n" + "=" * 60)
        print("Training RL Agent")
        print("=" * 60)

        # Create self-contained synthetic environment
        def make_env():
            return SyntheticRetrievalEnv(
                candidate_pool_size=self._candidate_pool_size,
                top_k=self._top_k,
                token_budget=self._token_budget,
                embedding_dim=self._embedding_dim,
                embedding_model=self._embedding_model,
            )

        env = make_vec_env(make_env, n_envs=1)

        # Create PPO agent
        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            learning_rate=3e-4,
            n_steps=512,
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

        # Training loop — just call learn() in chunks so we can log progress
        print(f"Training for {self._training_steps} timesteps...")
        print(f"Logging to {self._log_path}")

        rewards_history = []
        start_time = time.time()
        chunk = 1000

        try:
            for step in range(0, self._training_steps, chunk):
                model.learn(
                    total_timesteps=chunk,
                    progress_bar=False,
                    reset_num_timesteps=False,
                )

                # Evaluate: run a few episodes to get mean reward
                eval_rewards = []
                eval_env = make_env()
                for _ in range(10):
                    obs, _ = eval_env.reset()
                    total_r = 0.0
                    done = False
                    while not done:
                        action, _ = model.predict(obs, deterministic=True)
                        obs, reward, terminated, truncated, _ = eval_env.step(action)
                        total_r += reward
                        done = terminated or truncated
                    eval_rewards.append(total_r)

                mean_reward = float(np.mean(eval_rewards))
                rewards_history.append(mean_reward)

                # Log to CSV
                csv_writer.writerow([
                    step + chunk,
                    f"{mean_reward:.4f}",
                    1,  # single-step episodes
                ])
                log_file.flush()

                elapsed = time.time() - start_time
                print(f"  Step {step + chunk}/{self._training_steps} | "
                      f"Mean reward: {mean_reward:.4f} | "
                      f"Elapsed: {elapsed:.1f}s")

        except KeyboardInterrupt:
            print("\nTraining interrupted by user")

        finally:
            log_file.close()
            env.close()

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
            import matplotlib
            matplotlib.use("Agg")
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
