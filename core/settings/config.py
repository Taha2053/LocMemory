"""
Configuration management for LocMemory cognitive memory system.
"""

import os
from pathlib import Path
from typing import Any, Optional

import yaml


class Config:
    """Centralized configuration management."""

    _instance: Optional["Config"] = None

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self._data: dict[str, Any] = {}
        self._load()

    def _load(self):
        if self.config_path.exists():
            with open(self.config_path, "r") as f:
                self._data = yaml.safe_load(f) or {}
        else:
            self._data = self._default_config()
        self._resolve_env_vars()

    def _default_config(self) -> dict[str, Any]:
        return {
            "system": {
                "name": "LocMemory",
                "version": "1.0",
                "environment": "development",
            },
            "models": {
                "embedding": {
                    "provider": "sentence-transformers",
                    "name": "all-MiniLM-L6-v2",
                    "device": "cpu",
                },
                "llm": {
                    "provider": "ollama",
                    "model": "mistral",
                    "temperature": 0.3,
                    "max_tokens": 512,
                },
                "extractor_model": {
                    "provider": "ollama",
                    "model": "mistral",
                    "temperature": 0.1,
                },
                "summarizer_model": {
                    "provider": "ollama",
                    "model": "mistral",
                    "temperature": 0.2,
                },
            },
            "storage": {
                "sqlite_db_path": "data/memory.db",
                "auto_save": True,
                "backup_enabled": True,
                "backup_interval_minutes": 60,
            },
            "graph": {
                "max_nodes": 10000,
                "default_edge_weight": 0.1,
                "cross_domain_weight_threshold": 0.5,
                "traversal_depth": 2,
                "max_candidates": 20,
                "tiers": {
                    "context": 1,
                    "anchor": 2,
                    "leaf": 3,
                    "procedural": 4,
                },
            },
            "retrieval": {
                "semantic_weight": 0.7,
                "graph_weight": 0.3,
                "max_results": 20,
                "min_similarity": 0.35,
                "traversal_depth": 2,
            },
            "classification": {
                "domains": [
                    "health",
                    "engineering",
                    "programming",
                    "work",
                    "personal",
                    "finance",
                    "learning",
                ],
                "similarity_threshold": 0.45,
                "enable_llm_fallback": True,
            },
            "extraction": {
                "enable_background_extraction": True,
                "thread_pool_size": 2,
                "min_fact_length": 10,
                "max_facts_per_message": 5,
            },
            "hebbian": {
                "enabled": True,
                "decay_lambda": 0.01,
                "learning_rate": 0.2,
                "max_edge_weight": 5.0,
                "min_edge_weight": 0.01,
            },
            "consolidation": {
                "enabled": True,
                "cluster_min_size": 10,
                "run_every_n_additions": 30,
                "max_clusters_per_run": 5,
            },
            "procedural": {
                "enabled": True,
                "run_every_n_interactions": 50,
                "cross_domain_threshold": 0.6,
                "min_pattern_support": 3,
            },
            "rl": {
                "enabled": True,
                "model_path": "data/rl_agent.zip",
                "training_timesteps": 10000,
                "candidate_pool_size": 25,
                "top_k": 5,
                "token_budget": 512,
            },
            "security": {
                "pii_detection": True,
                "encryption_enabled": True,
                "encryption_key_path": "secrets/secret.key",
            },
            "logging": {
                "level": "INFO",
                "log_file": "logs/system.log",
                "retrieval_metrics": True,
                "latency_tracking": True,
            },
            "performance": {
                "max_retrieval_latency_ms": 100,
                "enable_caching": True,
                "embedding_batch_size": 16,
            },
            "debug": {
                "enable_trace_logs": False,
                "save_intermediate_results": False,
            },
        }

    def _resolve_env_vars(self):
        for section in self._data:
            for key, value in self._data[section].items():
                if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                    env_var = value[2:-1]
                    self._data[section][key] = os.environ.get(env_var, value)

    @classmethod
    def get_instance(cls, config_path: str = "config.yaml") -> "Config":
        if cls._instance is None:
            cls._instance = cls(config_path)
        return cls._instance

    @classmethod
    def reset(cls):
        cls._instance = None

    def get(self, section: str, key: str, default: Any = None) -> Any:
        return self._data.get(section, {}).get(key, default)

    def get_section(self, section: str) -> dict[str, Any]:
        return self._data.get(section, {})

    @property
    def system(self) -> dict[str, Any]:
        return self._data.get("system", {})

    @property
    def embedding_model(self) -> dict[str, Any]:
        return self._data.get("models", {}).get("embedding", {})

    @property
    def llm(self) -> dict[str, Any]:
        return self._data.get("models", {}).get("llm", {})

    @property
    def storage(self) -> dict[str, Any]:
        return self._data.get("storage", {})

    @property
    def graph(self) -> dict[str, Any]:
        return self._data.get("graph", {})

    @property
    def retrieval(self) -> dict[str, Any]:
        return self._data.get("retrieval", {})

    @property
    def classification(self) -> dict[str, Any]:
        return self._data.get("classification", {})

    @property
    def hebbian(self) -> dict[str, Any]:
        return self._data.get("hebbian", {})

    @property
    def consolidation(self) -> dict[str, Any]:
        return self._data.get("consolidation", {})

    @property
    def procedural(self) -> dict[str, Any]:
        return self._data.get("procedural", {})

    @property
    def rl(self) -> dict[str, Any]:
        return self._data.get("rl", {})

    @property
    def security(self) -> dict[str, Any]:
        return self._data.get("security", {})

    @property
    def logging(self) -> dict[str, Any]:
        return self._data.get("logging", {})

    @property
    def performance(self) -> dict[str, Any]:
        return self._data.get("performance", {})

    @property
    def debug(self) -> dict[str, Any]:
        return self._data.get("debug", {})

    def as_dict(self) -> dict[str, Any]:
        return self._data

    def update(self, new_data: dict[str, Any]):
        """Replace the in-memory config with new_data (deep merged by top-level section)."""
        for section, values in new_data.items():
            if isinstance(values, dict) and isinstance(self._data.get(section), dict):
                self._data[section].update(values)
            else:
                self._data[section] = values

    def save(self, path: Optional[str] = None):
        path = Path(path) if path else self.config_path
        with open(path, "w") as f:
            yaml.dump(self._data, f, default_flow_style=False, sort_keys=False)


def get_config(config_path: str = "config.yaml") -> Config:
    return Config.get_instance(config_path)


if __name__ == "__main__":
    config = Config()

    print(f"System: {config.system['name']} v{config.system['version']}")
    print(f"Embedding: {config.embedding_model['name']} on {config.embedding_model['device']}")
    print(f"LLM: {config.llm['model']} (temp={config.llm['temperature']})")
    print(f"Graph tiers: {config.graph['tiers']}")
    print(f"Hebbian learning: {config.hebbian['enabled']}")
    print(f"Domains: {config.classification['domains']}")