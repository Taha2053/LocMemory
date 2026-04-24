"""
Background memory extraction system for AI assistants.

Extracts structured facts from conversation messages and stores them in the graph.
"""

import json
import queue
import threading
import time
from typing import Optional

import requests

from core.memory.graph import GraphManager, TIER_LEAF
from core.memory.classifier import MemoryClassifier
from core.settings.config import get_config


OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "mistral:7b-instruct"


FACT_EXTRACTION_PROMPT = """Extract factual memories from this message.
Return ONLY valid JSON array of objects with 'fact' and 'domain' keys.

Example format:
[
  {"fact": "User started learning Rust", "domain": "programming"},
  {"fact": "User dislikes C++ memory bugs", "domain": "programming"}
]

Rules:
- Only extract STABLE facts (things that persist over time)
- Ignore temporary statements like "I'm hungry now" or "weather is nice today"
- Return 0-5 facts maximum
- Use lowercase for domain names: health, programming, work, personal, finance, learning, engineering
- If no stable facts found, return an empty array: []
- Return ONLY the JSON array, nothing else before or after."""


class MemoryExtractor:
    """Extracts structured facts from conversation messages using LLM."""

    def __init__(
        self,
        graph_manager: GraphManager,
        classifier: Optional[MemoryClassifier] = None,
        ollama_url: str = OLLAMA_URL,
        ollama_model: str = DEFAULT_MODEL,
    ):
        self.graph_manager = graph_manager
        self.classifier = classifier or MemoryClassifier()
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model

        self._config = get_config()
        self._task_queue: queue.Queue = queue.Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False

    def extract_facts(self, text: str) -> list[dict]:
        """
        Call Ollama to extract stable factual memories from the text.

        Returns list of dicts with 'fact' and 'domain' keys.
        """
        prompt = f"{FACT_EXTRACTION_PROMPT}\n\nMessage:\n{text}"

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.1,
                },
                timeout=60,
            )

            if response.status_code != 200:
                print(f"Ollama error: {response.status_code} - {response.text}")
                return []

            raw_response = response.json().get("response", "").strip()
            return self._parse_facts(raw_response)

        except requests.exceptions.ConnectionError:
            print("Error: Cannot connect to Ollama. Is it running?")
            return []
        except requests.exceptions.Timeout:
            print("Error: Ollama request timed out")
            return []
        except Exception as e:
            print(f"Error extracting facts: {e}")
            return []

    def _parse_facts(self, raw_response: str) -> list[dict]:
        """
        Safely parse JSON from LLM response.

        Handles common issues like trailing text, markdown code blocks, etc.
        """
        json_str = raw_response

        if json_str.startswith("```"):
            lines = json_str.split("\n")
            json_str = "\n".join(lines[1:] if lines[0].startswith("```") else lines)
            json_str = json_str.strip("` \n")

        try:
            facts = json.loads(json_str)
            if isinstance(facts, list):
                valid_facts = []
                for item in facts:
                    if isinstance(item, dict) and "fact" in item:
                        fact_text = str(item.get("fact", "")).strip()
                        if len(fact_text) >= 3:
                            valid_facts.append({
                                "fact": fact_text,
                                "domain": item.get("domain", "general").lower(),
                            })
                return valid_facts
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            print(f"Raw response: {raw_response[:200]}")

        return []

    def process_message(self, text: str) -> list[str]:
        """
        Extract facts, classify them, and store in graph as tier 3 nodes.

        Returns list of created node IDs.
        """
        facts = self.extract_facts(text)

        if not facts:
            return []

        node_ids = []
        for fact_data in facts:
            fact_text = fact_data["fact"]
            domain = fact_data.get("domain")
            subdomain = ""

            if not domain or domain == "general":
                classification = self.classifier.classify(fact_text)
                domain = classification.get("domain", "general")
                subdomain = classification.get("subdomain", "")
            else:
                subdomain, _ = self.classifier.detect_subdomain(fact_text, domain)

            try:
                node_id = self.graph_manager.add_node(
                    text=fact_text,
                    tier=TIER_LEAF,
                    domain=domain,
                    subdomain=subdomain,
                    embedding=None,
                )
                node_ids.append(node_id)
                tag = f"{domain}/{subdomain}" if subdomain else domain
                print(f"  + Stored: {fact_text[:50]}... [{tag}]")
            except Exception as e:
                print(f"Failed to store fact: {e}")

        return node_ids

    def start_background_extraction(self, text: str):
        """
        Launch process_message in a background thread.

        Does not block the main chat loop.
        """
        if not self._running:
            self._start_worker()

        self._task_queue.put(text)
        print(f"[Background] Queued extraction task for: {text[:50]}...")

    def _start_worker(self):
        """Start the background worker thread."""
        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="MemoryExtractor",
        )
        self._worker_thread.start()
        print("[Background] Extractor worker started")

    def _worker_loop(self):
        """Background worker that processes extraction tasks."""
        while self._running:
            try:
                text = self._task_queue.get(timeout=1.0)
                if text is None:
                    break

                try:
                    node_ids = self.process_message(text)
                    if node_ids:
                        print(f"[Background] Extracted {len(node_ids)} facts")
                except Exception as e:
                    print(f"[Background] Extraction error: {e}")

                self._task_queue.task_done()

            except queue.Empty:
                continue

    def stop(self, drain_timeout: float = 120.0):
        """
        Stop the background worker, waiting for pending extractions to finish
        so facts are actually persisted before shutdown.
        """
        if not self._running:
            return

        pending = self._task_queue.qsize()
        if pending:
            print(f"[Background] Draining {pending} pending extraction(s) "
                  f"(up to {drain_timeout:.0f}s)...")

        try:
            deadline = time.time() + drain_timeout
            while not self._task_queue.empty() and time.time() < deadline:
                time.sleep(0.2)
        except Exception:
            pass

        self._running = False
        self._task_queue.put(None)
        if self._worker_thread:
            self._worker_thread.join(timeout=drain_timeout)
        print("[Background] Extractor worker stopped")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


if __name__ == "__main__":
    from core.memory.graph import GraphManager

    with GraphManager("data/test_extract.db") as gm:
        extractor = MemoryExtractor(gm)

        test_messages = [
            "I've been working on a new Python project using FastAPI and it's really great for building APIs quickly. I deployed it on Render yesterday.",
            "Just finished reading a book about reinforcement learning. The chapter on PPO was confusing but I think I understand it now.",
            "My knee has been hurting for a week. I should probably see a doctor about it.",
            "I'm thinking about learning Rust next. Heard it's great for performance-critical code.",
            "Had lunch with Sarah today. She's moving to Berlin next month for a new job.",
        ]

        print("=" * 60)
        for msg in test_messages:
            print(f"\nProcessing: {msg[:60]}...")
            node_ids = extractor.process_message(msg)
            print(f"  Created {len(node_ids)} memory nodes")

        print("\n" + "=" * 60)
        print("Background extraction test:")
        extractor.start_background_extraction("Testing background extraction with a random thought about coffee preferences.")
        extractor.start_background_extraction("Another background task for hiking this weekend.")

        import time
        time.sleep(3)
        extractor.stop()

        print("\nFinal graph stats:")
        print(f"  Nodes: {gm.graph.number_of_nodes()}")
        print(f"  Edges: {gm.graph.number_of_edges()}")