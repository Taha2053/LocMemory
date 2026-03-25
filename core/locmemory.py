"""
LocMemory CLI - RAG with local Ollama
Usage: python locmemory.py "your question here"
"""

import sys
import json
import requests
from pathlib import Path
from core.memory import MemoryStore

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "mistral:7b-instruct"

store = MemoryStore(db_path="data/memories.db", md_dir="memories")


def search_memories(query: str, top_k: int = 3) -> list[str]:
    results = store.search(query, top_k=top_k)
    return [f"- {mem.text}" for mem, score in results]


def generate_with_context(prompt: str, context: str, model: str = DEFAULT_MODEL) -> str:
    full_prompt = f"""Based on the following context from my memory:

{context}

Question: {prompt}

Answer:"""

    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": model, "prompt": full_prompt, "stream": False},
        timeout=120,
    )

    if response.status_code != 200:
        return f"Error: {response.text}"

    return response.json().get("response", "No response")


def main():
    if len(sys.argv) < 2:
        print('Usage: python locmemory.py "your question here"')
        print('Example: python locmemory.py "What did I do last week?"')
        sys.exit(1)

    query = " ".join(sys.argv[1:])

    print(f'\nSearching memories for: "{query}"\n')

    memories = search_memories(query, top_k=3)

    if not memories:
        print("No memories found. Add some memories first with store.add()")
        context = "No relevant memories found."
    else:
        context = "\n".join(memories)
        print(f"Found {len(memories)} relevant memories:\n")
        for m in memories:
            print(m)
        print()

    print("Asking Ollama...\n")

    answer = generate_with_context(query, context)
    print(f"Answer:\n{answer}")


if __name__ == "__main__":
    main()
