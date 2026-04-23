"""
test_integration.py
-------------------
Integration test for LocMemory core modules.
Tests memory.py, context.py, and llm.py working together
in the exact same sequence that chat.py will use.

Run with:
    uv run python test_integration.py
"""

import sys
import os

# ---------------------------------------------------------------------------
# PATH SETUP
# Adjust this if your modules live in a subfolder (e.g. core/)
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))

from core.memory import MemoryStore
from core.context import pack_context, build_prompt, count_tokens
from core.llm import load_config, call_llm, is_model_available


# ---------------------------------------------------------------------------
# ANSI colors — no external dependency, just makes output readable
# ---------------------------------------------------------------------------
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"

def ok(msg):   print(f"  {GREEN}✓{RESET} {msg}")
def fail(msg): print(f"  {RED}✗{RESET} {msg}")
def info(msg): print(f"  {CYAN}→{RESET} {msg}")
def section(title): print(f"\n{YELLOW}{'─'*50}{RESET}\n{YELLOW}{title}{RESET}")


# ---------------------------------------------------------------------------
# TEST 1 — MemoryStore initializes correctly
# ---------------------------------------------------------------------------
section("TEST 1 · MemoryStore init")

try:
    # Use a temporary test database so we don't pollute the real one
    store = MemoryStore(db_path="test_integration.db")
    ok("MemoryStore created with test_integration.db")
except Exception as e:
    fail(f"MemoryStore init failed: {e}")
    sys.exit(1)  # No point continuing if store fails


# ---------------------------------------------------------------------------
# TEST 2 — Adding memories
# ---------------------------------------------------------------------------
section("TEST 2 · Adding memories")

test_memories = [
    ("My name is Eya and I study Mathematical Engineering.", "personal"),
    ("LocMemory is a local private memory system for LLMs.", "project"),
    ("The token budget for context packing is 500 tokens.",  "config"),
    ("Taha is responsible for the Data and Intelligence part.", "project"),
    ("We are using mistral:7b-instruct as the LLM model.",  "config"),
]

for text, category in test_memories:
    try:
        memory = store.add(text, category=category)
        ok(f"Added [{category}]: '{text[:50]}...' " if len(text) > 50 else f"Added [{category}]: '{text}'")
    except Exception as e:
        fail(f"Failed to add memory: {e}")

count = store.count()
info(f"Total memories in DB: {count}")
assert count == len(test_memories), f"Expected {len(test_memories)} memories, got {count}"
ok("Memory count matches")


# ---------------------------------------------------------------------------
# TEST 3 — Searching memories (semantic retrieval)
# ---------------------------------------------------------------------------
section("TEST 3 · Semantic search")

query = "Who is working on the data part of the project?"
info(f"Query: '{query}'")

try:
    results = store.search(query, top_k=3)
    ok(f"Search returned {len(results)} results")

    for memory, score in results:
        info(f"  score={score:.4f} | {memory.text[:70]}")

    # The result about Taha should be in the top results
    top_texts = [m.text for m, _ in results]
    assert any("Taha" in t for t in top_texts), "Expected Taha memory in top results"
    ok("Relevant memory ranked in top results")

except Exception as e:
    fail(f"Search failed: {e}")


# ---------------------------------------------------------------------------
# TEST 4 — Context packing (token budget enforcement)
# ---------------------------------------------------------------------------
section("TEST 4 · Context packing")

TOKEN_BUDGET = 500
info(f"Token budget: {TOKEN_BUDGET}")

try:
    # pack_context takes the search results and trims to fit the budget
    packed = pack_context(results, token_budget=TOKEN_BUDGET)
    ok(f"pack_context returned {len(packed)} memories (from {len(results)} candidates)")

    # Verify the packed memories respect the budget
    total_tokens = sum(count_tokens(m.text) for m, _ in packed)
    info(f"Total tokens used by packed memories: {total_tokens}")
    assert total_tokens <= TOKEN_BUDGET, f"Token budget exceeded: {total_tokens} > {TOKEN_BUDGET}"
    ok("Token budget respected")

except Exception as e:
    fail(f"pack_context failed: {e}")


# ---------------------------------------------------------------------------
# TEST 5 — Prompt building
# ---------------------------------------------------------------------------
section("TEST 5 · Prompt building")

try:
    prompt = build_prompt(query=query, packed_memories=packed)
    ok("build_prompt succeeded")

    token_count = count_tokens(prompt)
    info(f"Final prompt token count: {token_count}")
    info(f"Prompt preview:\n{'─'*40}\n{prompt[:300]}...\n{'─'*40}")

except Exception as e:
    fail(f"build_prompt failed: {e}")


# ---------------------------------------------------------------------------
# TEST 6 — LLM config loading
# ---------------------------------------------------------------------------
section("TEST 6 · LLM config")

try:
    config = load_config()
    print("RAW CONFIG:", config)
    ok("config.yaml loaded successfully")
    info(f"Model from config: {config.get('LLM_MODEL', 'NOT FOUND')}")
except Exception as e:
    fail(f"load_config failed: {e}")


# ---------------------------------------------------------------------------
# TEST 7 — Model availability check
# ---------------------------------------------------------------------------
section("TEST 7 · Model availability (Ollama)")

model = config.get("LLM_MODEL", "mistral:7b-instruct-v0.3-q4_0")

try:
    available = is_model_available(model)
    if available:
        ok(f"Model '{model}' is available in Ollama")
    else:
        fail(f"Model '{model}' NOT found in Ollama — is Ollama running?")
        sys.exit(1)  # No point calling LLM if model isn't available
except Exception as e:
    fail(f"is_model_available failed: {e}")


# ---------------------------------------------------------------------------
# TEST 8 — Full pipeline: memory → pack → prompt → LLM
# This is the exact sequence chat.py will run every turn
# ---------------------------------------------------------------------------
section("TEST 8 · FULL PIPELINE (the real test)")

info("Simulating one full chat exchange...")
info(f"User query: '{query}'")

try:
    # Step 1: retrieve relevant memories
    candidates = store.search(query, top_k=5)
    ok(f"Step 1 — Retrieved {len(candidates)} memory candidates")

    # Step 2: pack within token budget
    packed = pack_context(candidates, token_budget=TOKEN_BUDGET)
    ok(f"Step 2 — Packed to {len(packed)} memories within budget")

    # Step 3: build the final prompt
    final_prompt = build_prompt(query=query, packed_memories=packed)
    ok(f"Step 3 — Prompt built ({count_tokens(final_prompt)} tokens)")

    # Step 4: call the LLM
    info("Step 4 — Calling LLM (this may take a few seconds)...")
    response = call_llm(prompt=final_prompt, model=model)
    ok("Step 4 — LLM responded successfully")

    # Step 5: display the response
    info(f"LLM response:\n{'─'*40}\n{response.text}\n{'─'*40}")
    info(f"Tokens used — input: {response.input_tokens} | output: {response.output_tokens} | total: {response.total_tokens}")

    # Step 6: save the exchange as a new memory (what chat.py will do after every turn)
    exchange_text = f"User asked: {query} | Assistant replied: {response.text[:100]}"
    store.add(exchange_text, category="exchange")
    ok("Step 6 — Exchange saved to memory DB")

except Exception as e:
    fail(f"Full pipeline failed: {e}")
    import traceback
    traceback.print_exc()


# ---------------------------------------------------------------------------
# CLEANUP — remove the test database
# ---------------------------------------------------------------------------
section("CLEANUP")

try:
    store.close()
    ok("DB connection closed")
    os.remove("test_integration.db")
    ok("test_integration.db removed")
except Exception as e:
    info(f"Could not remove test DB: {e}")


# ---------------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------------
print(f"\n{YELLOW}{'─'*50}{RESET}")
print(f"{GREEN}Integration test complete.{RESET}")
print(f"If all steps showed ✓, chat.py is safe to build.\n")