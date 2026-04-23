# tests/test_context.py
# ─────────────────────────────────────────────
# Manual test script for context.py
# Run from the root of the project:
#   uv run python tests/test_context.py
# ─────────────────────────────────────────────

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.memory import MemoryStore
from core.context import count_tokens, pack_context, build_prompt, build_context_prompt


# ── Pretty printer helper ─────────────────────
def section(title: str):
    print(f"\n{'─'*50}")
    print(f"  {title}")
    print(f"{'─'*50}")


# ─────────────────────────────────────────────
# TEST 1 — count_tokens()
# ─────────────────────────────────────────────
section("TEST 1 — count_tokens()")

texts = [
    "Hello",
    "My name is Eya",
    "Tonight I need to finish coding context.py",
    "I am working on a local memory system for LLMs using Python and SQLite",
]

for text in texts:
    tokens = count_tokens(text)
    words  = len(text.strip().split())
    print(f"  words={words:2d} → tokens={tokens:2d} | '{text}'")

print("\n✅ count_tokens() works correctly")


# ─────────────────────────────────────────────
# TEST 2 — pack_context() fits all memories
# ─────────────────────────────────────────────
section("TEST 2 — pack_context() with large budget (fits all)")

store = MemoryStore(
    db_path="data/test_memories.db",
    md_dir="memories/test"
)

# Add fresh memories
store.add("My name is Eya and I study Mathematical Engineering.", "fact")
store.add("I am working on a local memory system for LLMs.", "project")
store.add("Tonight I need to finish coding context.py.", "todo")
store.add("My partner Taha handles the data and intelligence layer.", "fact")
store.add("I love Python and find AI fascinating.", "preference")

query = "What is Eya working on tonight?"
candidates = store.search(query, top_k=5)

print(f"\n  Query: '{query}'")
print(f"  Budget: 1000 tokens (large — should fit ALL memories)\n")

selected = pack_context(candidates, token_budget=1000)

assert len(selected) == 5, f"Expected 5, got {len(selected)}"
print(f"\n✅ All 5 memories packed correctly")


# ─────────────────────────────────────────────
# TEST 3 — pack_context() respects token budget
# ─────────────────────────────────────────────
section("TEST 3 — pack_context() with tiny budget (limited)")

print(f"\n  Query: '{query}'")
print(f"  Budget: 15 tokens (tiny — should only fit 1-2 memories)\n")

selected_limited = pack_context(candidates, token_budget=15)

assert len(selected_limited) < 5, "Should not fit all memories in 15 tokens"
print(f"\n✅ Token budget respected — only {len(selected_limited)} memories packed")


# ─────────────────────────────────────────────
# TEST 4 — build_prompt()
# ─────────────────────────────────────────────
section("TEST 4 — build_prompt()")

prompt = build_prompt(query, selected_limited)

print(f"\n{'═'*50}")
print(prompt)
print(f"{'═'*50}")

# NEW assertions — matching the new prompt format
assert "--- MEMORY CONTEXT" in prompt,     "Missing memory section"
assert "--- USER QUERY ---" in prompt,     "Missing query section"
assert query in prompt,                    "Query missing from prompt"

print(f"\n✅ Prompt structure is correct")
print(f"   Total prompt length: {len(prompt)} characters")
print(f"   Estimated tokens   : {count_tokens(prompt)}")


# ─────────────────────────────────────────────
# TEST 5 — build_context_prompt() full pipeline
# ─────────────────────────────────────────────
section("TEST 5 — build_context_prompt() full pipeline")

print(f"\n  Running full pipeline with budget=200 tokens\n")

prompt, selected = build_context_prompt(
    query=query,
    candidates=candidates,
    token_budget=200,
)

print(f"\n{'═'*50}")
print("  FINAL PROMPT:")
print(f"{'═'*50}")
print(prompt)
print(f"{'═'*50}")

assert isinstance(prompt, str),   "Prompt should be a string"
assert isinstance(selected, list),"Selected should be a list"
assert len(prompt) > 0,           "Prompt should not be empty"

print(f"\n✅ Full pipeline works correctly")
print(f"   Memories selected : {len(selected)}/{len(candidates)}")
print(f"   Prompt characters : {len(prompt)}")
print(f"   Prompt tokens     : {count_tokens(prompt)}")


# ─────────────────────────────────────────────
# TEST 6 — Edge case: empty candidates
# ─────────────────────────────────────────────
section("TEST 6 — Edge case: no candidates")

prompt_empty, selected_empty = build_context_prompt(
    query="A question with no relevant memories",
    candidates=[],
    token_budget=500,
)

assert "No relevant memories found" in prompt_empty, "Should handle empty gracefully"
assert selected_empty == [],                          "Selected should be empty list"

print(f"\n✅ Empty candidates handled gracefully")
print(f"\n  Prompt preview:")
print(f"  {prompt_empty[:150]}...")


# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
section("ALL TESTS COMPLETE 🎉")
print("  ✅ TEST 1 — count_tokens() works correctly")
print("  ✅ TEST 2 — pack_context() fits all when budget is large")
print("  ✅ TEST 3 — pack_context() respects token budget")
print("  ✅ TEST 4 — build_prompt() structure is correct")
print("  ✅ TEST 5 — build_context_prompt() full pipeline works")
print("  ✅ TEST 6 — Edge case: empty candidates handled")
print()
print("  context.py is ready! You can now connect it to llm.py 🚀")
print()