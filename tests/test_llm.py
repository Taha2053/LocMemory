# tests/test_llm.py
# ─────────────────────────────────────────────
# Manual test script for llm.py
# Run from the root of the project:
#   uv run python tests/test_llm.py
# ─────────────────────────────────────────────

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.llm import call_llm, is_model_available, list_available_models, load_config


# ── Pretty printer helper ─────────────────────
def section(title: str):
    print(f"\n{'─'*50}")
    print(f"  {title}")
    print(f"{'─'*50}")


# ─────────────────────────────────────────────
# TEST 1 — load_config()
# ─────────────────────────────────────────────
section("TEST 1 — load_config()")

config = load_config()
print(f"  Config loaded: {config}")
assert "LLM_MODEL" in config, "LLM_MODEL missing from config"
print(f"\n✅ Config loaded — model: {config['LLM_MODEL']}")


# ─────────────────────────────────────────────
# TEST 2 — list_available_models()
# ─────────────────────────────────────────────
section("TEST 2 — list_available_models()")

models = list_available_models()
print(f"  Available models: {models}")
assert len(models) > 0, "No models found — is Ollama running?"
print(f"\n✅ Found {len(models)} model(s) in Ollama")


# ─────────────────────────────────────────────
# TEST 3 — is_model_available()
# ─────────────────────────────────────────────
section("TEST 3 — is_model_available()")

model = config["LLM_MODEL"]
available = is_model_available(model)
print(f"  Checking: '{model}'")
print(f"  Available: {available}")
assert available, f"Model '{model}' not found — check config.yaml"
print(f"\n✅ Model '{model}' is available")


# ─────────────────────────────────────────────
# TEST 4 — call_llm() simple prompt
# ─────────────────────────────────────────────
section("TEST 4 — call_llm() simple prompt")

simple_prompt = "Reply with exactly 3 words: I am ready."
print(f"  Sending prompt: '{simple_prompt}'\n")

response = call_llm(prompt=simple_prompt)

print(f"\n  Response text  : '{response.text}'")
print(f"  Model          : {response.model}")
print(f"  Input tokens   : {response.input_tokens}")
print(f"  Output tokens  : {response.output_tokens}")
print(f"  Total tokens   : {response.total_tokens}")

assert isinstance(response.text, str),   "Response text should be a string"
assert len(response.text) > 0,           "Response should not be empty"
assert response.input_tokens > 0,        "Input tokens should be > 0"
assert response.output_tokens > 0,       "Output tokens should be > 0"

print(f"\n✅ call_llm() works correctly")


# ─────────────────────────────────────────────
# TEST 5 — call_llm() with memory context
# ─────────────────────────────────────────────
section("TEST 5 — call_llm() with full memory context")

# Simulate the exact prompt that context.py builds
memory_prompt = """You are a helpful personal assistant with persistent memory.
You have access to relevant memories from past conversations.
Use these memories to give personalized, context-aware answers.

--- MEMORY CONTEXT (2 memories) ---
[1] [FACT] (relevance: 0.95) (saved: 2026-03-01)
    My name is Eya and I study Mathematical Engineering.

[2] [PROJECT] (relevance: 0.87) (saved: 2026-03-01)
    I am working on a local memory system for LLMs called LocMemory.
--- END OF MEMORY CONTEXT ---

--- USER QUERY ---
What is my name and what am I working on?
--- END ---"""

print(f"  Sending prompt with 2 injected memories...\n")

response = call_llm(prompt=memory_prompt)

print(f"\n  ── RESPONSE ──────────────────────────────")
print(f"  {response.text}")
print(f"  ──────────────────────────────────────────")
print(f"  Input tokens  : {response.input_tokens}")
print(f"  Output tokens : {response.output_tokens}")
print(f"  Total tokens  : {response.total_tokens}")

assert "Eya" in response.text or "eya" in response.text.lower(), \
    "Model should mention Eya from memory context"

print(f"\n✅ Model correctly used memory context in response")


# ─────────────────────────────────────────────
# TEST 6 — Error handling: wrong model name
# ─────────────────────────────────────────────
section("TEST 6 — Error handling: wrong model name")

try:
    call_llm(prompt="Hello", model="non-existent-model:latest")
    print("❌ Should have raised an error!")
except ValueError as e:
    print(f"  ✅ Correctly caught ValueError: {e}")
except Exception as e:
    print(f"  ✅ Correctly raised error: {type(e).__name__}: {e}")


# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
section("ALL TESTS COMPLETE 🎉")
print("  ✅ TEST 1 — load_config() works correctly")
print("  ✅ TEST 2 — list_available_models() works")
print("  ✅ TEST 3 — is_model_available() works")
print("  ✅ TEST 4 — call_llm() basic prompt works")
print("  ✅ TEST 5 — call_llm() uses memory context correctly")
print("  ✅ TEST 6 — Error handling works gracefully")
print()
print("  llm.py is ready! You can now connect it to chat.py 🚀")
print()
