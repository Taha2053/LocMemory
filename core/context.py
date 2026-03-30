# core/context.py
# ─────────────────────────────────────────────
# Greedy token-budget packer
# Sits between memory.py (retrieval) and llm.py (local model)
#
# Improvements over v1:
#   - Token counting now uses chars/4 (more accurate, industry standard)
#   - Cleaner prompt format (better LLM comprehension)
#   - Detailed logging with efficiency stats
# ─────────────────────────────────────────────

from core.memory import Memory


# ─────────────────────────────────────────────
# 1. Token counter (IMPROVED)
# ─────────────────────────────────────────────

def count_tokens(text: str) -> int:
    """
    Estimate the number of tokens in a text.

    Formula: 1 token ≈ 4 characters
    This is the industry standard approximation used by OpenAI, Anthropic
    and most LLM providers. It is more accurate than word-based counting
    because tokenizers operate on characters, not words.

    Examples:
        "Hello"         →  5 chars →  2 tokens
        "My name is Eya"→ 14 chars →  4 tokens
        "I love Python" → 13 chars →  4 tokens

    Args:
        text: any string to estimate token count for

    Returns:
        token count (minimum 1, never 0)
    """
    # Count characters (excluding leading/trailing whitespace)
    char_count = len(text.strip())

    # Divide by 4 — the standard chars-per-token ratio
    # max(1, ...) ensures we never return 0 even for single characters
    return max(1, char_count // 4)


# ─────────────────────────────────────────────
# 2. Greedy packer (IMPROVED LOGGING)
# ─────────────────────────────────────────────

def pack_context(
    candidates: list[tuple[Memory, float]],
    token_budget: int = 500,
) -> list[tuple[Memory, float]]:
    """
    Greedily select memories that fit within the token budget.

    Algorithm (Greedy Knapsack):
        1. Iterate through candidates in score-descending order
           (already sorted by memory.py search())
        2. For each memory, check if it fits in the remaining budget
        3. If it fits  → add it, subtract its tokens from budget
        4. If it doesn't fit → skip it, move to next
        5. Return all selected memories

    Why greedy?
        Simple, fast, and good enough for Month 1.
        In Month 3, the RL agent will replace this with a
        learned policy that makes smarter tradeoffs.

    Args:
        candidates   : list of (Memory, score) tuples sorted by score desc
                       this is exactly what memory.py search() returns
        token_budget : maximum number of tokens allowed for all memories

    Returns:
        selected     : list of (Memory, score) tuples that fit in the budget
                       ordered by score descending (most relevant first)
    """
    selected  = []       # memories we decided to keep
    tokens_used = 0      # running total of tokens consumed
    tokens_skipped = 0   # tokens we had to skip (for logging)

    for memory, score in candidates:
        # Estimate token cost of this memory
        memory_tokens = count_tokens(memory.text)

        if tokens_used + memory_tokens <= token_budget:
            # ✅ fits within budget → pack it
            selected.append((memory, score))
            tokens_used += memory_tokens

        else:
            # ❌ would exceed budget → skip it
            tokens_skipped += memory_tokens

    # ── Logging summary ───────────────────────
    # Calculate how efficiently we used the budget
    efficiency = (tokens_used / token_budget * 100) if token_budget > 0 else 0
    skipped_count = len(candidates) - len(selected)

    print(f"\n[pack_context] Results:")
    print(f"  • Packed  : {len(selected)}/{len(candidates)} memories")
    print(f"  • Skipped : {skipped_count} memories "
          f"({tokens_skipped} tokens would have exceeded budget)")
    print(f"  • Budget  : {tokens_used}/{token_budget} tokens used "
          f"({efficiency:.1f}% efficiency)")

    # ── Per-memory breakdown ──────────────────
    print(f"\n  Packed memories (best → least relevant):")
    for i, (mem, score) in enumerate(selected, 1):
        tok = count_tokens(mem.text)
        print(f"    {i}. [{mem.category}] score={score:.3f} "
              f"tokens={tok} → \"{mem.text[:50]}...\"")

    return selected


# ─────────────────────────────────────────────
# 3. Prompt builder (IMPROVED FORMAT)
# ─────────────────────────────────────────────

def build_prompt(query: str, packed_memories: list[tuple[Memory, float]]) -> str:
    """
    Assemble the final prompt string to send to the local LLM.

    Prompt structure (optimized for local LLMs like Ollama/HuggingFace):
    ┌─────────────────────────────────────┐
    │  SYSTEM INSTRUCTION                 │
    │  ─────────────────                  │
    │  MEMORY CONTEXT (injected)          │
    │    Memory 1 (highest score)         │
    │    Memory 2                         │
    │    ...                              │
    │  ─────────────────                  │
    │  USER QUERY                         │
    └─────────────────────────────────────┘

    Why this structure?
        Local LLMs respond best when:
        - System instruction comes first (sets behavior)
        - Context is clearly labeled and separated
        - Query comes last (most recent = highest attention)

    Args:
        query          : the user's current message
        packed_memories: output of pack_context()

    Returns:
        prompt         : complete string ready to send to the LLM
    """
    # ── System instruction ────────────────────
    # Tells the LLM how to behave and how to use the memories
    system = (
        "You are a helpful personal assistant with persistent memory.\n"
        "You have access to relevant memories from past conversations.\n"
        "Use these memories to give personalized, context-aware answers.\n"
        "If the memories are not relevant to the question, "
        "answer from your own knowledge.\n"
    )

    # ── Memory context block ──────────────────
    if packed_memories:
        memory_lines = []
        for i, (memory, score) in enumerate(packed_memories, 1):
            # Each memory is labeled with its rank, category, score and timestamp
            # This gives the LLM rich metadata to reason about each memory
            memory_lines.append(
                f"[{i}] [{memory.category.upper()}] "
                f"(relevance: {score:.2f}) "
                f"(saved: {memory.timestamp[:10]})\n"   # only date, not full timestamp
                f"    {memory.text}"
            )

        context_block   = "\n\n".join(memory_lines)
        context_section = (
            f"--- MEMORY CONTEXT ({len(packed_memories)} memories) ---\n"
            f"{context_block}\n"
            f"--- END OF MEMORY CONTEXT ---\n"
        )
    else:
        # Graceful fallback — no memories found
        context_section = (
            "--- MEMORY CONTEXT ---\n"
            "No relevant memories found for this query.\n"
            "--- END OF MEMORY CONTEXT ---\n"
        )

    # ── User query ────────────────────────────
    # Comes last — local LLMs give highest attention to recent tokens
    query_section = f"--- USER QUERY ---\n{query}\n--- END ---"

    # ── Assemble full prompt ──────────────────
    prompt = f"{system}\n{context_section}\n{query_section}"

    return prompt


# ─────────────────────────────────────────────
# 4. Main pipeline function
# ─────────────────────────────────────────────

def build_context_prompt(
    query: str,
    candidates: list[tuple[Memory, float]],
    token_budget: int = 500,
) -> tuple[str, list[tuple[Memory, float]]]:
    """
    Full pipeline: candidates → pack → build prompt.
    This is the SINGLE function that llm.py will call.

    Flow:
        memory.py.search(query)           → candidates (ranked list)
              ↓
        pack_context(candidates, budget)  → selected memories
              ↓
        build_prompt(query, selected)     → final prompt string
              ↓
        llm.py.generate(prompt)           → LLM response

    Args:
        query        : the user's current message
        candidates   : raw output from memory.py search()
        token_budget : max tokens for memory context (default 500)

    Returns:
        prompt       : complete prompt string ready for the LLM
        selected     : memories that were packed (for dashboard logging)
    """
    print(f"\n{'='*55}")
    print(f"[context.py] New query received")
    print(f"  Query    : '{query}'")
    print(f"  Budget   : {token_budget} tokens")
    print(f"  Candidates: {len(candidates)} memories from retrieval")
    print(f"{'='*55}")

    # Step 1 — greedily pack memories within token budget
    selected = pack_context(candidates, token_budget)

    # Step 2 — assemble the final prompt string
    prompt = build_prompt(query, selected)

    # Step 3 — log final prompt stats
    prompt_tokens = count_tokens(prompt)
    print(f"\n[context.py] Prompt ready")
    print(f"  Characters : {len(prompt)}")
    print(f"  Est. tokens: {prompt_tokens}")

    return prompt, selected