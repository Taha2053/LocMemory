# core/context.py
# ─────────────────────────────────────────────
# Greedy token-budget packer
# Sits between memory.py (retrieval) and llm.py (local model)
#
# Main responsibilities:
#   1. Count tokens for each memory (lightweight approximation)
#   2. Greedily pack the most relevant memories within token budget
#   3. Build the final prompt string to send to the LLM
# ─────────────────────────────────────────────

from core.memory import Memory


# ─────────────────────────────────────────────
# 1. Token counter
# ─────────────────────────────────────────────

def count_tokens(text: str) -> int:
    """
    Estimate the number of tokens in a text.
    Approximation: 1 token ≈ 0.75 words (standard rule for English).
    No external library needed — works fully locally.
    """
    words = len(text.strip().split())
    return max(1, int(words / 0.75))


# ─────────────────────────────────────────────
# 2. Greedy packer
# ─────────────────────────────────────────────

def pack_context(
    candidates: list[tuple[Memory, float]],
    token_budget: int = 500,
) -> list[tuple[Memory, float]]:
    """
    Greedily select memories that fit within the token budget.

    Args:
        candidates   : list of (Memory, score) tuples, sorted by score descending
                       (this is exactly what memory.py search() returns)
        token_budget : maximum number of tokens allowed for memories

    Returns:
        selected     : list of (Memory, score) tuples that fit in the budget
                       ordered by score descending (most relevant first)
    """
    selected = []
    tokens_used = 0

    for memory, score in candidates:
        memory_tokens = count_tokens(memory.text)

        if tokens_used + memory_tokens <= token_budget:
            # ✅ fits → add it
            selected.append((memory, score))
            tokens_used += memory_tokens
            print(f"  ✅ packed [{memory.category}] score={score:.3f} "
                  f"tokens={memory_tokens} (used={tokens_used}/{token_budget})")
        else:
            # ❌ doesn't fit → skip
            print(f"  ❌ skipped [{memory.category}] score={score:.3f} "
                  f"tokens={memory_tokens} (would exceed budget)")

    print(f"\n  Total: {len(selected)} memories packed | "
          f"{tokens_used}/{token_budget} tokens used")

    return selected


# ─────────────────────────────────────────────
# 3. Prompt builder
# ─────────────────────────────────────────────

def build_prompt(query: str, packed_memories: list[tuple[Memory, float]]) -> str:
    """
    Assemble the final prompt string to send to the local LLM.

    Structure:
        [System instruction]
        [Relevant memories injected as context]
        [User query]

    Args:
        query          : the user's current message
        packed_memories: output of pack_context()

    Returns:
        prompt         : the complete string ready to send to the LLM
    """
    # ── System instruction ────────────────────
    system = (
        "You are a helpful personal assistant with memory. "
        "Use the context below to answer the user's question. "
        "If the context is not relevant, answer from your own knowledge.\n"
    )

    # ── Memory context block ──────────────────
    if packed_memories:
        memory_lines = []
        for i, (memory, score) in enumerate(packed_memories, 1):
            memory_lines.append(
                f"[Memory {i} | category={memory.category} | score={score:.3f}]\n"
                f"{memory.text}"
            )
        context_block = "\n\n".join(memory_lines)
        context_section = f"### Relevant Memories:\n{context_block}\n"
    else:
        context_section = "### Relevant Memories:\nNo relevant memories found.\n"

    # ── User query ────────────────────────────
    query_section = f"### User Query:\n{query}"

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
    This is the single function that llm.py will call.

    Args:
        query        : the user's current message
        candidates   : raw output from memory.py search()
        token_budget : max tokens for memory context

    Returns:
        prompt       : complete prompt string ready for the LLM
        selected     : the memories that were packed (for logging/dashboard)
    """
    print(f"\n[context.py] Packing context for query: '{query}'")
    print(f"[context.py] Budget: {token_budget} tokens | "
          f"Candidates: {len(candidates)}\n")

    selected = pack_context(candidates, token_budget)
    prompt = build_prompt(query, selected)

    return prompt, selected