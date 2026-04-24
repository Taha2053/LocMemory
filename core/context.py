# core/context.py
# ─────────────────────────────────────────────
# Greedy token-budget packer
# Sits between the GraphRetriever (core.memory.retriever) and llm.py.
#
# The new memory layer returns a list of dicts from
# GraphRetriever.retrieve(query):
#     {"node_id", "text", "domain", "tier", "score", "depth"}
# ─────────────────────────────────────────────


TIER_NAMES = {1: "context", 2: "anchor", 3: "leaf", 4: "procedural"}


# ─────────────────────────────────────────────
# 1. Token counter
# ─────────────────────────────────────────────

def count_tokens(text: str) -> int:
    """Approximate token count: 1 token ≈ 4 characters."""
    char_count = len(text.strip())
    return max(1, char_count // 4)


# ─────────────────────────────────────────────
# 2. Greedy packer
# ─────────────────────────────────────────────

def pack_context(
    candidates: list[dict],
    token_budget: int = 500,
) -> list[dict]:
    """
    Greedily select retrieved memories that fit within the token budget.

    Args:
        candidates   : list of memory dicts returned by GraphRetriever.retrieve()
                       (already sorted by score desc)
        token_budget : max tokens allowed for the packed memory block

    Returns:
        selected memory dicts ordered by score desc
    """
    selected: list[dict] = []
    tokens_used = 0
    tokens_skipped = 0

    for memory in candidates:
        memory_tokens = count_tokens(memory["text"])

        if tokens_used + memory_tokens <= token_budget:
            selected.append(memory)
            tokens_used += memory_tokens
        else:
            tokens_skipped += memory_tokens

    efficiency = (tokens_used / token_budget * 100) if token_budget > 0 else 0
    skipped_count = len(candidates) - len(selected)

    print(f"\n[pack_context] Results:")
    print(f"  • Packed  : {len(selected)}/{len(candidates)} memories")
    print(f"  • Skipped : {skipped_count} memories "
          f"({tokens_skipped} tokens would have exceeded budget)")
    print(f"  • Budget  : {tokens_used}/{token_budget} tokens used "
          f"({efficiency:.1f}% efficiency)")

    print(f"\n  Packed memories (best → least relevant):")
    for i, mem in enumerate(selected, 1):
        tok = count_tokens(mem["text"])
        tier_name = TIER_NAMES.get(mem.get("tier", 3), "leaf")
        domain = mem.get("domain", "") or "—"
        print(f"    {i}. [{tier_name}/{domain}] score={mem['score']:.3f} "
              f"tokens={tok} → \"{mem['text'][:50]}...\"")

    return selected


# ─────────────────────────────────────────────
# 3. Prompt builder
# ─────────────────────────────────────────────

def build_prompt(query: str, packed_memories: list[dict]) -> str:
    """Assemble the final prompt string to send to the local LLM."""
    system = (
        "You are a helpful personal assistant with persistent memory.\n"
        "The MEMORY CONTEXT below contains candidate facts from past conversations.\n"
        "Only use a memory if it is DIRECTLY relevant to the user's current query.\n"
        "Ignore memories that are unrelated to the topic the user is asking about —\n"
        "do NOT mention or reference them. If none of the memories are relevant,\n"
        "answer purely from general knowledge without referring to the memory block.\n"
    )

    if packed_memories:
        memory_lines = []
        for i, memory in enumerate(packed_memories, 1):
            tier_name = TIER_NAMES.get(memory.get("tier", 3), "leaf")
            domain = memory.get("domain", "") or "general"
            memory_lines.append(
                f"[{i}] [{tier_name.upper()}/{domain}] "
                f"(relevance: {memory['score']:.2f})\n"
                f"    {memory['text']}"
            )

        context_block = "\n\n".join(memory_lines)
        context_section = (
            f"--- MEMORY CONTEXT ({len(packed_memories)} memories) ---\n"
            f"{context_block}\n"
            f"--- END OF MEMORY CONTEXT ---\n"
        )
    else:
        context_section = (
            "--- MEMORY CONTEXT ---\n"
            "No relevant memories found for this query.\n"
            "--- END OF MEMORY CONTEXT ---\n"
        )

    query_section = f"--- USER QUERY ---\n{query}\n--- END ---"

    return f"{system}\n{context_section}\n{query_section}"


# ─────────────────────────────────────────────
# 4. Main pipeline function
# ─────────────────────────────────────────────

def build_context_prompt(
    query: str,
    candidates: list[dict],
    token_budget: int = 500,
) -> tuple[str, list[dict]]:
    """
    Full pipeline: candidates → pack → build prompt.

    Args:
        query        : the user's current message
        candidates   : output of GraphRetriever.retrieve()
        token_budget : max tokens for memory context

    Returns:
        (prompt string, packed memory dicts)
    """
    print(f"\n{'='*55}")
    print(f"[context.py] New query received")
    print(f"  Query     : '{query}'")
    print(f"  Budget    : {token_budget} tokens")
    print(f"  Candidates: {len(candidates)} memories from retrieval")
    print(f"{'='*55}")

    selected = pack_context(candidates, token_budget)
    prompt = build_prompt(query, selected)

    prompt_tokens = count_tokens(prompt)
    print(f"\n[context.py] Prompt ready")
    print(f"  Characters : {len(prompt)}")
    print(f"  Est. tokens: {prompt_tokens}")

    return prompt, selected
