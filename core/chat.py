"""
chat.py
-------
Main entry point for LocMemory.

Orchestrates the new cognitive memory stack (graph + retriever + extractor),
context.py (token-budget packer / prompt builder), and llm.py (Ollama caller)
into a continuous terminal chat loop.

Run with:
    uv run python -m core.chat
"""

import os
import sys

from core.memory import GraphManager, GraphRetriever, MemoryExtractor, MemoryClassifier
from core.context import pack_context, build_prompt
from core.llm import load_config, call_llm, is_model_available, resolve_model


# ─────────────────────────────────────────────
# ANSI colors — no external dependencies
# ─────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"
YELLOW = "\033[93m"
DIM    = "\033[2m"
GREEN  = "\033[92m"
RED    = "\033[91m"


# ─────────────────────────────────────────────
# LOGO
# ─────────────────────────────────────────────

LOGO = r"""
 █████                         ██████   ██████
▒▒███                         ▒▒██████ ██████
 ▒███         ██████   ██████  ▒███▒█████▒███   ██████  █████████████    ██████  ████████  █████ ████
 ▒███        ███▒▒███ ███▒▒███ ▒███▒▒███ ▒███  ███▒▒███▒▒███▒▒███▒▒███  ███▒▒███▒▒███▒▒███▒▒███ ▒███
 ▒███       ▒███ ▒███▒███ ▒▒▒  ▒███ ▒▒▒  ▒███ ▒███████  ▒███ ▒███ ▒███ ▒███ ▒███ ▒███ ▒▒▒  ▒███ ▒███
 ▒███      █▒███ ▒███▒███  ███ ▒███      ▒███ ▒███▒▒▒   ▒███ ▒███ ▒███ ▒███ ▒███ ▒███      ▒███ ▒███
 ███████████▒▒██████ ▒▒██████  █████     █████▒▒██████  █████▒███ █████▒▒██████  █████     ▒▒███████
▒▒▒▒▒▒▒▒▒▒▒  ▒▒▒▒▒▒   ▒▒▒▒▒▒  ▒▒▒▒▒     ▒▒▒▒▒  ▒▒▒▒▒▒  ▒▒▒▒▒ ▒▒▒ ▒▒▒▒▒  ▒▒▒▒▒▒  ▒▒▒▒▒       ▒▒▒▒▒███
                                                                                            ███ ▒███
                                                                                           ▒▒██████
                                                                                            ▒▒▒▒▒▒
"""

TAGLINE = "local memory · private · yours"


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

TOKEN_BUDGET = 500   # max tokens for packed memory context


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def print_logo():
    print(DIM+ LOGO + RESET)
    print(DIM + f"  {TAGLINE}" + RESET)
    print()


def print_startup_info(model: str, memory_count: int):
    print(DIM + f"  model   : {model}" + RESET)
    print(DIM + f"  memories: {memory_count} nodes in graph" + RESET)
    print(DIM + "  type 'exit' to quit · 'clear' to reset screen" + RESET)
    print()


def print_response(text: str):
    print(CYAN + BOLD + "assistant" + RESET)
    print(text.strip())
    print()


def print_error(msg: str):
    print(RED + f"  error: {msg}" + RESET)
    print()


# ─────────────────────────────────────────────
# CORE PIPELINE — one full turn
# retrieve (graph) → pack → prompt → LLM → background extract
# ─────────────────────────────────────────────

def run_pipeline(
    user_input: str,
    retriever: GraphRetriever,
    extractor: MemoryExtractor,
    model: str,
) -> str:
    """
    Execute one chat turn:
      1. Retrieve relevant memory nodes from the cognitive graph
      2. Pack them within the token budget
      3. Build the final prompt
      4. Call the LLM
      5. Queue background fact extraction for the user message + response
    """

    # Step 1 — graph retrieval (returns list[dict])
    candidates = retriever.retrieve(user_input)

    # Step 2 — greedy pack
    packed = pack_context(candidates, token_budget=TOKEN_BUDGET)

    # Step 3 — build prompt
    prompt = build_prompt(query=user_input, packed_memories=packed)

    # Step 4 — call Ollama
    response = call_llm(prompt=prompt, model=model)

    # Step 5 — save the exchange as extracted facts (background, non-blocking)
    exchange = f"User: {user_input}\nAssistant: {response.text.strip()}"
    try:
        extractor.start_background_extraction(exchange)
    except Exception as e:
        print(DIM + f"  [warn] background extraction failed: {e}" + RESET)

    return response.text


# ─────────────────────────────────────────────
# STARTUP
# ─────────────────────────────────────────────

def startup() -> tuple[GraphManager, GraphRetriever, MemoryExtractor, str]:
    """
    Load config, initialize the graph memory stack, verify Ollama is running.
    Returns (graph_manager, retriever, extractor, model_name).
    """
    config = load_config()
    model   = config.get("LLM_MODEL", "mistral:7b-instruct")
    db_path = config.get("DB_PATH", "data/memory.db")

    if not is_model_available(model):
        print_error(f"model '{model}' not found in Ollama.")
        print(DIM + "  make sure Ollama is running: ollama serve" + RESET)
        print(DIM + f"  and the model is pulled: ollama pull {model}" + RESET)
        sys.exit(1)

    # Resolve to the installed tag (e.g. "mistral:7b-instruct" →
    # "mistral:7b-instruct-v0.3-q4_0") so every downstream caller uses
    # the exact name Ollama recognizes.
    resolved = resolve_model(model)
    if resolved and resolved != model:
        print(DIM + f"  using installed tag: {resolved}" + RESET)
        model = resolved

    # Normalize if db_path points at a directory
    if db_path.endswith("/") or db_path.endswith("\\"):
        db_path = db_path + "memory.db"

    # Boot the cognitive graph
    gm = GraphManager(db_path=db_path)
    gm.initialize_db()
    gm.load_graph()

    classifier = MemoryClassifier()
    retriever  = GraphRetriever(gm, classifier=classifier)
    extractor  = MemoryExtractor(gm, classifier=classifier, ollama_model=model)

    return gm, retriever, extractor, model


# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────

def main():
    clear_screen()
    print_logo()

    print(DIM + "  initializing..." + RESET)
    gm, retriever, extractor, model = startup()

    clear_screen()
    print_logo()
    print_startup_info(model=model, memory_count=gm.graph.number_of_nodes())

    while True:
        try:
            user_input = input(YELLOW + BOLD + "you  " + RESET + "> ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            break

        if user_input.lower() == "clear":
            clear_screen()
            print_logo()
            print_startup_info(model=model, memory_count=gm.graph.number_of_nodes())
            continue

        print()
        try:
            response_text = run_pipeline(
                user_input=user_input,
                retriever=retriever,
                extractor=extractor,
                model=model,
            )
            print_response(response_text)

        except Exception as e:
            print_error(str(e))

    # Graceful shutdown
    print()
    print(DIM + "  stopping background extractor..." + RESET)
    try:
        extractor.stop()
    except Exception:
        pass

    print(DIM + "  closing graph..." + RESET)
    try:
        gm.save_graph()
    except Exception:
        pass
    gm.close()
    print(DIM + "  goodbye." + RESET)
    print()


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    main()
