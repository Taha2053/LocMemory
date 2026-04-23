"""
chat.py
-------
Main entry point for LocMemory.
Orchestrates memory.py, context.py, and llm.py
into a continuous terminal chat loop.

Run with:
    uv run python chat.py
"""

import os
import sys

from core.memory import MemoryStore
from core.context import pack_context, build_prompt, count_tokens
from core.llm import load_config, call_llm, is_model_available


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
# LOGO — paste your ASCII art between the quotes
# Keep the triple quotes, just replace the content
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

# Tagline printed just below the logo
TAGLINE = "local memory · private · yours"


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

TOKEN_BUDGET   = 500   # max tokens for packed memory context
TOP_K_MEMORIES = 5     # how many memories to retrieve per turn


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def clear_screen():
    """Clear the terminal screen — Windows and Unix compatible."""
    os.system("cls" if os.name == "nt" else "clear")


def print_logo():
    """Print the logo and tagline."""
    print(YELLOW + LOGO + RESET)
    print(DIM + f"  {TAGLINE}" + RESET)
    print()


def print_startup_info(model: str, memory_count: int):
    """Print system info line after the logo."""
    print(DIM + f"  model   : {model}" + RESET)
    print(DIM + f"  memories: {memory_count} stored" + RESET)
    print(DIM + "  type 'exit' to quit · 'clear' to reset screen" + RESET)
    print()


def print_response(text: str):
    """Print the LLM response with a subtle label."""
    print(CYAN + BOLD + "assistant" + RESET)
    print(text.strip())
    print()


def print_error(msg: str):
    """Print an error message in red."""
    print(RED + f"  error: {msg}" + RESET)
    print()


# ─────────────────────────────────────────────
# CORE PIPELINE — one full turn
# This is the heart of chat.py:
# retrieve → pack → prompt → LLM → save
# ─────────────────────────────────────────────

def run_pipeline(user_input: str, store: MemoryStore, model: str) -> str:
    """
    Execute one full chat turn:
    1. Retrieve relevant memories for the user input
    2. Pack them within the token budget
    3. Build the final prompt
    4. Call the LLM
    5. Save the exchange as a new memory
    Returns the LLM response text.
    """

    # Step 1 — retrieve the most relevant memories
    # search() returns [(Memory, score), ...] sorted by relevance
    candidates = store.search(user_input, top_k=TOP_K_MEMORIES)

    # Step 2 — pack candidates within the token budget
    # pack_context() drops low-relevance memories if budget is tight
    packed = pack_context(candidates, token_budget=TOKEN_BUDGET)

    # Step 3 — build the full prompt (system instructions + memories + user query)
    prompt = build_prompt(query=user_input, packed_memories=packed)

    # Step 4 — call the LLM via Ollama
    response = call_llm(prompt=prompt, model=model)

    # Step 5 — save the full exchange as a new memory
    # Format: "User: ... \nAssistant: ..." so future retrievals have full context
    exchange = f"User: {user_input}\nAssistant: {response.text.strip()}"
    store.add(exchange, category="exchange")

    return response.text


# ─────────────────────────────────────────────
# STARTUP — initialize everything before the loop
# ─────────────────────────────────────────────

def startup() -> tuple[MemoryStore, str]:
    """
    Load config, initialize MemoryStore, verify Ollama is running.
    Returns (store, model_name) ready for the chat loop.
    Exits with a clear message if anything is wrong.
    """

    # Load config.yaml
    config = load_config()
    model  = config.get("LLM_MODEL", "mistral:7b-instruct-v0.3-q4_0")
    db_path = config.get("DB_PATH", "data/memories.db")
    md_dir  = config.get("MEMORIES_DIRECTORY", "memories/")

    # Check Ollama is running and model is available
    if not is_model_available(model):
        print_error(f"model '{model}' not found in Ollama.")
        print(DIM + "  make sure Ollama is running: ollama serve" + RESET)
        print(DIM + f"  and the model is pulled: ollama pull {model}" + RESET)
        sys.exit(1)

    # Initialize the memory store
    # db_path may be a directory — append filename if so
    if db_path.endswith("/") or db_path.endswith("\\"):
        db_path = db_path + "memories.db"

    store = MemoryStore(db_path=db_path, md_dir=md_dir)

    return store, model


# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────

def main():
    # ── Startup ──────────────────────────────
    clear_screen()
    print_logo()

    print(DIM + "  initializing..." + RESET)
    store, model = startup()

    clear_screen()
    print_logo()
    print_startup_info(model=model, memory_count=store.count())

    # ── Chat loop ────────────────────────────
    while True:
        try:
            # Print the user prompt indicator and wait for input
            user_input = input(YELLOW + BOLD + "you  " + RESET + "> ").strip()

        except (KeyboardInterrupt, EOFError):
            # Handle Ctrl+C and Ctrl+D gracefully
            print()
            break

        # ── Empty input — just loop again
        if not user_input:
            continue

        # ── Exit command
        if user_input.lower() in ("exit", "quit"):
            break

        # ── Clear command — clears screen but NOT the database
        if user_input.lower() == "clear":
            clear_screen()
            print_logo()
            print_startup_info(model=model, memory_count=store.count())
            continue

        # ── Normal input — run the full pipeline
        print()
        try:
            response_text = run_pipeline(
                user_input=user_input,
                store=store,
                model=model
            )
            print_response(response_text)

        except Exception as e:
            print_error(str(e))

    # ── Graceful shutdown
    print()
    print(DIM + "  closing memory store..." + RESET)
    store.close()
    print(DIM + "  goodbye." + RESET)
    print()


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    main()