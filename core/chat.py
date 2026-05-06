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
import threading
import time

from rich.console import Console

from core.memory import GraphManager, GraphRetriever, MemoryExtractor, MemoryClassifier
from core.memory import HebbianUpdater, MemoryConsolidator
from core.memory.necessity import RetrievalNecessityHeuristic
from core.rl.agent import RLAgent
from core.context import pack_context, build_prompt
from core.llm import load_config, call_llm, is_model_available, resolve_model
from core.settings.config import get_config
from core.tui import CommandHandler


# ─────────────────────────────────────────────
# ANSI colors — blue/green theme inspired by dashboard
# ─────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
BLUE   = "\033[38;2;0;204;106m"   # #00CC6A (green - vert)
GREEN  = "\033[38;2;0;196;188m"  # #00C4BC (cyan-green)
WHITE  = "\033[38;2;217;208;194m" # #d9d0c2 (cream)
CYAN   = "\033[38;2;0;180;200m"   # cyan-blue
YELLOW = "\033[38;2;255;200;50m"   # warm yellow
DIM    = "\033[2m"
RED    = "\033[38;2;220;50;50m"  # soft red


# ─────────────────────────────────────────────

# ──────────────────────────────────────

_spinner_active = False
_spinner_thread = None


def _spinner_animation():
    """Background thread that shows a spinning animation."""
    frames = ["⠋", "⠙", "⠸", "⠴", "⠦", "⠧", "⠇", "⠏"]
    i = 0
    while _spinner_active:
        sys.stdout.write(f"\r{YELLOW}  {frames[i % len(frames)]} Thinking...{RESET}")
        sys.stdout.flush()
        time.sleep(0.12)
        i += 1
    sys.stdout.write(f"\r{' '*30}")  # Clear the line


def start_spinner():
    """Start the spinner animation in a background thread."""
    global _spinner_active, _spinner_thread
    _spinner_active = True
    _spinner_thread = threading.Thread(target=_spinner_animation, daemon=True)
    _spinner_thread.start()


def stop_spinner():
    """Stop the spinner animation."""
    global _spinner_active, _spinner_thread
    _spinner_active = False
    if _spinner_thread:
        _spinner_thread.join(timeout=0.5)
        _spinner_thread = None

# LOGO
# ─────────────────────────────────────────────

LOGO_LOC = [
" █████                        ",
"▒▒███                         ",
" ▒███         ██████   ██████ ",
" ▒███        ███▒▒███ ███▒▒███",
" ▒███       ▒███ ▒███▒███ ▒▒▒ ",
" ▒███      █▒███ ▒███▒███  ███",
" ███████████▒▒██████ ▒▒██████ ",
"▒▒▒▒▒▒▒▒▒▒▒  ▒▒▒▒▒▒   ▒▒▒▒▒▒  ",
"                              ",
"                              ",
"                              "
]

LOGO_MEMORY = [
" ██████   ██████                                                       ",
"▒▒██████ ██████                                                        ",
" ▒███▒█████▒███   ██████  █████████████    ██████  ████████  █████ ████",
" ▒███▒▒███ ▒███  ███▒▒███▒▒███▒▒███▒▒███  ███▒▒███▒▒███▒▒███▒▒███ ▒███ ",
" ▒███ ▒▒▒  ▒███ ▒███████  ▒███ ▒███ ▒███ ▒███ ▒███ ▒███ ▒▒▒  ▒███ ▒███ ",
" ▒███      ▒███ ▒███▒▒▒   ▒███ ▒███ ▒███ ▒███ ▒███ ▒███      ▒███ ▒███ ",
" █████     █████▒▒██████  █████▒███ █████▒▒██████  █████     ▒▒███████ ",
"▒▒▒▒▒     ▒▒▒▒▒  ▒▒▒▒▒▒  ▒▒▒▒▒ ▒▒▒ ▒▒▒▒▒  ▒▒▒▒▒▒  ▒▒▒▒▒       ▒▒▒▒▒███ ",
"                                                              ███ ▒███ ",
"                                                             ▒▒██████  ",
"                                                              ▒▒▒▒▒▒   "
]

TAGLINE = "local memory · private · yours"


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

TOKEN_BUDGET = 1500   # max tokens for packed memory context


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def print_logo():
    # Render "Loc" (cyan-green) and "Memory" (cream)
    print()
    for i in range(max(len(LOGO_LOC), len(LOGO_MEMORY))):
        left  = LOGO_LOC[i]    if i < len(LOGO_LOC)    else " " * len(LOGO_LOC[0])
        right = LOGO_MEMORY[i] if i < len(LOGO_MEMORY) else ""
        print(BOLD + GREEN + left + RESET + BOLD + WHITE + right + RESET)
    print(DIM + f"  {TAGLINE}" + RESET)
    print()


def print_startup_info(model: str, memory_count: int):
    print(DIM + f"  model   : {model}" + RESET)
    print(DIM + f"  memories: {memory_count} nodes in graph" + RESET)
    print(DIM + "  type /help for commands · /exit to quit · /clear to reset screen" + RESET)
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
    hebbian: HebbianUpdater,
    consolidator: MemoryConsolidator,
    rl_agent: RLAgent,
    state: dict,
    model: str,
    extraction_enabled: bool = True,
    use_necessity_heuristic: bool = True,
) -> str:
    """
    Execute one chat turn:
      0. Check retrieval necessity heuristic (optional)
      1. Retrieve relevant memory nodes from the cognitive graph
      1.5. RL agent selection (if available)
      2. Pack them within the token budget
      3. Build the final prompt
      4. Call the LLM
      5. Queue background: fact extraction + Hebbian update + consolidation check
    """
    candidates = []
    retrieved_node_ids = []

    if use_necessity_heuristic:
        heuristic = RetrievalNecessityHeuristic()
        requires_retrieval, _ = heuristic.should_retrieve(user_input)
        
        if not requires_retrieval:
            candidates = []
        else:
            candidates = retriever.retrieve(user_input)
            retrieved_node_ids = [c.get("node_id") for c in candidates if c.get("node_id")]
    else:
        candidates = retriever.retrieve(user_input)
        retrieved_node_ids = [c.get("node_id") for c in candidates if c.get("node_id")]

    if rl_agent is not None and retrieved_node_ids:
        from core.rl.agent import RetrievalResult as RLRetrievalResult
        import numpy as np
        
        rl_result = RLRetrievalResult(
            candidates=[
                {"node_id": c.get("node_id", ""), "text": c.get("text", ""),
                 "domain": c.get("domain", ""), "tier": c.get("tier", 3),
                 "score": c.get("score", 0), "hebbian": 0.5, "last_accessed": ""}
                for c in candidates
            ],
            context_str=user_input[:200]
        )
        
        try:
            query_emb = retriever._query_embedding if hasattr(retriever, '_query_embedding') else None
            if query_emb is None:
                query_emb = retriever.classifier._embed([user_input])[0]
            query_emb = np.array(query_emb, dtype=np.float32)
            
            token_budget = state.get("rl_token_budget", 512)
            selected = rl_agent.select(rl_result, query_emb, token_budget)
            
            candidates = selected if selected else candidates
        except Exception:
            pass

    state["retrieval_count"] = state.get("retrieval_count", 0) + 1

    # Step 2 — greedy pack
    packed = pack_context(candidates, token_budget=TOKEN_BUDGET)
    
    # Debug: show what was retrieved
    if candidates:
        print(f"[Recall] {len(candidates)} memories found")
        for c in candidates[:3]:
            print(f"  - {c.get('text', '')[:50]}")

    # Step 3 — build prompt
    prompt = build_prompt(query=user_input, packed_memories=packed)

    # Step 4 — call Ollama
    start_spinner()
    try:
        response = call_llm(prompt=prompt, model=model)
    finally:
        stop_spinner()
    print()

    # Step 5 — extract facts and update hebbian
    if extraction_enabled:
        exchange = f"User: {user_input}\nAssistant: {response.text.strip()}"
        
        # Run extraction synchronously to show summary
        try:
            stored = extractor.process_message(exchange)
            if stored:
                domains = ", ".join(set(s["domain"] for s in stored))
                print(f"[Saved {len(stored)} memory: {domains}]")
        except Exception:
            pass

        if retrieved_node_ids and len(retrieved_node_ids) >= 2:
            try:
                hebbian.update_after_retrieval(retrieved_node_ids)
            except Exception:
                pass

        consolidation_config = get_config().get_section("consolidation")
        if consolidation_config.get("enabled", True):
            run_every_n = consolidation_config.get("run_every_n_additions", 30)
            state["addition_count"] = state.get("addition_count", 0) + 1
            if consolidator.should_run(state["addition_count"], run_every_n):
                try:
                    consolidator.run()
                except Exception:
                    pass

        hebbian_config = get_config().get_section("hebbian")
        if hebbian_config.get("enabled", True):
            decay_interval = hebbian_config.get("decay_interval_retrievals", 100)
            if state.get("retrieval_count", 0) % decay_interval == 0 and state.get("retrieval_count", 0) > 0:
                try:
                    hebbian.apply_decay()
                except Exception:
                    pass

    return response.text


# ─────────────────────────────────────────────
# STARTUP
# ─────────────────────────────────────────────

def startup() -> tuple[GraphManager, GraphRetriever, MemoryExtractor, HebbianUpdater, MemoryConsolidator, RLAgent, str, dict]:
    """
    Load config, initialize the graph memory stack, verify Ollama is running.
    Returns (graph_manager, retriever, extractor, hebbian, consolidator, rl_agent, model_name, state).
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

    config = get_config()
    threshold = config.get("classification", "similarity_threshold", 0.45)
    classifier = MemoryClassifier(confidence_threshold=threshold)
    retriever  = GraphRetriever(gm, classifier=classifier)
    extractor  = MemoryExtractor(gm, classifier=classifier, ollama_model=model)
    
    hebbian = HebbianUpdater(gm)
    consolidator = MemoryConsolidator(gm)
    
    rl_agent = None
    if config.get("rl", "enabled", False):
        try:
            rl_agent = RLAgent()
            if not rl_agent.is_available():
                rl_agent = None
        except Exception:
            rl_agent = None
    
    state = {
        "addition_count": 0,
        "retrieval_count": 0,
    }

    return gm, retriever, extractor, hebbian, consolidator, rl_agent, model, state


# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────

def main():
    clear_screen()
    print_logo()
    gm, retriever, extractor, hebbian, consolidator, rl_agent, model, state = startup()

    def render_banner():
        print_logo()
        print_startup_info(model=model, memory_count=gm.graph.number_of_nodes())

    clear_screen()
    render_banner()

    handler = CommandHandler(gm, extractor, on_clear=render_banner)

    while True:
        try:
            user_input = input(YELLOW + BOLD + "you  " + RESET + "> ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break

        if not user_input:
            continue

        if handler.is_command(user_input):
            result = handler.handle(user_input)
            if result.should_exit:
                break
            if result.skip_pipeline:
                continue

        print()
        try:
            response_text = run_pipeline(
                user_input=user_input,
                retriever=retriever,
                extractor=extractor,
                hebbian=hebbian,
                consolidator=consolidator,
                rl_agent=rl_agent,
                state=state,
                model=model,
                extraction_enabled=handler.extraction_enabled,
            )
            print_response(response_text)

        except Exception as e:
            print_error(str(e))

    # Graceful shutdown
    print()
    print(DIM + "  flushing background extractor (waiting for pending saves)..." + RESET)
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
