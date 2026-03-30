# core/llm.py
# ─────────────────────────────────────────────
# LLM caller — model-agnostic Ollama backend
# Sits between context.py (prompt builder) and chat.py (terminal loop)
#
# Main responsibilities:
#   1. Connect to local Ollama server
#   2. Send the prompt built by context.py to the model
#   3. Return the response + exact token usage stats
#
# Design principles:
#   - Model-agnostic: model name comes from config.yaml, not hardcoded
#   - Fully local: zero internet, zero API keys
#   - Graceful errors: clear messages if Ollama is not running or model not pulled
# ─────────────────────────────────────────────

import ollama
import yaml
from pathlib import Path


# ─────────────────────────────────────────────
# 1. Config loader
# ─────────────────────────────────────────────

def load_config(config_path: str = "core/config.yaml") -> dict:
    """
    Load configuration from config.yaml.
    This is where the model name lives — so changing the model
    never requires touching the code, just the config file.

    Returns:
        config dict with keys like LLM_MODEL, DB_PATH, etc.
    """
    path = Path(config_path)
    if not path.exists():
        # Fallback defaults if config file is missing
        return {"LLM_MODEL": "mistral:7b-instruct"}

    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


# ─────────────────────────────────────────────
# 2. LLMResponse dataclass
# ─────────────────────────────────────────────

from dataclasses import dataclass

@dataclass
class LLMResponse:
    """
    Clean container for everything the LLM returns.
    Instead of passing raw dicts around, we use a typed object.

    Fields:
        text          : the actual response text from the model
        model         : which model generated it (e.g. "mistral:7b-instruct")
        input_tokens  : exact tokens used for the prompt (from Ollama)
        output_tokens : exact tokens used for the response (from Ollama)
        total_tokens  : input + output (useful for logging/dashboard)
    """
    text:          str
    model:         str
    input_tokens:  int
    output_tokens: int
    total_tokens:  int


# ─────────────────────────────────────────────
# 3. Core LLM caller
# ─────────────────────────────────────────────

def call_llm(
    prompt: str,
    model:  str | None = None,
    system: str | None = None,
) -> LLMResponse:
    """
    Send a prompt to the local Ollama model and return the response.

    How it works:
        1. Load model name from config (or use the one passed in)
        2. Split the prompt into system + user messages
           (Ollama chat API expects this format)
        3. Call ollama.chat() — this talks to the local Ollama server
        4. Extract response text + exact token counts
        5. Return everything as a clean LLMResponse object

    Args:
        prompt : the complete prompt string from context.py
                 (already contains system instruction + memories + query)
        model  : optional model override — if None, reads from config.yaml
        system : optional system message override

    Returns:
        LLMResponse object with text, model name, and token counts

    Raises:
        ConnectionError : if Ollama server is not running
        ValueError      : if the model is not pulled yet
    """
    # ── Step 1: Get model name ────────────────
    # Priority: passed argument > config.yaml > hardcoded default
    if model is None:
        config = load_config()
        model  = config.get("LLM_MODEL", "mistral:7b-instruct")

    # ── Step 2: Build messages list ───────────
    # Ollama chat() expects a list of {"role": ..., "content": ...} dicts
    # We split our prompt into:
    #   - "system" role: instructions + memory context
    #   - "user" role: the actual user query
    #
    # Why split? Because instruct models (like mistral:7b-instruct)
    # are trained to treat system and user messages differently.
    # System = persistent instructions, User = current request.

    if system:
        # If a custom system message is provided, use it separately
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ]
    else:
        # Otherwise send the full prompt as a single user message
        # context.py already formats everything cleanly
        messages = [
            {"role": "user", "content": prompt},
        ]

    # ── Step 3: Call Ollama ───────────────────
    try:
        response = ollama.chat(
            model=model,
            messages=messages,
        )

    except ollama.ResponseError as e:
        # Model not found — user needs to pull it first
        if e.status_code == 404:
            raise ValueError(
                f"Model '{model}' not found in Ollama.\n"
                f"Pull it with: ollama pull {model}"
            ) from e
        raise  # re-raise any other Ollama errors

    except Exception as e:
        # Ollama server is probably not running
        raise ConnectionError(
            "Cannot connect to Ollama. Is it running?\n"
            "Start it with: ollama serve"
        ) from e

    # ── Step 4: Extract response + token counts ──
    # response["message"]["content"] → the actual text response
    # response["prompt_eval_count"]  → exact input tokens (from Ollama)
    # response["eval_count"]         → exact output tokens (from Ollama)
    response_text  = response["message"]["content"]
    input_tokens   = response.get("prompt_eval_count", 0)
    output_tokens  = response.get("eval_count", 0)
    total_tokens   = input_tokens + output_tokens

    # ── Step 5: Log stats ─────────────────────
    print(f"\n[llm.py] Response received")
    print(f"  Model         : {model}")
    print(f"  Input tokens  : {input_tokens}")
    print(f"  Output tokens : {output_tokens}")
    print(f"  Total tokens  : {total_tokens}")
    print(f"  Response length: {len(response_text)} chars")

    # ── Step 6: Return clean object ───────────
    return LLMResponse(
        text=response_text,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


# ─────────────────────────────────────────────
# 4. Model checker utility
# ─────────────────────────────────────────────

def is_model_available(model: str) -> bool:
    """
    Check if a model is already pulled in Ollama.
    Useful before starting a session — avoids cryptic errors later.

    Args:
        model: model name to check (e.g. "mistral:7b-instruct")

    Returns:
        True if model is available, False otherwise
    """
    try:
        models = ollama.list()
        # ollama.list() returns a dict with "models" key
        available = [m.model for m in models.models]
        return model in available
    except Exception:
        return False


# ─────────────────────────────────────────────
# 5. List available models utility
# ─────────────────────────────────────────────

def list_available_models() -> list[str]:
    """
    Return all models currently pulled in Ollama.
    Useful for the dashboard and for letting users pick a model.

    Returns:
        list of model name strings
    """
    try:
        models = ollama.list()
        return [m.model for m in models.models]
    except Exception:
        return []