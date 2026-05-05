# core/llm.py
# ─────────────────────────────────────────────
# Multi-backend LLM caller
# Supported providers: ollama | huggingface | anthropic
#
# Provider is read from config.yaml → models.llm.provider
# (defaults to "ollama" for backwards compatibility).
#
# All backends return the same LLMResponse dataclass so the rest of
# the codebase (chat.py, context.py, etc.) needs zero changes.
# ─────────────────────────────────────────────

import os
import yaml
from dataclasses import dataclass
from pathlib import Path


# ─────────────────────────────────────────────
# 1. Config loader
# ─────────────────────────────────────────────

def load_config(config_path: str = "config.yaml") -> dict:
    """
    Load configuration from config.yaml and expose flat convenience keys.
    Backwards-compatible: always sets LLM_MODEL, DB_PATH, MEMORIES_DIRECTORY.
    """
    path = Path(config_path)
    if not path.exists():
        return {
            "LLM_MODEL": "mistral:7b-instruct",
            "DB_PATH": "data/memory.db",
            "MEMORIES_DIRECTORY": "memories/",
        }

    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}

    llm_section     = (data.get("models") or {}).get("llm") or {}
    storage_section = data.get("storage") or {}

    data["LLM_MODEL"]          = llm_section.get("model", "mistral:7b-instruct")
    data["DB_PATH"]             = storage_section.get("sqlite_db_path", "data/memory.db")
    data["MEMORIES_DIRECTORY"] = storage_section.get("memories_directory", "memories/")

    return data


# ─────────────────────────────────────────────
# 2. LLMResponse dataclass
# ─────────────────────────────────────────────

@dataclass
class LLMResponse:
    """Unified response container for all backends."""
    text:          str
    model:         str
    input_tokens:  int
    output_tokens: int
    total_tokens:  int


# ─────────────────────────────────────────────
# 3. Backend implementations
# ─────────────────────────────────────────────

def _call_ollama(
    prompt: str,
    model: str,
    system: str | None,
) -> LLMResponse:
    """Call a local Ollama model."""
    try:
        import ollama as _ollama
    except ImportError:
        raise ImportError(
            "ollama package not installed. Run: pip install ollama"
        )

    # Resolve to the exact installed tag if needed
    resolved = resolve_model(model)
    if resolved:
        model = resolved

    if system:
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ]
    else:
        messages = [{"role": "user", "content": prompt}]

    try:
        response = _ollama.chat(model=model, messages=messages)
    except _ollama.ResponseError as e:
        if e.status_code == 404:
            raise ValueError(
                f"Model '{model}' not found in Ollama.\n"
                f"Pull it with: ollama pull {model}"
            ) from e
        raise
    except Exception as e:
        raise ConnectionError(
            "Cannot connect to Ollama. Is it running?\n"
            "Start it with: ollama serve"
        ) from e

    text          = response["message"]["content"]
    input_tokens  = response.get("prompt_eval_count", 0)
    output_tokens = response.get("eval_count", 0)

    return LLMResponse(
        text=text,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
    )


def _call_huggingface(
    prompt: str,
    model: str,
    system: str | None,
    max_new_tokens: int = 512,
    temperature: float = 0.3,
) -> LLMResponse:
    """
    Call a HuggingFace model via the transformers text-generation pipeline.
    The model is downloaded on first use and cached locally by HuggingFace.

    Requires: pip install transformers torch  (or transformers accelerate)
    """
    try:
        from transformers import pipeline as hf_pipeline
    except ImportError:
        raise ImportError(
            "transformers package not installed.\n"
            "Install it with: pip install transformers torch\n"
            "Or: uv add --optional huggingface transformers torch"
        )

    # Build a single string input — HF text-generation expects a prompt string
    full_prompt = f"{system}\n\n{prompt}" if system else prompt

    pipe = hf_pipeline(
        "text-generation",
        model=model,
        device_map="auto",
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
        pad_token_id=50256,  # EOS token fallback to suppress warnings
    )

    result = pipe(full_prompt)
    generated = result[0]["generated_text"]

    # Strip the prompt prefix that some models echo back
    if generated.startswith(full_prompt):
        generated = generated[len(full_prompt):].lstrip()

    # Approximate token counts (transformers doesn't always return usage)
    input_tokens  = len(full_prompt) // 4
    output_tokens = len(generated) // 4

    return LLMResponse(
        text=generated,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
    )


def _call_anthropic(
    prompt: str,
    model: str,
    system: str | None,
    max_tokens: int = 1024,
    temperature: float = 0.3,
) -> LLMResponse:
    """
    Call Anthropic's API (Claude models).
    Requires ANTHROPIC_API_KEY environment variable.

    Requires: pip install anthropic
    Or: uv add --optional anthropic anthropic
    """
    try:
        import anthropic as _anthropic
    except ImportError:
        raise ImportError(
            "anthropic package not installed.\n"
            "Install it with: pip install anthropic\n"
            "Or: uv add --optional anthropic anthropic"
        )

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY environment variable not set.\n"
            "Set it with: export ANTHROPIC_API_KEY=sk-ant-..."
        )

    client = _anthropic.Anthropic(api_key=api_key)

    kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        kwargs["system"] = system

    response = client.messages.create(**kwargs)

    text          = response.content[0].text
    input_tokens  = response.usage.input_tokens
    output_tokens = response.usage.output_tokens

    return LLMResponse(
        text=text,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
    )


# ─────────────────────────────────────────────
# 3b. Ollama streaming
# ─────────────────────────────────────────────

def call_ollama_stream(
    prompt: str,
    model: str,
    system: str | None = None,
):
    """Yield tokens from Ollama model as they arrive."""
    try:
        import ollama as _ollama
    except ImportError:
        raise ImportError("ollama package not installed. Run: pip install ollama")

    resolved = resolve_model(model)
    if resolved:
        model = resolved

    if system:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
    else:
        messages = [{"role": "user", "content": prompt}]

    try:
        stream = _ollama.chat(model=model, messages=messages, stream=True)
        for chunk in stream:
            if "message" in chunk and "content" in chunk["message"]:
                yield chunk["message"]["content"]
    except Exception as e:
        yield f"[Error: {e}]"


# ─────────────────────────────────────────────
# 4. Core dispatcher
# ─────────────────────────────────────────────

def call_llm(
    prompt: str,
    model: str | None = None,
    system: str | None = None,
    provider: str | None = None,
) -> LLMResponse:
    """
    Send a prompt to the configured LLM backend and return the response.

    Provider resolution order:
        1. `provider` argument (explicit override)
        2. config.yaml → models.llm.provider
        3. "ollama" (default)

    Model resolution order:
        1. `model` argument
        2. config.yaml → models.llm.model
        3. Backend-specific default

    Args:
        prompt   : complete prompt string (from context.py)
        model    : optional model override
        system   : optional system message
        provider : "ollama" | "huggingface" | "anthropic" | None

    Returns:
        LLMResponse with text, model name, and token counts
    """
    config = load_config()

    # Resolve provider
    if provider is None:
        llm_cfg  = (config.get("models") or {}).get("llm") or {}
        provider = llm_cfg.get("provider", "ollama")

    # Resolve model
    if model is None:
        model = config.get("LLM_MODEL", "mistral:7b-instruct")

    # Resolve generation params from config
    llm_cfg       = (config.get("models") or {}).get("llm") or {}
    temperature   = float(llm_cfg.get("temperature", 0.3))
    max_tokens    = int(llm_cfg.get("max_tokens", 512))

    provider = provider.lower().strip()

    if provider == "ollama":
        resp = _call_ollama(prompt, model, system)
    elif provider in ("huggingface", "hf", "transformers"):
        resp = _call_huggingface(
            prompt, model, system,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
    elif provider in ("anthropic", "claude"):
        resp = _call_anthropic(
            prompt, model, system,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    else:
        raise ValueError(
            f"Unknown LLM provider: '{provider}'. "
            "Supported: ollama | huggingface | anthropic"
        )

    print(f"[LLM] {resp.input_tokens}+{resp.output_tokens} tokens ({len(resp.text)} chars)")

    return resp


def call_llm_stream(
    prompt: str,
    model: str | None = None,
    system: str | None = None,
    provider: str | None = None,
):
    """
    Stream tokens from the configured LLM backend.
    Yields tokens as they arrive for real-time display.

    Currently only supports Ollama streaming.
    Other providers fall back to non-streaming.
    """
    config = load_config()

    if provider is None:
        llm_cfg = (config.get("models") or {}).get("llm") or {}
        provider = llm_cfg.get("provider", "ollama")

    if model is None:
        model = config.get("LLM_MODEL", "mistral:7b-instruct")

    provider = provider.lower().strip()

    if provider == "ollama":
        for token in call_ollama_stream(prompt, model, system):
            yield token
    else:
        # Non-streaming fallback - yield all at once
        resp = call_llm(prompt, model, system, provider)
        yield resp.text


# ─────────────────────────────────────────────
# 5. Ollama utilities (backwards-compatible)
# ─────────────────────────────────────────────

def resolve_model(model: str) -> str | None:
    """
    Resolve a requested Ollama model name against what is installed.
    Returns the exact installed tag, or None if nothing matches.
    Only relevant for the Ollama backend.
    """
    try:
        import ollama as _ollama
        models    = _ollama.list()
        available = [m.model for m in models.models]
    except Exception:
        return None

    if model in available:
        return model

    separators = ("-", ".", ":", "@", "_")

    def variant_prefix_match(a: str, b: str) -> bool:
        if not b.startswith(a):
            return False
        rest = b[len(a):]
        return rest == "" or rest[0] in separators

    for inst in available:
        if variant_prefix_match(model, inst):
            return inst
    for inst in available:
        if variant_prefix_match(inst, model):
            return inst

    req_base, _, req_var = model.partition(":")
    for inst in available:
        inst_base, _, inst_var = inst.partition(":")
        if req_base != inst_base:
            continue
        if req_var and inst_var and (
            variant_prefix_match(req_var, inst_var)
            or variant_prefix_match(inst_var, req_var)
        ):
            return inst

    return None


def is_model_available(model: str) -> bool:
    """True if `model` (or a derivative tag) is pulled in Ollama."""
    return resolve_model(model) is not None


def list_available_models() -> list[str]:
    """Return all models currently pulled in Ollama."""
    try:
        import ollama as _ollama
        models = _ollama.list()
        return [m.model for m in models.models]
    except Exception:
        return []
