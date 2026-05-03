"""
Tests for core/llm.py multi-backend support.

Uses unittest.mock to isolate each backend — no real Ollama, HuggingFace,
or Anthropic connections required.
"""

import os
import sys
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# Ensure project root is on the path when running with pytest directly
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from core.llm import (
    LLMResponse,
    call_llm,
    load_config,
    resolve_model,
    is_model_available,
    list_available_models,
)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _make_ollama_response(text="hello", input_tok=10, output_tok=5):
    resp = MagicMock()
    resp.__getitem__ = lambda self, key: {
        "message": {"content": text},
        "prompt_eval_count": input_tok,
        "eval_count": output_tok,
    }[key]
    resp.get = lambda key, default=None: {
        "prompt_eval_count": input_tok,
        "eval_count": output_tok,
    }.get(key, default)
    return resp


# ─────────────────────────────────────────────
# LLMResponse dataclass
# ─────────────────────────────────────────────

class TestLLMResponse:
    def test_fields(self):
        r = LLMResponse(text="hi", model="m", input_tokens=1, output_tokens=2, total_tokens=3)
        assert r.text == "hi"
        assert r.model == "m"
        assert r.total_tokens == 3


# ─────────────────────────────────────────────
# load_config
# ─────────────────────────────────────────────

class TestLoadConfig:
    def test_missing_file_returns_defaults(self, tmp_path):
        cfg = load_config(str(tmp_path / "nonexistent.yaml"))
        assert cfg["LLM_MODEL"] == "mistral:7b-instruct"
        assert "DB_PATH" in cfg
        assert "MEMORIES_DIRECTORY" in cfg

    def test_reads_nested_llm_model(self, tmp_path):
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(
            "models:\n  llm:\n    model: gpt4all\nstorage:\n  sqlite_db_path: data/test.db\n"
        )
        cfg = load_config(str(yaml_file))
        assert cfg["LLM_MODEL"] == "gpt4all"
        assert cfg["DB_PATH"] == "data/test.db"


# ─────────────────────────────────────────────
# Ollama backend
# ─────────────────────────────────────────────

class TestOllamaBackend:
    @patch("core.llm.resolve_model", return_value="mistral:7b-instruct")
    @patch("core.llm.load_config")
    def test_call_llm_ollama_success(self, mock_cfg, mock_resolve):
        mock_cfg.return_value = {
            "LLM_MODEL": "mistral:7b-instruct",
            "models": {"llm": {"provider": "ollama", "temperature": 0.3, "max_tokens": 512}},
        }

        ollama_mod = MagicMock()
        fake_resp = {
            "message": {"content": "I am Mistral."},
            "prompt_eval_count": 20,
            "eval_count": 8,
        }
        ollama_mod.chat.return_value = fake_resp

        with patch.dict("sys.modules", {"ollama": ollama_mod}):
            # Re-import inside patch so _call_ollama picks up the mock
            import importlib
            import core.llm as llm_mod
            importlib.reload(llm_mod)

            result = llm_mod.call_llm("hello", provider="ollama")

        assert result.text == "I am Mistral."
        assert result.input_tokens == 20
        assert result.output_tokens == 8
        assert result.total_tokens == 28

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            call_llm("test", provider="unknown_backend")


# ─────────────────────────────────────────────
# HuggingFace backend
# ─────────────────────────────────────────────

class TestHuggingFaceBackend:
    def test_missing_transformers_raises_import_error(self):
        """When transformers is not installed, a clear ImportError is raised."""
        with patch.dict("sys.modules", {"transformers": None}):
            import importlib
            import core.llm as llm_mod
            importlib.reload(llm_mod)

            with pytest.raises((ImportError, Exception)):
                llm_mod._call_huggingface("prompt", "gpt2", None)

    def test_call_huggingface_success(self):
        fake_pipeline_output = [{"generated_text": "The answer is 42."}]
        mock_pipe_instance = MagicMock(return_value=fake_pipeline_output)

        transformers_mod = MagicMock()
        transformers_mod.pipeline.return_value = mock_pipe_instance

        with patch.dict("sys.modules", {"transformers": transformers_mod}):
            import importlib
            import core.llm as llm_mod
            importlib.reload(llm_mod)

            result = llm_mod._call_huggingface("What is 6x7?", "gpt2", None)

        assert result.text == "The answer is 42."
        assert result.model == "gpt2"
        assert result.input_tokens > 0
        assert result.output_tokens > 0

    def test_prompt_prefix_stripped(self):
        prompt = "My prompt: "
        generated = "My prompt: The answer."
        fake_pipeline_output = [{"generated_text": generated}]
        mock_pipe_instance = MagicMock(return_value=fake_pipeline_output)

        transformers_mod = MagicMock()
        transformers_mod.pipeline.return_value = mock_pipe_instance

        with patch.dict("sys.modules", {"transformers": transformers_mod}):
            import importlib
            import core.llm as llm_mod
            importlib.reload(llm_mod)

            result = llm_mod._call_huggingface(prompt, "gpt2", None)

        assert result.text == "The answer."

    def test_system_prepended_to_prompt(self):
        captured = {}

        def fake_pipe_factory(task, model, **kwargs):
            def run(full_prompt):
                captured["prompt"] = full_prompt
                return [{"generated_text": full_prompt + " response"}]
            return run

        transformers_mod = MagicMock()
        transformers_mod.pipeline.side_effect = fake_pipe_factory

        with patch.dict("sys.modules", {"transformers": transformers_mod}):
            import importlib
            import core.llm as llm_mod
            importlib.reload(llm_mod)

            llm_mod._call_huggingface("user message", "gpt2", "be helpful")

        assert captured["prompt"].startswith("be helpful")
        assert "user message" in captured["prompt"]


# ─────────────────────────────────────────────
# Anthropic backend
# ─────────────────────────────────────────────

class TestAnthropicBackend:
    def test_missing_api_key_raises(self):
        anthropic_mod = MagicMock()
        with patch.dict("sys.modules", {"anthropic": anthropic_mod}):
            with patch.dict(os.environ, {}, clear=True):
                os.environ.pop("ANTHROPIC_API_KEY", None)

                import importlib
                import core.llm as llm_mod
                importlib.reload(llm_mod)

                with pytest.raises(EnvironmentError, match="ANTHROPIC_API_KEY"):
                    llm_mod._call_anthropic("hello", "claude-haiku-4-5", None)

    def test_call_anthropic_success(self):
        mock_content = MagicMock()
        mock_content.text = "I am Claude."
        mock_usage = MagicMock()
        mock_usage.input_tokens = 15
        mock_usage.output_tokens = 6
        mock_message = MagicMock()
        mock_message.content = [mock_content]
        mock_message.usage = mock_usage

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_message

        anthropic_mod = MagicMock()
        anthropic_mod.Anthropic.return_value = mock_client

        with patch.dict("sys.modules", {"anthropic": anthropic_mod}):
            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}):
                import importlib
                import core.llm as llm_mod
                importlib.reload(llm_mod)

                result = llm_mod._call_anthropic("hello", "claude-haiku-4-5", None)

        assert result.text == "I am Claude."
        assert result.model == "claude-haiku-4-5"
        assert result.input_tokens == 15
        assert result.output_tokens == 6
        assert result.total_tokens == 21

    def test_system_passed_to_anthropic(self):
        mock_content = MagicMock()
        mock_content.text = "reply"
        mock_usage = MagicMock()
        mock_usage.input_tokens = 5
        mock_usage.output_tokens = 2
        mock_message = MagicMock()
        mock_message.content = [mock_content]
        mock_message.usage = mock_usage

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_message

        anthropic_mod = MagicMock()
        anthropic_mod.Anthropic.return_value = mock_client

        with patch.dict("sys.modules", {"anthropic": anthropic_mod}):
            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}):
                import importlib
                import core.llm as llm_mod
                importlib.reload(llm_mod)

                llm_mod._call_anthropic("hello", "claude-haiku-4-5", "be concise")

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs.get("system") == "be concise"

    def test_missing_anthropic_package_raises(self):
        with patch.dict("sys.modules", {"anthropic": None}):
            import importlib
            import core.llm as llm_mod
            importlib.reload(llm_mod)

            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}):
                with pytest.raises((ImportError, Exception)):
                    llm_mod._call_anthropic("hello", "claude-haiku-4-5", None)


# ─────────────────────────────────────────────
# Ollama utilities
# ─────────────────────────────────────────────

class TestOllamaUtilities:
    def _make_ollama_list(self, models: list[str]):
        mod = MagicMock()
        model_objs = [MagicMock(model=m) for m in models]
        mod.list.return_value = MagicMock(models=model_objs)
        return mod

    def test_resolve_exact_match(self):
        ollama_mod = self._make_ollama_list(["mistral:7b-instruct"])
        with patch.dict("sys.modules", {"ollama": ollama_mod}):
            import importlib
            import core.llm as llm_mod
            importlib.reload(llm_mod)
            assert llm_mod.resolve_model("mistral:7b-instruct") == "mistral:7b-instruct"

    def test_resolve_prefix_match(self):
        ollama_mod = self._make_ollama_list(["mistral:7b-instruct-v0.3-q4_0"])
        with patch.dict("sys.modules", {"ollama": ollama_mod}):
            import importlib
            import core.llm as llm_mod
            importlib.reload(llm_mod)
            assert llm_mod.resolve_model("mistral:7b-instruct") == "mistral:7b-instruct-v0.3-q4_0"

    def test_resolve_no_match_returns_none(self):
        ollama_mod = self._make_ollama_list(["llama3:8b"])
        with patch.dict("sys.modules", {"ollama": ollama_mod}):
            import importlib
            import core.llm as llm_mod
            importlib.reload(llm_mod)
            assert llm_mod.resolve_model("mistral:7b-instruct") is None

    def test_is_model_available_true(self):
        ollama_mod = self._make_ollama_list(["llama3:8b"])
        with patch.dict("sys.modules", {"ollama": ollama_mod}):
            import importlib
            import core.llm as llm_mod
            importlib.reload(llm_mod)
            assert llm_mod.is_model_available("llama3:8b") is True

    def test_is_model_available_false(self):
        ollama_mod = self._make_ollama_list(["llama3:8b"])
        with patch.dict("sys.modules", {"ollama": ollama_mod}):
            import importlib
            import core.llm as llm_mod
            importlib.reload(llm_mod)
            assert llm_mod.is_model_available("mistral:7b") is False

    def test_list_available_models(self):
        ollama_mod = self._make_ollama_list(["llama3:8b", "mistral:7b"])
        with patch.dict("sys.modules", {"ollama": ollama_mod}):
            import importlib
            import core.llm as llm_mod
            importlib.reload(llm_mod)
            result = llm_mod.list_available_models()
            assert "llama3:8b" in result
            assert "mistral:7b" in result

    def test_list_available_models_empty_on_error(self):
        ollama_mod = MagicMock()
        ollama_mod.list.side_effect = Exception("no server")
        with patch.dict("sys.modules", {"ollama": ollama_mod}):
            import importlib
            import core.llm as llm_mod
            importlib.reload(llm_mod)
            assert llm_mod.list_available_models() == []
