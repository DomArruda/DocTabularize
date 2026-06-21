"""
Data models for Document Intelligence Refinery.
"""

from dataclasses import dataclass, field
from typing import Optional, Any, Dict, List
import numpy as np
from .config import Config


@dataclass
class Chunk:
    chunk_id: int
    page_num: int
    text: str
    source: str
    embedding: Optional[np.ndarray] = None
    umap_coords: Optional[np.ndarray] = None
    cluster_id: Optional[int] = None


@dataclass
class DiscoveredSchema:
    granularity: str
    rationale: str
    fields: Dict[str, str]
    source: str = "discovered"


@dataclass
class ExtractedPage:
    page_num: int
    score: float
    tables: List[Dict[str, Any]]
    feedback: str
    source_type: str = field(default="pdf")   # "pdf" or "text"

# ==============================================================================
# LLM CONNECTORS for OpenAI, Gemini, and OpenAI-compatible APIs (Ollama, vLLM, etc.)
# ==============================================================================

import os
import logging
from typing import Any, Dict, List, Optional

log = logging.getLogger("Refinery")

try:
    from openai import OpenAI, APIError
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import google.genai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False


class GeminiCompletions:
    """Adapter to make Gemini behave like OpenAI chat.completions for text calls."""

    def __init__(self, model_name: str, api_key: Optional[str] = None):
        if not HAS_GEMINI:
            raise ImportError("google-generativeai not installed. pip install google-generativeai")
        if api_key:
            genai.configure(api_key=api_key)
        self.model_name = model_name
        self._model = genai.GenerativeModel(model_name)

    def create(self, model: str = None, messages: List[Dict] = None, temperature: float = 0.0, timeout: int = 30, **kwargs):
        # Convert OpenAI messages format to Gemini
        prompt_parts = []
        for m in messages or []:
            role = m.get("role", "user")
            content = m.get("content", "")
            if isinstance(content, list):  # vision etc, but basic text for now
                content = " ".join(str(c.get("text", c)) for c in content if isinstance(c, dict))
            if role == "system":
                prompt_parts.append(f"System: {content}")
            else:
                prompt_parts.append(content)
        prompt = "\n".join(prompt_parts)

        try:
            resp = self._model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=4096,
                ),
            )
            # Fake OpenAI response structure
            class FakeChoice:
                def __init__(self, text):
                    self.message = type("obj", (object,), {"content": text})()
            class FakeResp:
                def __init__(self, text):
                    self.choices = [FakeChoice(text)]
            text = resp.text if hasattr(resp, "text") else str(resp)
            return FakeResp(text)
        except Exception as e:
            log.error("Gemini generation error: %s", e)
            raise


class LLMClient:
    """
    Unified client factory supporting:
    - openai (or any OpenAI-compatible: ollama, vllm, lmstudio, etc.)
    - gemini (text only; vision via image_url not fully mapped here - use litellm or OpenAI proxy for vision+gemini)
    """

    def __init__(self, provider: str = "openai", model: str = None, api_key: str = None, base_url: str = None, **kwargs):
        self.provider = provider.lower()
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY")

        if self.provider in ("openai", "ollama", "vllm", "lmstudio", "compatible"):
            if not HAS_OPENAI:
                raise ImportError("openai package required")
            self._client = OpenAI(
                base_url=base_url or os.getenv("OPENAI_BASE_URL"),
                api_key=self.api_key or "ollama" if self.provider == "ollama" else self.api_key,
            )
            self.chat = self._client.chat
        elif self.provider == "gemini":
            if not HAS_GEMINI:
                raise ImportError("google-generativeai package required. pip install google-generativeai")
            self._gemini_completions = GeminiCompletions(model or "gemini-1.5-flash", api_key=self.api_key)
            # Provide .chat.completions.create interface
            class _Chat:
                def __init__(self, comp):
                    self.completions = comp
            self.chat = _Chat(self._gemini_completions)
        else:
            # Fallback to OpenAI compatible
            if not HAS_OPENAI:
                raise ImportError("openai package required")
            self._client = OpenAI(base_url=base_url, api_key=self.api_key or "sk-xxx")
            self.chat = self._client.chat
            log.warning("Unknown provider %s, falling back to OpenAI client", provider)

    def chat_completions_create(self, **kwargs):
        """Direct access if needed"""
        return self.chat.completions.create(**kwargs)


def get_llm_client(cfg: Config) -> LLMClient:  # forward ref to avoid circular
    """Factory based on config values."""
    from .config import Config  # local import
    provider = cfg.get("pipeline.models.llm_provider", "ollama")
    model = cfg.get("pipeline.models.vision_model") or cfg.get("pipeline.models.schema_model")
    api_key = cfg.get("pipeline.models.api_key")
    base_url = cfg.get("pipeline.models.ollama_url") or cfg.get("pipeline.models.base_url")
    return LLMClient(provider=provider, model=model, api_key=api_key, base_url=base_url)
