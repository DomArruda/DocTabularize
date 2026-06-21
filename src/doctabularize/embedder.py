"""
embedder.py — one embedding backend shared by ingest and verify.

Encapsulates the "use the OpenAI-compatible API if a cloud endpoint is
configured, otherwise fall back to a local SentenceTransformer" logic that
used to be duplicated in ingest.embed() and verify.Verifier.

The local SentenceTransformer is loaded lazily: it is only constructed if the
API path is unavailable or fails. So an API-only deployment never downloads a
model from the Hugging Face Hub and never imports torch.
"""

import logging
from typing import List, Optional, Any

import numpy as np
from openai import OpenAI

from .config import Config

log = logging.getLogger("DocTabularize")


class Embedder:
    """
    Embed text via an OpenAI-compatible /v1/embeddings endpoint, or locally.

    Parameters
    ----------
    model_name:
        Embedding model id. For the API path this is the served model name
        (e.g. "Qwen/Qwen3-Embedding-0.6B"); for the local path it's whatever
        SentenceTransformer can load.
    cfg:
        Config instance. Defaults to Config("pipeline_config.toml").
    batch_size:
        Texts per request / encode batch.
    local_model:
        Optional preloaded SentenceTransformer. If given, the local path uses
        it instead of constructing one (useful for fully-local runs). Leave
        None to keep the load lazy.
    """

    def __init__(
        self,
        model_name: str,
        cfg: Optional[Config] = None,
        batch_size: int = 32,
        local_model: Optional[Any] = None,
    ):
        self.model_name = model_name
        self.cfg = cfg or Config("pipeline_config.toml")
        self.batch_size = batch_size
        self._local = local_model
        self._client: Optional[OpenAI] = None

        self.base_url = self.cfg.get("pipeline.models.base_url") or self.cfg.get("pipeline.models.ollama_url")
        self.api_key = self.cfg.get("pipeline.models.api_key", "ollama")
        # Treat localhost endpoints as "local": use them only as the LLM/embeddings
        # server when explicitly cloud; otherwise prefer the local model path.
        self.use_api = bool(self.base_url) and "localhost" not in self.base_url and "127.0.0.1" not in self.base_url

    # ------------------------------------------------------------------ #
    # Backends
    # ------------------------------------------------------------------ #
    def _api_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        return self._client

    def _local_model(self):
        if self._local is None:
            if not self.model_name:
                raise RuntimeError(
                    "[EMBED] No embedding model configured "
                    "(pipeline.models.embedding_model) and no local model passed."
                )
            from sentence_transformers import SentenceTransformer
            log.info("[EMBED] Loading local SentenceTransformer %s ...", self.model_name)
            self._local = SentenceTransformer(self.model_name)
        return self._local

    def _encode_api(self, texts: List[str]) -> np.ndarray:
        client = self._api_client()
        log.info("[EMBED] Using API endpoint: %s", self.base_url)
        out: List[List[float]] = []
        n_batches = (len(texts) - 1) // self.batch_size + 1
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            log.info("[EMBED] Processing batch %d/%d...", i // self.batch_size + 1, n_batches)
            resp = client.embeddings.create(
                model=self.model_name,
                input=batch,
                encoding_format="float",
            )
            out.extend(item.embedding for item in resp.data)
        return np.array(out, dtype="float32")

    def _encode_local(self, texts: List[str]) -> np.ndarray:
        model = self._local_model()
        log.info("[EMBED] Using local SentenceTransformer.")
        embs = model.encode(texts, batch_size=self.batch_size, show_progress_bar=False)
        return np.asarray(embs, dtype="float32")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def encode(self, texts: List[str]) -> np.ndarray:
        """Return a (n, dim) float32 array of embeddings for `texts`."""
        if not texts:
            return np.empty((0, 0), dtype="float32")

        if self.use_api:
            try:
                return self._encode_api(texts)
            except Exception as e:
                log.error("[EMBED] API call failed: %s. Falling back to local model.", e)

        return self._encode_local(texts)

    def embed_chunks(self, chunks: List[Any]) -> List[Any]:
        """Embed each chunk's `.text` in place and return the same list."""
        embs = self.encode([c.text for c in chunks])
        for c, e in zip(chunks, embs):
            c.embedding = e
        if len(embs):
            log.info("[EMBED] Done (%d vectors, dim=%d)", len(embs), embs.shape[1])
        return chunks
