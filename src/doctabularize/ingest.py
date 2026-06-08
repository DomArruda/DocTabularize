import logging
from pathlib import Path
from typing import List
import fitz
import numpy as np
from sentence_transformers import SentenceTransformer
from contextlib import contextmanager

from .models import Chunk

log = logging.getLogger("Refinery")


@contextmanager
def open_pdf(path: str):
    """Context manager for PDF."""
    doc = fitz.open(path)
    try:
        yield doc
    finally:
        doc.close()


def ingest(path: str) -> List[Chunk]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix.lower() == ".pdf":
        with open_pdf(path) as doc:
            chunks = []
            for i in range(len(doc)):
                text = doc[i].get_text("text").strip()
                if len(text) >= 15:
                    chunks.append(Chunk(len(chunks), i, text, str(path)))
            log.info("[INGEST] %d pages from %s", len(chunks), p.name)
            return chunks
    else:
        words = p.read_text(encoding="utf-8").split()
        chunks = [Chunk(i, i, " ".join(words[i:i+400]), str(path))
                  for i in range(0, len(words), 400)]
        log.info("[INGEST] %d text chunks from %s", len(chunks), p.name)
        return chunks


def embed(chunks: List[Chunk], model_name: str) -> List[Chunk]:
    log.info("[EMBED] Loading %s...", model_name)
    model = SentenceTransformer(model_name)
    embs = model.encode([c.text for c in chunks], batch_size=32, show_progress_bar=False)
    for c, e in zip(chunks, embs):
        c.embedding = e
    log.info("[EMBED] Done (%d vectors, dim=%d)", len(embs), embs.shape[1])
    return chunks