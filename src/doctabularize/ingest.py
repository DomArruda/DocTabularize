import logging
from pathlib import Path
from typing import List, Dict, Any
import fitz
import numpy as np
from sentence_transformers import SentenceTransformer
from contextlib import contextmanager
from dataclasses import asdict

from langchain_community.document_loaders import (
    TextLoader,
    CSVLoader,
    Docx2txtLoader,
    UnstructuredHTMLLoader,
    JSONLoader,
)
from langchain_core.documents import Document

from .models import Chunk
from openai import OpenAI
from .config import Config
cfg = Config("pipeline_config.toml")

base_url = cfg.get("pipeline.models.ollama_url", "http://localhost:11434/v1")
embedding_client = OpenAI(base_url=base_url, api_key=cfg.get("pipeline.models.api_key", "ollama"))


log = logging.getLogger("DocTabularize")


@contextmanager
def open_pdf(path: str):
    doc = fitz.open(path)
    try:
        yield doc
    finally:
        doc.close()


def _pdf_to_chunks(path: str) -> List[Chunk]:
    p = Path(path)
    chunks = []
    with open_pdf(path) as doc:
        for i in range(len(doc)):
            text = doc[i].get_text("text").strip()
            if len(text) >= 15:
                chunks.append(Chunk(
                    chunk_id=len(chunks),
                    page_num=i,
                    text=text,
                    source=str(path)
                ))
    log.info("[INGEST] %d pages from %s", len(chunks), p.name)
    return chunks


def _langchain_to_chunks(docs: List[Document], source: str) -> List[Chunk]:
    """Convert LangChain Documents → your Chunk objects."""
    chunks = []
    for i, doc in enumerate(docs):
        # Try to preserve page number if the loader provided it
        page_num = doc.metadata.get("page", 0)
        if isinstance(page_num, str):
            page_num = int(page_num) if page_num.isdigit() else 0

        chunks.append(Chunk(
            chunk_id=i,
            page_num=page_num,
            text=doc.page_content.strip(),
            source=source,
            # You can later extend Chunk to accept metadata=doc.metadata
        ))
    return chunks


def ingest(path: str) -> List[Chunk]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    ext = p.suffix.lower()

    if ext == ".pdf":
        return _pdf_to_chunks(str(p))

    elif ext in {".txt", ".md", ".markdown"}:
        loader = TextLoader(str(p), encoding="utf-8")
        docs = loader.load()
        return _langchain_to_chunks(docs, str(p))

    elif ext == ".csv":
        # CSVLoader makes one Document per row — often very useful
        loader = CSVLoader(str(p))
        docs = loader.load()
        return _langchain_to_chunks(docs, str(p))

    elif ext == ".docx":
        loader = Docx2txtLoader(str(p))
        docs = loader.load()
        return _langchain_to_chunks(docs, str(p))

    elif ext in {".html", ".htm"}:
        loader = UnstructuredHTMLLoader(str(p))
        docs = loader.load()
        return _langchain_to_chunks(docs, str(p))

    elif ext == ".json":
        # You can customize the jq schema if needed
        loader = JSONLoader(str(p), jq_schema=".", text_content=False)
        docs = loader.load()
        return _langchain_to_chunks(docs, str(p))

    else:
        # Fallback: try to read as plain text
        log.warning("[INGEST] Unknown extension %s — treating as plain text", ext)
        text = p.read_text(encoding="utf-8", errors="ignore")
        # Simple fallback chunking (you can improve this later)
        words = text.split()
        chunks = [
            Chunk(i, 0, " ".join(words[i:i+400]), str(p))
            for i in range(0, len(words), 400)
        ]
        return chunks

def embed(chunks: List[Chunk], model_name: str) -> List[Chunk]:
    log.info("[EMBED] Loading %s...", model_name)
    
    # Get API configuration from config
    base_url = cfg.get("pipeline.models.base_url") or cfg.get("pipeline.models.ollama_url")
    api_key = cfg.get("pipeline.models.api_key", "ollama")
    
    # If we have a cloud base_url (like DeepInfra), use the API. 
    # Otherwise, fall back to local SentenceTransformer.
    if base_url and "localhost" not in base_url and "127.0.0.1" not in base_url:
        try:
            client = OpenAI(base_url=base_url, api_key=api_key)
            log.info("[EMBED] Using API endpoint: %s", base_url)
            
            texts = [c.text for c in chunks]
            all_embeddings = []
            batch_size = 32 # Keep batches small to avoid API rate limits/payload errors
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                log.info("[EMBED] Processing batch %d/%d...", i//batch_size + 1, (len(texts) - 1)//batch_size + 1)
                
                # Call the OpenAI-compatible embeddings endpoint
                response = client.embeddings.create(
                    model=model_name,
                    input=batch,
                    encoding_format="float" # As requested in your curl example
                )
                
                # Extract embeddings from the response
                batch_embs = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embs)
            
            # Convert list of lists to a single numpy array (matching local model output format)
            embs = np.array(all_embeddings)
            
            for c, e in zip(chunks, embs):
                c.embedding = e
                
            log.info("[EMBED] Done (%d vectors, dim=%d)", len(embs), embs.shape[1])
            return chunks
            
        except Exception as e:
            log.error("[EMBED] API call failed: %s. Falling back to local model.", e)
    
    # FALLBACK: Local SentenceTransformer execution
    try:
        from sentence_transformers import SentenceTransformer
        log.info("[EMBED] Using local SentenceTransformer.")
        model = SentenceTransformer(model_name)
        embs = model.encode([c.text for c in chunks], batch_size=32, show_progress_bar=False)
        for c, e in zip(chunks, embs):
            c.embedding = e
        log.info("[EMBED] Done (%d vectors, dim=%d)", len(embs), embs.shape[1])
        return chunks
    except ImportError:
        log.error("[EMBED] sentence-transformers not installed and no API configured!")
