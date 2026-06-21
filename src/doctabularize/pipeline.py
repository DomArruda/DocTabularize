"""
Main pipeline orchestration.
Supports PDFs (vision-based) + other formats (text-based extraction).
"""

import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List
from dataclasses import asdict

import numpy as np
import pandas as pd
import faiss

from .config import Config
from .ingest import ingest, embed, open_pdf
from .embedder import Embedder
from .cluster import clusterize, medoids
from .schema import discover_schema, load_config_schema
from .models import DiscoveredSchema, get_llm_client
from .verify import Verifier
from .extract import extract_page, extract_from_text
from .memory import SchemaMemory


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("DocTabularize")


def run():
    start = time.time()
    cfg = Config("pipeline_config.toml")

    # LLM client (supports OpenAI, Ollama, Gemini, etc.)
    try:
        client = get_llm_client(cfg)
        log.info("[LLM] Using provider: %s", getattr(client, 'provider', 'unknown'))
    except Exception as e:
        log.warning("[LLM] Connector init failed (%s), falling back to OpenAI-compatible", e)
        from openai import OpenAI
        base_url = cfg.get("pipeline.models.ollama_url", "http://localhost:11434/v1")
        client = OpenAI(base_url=base_url, api_key=cfg.get("pipeline.models.api_key", "ollama"))

    vision_model = cfg.get("pipeline.models.vision_model", "qwen2.5-vl:7b")
    text_model   = cfg.get("pipeline.models.text_model", vision_model)  # fallback to vision model if not set
    schema_model = cfg.get("pipeline.models.schema_model", "qwen2.5-vl:7b")
    json_mode    = cfg.get("pipeline.ollama_json_mode", "json")

    # One embedding backend, shared by the embedding step and the Verifier.
    # Embedder loads its local SentenceTransformer lazily, so nothing is pulled
    # from the Hugging Face Hub when an API endpoint is configured.
    embedder = Embedder(cfg.get("pipeline.models.embedding_model"), cfg)
    verifier = Verifier(cfg.get("pipeline.models.cross_encoder_model"), embedder, cfg)

    mem = SchemaMemory(cfg.get("pipeline.cache_path", "cache/schema_memory.json") or "cache/schema_memory.json")

    doc_path = cfg.get("pipeline.doc_path", "")
    if not doc_path:
        log.error("No doc_path configured in pipeline_config.toml")
        return

    # === INGEST (now supports PDF, txt, csv, docx, md, etc.) ===
    chunks = ingest(doc_path)
    if not chunks:
        log.error("No chunks extracted from %s. Exiting.", doc_path)
        return

    chunks = embed(chunks, embedder)

    # === CLUSTERING ===
    stats = clusterize(chunks, cfg)
    valid_cids = [c for c in stats if c != -1]
    if not valid_cids:
        log.error("No valid clusters found. Adjust HDBSCAN parameters.")
        return

    samples_map = medoids(chunks, cfg.cluster_cfg(-1)["samples_per_cluster"])

    # === SCHEMA DISCOVERY PER CLUSTER ===
    schemas: Dict[int, DiscoveredSchema] = {}
    for cid in valid_cids:
        ccfg = cfg.cluster_cfg(cid)
        if not ccfg.get("enabled", True):
            log.info("[CLUSTER %d] Disabled in config, skipping.", cid)
            continue

        predefined = cfg.schemas_for_cluster(cid)
        if predefined:
            log.info("[CLUSTER %d] Using predefined schema from config.", cid)
            merged = {}
            for p in predefined:
                merged.update(p.get("fields", {}))
            schemas[cid] = DiscoveredSchema("config", "From TOML config", merged, "config")
        else:
            fewshot = mem.get_fewshot(cid, cfg.get("job.schema_evolution_top_k", 3)) if cfg.get("job.enable_schema_evolution") else ""
            schemas[cid] = discover_schema(cid, samples_map.get(cid, []), cfg, client)

    # === EXTRACTION ===
    output = {"clusters": {}}
    is_pdf = str(doc_path).lower().endswith(".pdf")

    if is_pdf:
        # ==================== PDF VISION PATH ====================
        with open_pdf(doc_path) as doc:
            for cid, schema in schemas.items():
                ccfg = cfg.cluster_cfg(cid)
                cstart = time.time()
                log.info("\n[CLUSTER %d] %s — %d fields (VISION)", cid, schema.granularity, len(schema.fields))

                cid_chunks = [c for c in chunks if c.cluster_id == cid]
                pages = sorted(set(c.page_num for c in cid_chunks))
                extracted = []
                total_rows = 0

                # Optional fact verification setup
                faiss_idx, chunk_map = None, None
                if ccfg.get("enable_fact_verification"):
                    ve = [c for c in cid_chunks if c.embedding is not None]
                    if ve:
                        em = np.vstack([c.embedding for c in ve]).astype("float32")
                        faiss_idx = faiss.IndexFlatL2(em.shape[1])
                        faiss_idx.add(em)
                        chunk_map = ve

                for pnum in pages:
                    page_text = next((c.text for c in cid_chunks if c.page_num == pnum), "")
                    result = extract_page(
                        doc=doc,
                        page_num=pnum,
                        schema=schema,
                        cluster_cfg=ccfg,
                        page_text=page_text,
                        client=client,
                        verifier=verifier,
                        vision_model=vision_model,
                        json_mode=json_mode,
                    )
                    extracted.append(result)
                    total_rows += sum(len(t.get("rows", [])) for t in result.tables)

                    if result.score < 0:
                        log.warning("[PAGE %d] Failed all retries.", pnum)

                # Fact verification (PDF path)
                verified = 0
                if ccfg.get("enable_fact_verification") and faiss_idx is not None:
                    log.info("[CLUSTER %d] Fact-checking %d rows...", cid, total_rows)
                    for ex in extracted:
                        for t in ex.tables:
                            for r in t.get("rows", []):
                                ok, conf, ev = verifier.fact_check(r, chunk_map, client, schema_model)
                                r["fact_verified"] = ok
                                r["fact_confidence"] = round(conf, 4)
                                r["source_evidence"] = ev[:300]
                                if ok:
                                    verified += 1
                    log.info("[CLUSTER %d] Verified %d/%d rows.", cid, verified, total_rows)

                scores = [e.score for e in extracted if e.score >= 0]
                avg_score = round(float(np.mean(scores)), 2) if scores else 0.0

                output["clusters"][str(cid)] = {
                    "schema": asdict(schema),
                    "pages_processed": len(extracted),
                    "avg_score": avg_score,
                    "total_rows": total_rows,
                    "verified_rows": verified,
                    "processing_time": round(time.time() - cstart, 1),
                    "extractions": [asdict(e) for e in extracted]
                }

                if cfg.get("job.enable_schema_evolution"):
                    mem.register(cid, schema, avg_score, total_rows)
    else:
        # ==================== TEXT / OTHER FORMATS PATH ====================
        for cid, schema in schemas.items():
            ccfg = cfg.cluster_cfg(cid)
            cstart = time.time()
            log.info("\n[CLUSTER %d] %s — %d fields (TEXT)", cid, schema.granularity, len(schema.fields))

            cid_chunks = [c for c in chunks if c.cluster_id == cid]
            extracted = []
            total_rows = 0

            for chunk in cid_chunks:
                result = extract_from_text(
                    text=chunk.text,
                    schema=schema,
                    cluster_cfg=ccfg,
                    client=client,
                    verifier=verifier,
                    model=text_model,
                    json_mode=json_mode,
                    source_id=f"chunk_{chunk.chunk_id}",
                )
                extracted.append(result)
                total_rows += sum(len(t.get("rows", [])) for t in result.tables)

            scores = [e.score for e in extracted if e.score >= 0]
            avg_score = round(float(np.mean(scores)), 2) if scores else 0.0

            output["clusters"][str(cid)] = {
                "schema": asdict(schema),
                "chunks_processed": len(extracted),
                "avg_score": avg_score,
                "total_rows": total_rows,
                "processing_time": round(time.time() - cstart, 1),
                "extractions": [asdict(e) for e in extracted]
            }

            if cfg.get("job.enable_schema_evolution"):
                mem.register(cid, schema, avg_score, total_rows)

    # === CROSS-CLUSTER RECONCILIATION ===
    log.info("\n[CROSS] Running cross-cluster reconciliation...")
    try:
        all_rows = {}
        for cid, info in output["clusters"].items():
            flat = []
            for ex in info.get("extractions", []):
                for t in ex.get("tables", []):
                    flat.extend(t.get("rows", []))
            if flat:
                all_rows[f"cluster_{cid}"] = flat

        for name, rows in all_rows.items():
            verifier.db.register(name, pd.DataFrame(rows))

        output["cross_cluster_views"] = list(all_rows.keys())
        log.info("[CROSS] Reconciled %d views.", len(all_rows))
    except Exception as e:
        log.error("[CROSS] Reconciliation failed: %s", e)
        output["cross_cluster_views"] = []

    # === FINAL OUTPUT ===
    output["pipeline_summary"] = {
        "total_time": round(time.time() - start, 1),
        "clusters": len(schemas),
        "input_file": str(doc_path),
        "file_type": "pdf" if is_pdf else "text/other",
        "output_path": cfg.get("pipeline.output_path")
    }

    out_path = Path(cfg.get("pipeline.output_path", "output.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    log.info("\n[DONE] Pipeline finished in %.1fs → %s", time.time() - start, out_path)


if __name__ == "__main__":
    try:
        run()
    except FileNotFoundError as e:
        log.critical("File not found: %s", e)
        sys.exit(1)
    except KeyboardInterrupt:
        log.warning("Interrupted by user.")
        sys.exit(130)
    except Exception as e:
        log.critical("Unexpected error: %s", e)
        traceback.print_exc()
        sys.exit(1)
