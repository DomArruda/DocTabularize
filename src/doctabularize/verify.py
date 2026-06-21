"""
Verification and fact-checking with per-row metadata.

Embedding is delegated to embedder.Embedder (API-or-local, lazy local model).
The Verifier no longer constructs or holds any embedding model itself.

Pair scoring (_score_pairs) tries three backends, in order:
  1. A dedicated reranker endpoint (DeepInfra-style POST /v1/inference/<model>,
     configured via pipeline.models.rerank_url). A true cross-encoder; scores
     come back already in [0, 1], so the 0.5 threshold in fact_check is valid
     as-is. Preferred path.
  2. Embedding cosine via the Embedder. A bi-encoder *approximation* of a
     cross-encoder; the score scale differs from sigmoid(logit), so the 0.5
     threshold may need recalibration in this mode.
  3. Local CrossEncoder (sentence-transformers), loaded lazily.
"""

import json
import logging
import math
import re
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import faiss
import pandas as pd
import duckdb
from openai import OpenAI

from .models import Chunk
from .config import Config
from .embedder import Embedder

log = logging.getLogger("Refinery")


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid (the local cross-encoder emits raw logits)."""
    x = float(x)
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def _cosine_to_score(a: np.ndarray, b: np.ndarray) -> float:
    """
    Map cosine similarity into [0, 1] as a relevance proxy for the cosine path.
    Negatives clamp to 0; this is *not* calibrated like sigmoid(cross_logit),
    so thresholds tuned against the local cross-encoder may need adjusting.
    """
    a = np.asarray(a, dtype="float32")
    b = np.asarray(b, dtype="float32")
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    cos = float(np.dot(a, b) / denom)
    return max(0.0, min(1.0, cos))


def _rerank_api(
    url: str,
    api_key: str,
    pairs: List[Tuple[str, str]],
    instruction: Optional[str] = None,
    batch_size: int = 32,
) -> List[float]:
    """
    Score (evidence, claim) pairs via a DeepInfra-style reranker endpoint:
        POST <url>  {"queries": [...], "documents": [...]}  ->  {"scores": [...]}

    queries/documents are aligned positionally (queries[i] scored against
    documents[i]), which maps one pair to one score. We send query = claim
    (the short thing being verified) and document = evidence (the passage).
    Scores come back already in [0, 1], so no sigmoid is applied — only a
    defensive clamp. Pairs are chunked to keep payloads small (page_text is
    repeated per row in score_with_metadata).
    """
    import httpx

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    scores: List[float] = []
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]
        payload: Dict[str, Any] = {
            "queries": [claim for _evidence, claim in batch],
            "documents": [evidence for evidence, _claim in batch],
        }
        if instruction:
            payload["instruction"] = instruction
        resp = httpx.post(url, json=payload, headers=headers, timeout=60.0)
        resp.raise_for_status()
        batch_scores = resp.json().get("scores", [])
        if len(batch_scores) != len(batch):
            raise ValueError(
                f"rerank returned {len(batch_scores)} scores for {len(batch)} pairs"
            )
        scores.extend(max(0.0, min(1.0, float(s))) for s in batch_scores)
    return scores


def extract_tables_from_text(raw_text: str, schema) -> List[Dict]:  # schema: DiscoveredSchema
    from difflib import get_close_matches
    schema_keys = list(schema.fields.keys())
    tables = []
    lines = raw_text.split('\n')

    # Markdown tables
    in_table = False
    table_lines = []
    for line in lines:
        if '|' in line and not line.strip().startswith('#'):
            in_table = True
            table_lines.append(line)
        elif in_table and table_lines:
            headers = [h.strip().lower().replace(' ', '_') for h in table_lines[0].split('|') if h.strip()]
            mapped = []
            for h in headers:
                m = get_close_matches(h, schema_keys, n=1, cutoff=0.6)
                mapped.append(m[0] if m else h)
            rows = []
            for tl in table_lines[2:]:
                cells = [c.strip() for c in tl.split('|') if c.strip()]
                if cells:
                    rows.append(dict(zip(mapped, cells)))
            if rows:
                tables.append({"table_name": "extracted", "rows": rows})
            in_table = False
            table_lines = []

    # Key-value pairs
    if not tables:
        kvs = {}
        for line in lines:
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    raw_key, val = parts[0].strip().lower().replace(' ', '_'), parts[1].strip()
                    m = get_close_matches(raw_key, schema_keys, n=1, cutoff=0.6)
                    key = m[0] if m else raw_key
                    if val and key in schema_keys:
                        kvs[key] = val
        if kvs:
            tables.append({"table_name": "key_values", "rows": [kvs]})

    return tables


class Verifier:
    def __init__(
        self,
        cross_encoder_model: str,
        embedder: Embedder,
        cfg: Optional[Config] = None,
    ):
        # Cross-encoder is loaded lazily, only when we fall back to the local path.
        self.cross_model_name = cross_encoder_model
        self._cross = None  # type: ignore[var-annotated]

        # Embedding is fully delegated to the shared Embedder.
        self.embedder = embedder
        self.cfg = cfg or Config("pipeline_config.toml")

        self.db = duckdb.connect(":memory:")

        if self.cfg.get("pipeline.models.rerank_url"):
            log.info("[VERIFY] Using reranker endpoint for pair scoring.")
        elif getattr(self.embedder, "use_api", False):
            log.info("[VERIFY] Using API embedding cosine for pair scoring.")
        else:
            log.info("[VERIFY] Local mode — cross-encoder loads on first use.")

    def _local_cross(self):
        """Lazily load the local cross-encoder (only on the local scoring path)."""
        if self._cross is None:
            from sentence_transformers.cross_encoder import CrossEncoder
            log.info("[VERIFY] Loading local cross-encoder %s ...", self.cross_model_name)
            self._cross = CrossEncoder(self.cross_model_name)
        return self._cross

    def _score_pairs(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """
        Score (evidence, claim) relevance in [0, 1] for each pair.

        Backend order:
          1. Reranker endpoint (pipeline.models.rerank_url) — true cross-encoder.
          2. Embedding cosine via the Embedder — deduped before embedding.
          3. Local CrossEncoder — logits through a sigmoid.
        """
        if not pairs:
            return []
        pairs = list(pairs)

        # 1. Dedicated reranker endpoint (preferred — faithful cross-encoder).
        rerank_url = self.cfg.get("pipeline.models.rerank_url")
        if rerank_url:
            api_key = self.cfg.get("pipeline.models.rerank_api_key") or self.cfg.get("pipeline.models.api_key", "ollama")
            instruction = self.cfg.get("pipeline.models.rerank_instruction")
            try:
                return _rerank_api(rerank_url, api_key, pairs, instruction)
            except Exception as e:
                log.error("[VERIFY] Rerank API failed: %s. Falling back.", e)

        # 2. Embedding cosine via the shared Embedder.
        if getattr(self.embedder, "use_api", False):
            try:
                # Dedupe before embedding — in score_with_metadata the left side
                # (page_text) repeats across every row.
                uniq: Dict[str, Optional[np.ndarray]] = {}
                for a, b in pairs:
                    uniq.setdefault(a, None)
                    uniq.setdefault(b, None)
                texts = list(uniq.keys())
                embs = self.embedder.encode(texts)
                for t, e in zip(texts, embs):
                    uniq[t] = e
                return [_cosine_to_score(uniq[a], uniq[b]) for a, b in pairs]
            except Exception as e:
                log.error("[VERIFY] Cosine scoring failed: %s. Falling back to local cross-encoder.", e)

        # 3. Local cross-encoder.
        cross = self._local_cross()
        logits = cross.predict(pairs)
        return [_sigmoid(l) for l in logits]

    def score_with_metadata(self, tables: List[Dict], schema, page_text: str) -> Tuple[float, str, List[Dict]]:
        """
        Score extraction and attach per-row confidence metadata.
        """
        expected = set(schema.fields.keys())
        all_rows = [r for t in tables for r in t.get("rows", [])]

        if not all_rows:
            return (100.0, "Empty page — valid.", tables) if not tables else (0.0, "Tables with no rows.", tables)

        # Per-row semantic fidelity — batched into a single scoring call.
        sents = [
            " ".join([f"The {k} is {v}." for k, v in r.items()
                      if v and str(v).lower() not in ("null", "none", "")])
            for r in all_rows
        ]
        pairs = [(page_text, s) for s in sents if s]
        pair_scores = iter(self._score_pairs(pairs))
        row_scores = [next(pair_scores) if s else 0.0 for s in sents]

        fidelity = (np.mean(row_scores) * 50.0) if row_scores else 0.0

        # Per-row structural density
        row_structural = []
        for r in all_rows:
            populated = sum(1 for k in expected if k in r and r[k] not in (None, "", "null"))
            density = populated / max(1, len(expected))
            row_structural.append(density)
        structural = np.mean(row_structural) * 30.0 if row_structural else 0.0

        # SQL compliance
        sql_ok = 0
        for t in tables:
            if t.get("rows"):
                try:
                    self.db.register("tmp", pd.DataFrame(t["rows"]))
                    self.db.sql("SELECT * FROM tmp").fetchall()
                    sql_ok += 1
                except Exception:
                    pass
        sql_score = (sql_ok / len(tables)) * 20.0 if tables else 0.0

        overall = fidelity + structural + sql_score
        feedback = f"fidelity={fidelity:.1f} struct={structural:.1f} sql={sql_score:.1f}"

        # Attach metadata to each row and table
        tables_with_meta = []
        row_idx = 0
        for t in tables:
            table_rows = []
            table_fidelities = []
            table_structurals = []
            for r in t.get("rows", []):
                r_copy = dict(r)
                r_copy["_meta"] = {
                    "semantic_confidence": round(row_scores[row_idx], 4),
                    "structural_density": round(row_structural[row_idx], 4),
                    "row_score": round((row_scores[row_idx] * 50.0) + (row_structural[row_idx] * 30.0), 2)
                }
                table_rows.append(r_copy)
                table_fidelities.append(row_scores[row_idx])
                table_structurals.append(row_structural[row_idx])
                row_idx += 1

            tables_with_meta.append({
                "table_name": t.get("table_name", "unknown"),
                "rows": table_rows,
                "_meta": {
                    "table_confidence": round(np.mean(table_fidelities) if table_fidelities else 0.0, 4),
                    "avg_structural_density": round(np.mean(table_structurals) if table_structurals else 0.0, 4),
                    "n_rows": len(table_rows),
                    "schema_coverage": f"{sum(1 for k in expected if any(k in r for r in table_rows))}/{len(expected)}"
                }
            })

        return max(0.0, min(100.0, overall)), feedback, tables_with_meta

    def fact_check(self, row: Dict, chunks: List[Chunk], client: OpenAI, model: str) -> Tuple[bool, float, str]:
        try:
            facts = json.dumps({k: v for k, v in row.items() if not k.startswith("_")}, indent=2)
            prompt = f"Given this data row, generate one concise question it answers.\n\n{facts}\n\nQuestion:"
            q = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                timeout=15,
            ).choices[0].message.content.strip()

            # Index only chunks that actually have embeddings, and search against
            # that same list so the returned id maps back correctly.
            valid = [c for c in chunks if c.embedding is not None]
            if not valid:
                return False, 0.0, "no embedded chunks"

            q_emb = self.embedder.encode([q])[0]
            embs = np.vstack([c.embedding for c in valid]).astype("float32")
            idx = faiss.IndexFlatL2(embs.shape[1])
            idx.add(embs)
            _, ids = idx.search(np.asarray(q_emb, dtype="float32").reshape(1, -1), 1)
            evidence = valid[ids[0][0]].text[:600]

            sent = " ".join([f"The {k} is {v}." for k, v in row.items()
                             if v and not k.startswith("_") and str(v).lower() not in ("null", "none", "")])
            conf = self._score_pairs([(evidence, sent)])[0] if sent else 0.0
            return conf >= 0.5, float(conf), evidence
        except Exception as e:
            log.debug("Fact-check error: %s", e)
            return False, 0.0, str(e)[:100]


def robust_json_parse(text: str) -> Optional[Dict]:
    from .schema import _strip_json
    text = _strip_json(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        import json_repair
        repaired = json_repair.repair_json(text)
        return json.loads(repaired) if isinstance(repaired, str) else repaired
    except Exception:
        pass
    try:
        matches = re.findall(r'\{.*\}', text, re.DOTALL)
        if matches:
            for candidate in sorted(matches, key=len, reverse=True):
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass
    try:
        lines = text.strip().split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('{'):
                for j in range(i+1, len(lines)+1):
                    blob = '\n'.join(lines[i:j])
                    try:
                        return json.loads(blob)
                    except json.JSONDecodeError:
                        continue
    except Exception:
        pass
    return None
