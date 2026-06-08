"""
Verification and fact-checking with per-row metadata.
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
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from openai import OpenAI

from .models import Chunk

log = logging.getLogger("Refinery")


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
    def __init__(self, model_name: str, embedder: SentenceTransformer):
        self.cross = CrossEncoder(model_name)
        self.embedder = embedder
        self.db = duckdb.connect(":memory:")
        log.info("[VERIFY] Cross-encoder loaded.")

    def score_with_metadata(self, tables: List[Dict], schema, page_text: str) -> Tuple[float, str, List[Dict]]:
        """
        Score extraction and attach per-row confidence metadata.
        """
        expected = set(schema.fields.keys())
        all_rows = [r for t in tables for r in t.get("rows", [])]

        if not all_rows:
            return (100.0, "Empty page — valid.", tables) if not tables else (0.0, "Tables with no rows.", tables)

        # Per-row semantic fidelity scores
        row_scores = []
        for r in all_rows:
            sent = " ".join([f"The {k} is {v}." for k, v in r.items() if v and str(v).lower() not in ("null", "none", "")])
            if sent:
                try:
                    logit = self.cross.predict([(page_text, sent)])[0]
                    prob = 1 / (1 + math.exp(-logit))
                    row_scores.append(prob)
                except Exception:
                    row_scores.append(0.0)
            else:
                row_scores.append(0.0)

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
            q = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}], temperature=0.0, timeout=15).choices[0].message.content.strip()

            q_emb = self.embedder.encode([q])[0]
            embs = np.vstack([c.embedding for c in chunks if c.embedding is not None]).astype("float32")
            idx = faiss.IndexFlatL2(embs.shape[1])
            idx.add(embs)
            _, ids = idx.search(q_emb.reshape(1, -1), 1)
            evidence = chunks[ids[0][0]].text[:600]

            sent = " ".join([f"The {k} is {v}." for k, v in row.items() if v and not k.startswith("_") and str(v).lower() not in ("null", "none", "")])
            logit = self.cross.predict([(evidence, sent)])[0]
            conf = 1 / (1 + math.exp(-logit))
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