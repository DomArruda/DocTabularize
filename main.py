"""
Document Intelligence Refinery — v2.3
Target-score-aware extraction + per-row confidence metadata
"""

import os, io, json, math, base64, logging, traceback, sys, time, re
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Any, Dict, List, Tuple
from contextlib import contextmanager
from difflib import get_close_matches

import numpy as np
import faiss
import umap
import hdbscan
import duckdb
import fitz
import pandas as pd
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from openai import OpenAI, APIError, APITimeoutError, APIConnectionError

try:
    import json_repair
    HAS_JSON_REPAIR = True
except ImportError:
    HAS_JSON_REPAIR = False

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("Refinery")


# ==============================================================================
# CONFIG
# ==============================================================================

class Config:
    def __init__(self, path: str = "pipeline_config.toml"):
        p = Path(path)
        if not p.exists():
            self._write_default(p)
        with open(p, "rb") as f:
            self.raw = tomllib.load(f)
        self.flat = self._flatten(self.raw)
        log.info("[CONFIG] Loaded from %s", path)

    def get(self, key: str, default=None):
        return self.flat.get(key, default)

    def _flatten(self, d, parent=""):
        out = {}
        for k, v in d.items():
            nk = f"{parent}.{k}" if parent else k
            if isinstance(v, dict):
                out.update(self._flatten(v, nk))
            else:
                out[nk] = v
        return out

    def _write_default(self, p: Path):
        p.write_text("""[pipeline]
doc_path = "input/document.pdf"
output_path = "output/extraction.json"
ollama_json_mode = "json"

[pipeline.models]
embedding_model = "all-MiniLM-L6-v2"
cross_encoder_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
ollama_url = "http://localhost:11434/v1"
schema_model = "qwen2.5-vl:7b"
vision_model = "qwen2.5-vl:7b"

[job]
target_score = 85.0
max_page_retries = 3
enable_fact_verification = false
enable_schema_evolution = true

[job.extraction]
target_features = 8
extract_dpi = 150
max_rows_per_cluster = 100

[job.clustering]
umap_components = 5
umap_neighbors = 15
umap_min_dist = 0.1
hdbscan_min_cluster = 3
hdbscan_min_samples = 2
samples_per_cluster = 3
""", encoding="utf-8")

    def cluster_cfg(self, cid: int) -> Dict[str, Any]:
        job = self.raw.get("job", {})
        clusters = {c["id"]: c for c in self.raw.get("cluster", [])}
        overrides = clusters.get(cid, {})
        effective = {
            "target_score": job.get("target_score", 85.0),
            "max_page_retries": job.get("max_page_retries", 3),
            "extract_dpi": job.get("extraction", {}).get("extract_dpi", 150),
            "enable_fact_verification": job.get("enable_fact_verification", False),
            "target_features": job.get("extraction", {}).get("target_features", 8),
            "max_rows_per_cluster": job.get("extraction", {}).get("max_rows_per_cluster", 100),
            "hdbscan_min_cluster": job.get("clustering", {}).get("hdbscan_min_cluster", 3),
            "hdbscan_min_samples": job.get("clustering", {}).get("hdbscan_min_samples", 2),
            "samples_per_cluster": job.get("clustering", {}).get("samples_per_cluster", 3),
            "umap_components": job.get("clustering", {}).get("umap_components", 5),
            "umap_neighbors": job.get("clustering", {}).get("umap_neighbors", 15),
            "umap_min_dist": job.get("clustering", {}).get("umap_min_dist", 0.1),
            "custom_prompt_injection": "",
            "enabled": True,
        }
        for k, v in overrides.items():
            if k not in ("id", "name"):
                effective[k] = v
        return effective

    def schemas_for_cluster(self, cid: int) -> List[Dict[str, Any]]:
        return [s for s in self.raw.get("schema", []) if s.get("cluster_id") == cid and s.get("enabled", True)]


# ==============================================================================
# DATA MODELS
# ==============================================================================

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


# ==============================================================================
# INGESTION + CLUSTERING
# ==============================================================================

@contextmanager
def open_pdf(path: str):
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
                    chunks.append(Chunk(len(chunks), i, text, path))
            log.info("[INGEST] %d pages from %s", len(chunks), p.name)
            return chunks
    else:
        words = p.read_text(encoding="utf-8").split()
        chunks = [Chunk(i, i, " ".join(words[i:i+400]), path)
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


def clusterize(chunks: List[Chunk], cfg: Config) -> Dict[int, int]:
    ccfg = cfg.cluster_cfg(-1)
    log.info("[UMAP] Reducing to %d dims...", ccfg["umap_components"])
    mat = np.vstack([c.embedding for c in chunks])
    reducer = umap.UMAP(
        n_components=ccfg["umap_components"],
        n_neighbors=min(ccfg["umap_neighbors"], len(chunks)-1),
        min_dist=ccfg["umap_min_dist"],
        metric="cosine", random_state=42
    )
    coords = reducer.fit_transform(mat)
    for c, co in zip(chunks, coords):
        c.umap_coords = co

    log.info("[HDBSCAN] Clustering (min_cluster=%d, min_samples=%d)...",
             ccfg["hdbscan_min_cluster"], ccfg["hdbscan_min_samples"])
    labels = hdbscan.HDBSCAN(
        min_cluster_size=ccfg["hdbscan_min_cluster"],
        min_samples=ccfg["hdbscan_min_samples"],
        metric="euclidean", cluster_selection_method="eom"
    ).fit_predict(np.vstack([c.umap_coords for c in chunks]))

    stats = {}
    for c, lab in zip(chunks, labels):
        c.cluster_id = int(lab)
        stats[int(lab)] = stats.get(int(lab), 0) + 1
    valid = {k: v for k, v in stats.items() if k != -1}
    log.info("[HDBSCAN] %d clusters, %d noise", len(valid), stats.get(-1, 0))
    return stats


def medoids(chunks: List[Chunk], samples_per_cluster: int) -> Dict[int, List[Chunk]]:
    out = {}
    for cid in sorted(set(c.cluster_id for c in chunks if c.cluster_id != -1)):
        members = [c for c in chunks if c.cluster_id == cid]
        coords = np.vstack([c.umap_coords for c in members]).astype("float32")
        idx = faiss.IndexFlatL2(coords.shape[1])
        idx.add(coords)
        _, ids = idx.search(coords.mean(axis=0, keepdims=True), min(samples_per_cluster, len(members)))
        out[cid] = [members[i] for i in ids[0]]
    return out


# ==============================================================================
# SCHEMA DISCOVERY
# ==============================================================================

def discover_schema(cid: int, samples: List[Chunk], cfg: Config, client: OpenAI) -> DiscoveredSchema:
    sample_text = "\n\n---\n\n".join(s.text[:800] for s in samples)
    model = cfg.get("pipeline.models.schema_model", "qwen2.5-vl:7b")
    target = cfg.cluster_cfg(cid).get("target_features", 8)

    err_feedback = ""
    for attempt in range(1, 4):
        prompt = f"""Design a flat JSON schema for tabular data found in these document samples.

SAMPLES:
{sample_text}

TARGET: Extract roughly {target} fields.
Output ONLY valid JSON matching this exact shape:
{{"granularity": "medium", "rationale": "...", "fields": {{"field_name": "type | description"}}}}
{err_feedback}"""

        try:
            resp = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}],
                temperature=0.0, timeout=30
            )
            raw = _strip_json(resp.choices[0].message.content)
            data = json.loads(raw)
            return DiscoveredSchema(
                granularity=data.get("granularity", "medium"),
                rationale=data.get("rationale", ""),
                fields=data.get("fields", {}),
                source="discovered"
            )
        except (json.JSONDecodeError, KeyError) as e:
            log.warning("  [SCHEMA] Parse fail attempt %d: %s", attempt, str(e)[:80])
            err_feedback = f"\n\nPREVIOUS ERROR: {str(e)[:120]}. Output ONLY valid JSON."
        except Exception as e:
            log.error("  [SCHEMA] Error attempt %d: %s", attempt, e)
            break

    log.error("[SCHEMA] All discovery attempts failed for cluster %d, using fallback.", cid)
    return DiscoveredSchema("coarse", "Fallback after discovery failure", {"raw_text": "string"}, "fallback")


def load_config_schema(schema_def: Dict[str, Any]) -> DiscoveredSchema:
    fields = {}
    for name, fdef in schema_def.get("fields", {}).items():
        desc = fdef.get("type", "string")
        if fdef.get("description"):
            desc += f" | {fdef['description']}"
        if fdef.get("validation"):
            desc += f" | validation:{fdef['validation']}"
        if fdef.get("enum"):
            desc += f" | enum:{','.join(fdef['enum'])}"
        fields[name] = desc
    return DiscoveredSchema(
        granularity="config",
        rationale=f"Loaded from config schema '{schema_def.get('name')}'",
        fields=fields,
        source="config"
    )


# ==============================================================================
# JSON REPAIR
# ==============================================================================

def robust_json_parse(text: str) -> Optional[Dict]:
    text = _strip_json(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    if HAS_JSON_REPAIR:
        try:
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


def extract_tables_from_text(raw_text: str, schema: DiscoveredSchema) -> List[Dict]:
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


# ==============================================================================
# VERIFICATION — with per-row confidence metadata
# ==============================================================================

class Verifier:
    def __init__(self, model_name: str, embedder: SentenceTransformer):
        self.cross = CrossEncoder(model_name)
        self.embedder = embedder
        self.db = duckdb.connect(":memory:")
        log.info("[VERIFY] Cross-encoder loaded.")

    def score_with_metadata(self, tables: List[Dict], schema: DiscoveredSchema, page_text: str) -> Tuple[float, str, List[Dict]]:
        """
        Score extraction and attach per-row confidence metadata.

        Returns: (overall_score, feedback, tables_with_metadata)
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


# ==============================================================================
# VISION EXTRACTION — target-score aware
# ==============================================================================

def page_to_b64(doc: fitz.Document, page_num: int, dpi: int) -> str:
    page = doc[page_num]
    pix = page.get_pixmap(dpi=dpi)
    img_bytes = pix.tobytes("png")
    return base64.b64encode(img_bytes).decode()


def _schema_prompt_block(schema: DiscoveredSchema) -> str:
    lines = ["Extract tables with these exact fields:"]
    for name, desc in schema.fields.items():
        lines.append(f'  "{name}": {desc}')
    return "\n".join(lines)


def extract_page(
    doc: fitz.Document,
    page_num: int,
    schema: DiscoveredSchema,
    cluster_cfg: Dict[str, Any],
    page_text: str,
    client: OpenAI,
    verifier: Verifier,
    vision_model: str,
    json_mode: str,
) -> ExtractedPage:
    img = page_to_b64(doc, page_num, cluster_cfg["extract_dpi"])
    schema_hint = _schema_prompt_block(schema)
    target_score = cluster_cfg.get("target_score", 85.0)
    best_score, best_result, best_feedback = -1.0, None, ""

    for attempt in range(1, cluster_cfg["max_page_retries"] + 1):
        try:
            injections = []
            if cluster_cfg.get("custom_prompt_injection"):
                injections.append(cluster_cfg["custom_prompt_injection"])

            # TARGET-SCORE-AWARE PROMPT
            system = f"""You are a data extraction assistant. Look at the page image and extract tables.

{schema_hint}

TARGET QUALITY SCORE: {target_score}/100
Your extraction will be scored on:
- Semantic fidelity (do the extracted values match the image?)
- Structural density (are all required fields present?)
- SQL compliance (is the output well-formed?)

RULES:
1. Output ONLY valid JSON with shape: {{"reasoning": "...", "tables": [{{"table_name": "...", "rows": [{{"field_name": "value", ...}}]}}]}}
2. If a field is missing or unreadable, use null — do NOT hallucinate values.
3. If no tables exist, output: {{"reasoning": "No tables found", "tables": []}}
4. Be precise. A low-confidence correct extraction is better than a high-confidence wrong one.

{chr(10).join(injections)}"""

            kwargs = {
                "model": vision_model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}},
                        {"type": "text", "text": "Extract all tables from this page."}
                    ]}
                ],
                "temperature": 0.1,
                "timeout": 60
            }
            if json_mode == "json_schema":
                kwargs["extra_body"] = {
                    "format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "extraction",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "reasoning": {"type": "string"},
                                    "tables": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "table_name": {"type": "string"},
                                                "rows": {"type": "array", "items": {"type": "object"}}
                                            },
                                            "required": ["table_name", "rows"]
                                        }
                                    }
                                },
                                "required": ["reasoning", "tables"]
                            }
                        }
                    }
                }
            elif json_mode == "json":
                kwargs["response_format"] = {"type": "json_object"}

            resp = client.chat.completions.create(**kwargs)
            raw_text = resp.choices[0].message.content

            data = robust_json_parse(raw_text)
            if data is None:
                log.warning("  [PAGE %d] All JSON parsing failed, attempting text extraction...", page_num)
                tables = extract_tables_from_text(raw_text, schema)
                if tables:
                    data = {"reasoning": "Recovered from malformed JSON", "tables": tables}
                else:
                    raise json.JSONDecodeError("Could not parse or recover JSON", raw_text, 0)

            tables = data.get("tables", [])
            if not isinstance(tables, list):
                tables = []

            # Score with per-row metadata
            score, feedback, tables_with_meta = verifier.score_with_metadata(tables, schema, page_text)
            log.info("  [PAGE %d] Attempt %d: score=%.1f (%s)", page_num, attempt, score, feedback)

            if score > best_score:
                best_score, best_result, best_feedback = score, tables_with_meta, feedback
            if score >= target_score:
                break

        except (APITimeoutError, APIConnectionError) as e:
            log.warning("  [PAGE %d] API error attempt %d: %s", page_num, attempt, e)
            time.sleep(2 * attempt)
        except Exception as e:
            log.error("  [PAGE %d] Unexpected attempt %d: %s", page_num, attempt, e)

    return ExtractedPage(page_num, best_score, best_result or [], best_feedback)


# ==============================================================================
# GENETIC MEMORY
# ==============================================================================

class SchemaMemory:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {}

    def save(self):
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    def get_fewshot(self, cid: int, top_k: int = 3) -> str:
        key = f"cluster_{cid}"
        if key not in self.data:
            return ""
        schemas = sorted(self.data[key], key=lambda x: x.get("score", 0), reverse=True)[:top_k]
        if not schemas:
            return ""
        out = "\n\nHISTORICALLY SUCCESSFUL SCHEMAS:\n"
        for i, s in enumerate(schemas):
            out += f"\nSchema {i+1} (score {s.get('score', 0):.1f}):\n{json.dumps(s.get('fields', {}), indent=2)}\n"
        return out

    def register(self, cid: int, schema: DiscoveredSchema, score: float, rows: int):
        key = f"cluster_{cid}"
        if key not in self.data:
            self.data[key] = []
        self.data[key].append({
            "timestamp": time.time(),
            "score": score,
            "rows": rows,
            "fields": schema.fields,
            "granularity": schema.granularity,
            "rationale": schema.rationale
        })
        self.save()


# ==============================================================================
# UTILS
# ==============================================================================

def _strip_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1] if "```" in text[3:] else text
        if text.startswith("json"):
            text = text[4:]
    return text.strip()


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def run():
    start = time.time()
    cfg = Config("pipeline_config.toml")

    client = OpenAI(base_url=cfg.get("pipeline.models.ollama_url"), api_key="ollama")
    vision_model = cfg.get("pipeline.models.vision_model", "qwen2.5-vl:7b")
    schema_model = cfg.get("pipeline.models.schema_model", "qwen2.5-vl:7b")
    json_mode = cfg.get("pipeline.ollama_json_mode", "json")

    embedder = SentenceTransformer(cfg.get("pipeline.models.embedding_model", "all-MiniLM-L6-v2"))
    verifier = Verifier(cfg.get("pipeline.models.cross_encoder_model"), embedder)

    mem = SchemaMemory(cfg.get("pipeline.cache_path", "cache/schema_memory.json") or "cache/schema_memory.json")

    chunks = ingest(cfg.get("pipeline.doc_path", ""))
    if not chunks:
        log.error("No chunks extracted. Exiting.")
        return

    chunks = embed(chunks, cfg.get("pipeline.models.embedding_model"))
    stats = clusterize(chunks, cfg)
    valid_cids = [c for c in stats if c != -1]
    if not valid_cids:
        log.error("No valid clusters. Adjust hdbscan_min_cluster.")
        return

    samples_map = medoids(chunks, cfg.cluster_cfg(-1)["samples_per_cluster"])

    schemas: Dict[int, DiscoveredSchema] = {}
    for cid in valid_cids:
        ccfg = cfg.cluster_cfg(cid)
        if not ccfg.get("enabled", True):
            log.info("[CLUSTER %d] Disabled, skipping.", cid)
            continue

        predefined = cfg.schemas_for_cluster(cid)
        if predefined:
            log.info("[CLUSTER %d] Using %d predefined schema(s).", cid, len(predefined))
            merged = {}
            for p in predefined:
                merged.update(p.get("fields", {}))
            schemas[cid] = DiscoveredSchema("config", "From TOML config", merged, "config")
        else:
            fewshot = mem.get_fewshot(cid, cfg.get("job.schema_evolution_top_k", 3)) if cfg.get("job.enable_schema_evolution") else ""
            if fewshot:
                log.info("[CLUSTER %d] Using genetic memory for schema discovery.", cid)
            schemas[cid] = discover_schema(cid, samples_map[cid], cfg, client)

    output = {"clusters": {}}
    with open_pdf(cfg.get("pipeline.doc_path")) as doc:
        for cid, schema in schemas.items():
            ccfg = cfg.cluster_cfg(cid)
            cstart = time.time()
            log.info("\n[CLUSTER %d] %s — %d fields", cid, schema.granularity, len(schema.fields))

            cid_chunks = [c for c in chunks if c.cluster_id == cid]
            pages = sorted(set(c.page_num for c in cid_chunks))
            extracted: List[ExtractedPage] = []
            total_rows = 0

            faiss_idx = None
            chunk_map = None
            if ccfg.get("enable_fact_verification"):
                ve = [c for c in cid_chunks if c.embedding is not None]
                if ve:
                    em = np.vstack([c.embedding for c in ve]).astype("float32")
                    faiss_idx = faiss.IndexFlatL2(em.shape[1])
                    faiss_idx.add(em)
                    chunk_map = ve

            for pnum in pages:
                page_text = next((c.text for c in cid_chunks if c.page_num == pnum), "")
                result = extract_page(doc, pnum, schema, ccfg, page_text, client, verifier, vision_model, json_mode)
                extracted.append(result)
                total_rows += sum(len(t.get("rows", [])) for t in result.tables)

                if result.score < 0:
                    log.warning("[PAGE %d] Failed all retries.", pnum)

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

    log.info("\n[CROSS] Running cross-cluster reconciliation...")
    try:
        all_rows = {}
        for cid, info in output["clusters"].items():
            flat = [r for ex in info["extractions"] for t in ex["tables"] for r in t.get("rows", [])]
            if flat:
                all_rows[f"cluster_{cid}"] = flat
        for name, rows in all_rows.items():
            verifier.db.register(name, pd.DataFrame(rows))
        output["cross_cluster_views"] = list(all_rows.keys())
        log.info("[CROSS] Reconciled %d views.", len(all_rows))
    except Exception as e:
        log.error("[CROSS] Failed: %s", e)
        output["cross_cluster_views"] = []

    output["pipeline_summary"] = {
        "total_time": round(time.time() - start, 1),
        "clusters": len(schemas),
        "output_path": cfg.get("pipeline.output_path")
    }

    out_path = Path(cfg.get("pipeline.output_path", "output.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("\n[DONE] Output written to %s (%.1fs)", out_path, time.time() - start)


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
        log.critical("Fatal: %s", e)
        traceback.print_exc()
        sys.exit(1)