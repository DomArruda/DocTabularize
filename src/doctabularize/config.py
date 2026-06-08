"""
Configuration management for Document Intelligence Refinery.
"""

import tomllib
from pathlib import Path
from typing import Dict, Any, List
import logging

log = logging.getLogger("Refinery")

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
        """Write a minimal default config."""
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