"""
Genetic schema memory for evolution.
"""

import json
import time
from pathlib import Path
from typing import Dict

from .models import DiscoveredSchema

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