"""
Schema discovery and loading.
"""

from __future__ import annotations

import json
import logging
import re
from typing import List, Optional, Dict, Any

from openai import OpenAI

from .models import DiscoveredSchema, Chunk
from .config import Config

log = logging.getLogger("Refinery")


def _strip_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1] if "```" in text[3:] else text
        if text.startswith("json"):
            text = text[4:]
    return text.strip()


def discover_schema(cid: int, samples: List[Chunk], cfg: Config, client: OpenAI) -> DiscoveredSchema:
    from .models import Chunk  # avoid circular if needed
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