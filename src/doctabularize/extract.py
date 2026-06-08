"""
Vision-based page extraction with target-score awareness.
"""

import base64
import json
import logging
import time
from typing import List, Dict, Any

import fitz
from openai import OpenAI, APITimeoutError, APIConnectionError

from .models import ExtractedPage, DiscoveredSchema
from .verify import robust_json_parse, extract_tables_from_text
from .config import Config

log = logging.getLogger("Refinery")


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
    verifier,
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