"""
Vision-based (PDF) + Text-based extraction with target-score awareness.
Supports both PDF pages (with images) and other document types (text-only).
"""

import base64
import json
import logging
import time
from typing import List, Dict, Any, Optional

import fitz
from openai import OpenAI, APITimeoutError, APIConnectionError

from .models import ExtractedPage, DiscoveredSchema
from .verify import robust_json_parse, extract_tables_from_text

log = logging.getLogger("DocTabularize")


# ============================================================
# SHARED HELPERS
# ============================================================

def _schema_prompt_block(schema: DiscoveredSchema) -> str:
    lines = ["Extract tables with these exact fields:"]
    for name, desc in schema.fields.items():
        lines.append(f'  "{name}": {desc}')
    return "\n".join(lines)


def _build_json_schema() -> Dict[str, Any]:
    """Reusable JSON schema for structured output."""
    return {
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


# ============================================================
# PDF VISION PATH (kept from your original implementation)
# ============================================================

def page_to_b64(doc: fitz.Document, page_num: int, dpi: int) -> str:
    """Render a PDF page to base64 PNG."""
    page = doc[page_num]
    pix = page.get_pixmap(dpi=dpi)
    img_bytes = pix.tobytes("png")
    return base64.b64encode(img_bytes).decode()


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
    """
    Vision-based table extraction for a single PDF page.
    Uses image + text + target score awareness with retries.
    """
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
                            "schema": _build_json_schema()
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


# ============================================================
# NEW: TEXT-ONLY EXTRACTION PATH (for .txt, .csv, .md, .docx, etc.)
# ============================================================

def extract_from_text(
    text: str,
    schema: DiscoveredSchema,
    cluster_cfg: Dict[str, Any],
    client: OpenAI,
    verifier,
    model: str,
    json_mode: str,
    source_id: str = "text",
) -> ExtractedPage:
    """
    Text-based table extraction for non-PDF documents.
    Cheaper, faster, and often more accurate than vision for text formats.
    """
    schema_hint = _schema_prompt_block(schema)
    target_score = cluster_cfg.get("target_score", 85.0)
    max_retries = cluster_cfg.get("max_text_retries", 3)
    best_score, best_result, best_feedback = -1.0, None, ""

    # Truncate very long text to stay within context
    text_for_prompt = text[:15000] if len(text) > 15000 else text

    for attempt in range(1, max_retries + 1):
        try:
            injections = []
            if cluster_cfg.get("custom_prompt_injection"):
                injections.append(cluster_cfg["custom_prompt_injection"])

            system = f"""You are a data extraction assistant. Extract tables from the provided text.

{schema_hint}

TARGET QUALITY SCORE: {target_score}/100
Score based on semantic fidelity, structural completeness, and valid JSON.

RULES:
1. Output ONLY valid JSON: {{"reasoning": "...", "tables": [{{"table_name": "...", "rows": [...]}}]}}
2. Use null for missing/unreadable fields. Never hallucinate.
3. If no tables exist, return: {{"reasoning": "No tables found", "tables": []}}
4. Be precise and conservative.

{chr(10).join(injections)}"""

            user_content = f"Extract all tables from this document:\n\n{text_for_prompt}"

            kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_content}
                ],
                "temperature": 0.1,
            }

            if json_mode == "json_schema":
                kwargs["extra_body"] = {
                    "format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "extraction",
                            "schema": _build_json_schema()
                        }
                    }
                }
            elif json_mode == "json":
                kwargs["response_format"] = {"type": "json_object"}

            resp = client.chat.completions.create(**kwargs)
            raw_text = resp.choices[0].message.content

            data = robust_json_parse(raw_text)
            if data is None:
                tables = extract_tables_from_text(raw_text, schema)
                data = {"reasoning": "Recovered from text", "tables": tables} if tables else {"reasoning": "Failed", "tables": []}

            tables = data.get("tables", []) or []
            score, feedback, tables_with_meta = verifier.score_with_metadata(tables, schema, text)

            log.info("  [%s] Attempt %d: score=%.1f (%s)", source_id, attempt, score, feedback)

            if score > best_score:
                best_score, best_result, best_feedback = score, tables_with_meta, feedback
            if score >= target_score:
                break

        except (APITimeoutError, APIConnectionError) as e:
            log.warning("  [%s] API error attempt %d: %s", source_id, attempt, e)
            time.sleep(1.5 * attempt)
        except Exception as e:
            log.error("  [%s] Unexpected error attempt %d: %s", source_id, attempt, e)

    return ExtractedPage(
        page_num=0,           # Use 0 for non-paginated sources
        score=best_score,
        tables=best_result or [],
        feedback=best_feedback
    )