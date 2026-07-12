import json
import os
import re
from datetime import datetime

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Any, Dict

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-3.5-flash"

SUPPORTED_TYPES = {
    "string", "integer", "float", "boolean", "date",
    "array[string]", "array[integer]",
}


def gemini_type_for(field_type: str) -> dict:
    """Map a requested field type to a Gemini responseSchema fragment."""
    if field_type == "string":
        return {"type": "STRING", "nullable": True}
    if field_type == "integer":
        return {"type": "INTEGER", "nullable": True}
    if field_type == "float":
        return {"type": "NUMBER", "nullable": True}
    if field_type == "boolean":
        return {"type": "BOOLEAN", "nullable": True}
    if field_type == "date":
        return {"type": "STRING", "nullable": True, "description": "ISO format YYYY-MM-DD"}
    if field_type == "array[string]":
        return {"type": "ARRAY", "items": {"type": "STRING"}, "nullable": True}
    if field_type == "array[integer]":
        return {"type": "ARRAY", "items": {"type": "INTEGER"}, "nullable": True}
    # Unknown type -> treat as string, safest default
    return {"type": "STRING", "nullable": True}


def build_response_schema(schema: Dict[str, str]) -> dict:
    properties = {}
    for key, ftype in schema.items():
        properties[key] = gemini_type_for(ftype)
    return {
        "type": "OBJECT",
        "properties": properties,
        "required": list(schema.keys()),
    }


DATE_FORMATS = [
    "%Y-%m-%d", "%d %B %Y", "%B %d, %Y", "%B %d %Y", "%d %b %Y", "%b %d, %Y",
    "%b %d %Y", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d", "%d-%m-%Y", "%m-%d-%Y",
]


def normalize_date(raw) -> Any:
    if raw is None:
        return None
    raw = str(raw).strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
        return raw
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    # Last resort: try to pull YYYY, Month name/number, and Day out of the string
    m = re.search(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", raw)
    if m:
        try:
            return datetime.strptime(m.group(0), "%d %B %Y").strftime("%Y-%m-%d")
        except ValueError:
            try:
                return datetime.strptime(m.group(0), "%d %b %Y").strftime("%Y-%m-%d")
            except ValueError:
                pass
    return None


def coerce_value(value: Any, field_type: str) -> Any:
    if value is None:
        return None
    try:
        if field_type == "string":
            return str(value)
        if field_type == "integer":
            if isinstance(value, bool):
                return None
            return int(float(str(value).replace(",", "")))
        if field_type == "float":
            return float(str(value).replace(",", ""))
        if field_type == "boolean":
            if isinstance(value, bool):
                return value
            return str(value).strip().lower() in ("true", "1", "yes", "on")
        if field_type == "date":
            return normalize_date(value)
        if field_type == "array[string]":
            if isinstance(value, list):
                return [str(v) for v in value]
            return [str(value)]
        if field_type == "array[integer]":
            if isinstance(value, list):
                return [int(float(str(v))) for v in value]
            return [int(float(str(value)))]
    except (ValueError, TypeError):
        return None
    return None


def validate_and_coerce(raw: dict, schema: Dict[str, str]) -> dict:
    """Return exactly the requested keys, correctly typed, no extras."""
    result = {}
    for key, ftype in schema.items():
        val = raw.get(key) if isinstance(raw, dict) else None
        result[key] = coerce_value(val, ftype)
    return result


# ---- Heuristic fallback (used only if the LLM call fails outright) ----

def heuristic_fallback(text: str, schema: Dict[str, str]) -> dict:
    result = {}
    used_spans = []

    def find_first_unused(pattern):
        for m in re.finditer(pattern, text):
            if not any(a <= m.start() < b for a, b in used_spans):
                used_spans.append((m.start(), m.end()))
                return m
        return None

    for key, ftype in schema.items():
        val = None
        if ftype == "integer":
            m = find_first_unused(r"\b\d+\b")
            if m:
                val = int(m.group(0))
        elif ftype == "float":
            m = find_first_unused(r"\b\d[\d,]*\.\d+\b|\b\d[\d,]*\b")
            if m:
                val = float(m.group(0).replace(",", ""))
        elif ftype == "date":
            m = re.search(
                r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}\s+[A-Za-z]+\s+\d{4}\b|"
                r"\b[A-Za-z]+\s+\d{1,2},?\s+\d{4}\b",
                text,
            )
            if m:
                val = normalize_date(m.group(0))
        elif ftype == "boolean":
            val = bool(re.search(r"\btrue\b|\byes\b", text, re.IGNORECASE))
        elif ftype in ("array[string]", "array[integer]"):
            val = None
        else:
            val = None
        result[key] = val
    return result


@app.post("/dynamic-extract")
async def dynamic_extract(payload: dict):
    text = payload.get("text", "")
    schema = payload.get("schema", {})

    if not isinstance(schema, dict) or not schema:
        return JSONResponse(status_code=400, content={"error": "schema must be a non-empty object"})

    prompt = (
        "Extract the following fields from the text below, matching the exact field "
        "names and types given. Use null for any field that cannot be found or inferred. "
        "For 'date' typed fields, always output ISO format YYYY-MM-DD. "
        "For numeric fields, output plain numbers with no currency symbols or commas. "
        "Return ONLY the requested fields, nothing extra.\n\n"
        f"Fields and types: {json.dumps(schema)}\n\n"
        f"Text:\n{text}"
    )

    response_schema = build_response_schema(schema)

    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": response_schema,
        },
    }

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, json=body)
            if resp.status_code != 200:
                return validate_and_coerce(heuristic_fallback(text, schema), schema)
            data = resp.json()

        candidates = data.get("candidates", [])
        if not candidates:
            return validate_and_coerce(heuristic_fallback(text, schema), schema)
        parts = candidates[0].get("content", {}).get("parts", [])
        raw_text = "".join(p.get("text", "") for p in parts)
        parsed = json.loads(raw_text)
        return validate_and_coerce(parsed, schema)
    except Exception:
        return validate_and_coerce(heuristic_fallback(text, schema), schema)


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}
