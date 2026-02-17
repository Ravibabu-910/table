from __future__ import annotations

import json
from typing import Any, Dict, List

import requests


class OllamaClient:
    def __init__(self, host: str = "http://127.0.0.1:11434") -> None:
        self.host = host.rstrip("/")

    def extract_structured_json(
        self,
        *,
        model_name: str,
        ocr_table_markdown: str,
        raw_text: str,
        title_hint: str | None = None,
    ) -> List[Dict[str, Any]]:
        schema_hint = {
            "output": [
                {
                    "<arm_key_like_ABBV154_150mg_Q2W_n94>": "<value string>",
                    "<arm_key>_significance": "<optional significance e.g. P < 0.01>",
                    "PBO_n96": "<placebo value>",
                    "assessment": "<row assessment>",
                    "outcome_type": "<row group context>",
                }
            ]
        }
        prompt = f"""
You are a precise table-to-JSON parser.

Goal:
Convert OCR'ed clinical table content to a JSON array similar to this schema:
{json.dumps(schema_hint, indent=2)}

Requirements:
1. Return ONLY valid JSON, no markdown, no commentary.
2. Keep values as strings exactly as present.
3. Build dynamic arm keys using treatment, frequency and n-count.
   Example: ABBV154_150mg_Q2W_n94
4. Include *_significance keys when footnote markers indicate significance.
5. Preserve one JSON object per assessment row.
6. Include fields: assessment, outcome_type, placebo arm key if present.
7. If uncertain, keep best effort and do not invent impossible values.

Table title hint: {title_hint or 'N/A'}

OCR table (markdown approximation):
{ocr_table_markdown}

Raw OCR text:
{raw_text}
""".strip()

        payload: Dict[str, Any] = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
            },
            "format": "json",
        }
        resp = requests.post(f"{self.host}/api/generate", json=payload, timeout=180)
        resp.raise_for_status()
        body = resp.json()
        parsed = json.loads(body["response"])

        if isinstance(parsed, dict) and "output" in parsed:
            parsed = parsed["output"]
        if not isinstance(parsed, list):
            raise ValueError("LLM output is not a JSON array.")
        return parsed
