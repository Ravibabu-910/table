from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ExtractRequestConfig(BaseModel):
    model_name: str = Field(
        default="llama3.1:8b-instruct-q4_K_M",
        description="Local Ollama model used to structure OCR output.",
    )
    table_title_hint: Optional[str] = Field(
        default=None,
        description="Optional hint (for example: 'Table 2 secondary outcomes').",
    )
    expected_json: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Optional expected JSON for quality scoring against extraction.",
    )


class ExtractResponse(BaseModel):
    success: bool
    json_result: List[Dict[str, Any]]
    quality_score: Optional[float] = None
    comparison_summary: Optional[str] = None
    diagnostics: Dict[str, Any] = Field(default_factory=dict)
