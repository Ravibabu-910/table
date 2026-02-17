from __future__ import annotations

import json

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from app.llm_client import OllamaClient
from app.schema import ExtractRequestConfig, ExtractResponse
from app.table_extractor import TableImageAnalyzer, compare_json_results

app = FastAPI(title="Table-to-JSON Extractor", version="1.0.0")

analyzer = TableImageAnalyzer()
ollama = OllamaClient()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/extract", response_model=ExtractResponse)
async def extract_table(
    image: UploadFile = File(...),
    model_name: str = Form("llama3.1:8b-instruct-q4_K_M"),
    table_title_hint: str | None = Form(None),
    expected_json: str | None = Form(None),
) -> ExtractResponse:
    try:
        image_bytes = await image.read()
        artifacts = analyzer.extract(image_bytes)

        extracted = ollama.extract_structured_json(
            model_name=model_name,
            ocr_table_markdown=artifacts.markdown_table,
            raw_text=artifacts.raw_text,
            title_hint=table_title_hint,
        )

        quality_score = None
        comparison_summary = None
        if expected_json:
            cfg = ExtractRequestConfig(expected_json=json.loads(expected_json))
            quality_score, comparison_summary = compare_json_results(extracted, cfg.expected_json or [])

        return ExtractResponse(
            success=True,
            json_result=extracted,
            quality_score=quality_score,
            comparison_summary=comparison_summary,
            diagnostics=artifacts.diagnostics,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
