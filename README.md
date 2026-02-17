# Table Image -> Structured JSON (FastAPI + OpenCV + Local LLM)

This project extracts complex table content from images and converts it to a structured JSON format similar to your example (clinical-table style keys such as `ABBV154_150mg_Q2W_n94`).

## What this does

1. Uses **OpenCV** to detect table lines/cells.
2. Uses **Tesseract OCR** to read cell text.
3. Uses a **local LLM via Ollama (CPU-friendly)** to normalize OCR output into a consistent JSON schema.
4. Optionally compares extracted JSON vs expected JSON and reports a similarity score.

## Local setup (CPU)

### 1) System dependencies

Install Tesseract OCR:

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr
```

Install Ollama and pull a local model:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1:8b-instruct-q4_K_M
```

### 2) Python setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3) Run API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## API usage

### Health

```bash
curl http://127.0.0.1:8000/health
```

### Extract JSON

```bash
curl -X POST "http://127.0.0.1:8000/extract" \
  -F "image=@/path/to/table_image.png" \
  -F "model_name=llama3.1:8b-instruct-q4_K_M" \
  -F "table_title_hint=Selected secondary endpoint outcomes"
```

### Extract + compare with expected JSON

```bash
curl -X POST "http://127.0.0.1:8000/extract" \
  -F "image=@/path/to/table_image.png" \
  -F 'expected_json=[{"assessment":"LDA - CDAI score ≤10","PBO_n96":"21 (21.9) [13.6-30.1]"}]'
```

## Notes for accuracy

- Accuracy above 90% depends on image quality, font size, and table line visibility.
- For best results:
  - Use high-resolution images (>= 200 DPI).
  - Crop exactly around the target table.
  - Provide `table_title_hint`.
  - Use consistent model and prompt.
- The API returns `quality_score` when `expected_json` is provided.

## File overview

- `app/main.py`: FastAPI app and extraction endpoint.
- `app/table_extractor.py`: OpenCV preprocessing + OCR + similarity scoring.
- `app/llm_client.py`: local Ollama LLM call and JSON structuring prompt.
- `app/schema.py`: request/response models.
