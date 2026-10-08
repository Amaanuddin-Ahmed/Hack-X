# Medical Branch: Chest X-Ray Demo

The primary demo is a local FastAPI service using a chest-X-ray-specific classifier. It replaces the earlier Gemma-based X-ray experiment, because Gemma is not a chest-radiograph classifier.

The model is TorchXRayVision `densenet121-res224-all`, trained on multiple chest-radiograph datasets. It emits 18 model scores; it is research-only and does not diagnose, treat, or make medical decisions.

## Free local demo

Install the dependencies and run the service. The 28 MB model checkpoint downloads once on the first real analysis, then is cached locally. CPU inference is supported; CUDA is used automatically when available.

## Integrated model context API

`POST /api/v1/medical/scan/analyze` accepts multipart fields:

- `file`: a chest X-ray JPG, PNG, or WEBP.
- `health_model_summary` (optional): JSON emitted by the tabular/risk-model branch.

Example `health_model_summary`:

```json
{
  "source": "disease-risk-service",
  "overall_health_score": 72,
  "summary": "Demo health-model output.",
  "risk_signals": [
    {"model_name": "cardio-risk-v1", "label": "cardiovascular risk", "risk_level": "moderate", "score": 0.42}
  ]
}
```

The API retains this context in `health_model_summary` and produces `integrated_summary`. It never changes the X-ray scores based on another model's output. See `/docs` for the full UX contract.

## UX contract

`POST /api/v1/medical/scan/analyze` accepts multipart form field `file` (JPG, PNG, WEBP; max 12 MB). The interactive OpenAPI contract is available at `/docs`.

Responses include `image_type`, `image_quality`, `status`, cautious `observations`, and an always-on `clinical_review_required` flag. The UX must render the disclaimer and must not present observations as a diagnosis.

## Configure Gemma 4

1. Copy `.env.example` to `.env`.
2. Set `GEMINI_API_KEY` and the exact image-capable Gemma 4 model ID in `GEMMA_MODEL`.
3. Do not commit the real API key.

## Run

```powershell
cd medical-image-analysis
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8002
```

The UX can call `http://127.0.0.1:8002/api/v1/medical/scan/analyze` locally. Update CORS origins in `app/main.py` before deployment.
