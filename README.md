# Gemma Food Label API

Production-oriented API for the hackathon food-analysis path:

`Image Upload -> Gemma 4 -> Structured Nutrition JSON -> UI`

## 1. Setup

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env`:

```env
GEMINI_API_KEY=YOUR_KEY
GEMMA_MODEL=THE_EXACT_GEMMA_4_MODEL_ID_CONFIRMED_BY_ORGANIZERS
```

Do not guess the model ID: the challenge brief specifically says to confirm available Gemma 4 access with organizers.

## 2. Run

```bash
uvicorn app.main:app --reload
```

Open: http://127.0.0.1:8000

API documentation: http://127.0.0.1:8000/docs

Primary UX endpoint: `POST /api/v1/food/analyze` with multipart field `file`.
`POST /analyze` remains available for backward compatibility.

## 3. Operational behavior

- Accepts verified JPG, PNG, and WEBP payloads up to 8 MB.
- Limits concurrent upstream calls and retries temporary Gemini capacity errors.
- Returns `400` for invalid input, `413` for oversized payloads, `503` for unavailable/configuration failures, and `502` for malformed upstream responses.
- Sends an `X-Request-ID` header on every response. Configure browser access with `ALLOWED_ORIGINS` in `.env`.

## 4. First test

Use one clear photo of a packaged-food nutrition label.

Success criteria:
- request completes
- valid JSON is returned
- visible nutrition values match the label
- invisible values are `null`
- missing/uncertain values are not hallucinated

## 5. Suggested dataset

Place your own test images in `tests/fixtures/` using the filenames in `test_cases.json`.
Start with six images, then expand to 10-15.

## 6. Architecture

```text
Food image
   |
   v
FastAPI /analyze
   |
   v
Gemma 4 via Gemini API
   |
   v
Pydantic JSON schema
   |
   v
Structured food/nutrition data
   |
   v
Browser result
```

## Later phase (NOT in this smoke test)

```text
Structured nutrition + health context from ML models
                    |
                    v
              Gemma reasoning
                    |
                    v
       Personalized food recommendation
```
