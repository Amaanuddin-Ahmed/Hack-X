# Vitalis

Vitalis is a Hack Day health-assessment prototype combining three tabular risk models, a local research chest-X-ray model, Gemma health summaries, and personalized food-label guidance. It is an educational prototype, not a medical device.

## Run the integrated app

Create the environment and install dependencies:

```bash
npm install
cp .env.example .env
python3 -m venv .venv
.venv/bin/pip install -r models/requirements.txt
```

Add `GEMINI_API_KEY` to `.env`, then start the Python model service:

```bash
.venv/bin/uvicorn models.server:app --host 127.0.0.1 --port 8000
```

In another terminal, start the website:

```bash
npm run dev
```

The Python service exposes `/predict` and `/xray`. TorchXRayVision downloads and caches the approximately 28 MB `densenet121-res224-all` checkpoint on the first X-ray scan. The website defaults to `http://127.0.0.1:8000/predict`; use `MODEL_API_URL` or `XRAY_API_URL` to override the endpoints. Food analysis defaults to `gemma-4-26b-a4b-it`; use `GEMMA_FAST_MODEL` to override it. Keep all credentials server-side.

## Integrated API routes

- `POST /api/assess`: profile to three risk scores, combined health score, and BMI.
- `POST /api/summary`: profile and assessment to a Gemma health summary.
- `POST /api/food/extract`: image to visibly extracted, validated nutrition facts.
- `POST /api/food/recommend`: extracted label plus complete JSON health report to personalized guidance.
- `POST /api/scan`: chest-X-ray image plus separate health context to research model observations.

Images use `{name,mimeType,data}` with a base64 data URL and support JPG, PNG, and WebP up to the UI limit of 5 MB.

## Food-label flow

Food analysis has two explicit stages. Gemma first extracts only visible label facts, including confidence, allergens, missing fields, ingredients, and nutrients. The app normalizes duplicate or oversized model output for reliable charts. It then sends those facts together with the complete JSON health report, profile, health score, and three risk scores to Gemma. The response contains a direct Yes/Occasionally/No/Unknown answer, reasoning, portion guidance, benefits, cautions, and a transparent food-fit indicator that is not a medical probability.

The website shows progress and safe diagnostic details for each stage. Server errors contain a matching request ID without exposing secrets.

## Model contract

The tabular service returns `{scores:{obesity,diabetes,heart_disease}}` on a 0–100 scale. Diabetes and heart use positive-class probability. Obesity risk combines the overweight and obesity classes. The overall health score is `100 - mean(the three risks)` and is a summary indicator, not a validated clinical probability.

Chest X-rays are scored locally with the research-only TorchXRayVision DenseNet adapted from the Hacktoberfest26 source repository. The complete Vitalis report is retained as separate context and never changes the raw X-ray outputs. Scores are not diagnoses or clinical probabilities and always require qualified review.

## Reports and local session

The results page exports HTML and JSON reports containing the profile, BMI, risk scores, health score, AI summary, recommendations, and completed food analysis. Structured session data is stored under `vitalis-session-report` in browser localStorage so Food Lens can reuse the health context after a refresh. Uploaded image bytes are never stored there.

## Reference services

The original standalone FastAPI services from [Hacktoberfest26](https://github.com/mallurivikas/Hacktoberfest26) are retained in `app/` and `medical-image-analysis/` for comparison. The integrated Vitalis runtime uses the Next API routes and the combined `models.server` service described above.

Challenge: https://github.com/reacthyderabad/hacktoberfest-hack-day-2026/blob/main/challenges/gemma-4.md

Gemini API reference: https://ai.google.dev/api/generate-content
