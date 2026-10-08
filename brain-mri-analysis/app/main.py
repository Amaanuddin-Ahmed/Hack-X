import asyncio
import logging
import os
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from .mri_service import ImageValidationError, analyze_brain_mri
from .schemas import BrainMRIAnalysis, ErrorResponse, HealthModelSummary

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_BYTES = 12 * 1024 * 1024
analysis_slots = asyncio.Semaphore(2)
allowed_origins = [item.strip() for item in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",") if item.strip()]

app = FastAPI(title="Brain MRI Analysis API", version="1.0.0", description="Research-only 2D brain-MRI classification API.")
app.add_middleware(CORSMiddleware, allow_origins=allowed_origins, allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["Content-Type", "X-Request-ID"])
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid4()))
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    request_id = request.headers.get("X-Request-ID", "unknown")
    logger.exception("Unhandled MRI API failure request_id=%s", request_id)
    return JSONResponse(status_code=500, content={"detail": "Internal server error.", "request_id": request_id})


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "service": "brain-mri-analysis-api", "version": "1.0.0"}


def parse_health_summary(raw_summary: str | None) -> HealthModelSummary | None:
    if not raw_summary:
        return None
    try:
        return HealthModelSummary.model_validate_json(raw_summary)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail="health_model_summary must be valid JSON matching the documented schema.") from exc


def integrate_result(result: BrainMRIAnalysis, context: HealthModelSummary | None, request_id: str) -> BrainMRIAnalysis:
    if context is None:
        summary = "No tabular-model health summary was supplied. MRI model output requires clinical review."
    else:
        high_count = sum(signal.risk_level == "high" for signal in context.risk_signals)
        summary = (
            f"The MRI classifier's top label is {result.predicted_label}. The supplied summary from {context.source} "
            f"contains {len(context.risk_signals)} risk signal(s), including {high_count} high-risk signal(s). "
            "This context does not alter, validate, or diagnose the MRI classifier output."
        )
    return result.model_copy(update={"request_id": request_id, "health_model_summary": context, "integrated_summary": summary})


@app.post("/api/v1/medical/mri/analyze", response_model=BrainMRIAnalysis, responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}})
async def analyze_mri(request: Request, file: UploadFile = File(...), health_model_summary: str | None = Form(default=None)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Upload a JPG, PNG, or WEBP image.")
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Empty image file.")
    if len(payload) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="Image must be 12 MB or smaller.")
    context = parse_health_summary(health_model_summary)
    try:
        async with analysis_slots:
            result = await asyncio.to_thread(analyze_brain_mri, payload)
        return integrate_result(result, context, request.headers.get("X-Request-ID", str(uuid4())))
    except ImageValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Brain MRI inference failed")
        raise HTTPException(status_code=503, detail="Brain MRI model is unavailable. Verify local dependencies and retry.") from exc
