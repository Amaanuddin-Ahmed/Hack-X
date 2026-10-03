import asyncio
import logging
import os
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .gemma_service import GemmaConfigurationError, GemmaUnavailableError, ImageValidationError, analyze_food_image
from .schemas import FoodAnalysis

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_BYTES = 8 * 1024 * 1024
analysis_slots = asyncio.Semaphore(4)
allowed_origins = [item.strip() for item in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",") if item.strip()]

app = FastAPI(title="Food Analysis API", version="1.0.0", description="Structured packaged-food and nutrition-label extraction using Gemma 4.")
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
    logger.exception("Unhandled request failure request_id=%s", request_id)
    return JSONResponse(status_code=500, content={"detail": "Internal server error.", "request_id": request_id})


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "service": "food-analysis-api"}


@app.get("/ready")
def ready():
    configured = bool(os.getenv("GEMINI_API_KEY")) and bool(os.getenv("GEMMA_MODEL"))
    if not configured:
        raise HTTPException(status_code=503, detail="Service configuration is incomplete.")
    return {"status": "ready"}


async def _analyze(file: UploadFile) -> FoodAnalysis:
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Upload JPG, PNG, or WEBP only.")
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Empty image file.")
    if len(payload) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="Image must be 8 MB or smaller.")
    try:
        async with analysis_slots:
            return await asyncio.to_thread(analyze_food_image, payload)
    except ImageValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (GemmaConfigurationError, GemmaUnavailableError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Food analysis failed")
        raise HTTPException(status_code=502, detail="Food analysis provider returned an invalid response.") from exc


@app.post("/api/v1/food/analyze", response_model=FoodAnalysis)
async def analyze_v1(file: UploadFile = File(...)):
    return await _analyze(file)


@app.post("/analyze", response_model=FoodAnalysis, include_in_schema=False)
async def analyze_legacy(file: UploadFile = File(...)):
    return await _analyze(file)
