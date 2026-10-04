"""Vercel entrypoint for the dedicated chest X-ray model service."""

import base64
import binascii

from fastapi import FastAPI, HTTPException

from schemas import XrayAnalysis, XrayRequest
from service import ImageValidationError, analyze_chest_xray

app = FastAPI(title="Vitalis X-ray service", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok", "model": "densenet121-res224-all"}


@app.post("/xray", response_model=XrayAnalysis)
def xray(request: XrayRequest):
    try:
        payload = base64.b64decode(request.image_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 image encoding.") from exc
    if not payload:
        raise HTTPException(status_code=400, detail="Empty X-ray image.")
    if len(payload) > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="X-ray image must be 5 MB or smaller.")
    try:
        return analyze_chest_xray(payload, request.health_report)
    except ImageValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Chest X-ray model is unavailable. Verify its dependencies and retry.",
        ) from exc
