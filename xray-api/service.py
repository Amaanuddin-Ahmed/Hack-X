"""Research-only chest X-ray scoring service."""

from __future__ import annotations

from functools import lru_cache
from io import BytesIO
import os
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from schemas import Finding, HealthReport, XrayAnalysis

MODEL_NAME = "densenet121-res224-all"
DEFAULT_CACHE = Path(__file__).resolve().parent / "model_cache"


class ImageValidationError(ValueError):
    pass


@lru_cache(maxsize=1)
def load_model():
    import torch
    import torchxrayvision as xrv

    cache_dir = Path(os.getenv("XRAY_MODEL_CACHE", str(DEFAULT_CACHE)))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = xrv.models.DenseNet(
        weights=MODEL_NAME,
        cache_dir=str(cache_dir),
    ).to(device).eval()
    return model, device


def prepare_image(payload: bytes):
    import torch
    import torchxrayvision as xrv

    try:
        with Image.open(BytesIO(payload)) as candidate:
            candidate.verify()
        with Image.open(BytesIO(payload)) as source:
            grayscale = np.asarray(source.convert("L"))
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageValidationError("The uploaded file is not a readable image.") from exc
    if grayscale.ndim != 2 or min(grayscale.shape) < 32:
        raise ImageValidationError(
            "Upload a readable chest X-ray at least 32 pixels wide and high."
        )
    image = xrv.datasets.normalize(grayscale, 255).astype(np.float32)[None, ...]
    image = xrv.datasets.XRayCenterCrop()(image)
    image = xrv.datasets.XRayResizer(224)(image)
    return torch.from_numpy(image).unsqueeze(0)


def analyze_chest_xray(payload: bytes, report: HealthReport | None) -> XrayAnalysis:
    import torch

    tensor = prepare_image(payload)
    model, device = load_model()
    with torch.no_grad():
        output = model(tensor.to(device))[0].detach().cpu().numpy()
    scores = {
        label: round(float(value), 4)
        for label, value in zip(model.pathologies, output)
    }
    top = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:3]
    findings = [Finding(label=label, score=score) for label, score in top]
    labels = ", ".join(item.label for item in findings)
    if report:
        high_count = sum(signal.risk_level == "high" for signal in report.risk_signals)
        integrated = (
            f"The X-ray model's highest-scoring labels are {labels}. "
            f"The supplied Vitalis report contains {len(report.risk_signals)} "
            f"risk signals, including {high_count} high-risk signals. "
            "The health report is shown as separate context and does not modify "
            "or validate the X-ray scores."
        )
    else:
        integrated = (
            f"The X-ray model's highest-scoring labels are {labels}. "
            "No Vitalis health report was supplied."
        )
    return XrayAnalysis(
        model=MODEL_NAME,
        device=device,
        model_scores=scores,
        highest_scoring_labels=findings,
        integrated_summary=integrated,
        limitations=[
            "The model assumes a chest radiograph and does not verify view, acquisition, or anatomy.",
            "Scores can be wrong and are not probabilities, diagnoses, or treatment recommendations.",
            "A qualified clinician must review the original study and clinical context.",
        ],
    )
