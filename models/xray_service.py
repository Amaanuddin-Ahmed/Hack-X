"""Research-only chest X-ray scoring adapted from the supplied reference repo."""

from __future__ import annotations

from functools import lru_cache
from io import BytesIO

import numpy as np
from PIL import Image, UnidentifiedImageError

from .xray_schemas import Finding, HealthReport, XrayAnalysis

MODEL_NAME = "densenet121-res224-all"


class ImageValidationError(ValueError):
    pass


@lru_cache(maxsize=1)
def _model_and_device():
    import torch
    import torchxrayvision as xrv

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = xrv.models.DenseNet(weights=MODEL_NAME).to(device).eval()
    return model, device


def _prepare(payload: bytes):
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
        raise ImageValidationError("Upload a readable chest X-ray at least 32 pixels wide and high.")
    image = xrv.datasets.normalize(grayscale, 255).astype(np.float32)[None, ...]
    image = xrv.datasets.XRayCenterCrop()(image)
    image = xrv.datasets.XRayResizer(224)(image)
    return torch.from_numpy(image).unsqueeze(0)


def analyze_chest_xray(payload: bytes, report: HealthReport | None) -> XrayAnalysis:
    import torch

    tensor = _prepare(payload)
    model, device = _model_and_device()
    with torch.no_grad():
        output = model(tensor.to(device))[0].detach().cpu().numpy()
    scores = {label: round(float(value), 4) for label, value in zip(model.pathologies, output)}
    top = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:3]
    findings = [Finding(label=label, score=score) for label, score in top]
    labels = ", ".join(item.label for item in findings)
    if report:
        high_count = sum(signal.risk_level == "high" for signal in report.risk_signals)
        integrated = (
            f"The X-ray model's highest-scoring labels are {labels}. "
            f"The supplied Vitalis report contains {len(report.risk_signals)} risk signals, including {high_count} high-risk signals. "
            "The health report is shown as separate context and does not modify or validate the X-ray scores."
        )
    else:
        integrated = f"The X-ray model's highest-scoring labels are {labels}. No Vitalis health report was supplied."
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
