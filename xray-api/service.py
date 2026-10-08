"""Research-only chest X-ray scoring with a compact ONNX CPU runtime."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, UnidentifiedImageError
from skimage.transform import resize

from schemas import Finding, HealthReport, XrayAnalysis

MODEL_NAME = "densenet121-res224-all"
MODEL_PATH = Path(__file__).resolve().parent / "model.onnx"
PATHOLOGIES = [
    "Atelectasis",
    "Consolidation",
    "Infiltration",
    "Pneumothorax",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Effusion",
    "Pneumonia",
    "Pleural_Thickening",
    "Cardiomegaly",
    "Nodule",
    "Mass",
    "Hernia",
    "Lung Lesion",
    "Fracture",
    "Lung Opacity",
    "Enlarged Cardiomediastinum",
]
OPERATING_THRESHOLDS = np.asarray(
    [
        0.07422872,
        0.038290843,
        0.09814756,
        0.0098118475,
        0.023601074,
        0.0022490358,
        0.010060724,
        0.103246614,
        0.056810737,
        0.026791653,
        0.050318155,
        0.023985857,
        0.01939503,
        0.042889766,
        0.053369623,
        0.035975814,
        0.20204692,
        0.05015312,
    ],
    dtype=np.float32,
)


class ImageValidationError(ValueError):
    pass


SESSION = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])


def prepare_image(payload: bytes):
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
    image = (((grayscale.astype(np.float32) / 255.0) * 2.0) - 1.0) * 1024.0
    image = image[None, ...]
    _, height, width = image.shape
    crop_size = min(height, width)
    start_x = width // 2 - crop_size // 2
    start_y = height // 2 - crop_size // 2
    image = image[:, start_y:start_y + crop_size, start_x:start_x + crop_size]
    image = resize(
        image,
        (1, 224, 224),
        mode="constant",
        preserve_range=True,
    ).astype(np.float32)
    return image[None, ...]


def analyze_chest_xray(payload: bytes, report: HealthReport | None) -> XrayAnalysis:
    tensor = prepare_image(payload)
    logits = SESSION.run(["logits"], {"image": tensor})[0][0]
    raw = 1.0 / (1.0 + np.exp(-logits))
    output = np.where(
        raw < OPERATING_THRESHOLDS,
        raw / (OPERATING_THRESHOLDS * 2.0),
        1.0 - ((1.0 - raw) / ((1.0 - OPERATING_THRESHOLDS) * 2.0)),
    )
    scores = {
        label: round(float(value), 4)
        for label, value in zip(PATHOLOGIES, output)
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
        device="cpu",
        model_scores=scores,
        highest_scoring_labels=findings,
        integrated_summary=integrated,
        limitations=[
            "The model assumes a chest radiograph and does not verify view, acquisition, or anatomy.",
            "Scores can be wrong and are not probabilities, diagnoses, or treatment recommendations.",
            "A qualified clinician must review the original study and clinical context.",
        ],
    )
