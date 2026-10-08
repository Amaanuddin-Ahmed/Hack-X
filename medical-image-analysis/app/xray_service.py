from functools import lru_cache
from io import BytesIO

import numpy as np
from PIL import Image, UnidentifiedImageError

from .schemas import Finding, MedicalScanAnalysis

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
        raise ImageValidationError("Upload a readable chest X-ray image at least 32 pixels wide and high.")
    image = xrv.datasets.normalize(grayscale, 255).astype(np.float32)[None, ...]
    image = xrv.datasets.XRayCenterCrop()(image)
    image = xrv.datasets.XRayResizer(224)(image)
    return torch.from_numpy(image).unsqueeze(0)


def analyze_chest_xray(payload: bytes) -> MedicalScanAnalysis:
    import torch
    tensor = _prepare(payload)
    model, device = _model_and_device()
    with torch.no_grad():
        output = model(tensor.to(device))[0].detach().cpu().numpy()
    scores = {label: round(float(value), 4) for label, value in zip(model.pathologies, output)}
    top = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:3]
    return MedicalScanAnalysis(
        image_type="chest_xray",
        image_quality="adequate",
        status="requires_clinical_review",
        model=MODEL_NAME,
        device=device,
        model_scores=scores,
        highest_scoring_labels=[Finding(label=label, confidence="low", note="Highest model score only; not a diagnosis.") for label, _ in top],
        observations=[],
        limitations=["The model assumes a chest radiograph and does not verify view, acquisition, or anatomy.", "Scores can be wrong and are not probabilities, diagnoses, or treatment recommendations.", "A qualified clinician must review the original study and clinical context."],
    )
