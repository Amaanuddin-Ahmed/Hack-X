from functools import lru_cache
from io import BytesIO

from huggingface_hub import hf_hub_download
from PIL import Image, UnidentifiedImageError

from .schemas import BrainMRIAnalysis, RankedLabel

MODEL_REPOSITORY = "ThisenEkanayake/brain-tumor-detection"
MODEL_ID = "brain-mri-resnet18-brats2020"
LABELS = ("glioma", "meningioma", "no_tumor", "pituitary")


class ImageValidationError(ValueError):
    pass


def _checkpoint_path() -> str:
    return hf_hub_download(
        repo_id=MODEL_REPOSITORY,
        filename="multiclass-classification/multi_class_resnet.pth",
    )


@lru_cache(maxsize=1)
def _model_and_device():
    import torch
    from torch import nn
    from torchvision import models

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(LABELS))
    checkpoint = torch.load(_checkpoint_path(), map_location=device, weights_only=True)
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state)
    return model.to(device).eval(), device


def _prepare(payload: bytes):
    import torch
    from torchvision import transforms

    try:
        with Image.open(BytesIO(payload)) as candidate:
            candidate.verify()
        with Image.open(BytesIO(payload)) as source:
            image = source.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageValidationError("The uploaded file is not a readable image.") from exc
    if min(image.size) < 32:
        raise ImageValidationError("Upload a readable MRI image at least 32 pixels wide and high.")
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return preprocess(image).unsqueeze(0)


def analyze_brain_mri(payload: bytes) -> BrainMRIAnalysis:
    import torch

    tensor = _prepare(payload)
    model, device = _model_and_device()
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor.to(device))[0], dim=0).detach().cpu().tolist()
    scores = {label: round(float(score), 4) for label, score in zip(LABELS, probabilities)}
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return BrainMRIAnalysis(
        request_id="",
        model=MODEL_ID,
        device=device,
        predicted_label=ranked[0][0],
        class_scores=scores,
        ranked_labels=[RankedLabel(label=label, score=score) for label, score in ranked],
        integrated_summary="No tabular-model health summary was supplied with this MRI image.",
    )
