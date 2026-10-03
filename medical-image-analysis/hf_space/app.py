import numpy as np
import gradio as gr
import spaces
import torch
import torchxrayvision as xrv

MODEL_ID = "densenet121-res224-all"
DISCLAIMER = (
    "Research/demo only. These are model scores, not diagnoses or medical advice. "
    "A qualified clinician must review the original study and clinical context."
)

# ZeroGPU provides CUDA emulation during Space startup. Keeping the model on CUDA
# here avoids repeatedly transferring it whenever a shared GPU is allocated.
MODEL = xrv.models.DenseNet(weights=MODEL_ID).cuda().eval()


def _prepare(image):
    if image is None:
        raise gr.Error("Upload a chest X-ray image.")
    array = np.asarray(image.convert("L"))
    array = xrv.datasets.normalize(array, 255).astype(np.float32)[None, ...]
    array = xrv.datasets.XRayCenterCrop()(array)
    array = xrv.datasets.XRayResizer(224)(array)
    return torch.from_numpy(array).unsqueeze(0).cuda()


@spaces.GPU(duration=30)
def analyze_xray(image):
    tensor = _prepare(image)
    with torch.no_grad():
        scores = MODEL(tensor)[0].detach().cpu().numpy()
    return {
        "model": MODEL_ID,
        "image_type": "chest_xray_assumed",
        "model_scores": {
            label: round(float(score), 4)
            for label, score in zip(MODEL.pathologies, scores)
        },
        "clinical_review_required": True,
        "disclaimer": DISCLAIMER,
    }


with gr.Blocks(title="Chest X-Ray Research Demo") as demo:
    gr.Markdown("# Chest X-Ray Model Demo\n\n" + DISCLAIMER)
    image = gr.Image(type="pil", label="De-identified chest X-ray")
    run = gr.Button("Run model")
    result = gr.JSON(label="Structured model output")
    run.click(analyze_xray, inputs=image, outputs=result, api_name="analyze_xray")

demo.queue(default_concurrency_limit=1).launch()
