# Brain MRI Analysis API

Research-only local API for 2D brain-MRI image classification. It loads the MIT-licensed `ThisenEkanayake/brain-tumor-detection` ResNet-18 checkpoint from Hugging Face on its first analysis request, then caches it locally.

The source model was trained on a BraTS2020-derived dataset and predicts four research/demo labels: `glioma`, `meningioma`, `no_tumor`, and `pituitary`. It is not validated for clinical use and must not be used for diagnosis, treatment, or patient-care decisions. The model card documents the input format, classes, and limitations: https://huggingface.co/ThisenEkanayake/brain-tumor-detection

## Run

```powershell
cd brain-mri-analysis
C:\Siddu\H\gemma-food-smoke-test\.venv\Scripts\python.exe -m pip install -r requirements.txt
C:\Siddu\H\gemma-food-smoke-test\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8003
```

Open `http://127.0.0.1:8003` for the upload interface or `http://127.0.0.1:8003/docs` for the API contract.

## Endpoint

`POST /api/v1/medical/mri/analyze` accepts multipart fields:

- `file`: de-identified 2D brain MRI as JPG, PNG, or WEBP (maximum 12 MB).
- `health_model_summary` (optional): JSON produced by the tabular risk-model branch. It is preserved as context but never changes the MRI classifier scores.

The response contains the winning label, all softmax class scores, full ranking, and an integrated context summary.
