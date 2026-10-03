---
title: Chest X-Ray Research Demo
emoji: 🩻
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
suggested_hardware: zerogpu
---

# Chest X-Ray Research Demo

Research-only chest-radiograph demo built with TorchXRayVision's pretrained `densenet121-res224-all` DenseNet-121 model. The weights were trained on a combination of chest X-ray datasets (NIH, PadChest, CheXpert, MIMIC-CXR, Google, OpenI, and RSNA) and return 18 model scores.

It is not a medical device, does not diagnose disease, and must not inform treatment or urgent-care decisions. Do not upload identifiable health information.

## Deploy on Hugging Face ZeroGPU

1. Create a Hugging Face account with verified email. A personal account older than 30 days in good standing can host up to two ZeroGPU Spaces at no charge.
2. Create a **Gradio** Space, select **ZeroGPU** hardware, and make it public for the hackathon demo.
3. Upload the contents of this `hf_space` folder to that Space repository.
4. Wait for the build to finish. The model weights are downloaded by the Space, not by the user’s computer.

## UX integration

The Gradio function is named `analyze_xray`; the Space's **Use via API** page provides the live HTTP/JavaScript call for the deployed Space URL. It returns `model_scores`, `clinical_review_required`, and `disclaimer` as JSON.

Free ZeroGPU usage has daily quotas and can queue during demand, so this is appropriate for a demo—not production clinical use.
