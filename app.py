"""
RTIE — FastAPI Deployment

Endpoints:
    POST /predict — Upload thermal image → fault diagnosis + business metrics
    GET  /health  — Healthcheck
"""

import io
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

import config
from model import RTIEModel
from feature_engineering import extract_features
from business_metrics import compute_business_impact


app = FastAPI(
    title="RTIE — Radiator Thermal Intelligence Engine",
    description="AI-powered radiator fault detection and efficiency scoring",
    version="1.0.0",
)

# ── Global model state ──
_model = None
_device = None


def get_model():
    """Lazy-load model on first request."""
    global _model, _device

    if _model is None:
        _device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        _model = RTIEModel(pretrained=False).to(_device)
        ckpt_path = os.path.join(config.MODEL_DIR, "best_model.pth")

        if not os.path.exists(ckpt_path):
            raise RuntimeError(f"Model checkpoint not found at {ckpt_path}. Run train.py first.")

        checkpoint = torch.load(ckpt_path, map_location=_device, weights_only=False)
        _model.load_state_dict(checkpoint["model_state_dict"])
        _model.eval()
        print(f"Model loaded on {_device}")

    return _model, _device


def preprocess_image(image_bytes):
    """Convert uploaded image to model input tensor."""
    from torchvision import transforms

    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    img_rgb = Image.merge("RGB", [img, img, img])

    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(img_rgb).unsqueeze(0)  # (1, 3, 224, 224)
    return tensor


def extract_physics_from_image(image_bytes):
    """Extract physics features from uploaded image via temp matrix approximation."""
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    img_resized = img.resize((config.GRID_COLS, config.GRID_ROWS), Image.BICUBIC)
    # Map pixel values (0-255) back to approximate temperature range (30-80°C)
    matrix = np.array(img_resized, dtype=np.float32)
    matrix = matrix / 255.0 * 50.0 + 30.0  # approx 30-80°C range
    features = extract_features(matrix)
    return features


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "model": "RTIE v1.0"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict radiator fault from thermal image.

    Returns fault classification, efficiency score, confidence,
    uncertainty, escalation status, and business impact metrics.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()

    try:
        model, device = get_model()

        # Preprocess
        image_tensor = preprocess_image(image_bytes).to(device)
        physics_features = extract_physics_from_image(image_bytes)
        physics_tensor = torch.tensor(physics_features, dtype=torch.float32).unsqueeze(0).to(device)

        # MC Dropout inference
        start_time = time.time()
        result = model.predict_with_uncertainty(
            image_tensor, physics_tensor, T=config.MC_FORWARD_PASSES
        )
        inference_time_ms = (time.time() - start_time) * 1000

        # Extract predictions
        class_idx = result["predicted_class"][0].item()
        fault_label = config.CLASS_NAMES[class_idx]
        efficiency_score = round(result["efficiency_score"][0].item(), 1)
        confidence = round(result["confidence"][0].item(), 4)
        uncertainty_std = round(result["uncertainty_std"][0].item(), 4)
        status = result["status"][0]

        # Business impact
        impact = compute_business_impact(fault_label, efficiency_score)

        # Physics features dict
        from feature_engineering import feature_names
        physics_dict = dict(zip(feature_names(), [round(float(f), 4) for f in physics_features]))

        response = {
            "status": status,
            "fault_label": fault_label,
            "fault_class_index": class_idx,
            "efficiency_score": efficiency_score,
            "confidence": confidence,
            "uncertainty_std": uncertainty_std,
            "class_probabilities": {
                config.CLASS_NAMES[i]: round(result["class_probs"][0][i].item(), 4)
                for i in range(config.NUM_CLASSES)
            },
            "energy_loss_pct": impact["energy_loss_pct"],
            "wasted_kwh_per_year": impact["wasted_kwh_per_year"],
            "estimated_annual_cost_gbp": impact["estimated_annual_cost_gbp"],
            "co2_reduction_potential_kg": impact["co2_reduction_potential_kg"],
            "physics_features": physics_dict,
            "inference_time_ms": round(inference_time_ms, 1),
        }

        return JSONResponse(content=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
