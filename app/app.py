"""
Face Mask Detection API.

Project 4 - Advanced ML team project.
Wraps our trained MobileNetV2 model in a FastAPI endpoint.

Endpoints:
- GET  /         -> web UI
- GET  /health   -> health check (used by docker)
- GET  /metrics  -> model performance metrics
- POST /predict  -> upload an image, get back: status, confidence, action

Run locally:
    uvicorn app:app --reload --port 8000
Docs:
    http://localhost:8000/docs
"""

import io
import os
import time
import numpy as np
import torch
import cv2
from torch import nn
from torchvision import transforms, models
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles


# ------- constants -------

CLASSES       = ["WithMask", "WithoutMask"]  # alphabetical, same as ImageFolder
MODEL_PATH    = os.environ.get("MODEL_PATH", "mask_detector.pth")
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB — reject oversized uploads early


# ------- preprocessing -------
# Match the eval pipeline used in train_robust.py: Resize(256) + CenterCrop(224).
# Resizing then center-cropping preserves face aspect ratio better than
# squashing directly to 224x224, which matters for non-square inputs
# (wide webcam frames, phone portraits, etc).

preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ------- face detection -------
# OpenCV Haar cascade crops the face region before classification.
# The model was trained on tight face crops, so giving it a full portrait
# with shoulders/background hurts accuracy. Detecting first keeps the
# inference framing consistent with training.

HAAR_PATH = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
_face_cascade = cv2.CascadeClassifier(HAAR_PATH)


def detect_and_crop_face(pil_img: Image.Image) -> Image.Image:
    """
    Find the largest face in the image and return a tight crop around it.
    Falls back to the full image if no face is detected.
    """
    arr  = np.array(pil_img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    faces = _face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60),
    )
    if len(faces) == 0:
        return pil_img
    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
    margin = int(0.2 * max(w, h))
    H, W   = arr.shape[:2]
    x0 = max(0, x - margin)
    y0 = max(0, y - margin)
    x1 = min(W, x + w + margin)
    y1 = min(H, y + h + margin)
    return pil_img.crop((x0, y0, x1, y1))


# ------- model -------

def build_model(num_classes: int = 2):
    m = models.mobilenet_v2(weights=None)
    m.classifier[1] = nn.Sequential(
        nn.Linear(1280, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes),
    )
    return m


def load_model():
    m = build_model(num_classes=len(CLASSES)).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        state = torch.load(MODEL_PATH, map_location=DEVICE)
        m.load_state_dict(state)
        print(f"[startup] loaded weights from {MODEL_PATH}")
    else:
        print(f"[startup] WARNING: {MODEL_PATH} not found — predictions will be random")
    m.train(False)
    return m


# ------- app -------

app = FastAPI(
    title="Face Mask Detection API",
    description="Upload a face image, get back whether the person is wearing a mask.",
    version="1.0.0",
)

model = load_model()

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ------- helpers -------

def decide_action(predicted_class: str, confidence: float, threshold: float = 0.6) -> str:
    if confidence < threshold:
        return "Manual check (low confidence)"
    if predicted_class == "WithMask":
        return "Allow entry"
    return "Deny entry - please put on a mask"


# ------- routes -------

@app.get("/")
def root():
    index = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return JSONResponse({
        "service": "Face Mask Detection API",
        "model":   "MobileNetV2 (transfer learning)",
        "classes": CLASSES,
        "endpoints": ["/health", "/metrics", "/predict", "/docs"],
    })


@app.get("/health")
def health():
    return {
        "status":       "ok",
        "device":       str(DEVICE),
        "model_loaded": os.path.exists(MODEL_PATH),
    }


@app.get("/metrics")
def metrics():
    """Reported performance on the held-out test set (992 images, never seen during training)."""
    return {
        "test_accuracy":   0.9950,
        "f1_score":        0.998,
        "roc_auc":         1.0000,
        "test_set_size":   992,
        "inference_ms_cpu": 23,
        "model":           "MobileNetV2 (transfer learning, ImageNet)",
        "dataset":         "Face Mask 12K (Kaggle - ashishjangra27)",
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    threshold: float = Query(default=0.6, ge=0.0, le=1.0,
                             description="Confidence threshold for action decision (0-1)"),
                             ):
    # --- input validation ---
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="please upload an image file")

    contents = await file.read()

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="file too large — maximum size is 10 MB")

    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"could not read image: {e}")

    # --- face detection & inference ---
    t0       = time.perf_counter()
    face_img = detect_and_crop_face(img)
    face_detected = face_img is not img

    tensor = preprocess(face_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out   = model(tensor)
        probs = torch.softmax(out, dim=1)[0].cpu().numpy()

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    pred_idx   = int(probs.argmax())
    pred_class = CLASSES[pred_idx]
    conf       = float(probs[pred_idx])

    return JSONResponse({
        "status":         "mask_on" if pred_class == "WithMask" else "no_mask",
        "predicted_class": pred_class,
        "confidence":     round(conf, 4),
        "action":         decide_action(pred_class, conf, threshold),
        "face_detected":  face_detected,
        "inference_ms":   elapsed_ms,
        "all_probs":      {c: round(float(p), 4) for c, p in zip(CLASSES, probs)},
    })