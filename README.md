# Face Mask Detection

End-to-end deep learning project that detects whether a person in an image is wearing a face mask, served behind a FastAPI endpoint and packaged in Docker.

```
Kaggle dataset → PyTorch (transfer learning) → .pth → FastAPI → Docker
```

**Test accuracy: 99.50%** on 992 held-out images &nbsp;|&nbsp; F1: 0.998 &nbsp;|&nbsp; ROC-AUC: 1.0000

---

## Project Structure

```
face_mask_project/
├── notebooks/
│   ├── eda_notebook.ipynb        # 8+ visualizations and dataset insights
│   ├── training_notebook.ipynb   # training loop, augmentation, loss/accuracy curves
│   └── evaluation.ipynb          # confusion matrix, F1, ROC-AUC, error analysis
├── app/
│   ├── app.py                    # FastAPI service (/predict, /health, /ui)
│   ├── static/
│   │   └── index.html            # web UI — upload image or use live webcam
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── test_api.py               # smoke tests
│   └── webcam_test.py            # live webcam / phone-photo client
├── train_robust.py               # standalone training script
└── README.md
```

---

## Dataset

[Face Mask 12K Images Dataset](https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset) — 2 classes: `WithMask` / `WithoutMask`, pre-split by the dataset author.

| Split | Images |
|-------|--------|
| Train | ~10,000 |
| Validation | ~800 |
| Test | ~992 |

---

## Model

**MobileNetV2** pretrained on ImageNet, fine-tuned for binary mask classification.

| Component | Detail |
|-----------|--------|
| Backbone | MobileNetV2 — frozen except last 2 inverted-residual blocks |
| Head | `Linear(1280→256) → ReLU → Dropout(0.5) → Linear(256→2)` |
| Loss | CrossEntropy with label smoothing 0.05 |
| Optimizer | AdamW (lr=3e-4, weight_decay=1e-4) |
| Schedule | Cosine annealing |
| Augmentation | RandomResizedCrop, RandomAffine, ColorJitter, GaussianBlur, RandomErasing |
| Epochs / Batch | 10 / 32 |

The backbone is only partially unfrozen (last 2 blocks) so the model learns mask-specific features rather than just overfitting the folder names. Heavy augmentation means the model trains on harder images than it evaluates on — a healthy gap that confirms there is no overfitting.

---

## Input Pipeline — Face Detection

Before classification, the API runs an **OpenCV Haar cascade** to detect and crop the face from the incoming image. This ensures the model always sees the same tight face-crop framing it was trained on, regardless of whether the input is a close portrait or a wide webcam frame.

The response includes a `face_detected` field. If no face is found, the full image is used as a fallback.

---

## Results

| Metric | Value |
|--------|-------|
| Test accuracy | **99.50%** |
| F1 score | 0.998 |
| ROC-AUC | 1.0000 |
| Inference speed | ~23 ms / image (CPU) |
| Best val accuracy | 100.00% (epoch 2) |

Training curves and the full evaluation (confusion matrix, per-class metrics, misclassification analysis) are in the notebooks.

---

## Running the API Locally

```bash
cd app
pip install -r requirements.txt
# download mask_detector.pth (see Releases) and place it in app/
uvicorn app:app --reload --port 8000
```

| URL | Description |
|-----|-------------|
| `http://localhost:8000` | Web UI — upload a photo or use the webcam |
| `http://localhost:8000/docs` | Swagger UI — try `/predict` in the browser |
| `http://localhost:8000/health` | Health check |

**Smoke tests:**
```bash
python test_api.py path/to/face.jpg
```

**Live webcam / phone photos:**
```bash
pip install opencv-python
python webcam_test.py                                      # laptop webcam
python webcam_test.py --photos ./phone_photos             # folder of images
python webcam_test.py --source http://PHONE_IP:8080/video # phone IP camera
```

---

## Running with Docker

```bash
cd app
docker build -t face-mask-api .
docker run -p 8000:8000 face-mask-api
```

```bash
curl -X POST http://localhost:8000/predict \
     -F "file=@path/to/face.jpg"
```

Example response:

```json
{
  "status": "mask_on",
  "predicted_class": "WithMask",
  "confidence": 0.9812,
  "action": "Allow entry",
  "face_detected": true,
  "all_probs": { "WithMask": 0.9812, "WithoutMask": 0.0188 }
}
```

---

## Training From Scratch

Download the [Kaggle dataset](https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset) and extract it under `data/` so the structure is:

```
data/
├── Train/
│   ├── WithMask/
│   └── WithoutMask/
├── Validation/
└── Test/
```

Then run:

```bash
pip install torch torchvision
python train_robust.py
```

The best checkpoint is saved to `app/mask_detector.pth` automatically.

---

## Deliverables

- [x] EDA notebook with 8+ visualizations
- [x] Trained model exported as `.pth`
- [x] Training report — loss/accuracy curves
- [x] Evaluation — confusion matrix, per-class metrics, F1, ROC-AUC
- [x] FastAPI `/predict` endpoint
- [x] Dockerfile with health check
- [x] `test_api.py` smoke tests
- [x] Live webcam client

---

## Notes

- ImageNet mean/std normalization is used in both training and inference — changing one without the other will break predictions.
- Horizontal flip is safe for face data (faces are roughly symmetric). Vertical flip is intentionally excluded.
- The validation set (~800 images) is small so val accuracy is noisy. Final numbers are reported on the held-out test set (~992 images).