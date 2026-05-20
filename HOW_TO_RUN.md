# How to Run — Face Mask Detection

Requirements: Python 3.10+

---

## Option A — Run Locally

### 1. Install dependencies

```bash
cd app
pip install -r requirements.txt
```

This installs PyTorch, FastAPI, and OpenCV — allow a few minutes on first run.

### 2. Add the model weights

Download `mask_detector.pth` from the [Releases page](../../releases) and place it inside the `app/` folder:

```
app/
├── app.py
├── mask_detector.pth   ← here
└── ...
```

### 3. Start the server

```bash
uvicorn app:app --port 8000
```

You should see:

```
[startup] loaded weights from mask_detector.pth
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 4. Open in browser

| URL | What you get |
|-----|--------------|
| `http://localhost:8000` | Web UI — upload a photo or use the live webcam |
| `http://localhost:8000/docs` | Swagger UI — try `/predict` directly from the browser |
| `http://localhost:8000/health` | Health check |

---

## Option B — Run with Docker

```bash
cd app
docker build -t face-mask-api .
docker run -p 8000:8000 face-mask-api
```

Then open `http://localhost:8000` as above, or call the API directly:

```bash
curl -X POST http://localhost:8000/predict \
     -F "file=@path/to/face.jpg"
```

---

## Testing

**Smoke tests** (requires the server to be running):

```bash
cd app
python test_api.py path/to/some_face.jpg
```

No image? The script will download a sample face automatically.

**Live webcam / phone photos:**

```bash
pip install opencv-python

python webcam_test.py                                       # laptop webcam
python webcam_test.py --photos ./phone_photos              # folder of images
python webcam_test.py --source http://PHONE_IP:8080/video  # phone IP camera
```

Press `q` to quit, `s` to save the current frame.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `pip not found` | Use `python -m pip install -r requirements.txt` instead |
| `port 8000 already in use` | Add `--port 8001` to the uvicorn command |
| Camera won't open in browser | Use Chrome and allow camera permissions when prompted |
| `ModuleNotFoundError: torchvision` | Run `pip install torchvision` separately |
| Predictions are all wrong | Make sure `mask_detector.pth` is inside the `app/` folder |
| `[startup] WARNING: mask_detector.pth not found` | Same as above — model file is missing |