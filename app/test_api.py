"""
Smoke test for the Face Mask Detection API.

Usage:
    # 1) start the API in another terminal:
    #    uvicorn app:app --port 8000
    # 2) then run:
    #    python test_api.py /path/to/some_face.jpg

If no image path is given the script downloads a sample face image from a
public URL so you can quickly check that the API is alive.
"""

import sys
import os
import time
import requests

API = os.environ.get("API_URL", "http://localhost:8000")


def wait_for_api(timeout=20):
    # wait for the server to come up. useful when running in CI / docker
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = requests.get(f"{API}/health", timeout=2)
            if r.ok:
                return True
        except Exception:
            time.sleep(1)
    return False


def test_root():
    # GET / now serves the static index.html (the web UI), not JSON.
    # We just check it returns 200 and HTML.
    r = requests.get(f"{API}/")
    assert r.status_code == 200, f"GET / failed: {r.status_code}"
    ctype = r.headers.get("content-type", "")
    assert "html" in ctype.lower() or "json" in ctype.lower(), f"unexpected content-type: {ctype}"
    print(f"[ok] GET /            ->  ({ctype})")


def test_health():
    r = requests.get(f"{API}/health")
    assert r.status_code == 200, f"GET /health failed: {r.status_code}"
    data = r.json()
    assert data.get("status") == "ok"
    print("[ok] GET /health      -> ", data)


def test_predict(image_path):
    with open(image_path, "rb") as f:
        files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
        r = requests.post(f"{API}/predict", files=files, timeout=30)

    assert r.status_code == 200, f"predict failed: {r.status_code} - {r.text}"
    data = r.json()
    assert "predicted_class" in data
    assert "confidence" in data
    assert 0.0 <= data["confidence"] <= 1.0
    print(f"[ok] POST /predict    -> {data['predicted_class']} "
          f"(conf={data['confidence']}) action='{data['action']}'")


def test_bad_file():
    # non-image upload should be rejected
    files = {"file": ("not_an_image.txt", b"hello world", "text/plain")}
    r = requests.post(f"{API}/predict", files=files, timeout=10)
    assert r.status_code == 400, f"expected 400 for bad file, got {r.status_code}"
    print("[ok] reject non-image -> 400 as expected")


def get_sample_image():
    # use a Wikimedia Commons sample face for a quick smoke test
    url = "https://upload.wikimedia.org/wikipedia/commons/thumb/8/85/Elon_Musk_Royal_Society.jpg/220px-Elon_Musk_Royal_Society.jpg"
    out = "_sample.jpg"
    print(f"[info] downloading sample image: {url}")
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    with open(out, "wb") as f:
        f.write(r.content)
    return out


if __name__ == "__main__":
    print(f"[info] testing API at {API}")
    if not wait_for_api():
        print("[fail] API did not come up in time")
        sys.exit(1)

    test_root()
    test_health()

    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        img_path = get_sample_image()

    test_predict(img_path)
    test_bad_file()

    print("\nall tests passed.")
