"""
Live webcam / phone photo test for the Face Mask Detection API.

This is what the project doc means by "test with webcam/phone photos" -
grab frames from the laptop webcam (or any phone IP camera that exposes a
URL) and send them to the API in real time.

Usage:

    # 1) make sure the API is running (uvicorn app:app --port 8000)
    # 2) install opencv if you don't have it:
    #    pip install opencv-python
    # 3) for laptop webcam:
    #    python webcam_test.py
    # 4) for phone using IP Webcam app or similar:
    #    python webcam_test.py --source http://192.168.1.10:8080/video
    # 5) for a folder of phone photos:
    #    python webcam_test.py --photos ./phone_photos

Press 'q' to quit, 's' to save the current frame.
"""

import argparse
import io
import os
import sys
import time

import cv2
import requests


API = os.environ.get("API_URL", "http://localhost:8000")


def encode_jpg(frame):
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        return None
    return buf.tobytes()


def call_api(jpg_bytes):
    files = {"file": ("frame.jpg", jpg_bytes, "image/jpeg")}
    try:
        r = requests.post(f"{API}/predict", files=files, timeout=5)
        if r.ok:
            return r.json()
        return {"error": f"http {r.status_code}", "detail": r.text}
    except requests.RequestException as e:
        return {"error": str(e)}


def draw_overlay(frame, result):
    if "error" in result:
        text = f"API error: {result['error']}"
        color = (0, 0, 255)
    else:
        cls = result.get("predicted_class", "?")
        conf = result.get("confidence", 0)
        action = result.get("action", "")
        text = f"{cls} ({conf:.2f}) - {action}"
        color = (0, 200, 0) if cls == "WithMask" else (0, 0, 255)

    # background rectangle so the text is readable
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (0, 0, 0), -1)
    cv2.putText(frame, text, (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame


def run_live(source, every_n=10):
    # source can be 0 (laptop cam) or an http URL (phone IP cam)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[fail] cannot open video source: {source}")
        sys.exit(1)
    print(f"[info] live mode. press q to quit, s to save a frame")

    last_result = {"info": "warming up..."}
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[warn] dropped frame")
            time.sleep(0.05)
            continue

        # don't hammer the API every single frame, every Nth is enough
        if frame_idx % every_n == 0:
            jpg = encode_jpg(frame)
            if jpg:
                last_result = call_api(jpg)

        frame = draw_overlay(frame, last_result)
        cv2.imshow("Face Mask Detector - press q to quit", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            fname = f"capture_{int(time.time())}.jpg"
            cv2.imwrite(fname, frame)
            print(f"[info] saved {fname}")

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


def run_folder(folder):
    # send every image in a folder to the API and print the result
    files = [f for f in os.listdir(folder)
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not files:
        print(f"[fail] no images found in {folder}")
        sys.exit(1)

    print(f"[info] sending {len(files)} photos to {API}/predict")
    for f in sorted(files):
        path = os.path.join(folder, f)
        with open(path, "rb") as fh:
            data = fh.read()
        result = call_api(data)
        if "error" in result:
            print(f"  {f:30s} ERROR: {result['error']}")
        else:
            print(f"  {f:30s} -> {result['predicted_class']:12s} "
                  f"({result['confidence']:.3f})  action: {result['action']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=0,
                        help="webcam index (default 0) or video URL for phone IP cam")
    parser.add_argument("--photos", default=None,
                        help="folder of phone photos to send to the API")
    parser.add_argument("--every", type=int, default=10,
                        help="send every Nth frame to the API (live mode)")
    args = parser.parse_args()

    # quick health check first
    try:
        r = requests.get(f"{API}/health", timeout=3)
        if not r.ok:
            print(f"[fail] /health returned {r.status_code}")
            sys.exit(1)
        print(f"[info] API health: {r.json()}")
    except requests.RequestException as e:
        print(f"[fail] cannot reach API at {API}: {e}")
        sys.exit(1)

    if args.photos:
        run_folder(args.photos)
    else:
        # if --source is the string "0" turn it into int 0
        try:
            src = int(args.source)
        except (TypeError, ValueError):
            src = args.source
        run_live(src, every_n=args.every)


if __name__ == "__main__":
    main()
