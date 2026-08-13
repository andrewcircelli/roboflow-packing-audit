"""Zero-shot detection via YOLO-World on the LOCAL inference server.

Open-vocabulary: the classes are text prompts, not a trained label set. No
Universe model id, no training, no labelling.

Reports cold vs. warm latency separately, because the first call pays for the
weight download and quoting that number as "inference latency" is wrong by ~40x.
Writes an annotated frame, because scores alone cannot reveal that a model is
confidently looking at the wrong part of the image.

Usage:
    ./.venv/bin/python scripts/zeroshot.py <image> "<prompt>" ["<prompt>" ...]

Env:
    THRESHOLD  confidence floor (default 0.2)
    CALLS      timed calls; call 1 is cold, the rest are warm (default 4)
"""

import base64
import os
import statistics
import sys
import time

import cv2
import requests
import supervision as sv
from dotenv import load_dotenv

LOCAL_SERVER = "http://localhost:9001"
ENDPOINT = f"{LOCAL_SERVER}/yolo_world/infer"


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    image_path, prompts = sys.argv[1], sys.argv[2:]
    threshold = float(os.environ.get("THRESHOLD", 0.2))
    calls = int(os.environ.get("CALLS", 4))

    load_dotenv()
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print("ROBOFLOW_API_KEY missing from .env")
        return 1

    with open(image_path, "rb") as fh:
        encoded = base64.b64encode(fh.read()).decode()

    payload = {
        "id": "zeroshot",
        "api_key": api_key,
        "image": {"type": "base64", "value": encoded},
        "text": prompts,
        "confidence": threshold,
    }

    print(f"image   {image_path}")
    print(f"prompts {prompts}")
    print(f"thr     {threshold}\n")

    timings = []
    result = None
    for i in range(calls):
        start = time.perf_counter()
        response = requests.post(ENDPOINT, json=payload, timeout=900)
        elapsed = (time.perf_counter() - start) * 1000
        response.raise_for_status()
        result = response.json()
        timings.append(elapsed)
        tag = "cold" if i == 0 else "warm"
        print(f"  call {i + 1} ({tag:<4}) {elapsed:8.0f} ms   "
              f"{len(result.get('predictions', []))} det")

    warm = timings[1:] or timings
    mean_ms = statistics.mean(warm)
    print(f"\ncold           {timings[0]:.0f} ms   (includes weight download if uncached)")
    print(f"warm mean      {mean_ms:.0f} ms over {len(warm)} calls")
    print(f"throughput     {1000 / mean_ms:.2f} FPS   (CPU, {os.uname().machine})\n")

    for p in result.get("predictions", []):
        print(f"  {p['class']:<16} {p['confidence']:.3f}  "
              f"x={p['x']:.0f} y={p['y']:.0f} {p['width']:.0f}x{p['height']:.0f}")

    frame = cv2.imread(image_path)
    detections = sv.Detections.from_inference(result)
    labels = [
        f"{n} {c:.3f}"
        for n, c in zip(detections.data.get("class_name", []), detections.confidence)
    ]
    annotated = sv.BoxAnnotator(thickness=2).annotate(frame.copy(), detections)
    annotated = sv.LabelAnnotator(text_scale=0.5).annotate(annotated, detections, labels)

    os.makedirs("out", exist_ok=True)
    stem = os.path.splitext(os.path.basename(image_path))[0]
    out_path = f"out/{stem}__yolo-world.jpg"
    cv2.imwrite(out_path, annotated)
    print(f"\nwrote {out_path}  <-- OPEN THIS. Verify the box is on the parcel.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
