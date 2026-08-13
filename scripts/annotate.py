"""Render detections onto a frame with supervision.

Draws RAW model output -- no de-duplication -- so overlapping predictions stay
visible. Suppressing them is a build decision, not something this script should
hide.

Usage:
    ./.venv/bin/python scripts/annotate.py <image> <model-id> [threshold]
"""

import os
import sys

import cv2
import supervision as sv
from dotenv import load_dotenv
from inference_sdk import InferenceConfiguration, InferenceHTTPClient

LOCAL_SERVER = "http://localhost:9001"


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    image_path, model_id = sys.argv[1], sys.argv[2]
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2

    load_dotenv()
    client = InferenceHTTPClient(
        api_url=LOCAL_SERVER, api_key=os.environ["ROBOFLOW_API_KEY"]
    )
    client.configure(InferenceConfiguration(confidence_threshold=threshold))
    result = client.infer(image_path, model_id=model_id)

    frame = cv2.imread(image_path)
    detections = sv.Detections.from_inference(result)
    labels = [
        f"{name} {conf:.3f}"
        for name, conf in zip(
            detections.data.get("class_name", []), detections.confidence
        )
    ]

    annotated = sv.BoxAnnotator(thickness=2).annotate(frame.copy(), detections)
    annotated = sv.LabelAnnotator(text_scale=0.5).annotate(annotated, detections, labels)

    os.makedirs("out", exist_ok=True)
    stem = os.path.splitext(os.path.basename(image_path))[0]
    out_path = f"out/{stem}__{model_id.replace('/', '-')}__thr{threshold}.jpg"
    cv2.imwrite(out_path, annotated)

    print(f"{len(detections)} detection(s) at threshold {threshold}")
    for label, box in zip(labels, detections.xyxy):
        x1, y1, x2, y2 = (int(v) for v in box)
        print(f"  {label:<16} xyxy=({x1},{y1},{x2},{y2})  {x2 - x1}x{y2 - y1}px")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
