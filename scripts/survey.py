"""Diagnostic: what do both signals do across the whole clip?

NOT the pipeline. One inference call per frame, no zone, no correlation, no
decisions -- just the raw temporal picture, so the zone geometry and the
frame-sampling rate get chosen from data instead of intuition.

Prints, per frame: parcel confidence, the detection's center and coverage, and
whether the barcode decodes.

Usage:
    ./.venv/bin/python scripts/survey.py data/frames
"""

import os
import sys

import cv2
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient
from pyzbar.pyzbar import decode as zbar_decode

LOCAL_SERVER = "http://localhost:9001"


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    frames_dir = sys.argv[1]

    load_dotenv()
    client = InferenceHTTPClient(
        api_url=LOCAL_SERVER, api_key=os.environ["ROBOFLOW_API_KEY"]
    )
    workspace = os.environ["ROBOFLOW_WORKSPACE"]
    workflow_id = os.environ["ROBOFLOW_WORKFLOW_ID"]

    paths = sorted(
        os.path.join(frames_dir, f)
        for f in os.listdir(frames_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    print(f"{'frame':<18} {'conf':>6} {'cx':>6} {'cy':>6} {'%frame':>7}  barcode")
    print("-" * 62)

    for path in paths:
        result = client.run_workflow(
            workspace_name=workspace,
            workflow_id=workflow_id,
            images={"image": path},
        )
        outputs = result[0] if isinstance(result, list) else result
        predictions = outputs[next(iter(outputs))].get("predictions", [])

        image = cv2.imread(path)
        frame_h, frame_w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        codes = zbar_decode(gray)
        barcode = codes[0].data.decode("utf-8", errors="replace") if codes else "-"

        name = os.path.basename(path)
        if not predictions:
            print(f"{name:<18} {'-':>6} {'-':>6} {'-':>6} {'-':>7}  {barcode}")
            continue

        best = max(predictions, key=lambda p: p["confidence"])
        coverage = 100 * (best["width"] * best["height"]) / (frame_w * frame_h)
        print(
            f"{name:<18} {best['confidence']:6.3f} {best['x']:6.0f} {best['y']:6.0f} "
            f"{coverage:6.1f}%  {barcode}"
        )

    print(f"\nframe size {frame_w}x{frame_h}  ({len(paths)} frames sampled)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
