"""Pre-shoot check: will the camera side and the label side both work on THIS scene?

Two independent questions, which is the whole point of the correlation:

  CAMERA SIDE  Does YOLO-World detect the parcel? The prompt is a plain English
               string, so the right string is an empirical question -- a flat
               paper mailer may not match "cardboard box".

  LABEL SIDE   Does pyzbar decode the shipping label at this distance? Barcode
               decode degrades with resolution and rotation, and if it fails
               here it fails in every video frame too.

Run this BEFORE filming. Finding out the prompt is wrong after the shoot means
shooting twice.

Usage:
    ./.venv/bin/python scripts/check_frame.py data/bag.jpg
"""

import base64
import os
import sys

import cv2
import numpy as np
import requests
from dotenv import load_dotenv
from pyzbar.pyzbar import decode as zbar_decode

LOCAL_SERVER = "http://localhost:9001"

# Candidate prompts, plain English. YOLO-World scores prompts against each other,
# so each is sent alone -- sending them together would change the scores.
PROMPTS = [
    "cardboard box",
    "package",
    "parcel",
    "padded envelope",
    "brown paper bag",
    "shipping package",
    "mailer",
]


def camera_side(image_path: str, api_key: str) -> None:
    print("=" * 62)
    print("CAMERA SIDE - which prompt detects this parcel?")
    print("=" * 62)

    # Downscale before sending: YOLO-World resizes to its own input size anyway,
    # so a 5712px payload costs upload time and buys no accuracy.
    source = cv2.imread(image_path)
    height, width = source.shape[:2]
    resized = cv2.resize(source, (1920, int(height * 1920 / width)), interpolation=cv2.INTER_AREA)
    encoded = base64.b64encode(cv2.imencode(".jpg", resized)[1].tobytes()).decode()
    print(f"  (sent at 1920x{int(height * 1920 / width)}, source was {width}x{height})\n")

    for prompt in PROMPTS:
        payload = {
            "id": "check",
            "api_key": api_key,
            "image": {"type": "base64", "value": encoded},
            "text": [prompt],
            "confidence": 0.01,          # low floor: we want to SEE weak scores,
            "yolo_world_version_id": "l",  # not hide them behind the threshold
        }
        response = requests.post(f"{LOCAL_SERVER}/yolo_world/infer", json=payload, timeout=900)
        predictions = response.json().get("predictions", [])
        best = max((p["confidence"] for p in predictions), default=0.0)
        verdict = "PASSES thr=0.2" if best >= 0.2 else "below thr=0.2"
        print(f"  {prompt:<20} best={best:.3f}  n={len(predictions):<3} {verdict}")


def label_side(image_path: str) -> None:
    print()
    print("=" * 62)
    print("LABEL SIDE - does pyzbar decode the shipping label?")
    print("=" * 62)

    original = cv2.imread(image_path)
    height, width = original.shape[:2]
    print(f"  source image: {width}x{height}")

    grayscale = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    # Sweep resolution to find where decode BREAKS. That number is the answer to
    # "what resolution do I have to film at" -- measured, not guessed. Rotation is
    # swept too, because the label was shot sideways and zbar is orientation
    # sensitive.
    rotations = {
        "as shot": lambda im: im,
        "rot 90 CW": lambda im: cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE),
        "rot 90 CCW": lambda im: cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE),
    }

    for target_width in (width, 2560, 1920, 1280, 960, 640):
        if target_width > width:
            continue
        scaled = cv2.resize(
            grayscale,
            (target_width, int(height * target_width / width)),
            interpolation=cv2.INTER_AREA,
        )
        for rotation_name, rotate in rotations.items():
            found = zbar_decode(rotate(scaled))
            label = f"{target_width:>5}px {rotation_name:<11}"
            if not found:
                print(f"  {label} -- nothing")
                continue
            for barcode in found:
                data = barcode.data.decode("utf-8", errors="replace")
                print(f"  {label} -- {barcode.type:<9} '{data}'")


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    image_path = sys.argv[1]

    load_dotenv()
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print("ROBOFLOW_API_KEY missing from .env")
        return 1

    camera_side(image_path, api_key)
    label_side(image_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
