"""Model-selection harness: score candidate models against MY OWN frames.

Universe mAP is measured in-distribution on the model's own held-out split.
That number did not survive contact with a real frame (parcel-vcjiz/1 reported
66.9% mAP and returned nothing above 0.163 confidence on a clean studio image).
So: measure candidates locally, on our images, before committing to one.

All inference is against the LOCAL server. No hosted API in this path.

Usage:
    ./.venv/bin/python scripts/probe.py <image> <model-id> [<model-id> ...]
"""

import os
import sys
import time

from dotenv import load_dotenv
from inference_sdk import InferenceConfiguration, InferenceHTTPClient

LOCAL_SERVER = "http://localhost:9001"
SWEEP = (0.5, 0.2, 0.05)


def score(client: InferenceHTTPClient, image: str, model_id: str) -> None:
    print(f"\n=== {model_id} ===")

    # Cold call pays for the weight download; warm call is the honest local number.
    client.configure(InferenceConfiguration(confidence_threshold=SWEEP[0]))
    start = time.perf_counter()
    try:
        client.infer(image, model_id=model_id)
    except Exception as exc:  # unreachable model id, bad key, server error
        print(f"  FAILED: {type(exc).__name__}: {exc}")
        return
    cold_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    client.infer(image, model_id=model_id)
    warm_ms = (time.perf_counter() - start) * 1000
    print(f"  cold {cold_ms:.0f} ms -> warm {warm_ms:.0f} ms")

    for thr in SWEEP:
        client.configure(InferenceConfiguration(confidence_threshold=thr))
        preds = client.infer(image, model_id=model_id).get("predictions", [])
        top = ", ".join(f"{p['class']}:{p['confidence']:.3f}" for p in preds[:5])
        print(f"  thr={thr:<5} {len(preds):>3} det  {top}")


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    image, model_ids = sys.argv[1], sys.argv[2:]

    load_dotenv()
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print("ROBOFLOW_API_KEY missing from .env")
        return 1

    client = InferenceHTTPClient(api_url=LOCAL_SERVER, api_key=api_key)
    for model_id in model_ids:
        score(client, image, model_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
