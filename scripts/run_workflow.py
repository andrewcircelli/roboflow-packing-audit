"""R-0 gate: run the published Workflow against the LOCAL inference server.

This is the path the R-1 pipeline will use. The workflow is the deployable
artifact -- the model id, variant and threshold live inside it, not in this file,
which is the whole point of Workflows: the pipeline is versioned separately from
the code that calls it.

api_url is localhost:9001. There is no hosted API in this path.

Usage:
    ./.venv/bin/python scripts/run_workflow.py <image>
"""

import json
import os
import statistics
import sys
import time

import cv2
import supervision as sv
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient

LOCAL_SERVER = "http://localhost:9001"
CALLS = 4


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    image_path = sys.argv[1]

    load_dotenv()
    try:
        api_key = os.environ["ROBOFLOW_API_KEY"]
        workspace = os.environ["ROBOFLOW_WORKSPACE"]
        workflow_id = os.environ["ROBOFLOW_WORKFLOW_ID"]
    except KeyError as missing:
        print(f"missing {missing} in .env")
        return 1

    client = InferenceHTTPClient(api_url=LOCAL_SERVER, api_key=api_key)
    print(f"server    {LOCAL_SERVER}")
    print(f"workflow  {workspace}/{workflow_id}")
    print(f"image     {image_path}\n")

    timings, result = [], None
    for i in range(CALLS):
        start = time.perf_counter()
        result = client.run_workflow(
            workspace_name=workspace,
            workflow_id=workflow_id,
            images={"image": image_path},
        )
        timings.append((time.perf_counter() - start) * 1000)
        print(f"  call {i + 1} {'(cold)' if i == 0 else '(warm)'} {timings[-1]:8.0f} ms")

    warm = statistics.mean(timings[1:]) if len(timings) > 1 else timings[0]
    print(f"\nwarm mean   {warm:.0f} ms = {1000 / warm:.2f} FPS  (CPU, {os.uname().machine})")

    # run_workflow returns a list -- one entry per input image.
    outputs = result[0] if isinstance(result, list) else result
    key = next(iter(outputs))
    predictions = outputs[key].get("predictions", [])
    print(f"output key  '{key}'  ->  {len(predictions)} detection(s)")
    for p in predictions:
        print(f"  {p['class']:<16} {p['confidence']:.4f}  "
              f"x={p['x']:.0f} y={p['y']:.0f} {p['width']:.0f}x{p['height']:.0f}")

    frame = cv2.imread(image_path)
    detections = sv.Detections.from_inference(outputs[key])
    labels = [
        f"{n} {c:.3f}"
        for n, c in zip(detections.data.get("class_name", []), detections.confidence)
    ]
    annotated = sv.BoxAnnotator(thickness=2).annotate(frame.copy(), detections)
    annotated = sv.LabelAnnotator(text_scale=0.5).annotate(annotated, detections, labels)
    os.makedirs("out", exist_ok=True)
    out_path = f"out/{os.path.splitext(os.path.basename(image_path))[0]}__workflow.jpg"
    cv2.imwrite(out_path, annotated)
    print(f"\nwrote {out_path}")

    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/last-workflow-response.json", "w") as fh:
        json.dump(result, fh, indent=2)
    print("wrote artifacts/last-workflow-response.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
