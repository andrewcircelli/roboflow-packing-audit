"""Frame loop: video in, annotated video out, with both signals per frame.

This module is deliberately dumb. It moves frames, calls the workflow, times
things, draws boxes and writes files. Every decision with a defensible answer
lives in zones.py, decode.py or correlate.py.

Inference runs against the LOCAL server. No hosted API in this path.

Full-resolution frames go to decode (the barcode needs native pixels); the
output video is downscaled, because nothing about a written MP4 needs 4K.

Usage:
    ./.venv/bin/python -m audit.pipeline data/bag.MOV
    ./.venv/bin/python -m audit.pipeline data/bag.MOV --out out/audit.mp4
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import supervision as sv
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient

from audit import correlate, decode, zones

LOCAL_SERVER = "http://localhost:9001"
OUTPUT_MAX_WIDTH = 1080


@dataclass
class FrameResult:
    """One frame's worth of evidence. Deliberately records both signals
    separately -- the whole point is that they are independent."""

    index: int
    timestamp_s: float
    parcel_detected: bool
    parcel_confidence: float
    parcel_center: tuple[int, int] | None
    in_zone: bool
    barcode: str | None
    verdict: str
    inference_ms: float
    decode_ms: float
    total_ms: float

    # How many detections the model returned BEFORE duplicate suppression.
    # The single-parcel assumption means everything past the first is dropped;
    # recording the raw count means it is dropped visibly rather than silently.
    # A frame where this is >1 after NMS is a second parcel in the station.
    detections_raw: int = 0


@dataclass
class RunStats:
    """Everything accumulated across a run. Carries the source video's
    properties so the summary can state throughput against video rate, not just
    inference rate."""

    results: list[FrameResult] = field(default_factory=list)
    frames_read: int = 0
    frames_processed: int = 0
    source_fps: float = 30.0
    sample_every: int = 1


# hardcode LOCAL_SERVER as a module constant, not a parameter so no code path reaches a hosted API
def build_client() -> tuple[InferenceHTTPClient, str, str]:
    load_dotenv()
    client = InferenceHTTPClient(
        api_url=LOCAL_SERVER, api_key=os.environ["ROBOFLOW_API_KEY"]
    )
    return client, os.environ["ROBOFLOW_WORKSPACE"], os.environ["ROBOFLOW_WORKFLOW_ID"]


def infer(
    client, workspace, workflow_id, frame: np.ndarray
) -> tuple[sv.Detections, float]:
    """One workflow call. Returns detections and the wall-clock cost.

    The frame is handed over as a numpy array. inference_sdk accepts arrays
    directly, so encoding a 4K JPEG to disk and passing a path -- just to have
    the SDK read it straight back -- was pure overhead.
    """
    start = time.perf_counter()
    result = client.run_workflow(
        workspace_name=workspace, workflow_id=workflow_id, images={"image": frame}
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    outputs = (
        result[0] if isinstance(result, list) else result
    )  # run_workflow returns a list (one entry per input image)
    payload = outputs[next(iter(outputs))]
    return sv.Detections.from_inference(payload), elapsed_ms


def annotate(
    frame: np.ndarray, detections: sv.Detections, zone_polygon, result: FrameResult
) -> np.ndarray:
    """Draw the evidence: the zone, the detection, and the verdict.

    An operator watching this should be able to see WHY a parcel was diverted,
    not just that it was. That is the difference between a demo and a runbook.
    """
    annotated = frame.copy()

    if zone_polygon is not None:
        cv2.polylines(
            annotated, [zone_polygon.astype(np.int32)], True, (255, 200, 0), 3
        )

    # The zone is a target for the parcel's CENTER, not a container for the
    # parcel -- the parcel covers ~90% of the frame and could never fit inside
    # it. Drawing the center point makes the comparison legible: a viewer sees
    # the dot cross into the band rather than wondering why a huge box is being
    # judged by a thin stripe.
    if result.parcel_center is not None:
        cv2.circle(annotated, result.parcel_center, 18, (255, 200, 0), -1)
        cv2.circle(annotated, result.parcel_center, 18, (255, 255, 255), 3)

    if len(detections) > 0:
        annotated = sv.BoxAnnotator(thickness=3).annotate(annotated, detections)
        labels = [
            f"{name} {conf:.2f}"
            for name, conf in zip(
                detections.data.get("class_name", []), detections.confidence
            )
        ]
        annotated = sv.LabelAnnotator(text_scale=0.7).annotate(
            annotated, detections, labels
        )

    color = {"PASS": (0, 200, 0), "DIVERT": (0, 0, 255)}.get(
        result.verdict, (160, 160, 160)
    )
    banner = [
        f"frame {result.index}  t={result.timestamp_s:.1f}s",
        f"parcel: {'yes' if result.parcel_detected else 'no'} ({result.parcel_confidence:.2f})",
        f"in zone: {result.in_zone}",
        f"barcode: {result.barcode or '-'}",
        f"verdict: {result.verdict}",
        f"{result.inference_ms:.0f} ms inference",
    ]
    for i, line in enumerate(banner):
        cv2.putText(
            annotated,
            line,
            (20, 50 + i * 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            color,
            3,
            cv2.LINE_AA,
        )
    return annotated


def process(video_path: str, output_path: str) -> RunStats:
    client, workspace, workflow_id = build_client()

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise SystemExit(f"could not open {video_path}")

    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Process every 10th frame: 3 samples per second of video at 30 fps.
    #
    # The parcel dwells at the read position for roughly 4 seconds, so a stride
    # of 10 puts ~13 inferences inside a dwell -- far more than needed to reach a
    # verdict, and the redundancy is what absorbs the single-frame decode
    # dropouts seen mid-dwell.
    #
    # This does NOT keep up with a live feed. At ~680 ms per frame the pipeline
    # sustains 1.5 frames/sec against the 3.0 it would need at this stride,
    # measured at 0.5x real time. Processing a live camera on this hardware would
    # require a stride of ~21. The run summary prints both numbers rather than a
    # single "FPS", because inference throughput and video rate are not the same
    # figure and conflating them overstates what the system does.
    sample_every = 10

    zone_polygon = zones.build_zone(width, height)

    scaled_width = min(OUTPUT_MAX_WIDTH, width)
    scaled_height = int(height * scaled_width / width)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    writer = cv2.VideoWriter(
        output_path,
        # avc1 = H.264. QuickTime will not play mp4v (MPEG-4 Part 2), which
        # makes the output unwatchable on the machine doing the demo.
        cv2.VideoWriter_fourcc(*"avc1"),
        max(1.0, source_fps / sample_every),
        (scaled_width, scaled_height),
    )

    print(
        f"source     {video_path}  {width}x{height} @ {source_fps:.1f} fps, {total_frames} frames"
    )
    print(
        f"sampling   every {sample_every} frames -> ~{total_frames // sample_every} inferences"
    )
    print(f"output     {output_path}  {scaled_width}x{scaled_height}\n")

    stats = RunStats(source_fps=source_fps, sample_every=sample_every)
    index = -1
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        index += 1
        stats.frames_read += 1
        if index % sample_every:
            continue

        frame_start = time.perf_counter()
        detections, inference_ms = infer(client, workspace, workflow_id, frame)
        detections_raw = len(detections)

        # Single-parcel station by design: collapse to the highest-confidence
        # detection rather than tuning an NMS threshold. The raw count above is
        # what makes that assumption visible -- a frame reporting more than one
        # detection is a second parcel being discarded, and it shows in the log.
        if len(detections) > 1:
            keep = int(np.argmax(detections.confidence))
            detections = detections[[keep]]

        parcel_detected = len(detections) > 0
        confidence = float(detections.confidence.max()) if parcel_detected else 0.0
        center = None
        if parcel_detected:
            best = int(np.argmax(detections.confidence))
            x1, y1, x2, y2 = detections.xyxy[best]
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

        in_zone = zones.is_in_zone(detections, zone_polygon, (width, height))

        # The full-resolution frame is passed deliberately. Decode fails below
        # ~1920px frame width on this footage, so any downscaling here would
        # silently destroy the label side of the correlation.
        decode_start = time.perf_counter()
        barcode = decode.read_barcode(frame, detections, zone_polygon)
        decode_ms = (time.perf_counter() - decode_start) * 1000

        verdict = correlate.decide(
            parcel_detected=parcel_detected, in_zone=in_zone, barcode=barcode
        )

        result = FrameResult(
            index=index,
            timestamp_s=index / source_fps,
            parcel_detected=parcel_detected,
            parcel_confidence=confidence,
            parcel_center=center,
            in_zone=in_zone,
            barcode=barcode,
            verdict=verdict,
            inference_ms=inference_ms,
            decode_ms=decode_ms,
            total_ms=(time.perf_counter() - frame_start) * 1000,
            detections_raw=detections_raw,
        )
        stats.results.append(result)
        stats.frames_processed += 1

        annotated = annotate(frame, detections, zone_polygon, result)
        writer.write(
            cv2.resize(
                annotated, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA
            )
        )

        # `raw` is the detection count before suppression. A frame showing raw=2
        # is the model reporting two parcels -- either a duplicate box on one
        # parcel, or a genuine second parcel the single-parcel assumption is
        # about to discard. Printing it means that choice is never silent.
        print(
            f"  f{index:<5} {result.verdict:<12} parcel={confidence:5.3f} "
            f"raw={detections_raw} "
            f"zone={in_zone!s:<5} code={barcode or '-':<16} "
            f"{inference_ms:5.0f}ms inf {decode_ms:5.0f}ms dec"
        )

    capture.release()
    writer.release()
    return stats


def report(stats: RunStats) -> None:
    """Print the run summary.

    Four choices worth stating, because each one is a way a latency number can
    mislead:

      - The cold call is reported separately. It loads the model into the server
        and costs 8.4s after a container restart against ~680ms warm; averaging
        it in makes every figure quietly wrong.
      - p95 is reported alongside the mean, because a mean alone hides the tail.
      - Inference and decode are reported separately, which answers where
        optimization would pay before anyone has to guess.
      - Throughput is given as two labeled figures -- what the pipeline
        achieves and what tracking the video would require. They differ by the
        sampling stride, and quoting one number as "FPS" overstates the system.
    """
    if not stats.results:
        print("\nno frames processed")
        return

    def pct(values: list[float], p: float) -> float:
        ordered = sorted(values)
        return ordered[min(int(len(ordered) * p), len(ordered) - 1)]

    # The first call loads the model into the server. Averaging it in makes
    # every latency number quietly wrong, so it is reported on its own.
    cold, warm = stats.results[0], stats.results[1:] or stats.results
    inference = [r.inference_ms for r in warm]
    decoding = [r.decode_ms for r in warm]
    total = [r.total_ms for r in warm]

    print(f"\nframes read {stats.frames_read}, "
          f"processed {stats.frames_processed} (every {stats.sample_every})")
    print(f"cold start   {cold.total_ms:7.0f} ms  (first call, includes model load)\n")

    print(f"  {'warm latency':<14}{'mean':>8}{'p95':>8}{'min':>8}{'max':>8}   (ms)")
    for name, series in (("inference", inference), ("decode", decoding), ("total", total)):
        print(f"  {name:<14}{statistics.mean(series):8.0f}{pct(series, 0.95):8.0f}"
              f"{min(series):8.0f}{max(series):8.0f}")

    achieved = 1000 / statistics.mean(total)
    required = stats.source_fps / stats.sample_every
    verdict = "keeps up" if achieved >= required else "DOES NOT keep up"
    print(f"\n  achieved     {achieved:8.2f} frames/sec   (wall clock, this hardware)")
    print(f"  required     {required:8.2f} frames/sec   "
          f"(to track {stats.source_fps:.0f} fps video at stride {stats.sample_every})")
    print(f"  real-time    {achieved / required:8.2f}x          {verdict}")

    counts: dict[str, int] = {}
    for r in stats.results:
        counts[r.verdict] = counts.get(r.verdict, 0) + 1
    print("\n  verdicts")
    for name, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"    {name:<14}{n:>4}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Packing-station audit pipeline")
    parser.add_argument("video")
    parser.add_argument("--out", default="out/audit.mp4")
    args = parser.parse_args()

    stats = process(args.video, args.out)
    report(stats)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
