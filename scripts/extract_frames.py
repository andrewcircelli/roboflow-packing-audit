"""Report a video's real properties and pull sample frames out of it.

Two jobs:

  1. Print what the file ACTUALLY is -- resolution, fps, frame count, duration.
     A phone that shoots 24MP stills does not shoot 24MP video, and the barcode
     decode floor was measured on stills. This is where that assumption gets
     checked.

  2. Save evenly-spaced frames so the same checks that ran on the photos can run
     on real frames.

Frames are written at full resolution. Do not downscale here -- decode needs the
native pixels, and check_frame.py handles its own scaling.

Usage:
    ./.venv/bin/python scripts/extract_frames.py data/bag.mov [count]
"""

import os
import sys

import cv2


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    video_path = sys.argv[1]
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        print(f"could not open {video_path}")
        return 1

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps if fps else 0

    print(f"file       {video_path}")
    print(f"resolution {width} x {height}")
    print(f"fps        {fps:.2f}")
    print(f"frames     {total}")
    print(f"duration   {duration:.1f} s")

    # At ~2.4 FPS of inference, processing every frame is rarely realistic.
    # This is the number that forces the frame-sampling decision.
    if fps:
        print(f"\nprocessing EVERY frame at 2.42 FPS would take "
              f"{total / 2.42 / 60:.1f} minutes")

    os.makedirs("data/frames", exist_ok=True)
    written = []
    for i in range(count):
        index = int(total * i / count) if total else 0
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            print(f"  frame {index}: read failed")
            continue
        path = f"data/frames/frame_{index:05d}.jpg"
        cv2.imwrite(path, frame)
        written.append(path)
        print(f"  wrote {path}")

    capture.release()
    if written:
        print(f"\nnext:\n  ./.venv/bin/python scripts/check_frame.py {written[len(written) // 2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
