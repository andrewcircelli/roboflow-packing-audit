# Packing-station audit

A packing-station quality gate that runs entirely on a self-hosted Roboflow
inference server. It reconciles two independent signals — what the **barcode
says the parcel should be** and what the **camera says is actually there** — and
routes the disagreements to a QA lane.

Video in, annotated video out, with per-frame latency and a verdict per frame.
No hosted API anywhere in the inference path. Verified running with the network
physically down.

---

## Why two signals

A barcode scanner alone cannot tell you the difference between these two cases:

| Parcel present | Barcode read | Meaning |
|---|---|---|
| no | none | idle station, nothing happening |
| **yes** | **none** | **unlabelled parcel — must leave the line** |

The scanner reports *nothing* in both. One is a defect; the other is a quiet
Tuesday. Only a camera observing the station directly separates them.

That is the whole argument, and it is not a design preference — it turned out to
be forced by optics. See [Calibration](#calibration): a single camera cannot
simultaneously frame a whole parcel and resolve the barcode on it. Real
facilities mount a separate scanner for exactly this reason.

---

## Quick start

### Prerequisites

| | Verified against |
|---|---|
| macOS | 26.5.1, Apple Silicon (arm64) |
| Docker Desktop | 29.7.2, daemon running |
| Python | 3.12 (`brew install python@3.12`) |
| zbar | `brew install zbar` — required by `pyzbar` |
| Roboflow account | private API key, not the publishable one |

Python 3.12 specifically. The CV ecosystem does not yet ship wheels for 3.14,
which is the current macOS default.

### Install

```bash
git clone https://github.com/andrewcircelli/roboflow-packing-audit
cd roboflow-packing-audit

python3.12 -m venv .venv
./.venv/bin/pip install -r requirements.txt

cp .env.example .env      # then fill in the three values
```

`.env` needs:

```
ROBOFLOW_API_KEY=...        # private key, from workspace settings
ROBOFLOW_WORKSPACE=...      # workspace slug, e.g. andrew-c-fcfp2
ROBOFLOW_WORKFLOW_ID=...    # e.g. custom-workflow
```

The workspace slug is *not* the project name. If you don't know it:

```bash
curl -s "https://api.roboflow.com/?api_key=$ROBOFLOW_API_KEY"
```

### Start the inference server

```bash
./.venv/bin/inference server start
docker ps                                   # confirm the container and port mapping
curl -s http://localhost:9001/info          # {"name":"Roboflow Inference Server",...}
```

First run pulls `roboflow/roboflow-inference-server-cpu` (~15.6 GB). The image
publishes a real arm64/linux manifest, so it runs natively on Apple Silicon with
no emulation. This matters beyond convenience: emulated inference would still
work while making every latency figure below meaningless.

### Import the workflow

`workflow.json` is the exported Workflow definition. Import it in the Roboflow UI
(Workflows → Create Workflow → JSON editor), publish it, and put its ID in
`.env`. It pins the two settings that took longest to establish — model variant
`l` and confidence `0.2`.

### Run

```bash
./.venv/bin/python -m audit.pipeline data/your-clip.MOV --out out/audit.mp4
```

---

## Reading the output

Per frame:

```
f150   PASS         parcel=0.770 raw=1 zone=True  code=TBA333622790661   452ms inf  188ms dec
```

| Field | Meaning |
|---|---|
| `f150` | source frame index |
| `PASS` | verdict for this frame |
| `parcel=0.770` | detection confidence |
| `raw=1` | detections **before** duplicate suppression — see [Assumptions](#assumptions-stated-because-they-will-break-somewhere-else) |
| `zone=True` | the parcel's centre is at the read position |
| `code=` | validated tracking identifier, or `-` |

Verdicts:

| | Meaning |
|---|---|
| `IDLE` | no parcel detected |
| `APPROACHING` | parcel detected, not yet at the read position |
| `PASS` | parcel at the read position with a valid tracking identifier |
| `DIVERT` | parcel at the read position with **no valid identity** |

The annotated video draws the read window, the detection box, the parcel's centre
point, and a banner carrying both signals and the verdict. An operator should be
able to see *why* a parcel diverted, not only that it did.

---

## How it works

```
frame ──► workflow @ localhost:9001 ──► detections ──► collapse duplicates
                                                            │
                       ┌────────────────────────────────────┤
                       ▼                                    ▼
              zone: at read position?          decode: valid tracking code?
                       │                                    │
                       └──────────────► correlate ◄─────────┘
                                             │
                                             ▼
                                IDLE / APPROACHING / PASS / DIVERT
```

| File | Responsibility |
|---|---|
| `audit/pipeline.py` | frame loop, workflow calls, latency capture, annotation |
| `audit/zones.py` | the read window |
| `audit/decode.py` | barcode decode and identity validation |
| `audit/correlate.py` | reconciliation of the two signals |
| `workflow.json` | the Roboflow Workflow — model, variant, threshold |

The detection step lives in the Workflow rather than in code. The model ID,
variant and confidence threshold are versioned separately from the pipeline that
calls them, which is the point of Workflows: swapping the model is a workflow
change, not a code change.

---

## Calibration

Every number in this pipeline was derived from footage rather than chosen.

### The prompt is a tuning parameter

Detection is zero-shot via YOLO-World, so the class is a plain English string.
Same frame, same parcel, one prompt per request:

| Prompt | Confidence |
|---|---|
| **`brown paper bag`** | **0.970** |
| `cardboard box` | 0.338 |
| `padded envelope` | 0.131 |
| `package` | 0.090 |
| `parcel`, `mailer`, `shipping package` | ~0.02 or nothing |

A 4.5× swing from word choice alone. `cardboard box` was the obvious first guess
and nearly the worst usable option.

Prompts must be swept **one per request** — open-vocabulary models score prompts
relative to each other, so sending them together changes the scores.

### The single-camera trade-off

| Framing | Detection | Barcode decodes down to |
|---|---|---|
| Wide | 0.969 | 4284px only |
| **Mid + added light** | **0.929** | **2560px** |
| Close | 0.187 — fails | 1920px |

Moving closer improves decode and destroys detection: at close range the parcel
fills the frame and the model loses the surrounding context it relies on. Adding
light was the only change that improved both sides at once.

**One camera cannot simultaneously frame the parcel and resolve its barcode.**
That is optics, not tuning, and it is why the two-sensor architecture is forced
rather than chosen.

### Motion blur dominates resolution

Identical 2160×3840 source, same distance, same lighting:

| Frame | Detection | Barcode decode |
|---|---|---|
| Parcel moving | 0.446 — passed threshold | **failed at every scale** |
| Parcel at rest | 0.970 | **decodes down to 1920px** |

Detection tolerates motion blur; decode does not. Fixed-mount scanners use fast
shutters and strobe illumination for this reason.

Consequence for the deployment: the parcel comes to rest for ~2 seconds. That is
not a workaround — a packing station holds the parcel while it is processed,
unlike a sortation conveyor. The realistic behaviour and the working behaviour
coincide here; on a moving line they would not, and the answer there is a
separate scanner rather than a better prompt.

### The read window

Surveying the clip, with frame centre at x=1080:

```
cx= 734   entering, off-centre     barcode: no
cx=1073   arriving                 barcode: no
cx=1080   settled                  barcode: TBA333622790661
cx=1080   settled                  barcode: TBA333622790661
cx=1343   leaving, off-centre      barcode: no
```

The barcode decodes **only** inside the settled window, so the zone predicts
decodability. Half-width is 150px — between the settled cluster at 1080 and the
nearest reject at 1343, with margin either side.

The zone constrains **x only**. `cy` overlapped heavily between settled
(1992–2172) and in-transit (2035–2307) frames, so it does not discriminate.
Constraining a dimension that does not discriminate adds a way to fail without
adding a way to be right.

### Threshold

Confidence threshold is 0.2. What makes that defensible rather than arbitrary is
the negative control: **an empty station scored 0.000 on every prompt tested.**
The gap below 0.2 is genuinely empty, not merely unexamined.

### Model selection

Four Roboflow Universe models were evaluated against local frames and rejected
before settling on zero-shot detection:

| Model | Result |
|---|---|
| `parcel-vcjiz/1` | advertised 66.9% mAP; observed max 0.163 |
| `box-plzjm/1` | no detections at any threshold down to 0.05 |
| `parcel-box-jsl0q/1` | 0.905 on a pristine carton (class `BADBOX`), 0.474 on a photo of a blank wall |
| `box-box-hh0mq/4` | passed a negative control, then localised on the empty table rather than the parcel |

Three failures worth separating, because each needed a different check to catch:

1. **Published metrics did not transfer.** mAP is measured in-distribution on the
   model's own held-out split — 28 images, same camera, same day. Every frame in
   a real deployment is out of that distribution.
2. **High confidence is not correctness.** The best-scoring model of the four
   asserted a good parcel on a photograph of a blank wall. A **negative control**
   caught it; no threshold would have.
3. **Correct scores, wrong pixels.** The model that survived the negative control
   was still detecting the table. Only **rendering the boxes onto the frame**
   caught that — nothing numeric would have.

Score → negative control → visual verification. Each caught what the previous
one missed.

---

## Measured performance

Source: 2160×3840 @ 30 fps, 522 frames, sampled every 10th frame.

```
cold start       792 ms      (warm container, first call)
                8432 ms      (after container restart, no network)

warm latency      mean    p95    min    max   (ms)
  inference        498    648    452    695
  decode           180    207    141    237
  total            679    846    594    894

  achieved        1.47 frames/sec    (wall clock, this hardware)
  required        3.00 frames/sec    (to track 30 fps video at stride 10)
  real-time       0.49x              DOES NOT keep up
```

Read those last three lines together. **This pipeline does not run in real time
on this hardware.** Tracking a live camera at this sampling rate would need a
stride of ~21 rather than 10.

Two figures are reported rather than a single "FPS" because inference throughput
and video rate differ by the sampling stride, and quoting one as the other
overstates what the system does.

Inference plus decode account for 678 of 679 ms. The frame loop itself is free;
all cost is in the two signals.

Execution is genuinely CPU-only. ONNX Runtime reports:

```
'CUDAExecutionProvider' is not in available provider names.
'OpenVINOExecutionProvider' is not in available provider names.
'CoreMLExecutionProvider' is not in available provider names.
Available providers: 'AzureExecutionProvider, CPUExecutionProvider'
```

No GPU, no Neural Engine, no accelerator of any kind inside the container.

---

## Offline verification

The connectivity claim is tested, not asserted. `artifacts/offline-run.log` is
the captured evidence.

```bash
networksetup -setairportpower en0 off
bash scripts/offline_test.sh data/your-clip.MOV
```

The script refuses to produce a passing result under false conditions: it
records the Wi-Fi power state and the routing table, and attempts to reach both
`api.roboflow.com` and `1.1.1.1`. If either succeeds it says the test is invalid.

It then **restarts the container** before running. Testing against a
long-running server proves nothing — a resident model would serve requests
whether or not the disk cache works. Restarting forces the model *and* the
workflow definition to reload from `/private/tmp/model-cache` with no network
available. That is what a site reboot at 3am with no WAN looks like.

Result: identical to the online run — 9 PASS, 6 DIVERT, 6 APPROACHING, 32 IDLE,
warm latency within run-to-run noise.

**Time to service after restart with no network: 8.4 seconds.**

---

## Assumptions, stated because they will break somewhere else

**Single-parcel station.** Duplicate detections collapse to the highest-confidence
box rather than a tuned NMS threshold. If two parcels are genuinely in frame, the
second is discarded. The `raw=` count in the log is what makes that visible — a
frame reporting more than one detection is a second parcel being dropped, and it
is recorded rather than silent.

**Highest-confidence detection is the parcel.** True for this model — its
negative control scored 0.000 — but not model-independent. One of the rejected
Universe models confidently detected a wooden table. Swap the model, re-run the
negative control.

**Carrier prefix identifies the parcel.** `VALID_PREFIXES = ("TBA",)` is Amazon's
prefix. This is the deliberate answer to a real trap: the mailer carries its
**own** CODE128 (`FSA`) beside the shipping label. Both are CODE128, so symbology
cannot separate them; both are physically on the parcel, so location cannot
either. Accepting any decoded barcode passes the deliberately flipped parcel —
a false PASS on the one case staged to fail.

Prefix validation is per-deployment configuration and the first thing to change
for a customer shipping UPS or FedEx. It is a maintenance cost accepted with eyes
open, not an oversight.

**Fixed camera.** The read window is in pixel coordinates and is bound to this
exact camera position.

---

## Known limitations

**The verdict is per frame, not per parcel.** A parcel visible across ~13 sampled
frames produces 13 independent verdicts. On the source clip that yields **6
DIVERTs of which 1 is a real defect**: three while the label is still resolving,
one single-frame decode dropout mid-dwell, one on departure.

A per-parcel verdict — opening on zone entry, closing on exit, resolving with an
"any DIVERT means DIVERT" rule — is the fix. It is not implemented here. In a
warehouse a missed defect and a false divert are not equally expensive, and which
way that asymmetry runs is a customer conversation, not a default.

**`APPROACHING` cannot distinguish arrival from departure.** A stateless verdict
has no memory of whether this parcel has already been inspected. It assumes an
off-zone parcel is inbound, which holds on a continuously fed line. Resolved by
the same per-parcel state above.

**Decode scans the whole frame** rather than a crop of the detection. Cropping
would cut ~27% of per-frame cost, but it would make decode depend on detection —
and the architecture rests on the two signals failing independently. Survey
frames 443 and 495 had a readable barcode and no detection at all; a
crop-to-detection decode would be blind to exactly those.

**Not real time on this hardware**, as above.

---

## What changes on real hardware

This runs CPU-only on a laptop. Stated plainly: **no edge hardware was used and
none of this has been run on a Jetson.** What follows is reasoning, flagged as
such.

**Inference latency should improve substantially, but it is not the whole
story.** YOLO-World's published figure is 52 FPS at 35.4 AP on LVIS — measured on
an NVIDIA V100. Measured here: 2.4 FPS on an Apple Silicon CPU. Neither number is
wrong; both carry their measurement conditions with them, and those conditions do
not survive the trip to a customer site. The same trap appeared with the 66.9%
mAP model above, in a different dimension: one number was measured on data that
was not ours, the other on hardware that is not ours.

**The bottleneck likely moves.** Inference and decode are currently 678 of 679 ms.
With CUDA and TensorRT, inference should drop sharply while `pyzbar` — pure CPU
work on a full-resolution frame — will not. Decode would likely become the
dominant cost, which changes where optimisation pays and makes the
crop-to-detection trade-off worth revisiting.

**Memory is a real constraint, not a theoretical one.** The container was
OOM-killed (`Exited (137)`) during model selection while several YOLO-World
variants were resident in a 7.75 GiB VM. The failure mode is the point: no
graceful degradation, no error response — the entire service disappears. On a
4 GB edge device that argues for pinning one variant and setting explicit memory
limits.

**Model size against accuracy is a live trade-off.** The Workflow block defaults
to variant `v2-s`; this pipeline pins `l`.

| Variant | Size | Warm latency | Confidence on test frame |
|---|---|---|---|
| `l` | 91 MB | 414 ms | **0.580** |
| `v2-s` | 25 MB | **128 ms** | **0.037** — below threshold |

`v2-s` is 3.2× faster and 3.6× smaller, and cannot see the parcel. **The model
that fits the device is not always the model that works.** On constrained
hardware that trade-off has to be measured against real frames, not assumed from
a spec sheet.

**The model cache needs a named volume.** The stock container writes model weights
to `/tmp/model-cache` via a bind mount, with no named volume — `docker volume ls`
is empty. It survives container restarts, including a SIGKILL. But `/tmp` is
frequently tmpfs on Linux edge hardware, meaning RAM, meaning gone at every
reboot. A box that runs correctly for weeks reboots overnight at a site with no
WAN and comes up unable to fetch weights it cannot reach. Nothing alerts, because
nothing was wrong until the reboot.

**First on-site measurement** would be whether the constraint is inference, frame
acquisition, or the network hop — not an interpolation from these numbers.

---

## Re-calibration

The read window is in pixel coordinates, so a moved or bumped camera invalidates
it. To re-derive:

```bash
# 1. Confirm both signals work at the new framing before shooting anything.
./.venv/bin/python scripts/check_frame.py data/still.jpg

# 2. Pull sample frames from a clip.
./.venv/bin/python scripts/extract_frames.py data/clip.MOV 20

# 3. Survey both signals across the clip -- gives cx per frame and where the
#    barcode decodes.
./.venv/bin/python scripts/survey.py data/frames
```

Read `cx` where the parcel settles and where it enters and leaves, then set
`ZONE_HALF_WIDTH_PX` in `audit/zones.py` between the two.

`scripts/check_frame.py` is the one to run first at any new site. It answers both
questions that matter before a single frame is processed: does a prompt detect
this parcel at this framing, and does the label decode at this distance.

---

## Tooling

| Script | Purpose |
|---|---|
| `scripts/check_frame.py` | prompt sweep and barcode decode against a still — run first at a new site |
| `scripts/extract_frames.py` | video properties and sample frames |
| `scripts/survey.py` | both signals across a whole clip, no decisions |
| `scripts/probe.py` | score candidate Universe models against local frames |
| `scripts/annotate.py` | render detections onto a frame |
| `scripts/zeroshot.py` | YOLO-World directly, cold and warm latency reported separately |
| `scripts/run_workflow.py` | single-frame workflow call |
| `scripts/offline_test.sh` | offline verification with evidence capture |

---

## Stack

Roboflow Inference Server 1.4.0 (self-hosted, CPU, arm64) · Roboflow Workflows ·
YOLO-World (Tencent AI Lab, served by Roboflow) · `supervision` 0.30.0 ·
`inference-sdk` 1.4.0 · `pyzbar` / zbar · OpenCV 4.12
