# Project context

Floating trash detector for a river/canal. YOLO11s fine-tuned on a merged,
single-class dataset. Hackathon project.

## Current state

- Dataset merge: **done and verified**
- Training: **done** — 100 epochs, batch 32, ~11 min on an RTX 5060 Ti
- Inference scripts: **switched to local weights**; the hosted Roboflow API and
  `inference-sdk` are gone, so no API key is needed any more

Held-out test split (76 images, 126 boxes), from `runs/trash_yolo11s/weights/best.pt`:

| mAP50 | mAP50-95 | precision | recall |
|---|---|---|---|
| 0.734 | 0.515 | 0.839 | 0.667 |

Validation scored mAP50 0.801, so the drop on test is the usual small-set gap.
Inference is ~5 ms/image on the 5060 Ti.

## Known limitation: the labels are sparse

**The model detects regions of trash, not individual items.** Ground truth averages
1.6 boxes per image with a **median of 1**, even for canal scenes containing dozens
of visible bottles and bags, and ~4% of boxes cover more than a quarter of the frame.
The model reproduces that faithfully: on a busy test image it emits a handful of
boxes, some of them large regions, and misses dense piles entirely.

Consequences:

- The count `webcam_monitor.py` prints is a **count of detected regions, not of
  objects**. Do not present it as "N pieces of trash".
- Lowering `--conf` raises coverage but not granularity — at 0.05 the dense areas
  get covered by sprawling region boxes rather than resolved into items.
- Per-item counting would need re-annotation, not more training.
- Dense piles are missed outright, so `zone_patches.py` **under-reports the dirtiest
  areas**. A flagged zone is trustworthy; an unflagged one is not evidence of clean water.

**Reflections cause confident false positives.** On a clean-canal `test.jpg` (bright
sun, green railings mirrored in the water) the model's *most confident* detection, 0.65,
was a railing reflection, while the three real pieces of trash scored 0.22-0.27. The
confidence ordering is inverted, so no `--conf` value separates them: 0.25 keeps the
reflection and drops two real items; lowering it keeps the reflection too.

This is domain shift. Training images are packed with garbage; a mostly-clean canal with
strong reflections is out of distribution, and the 0.734 test mAP does not describe it.
The fix is data, not tuning -- add frames of clean water with railings, reflections and
glare as **background images** (an image with no label file), which Ultralytics uses as
negatives. It needs no box-drawing, so it is the cheapest labelling available.

**Confidence anti-correlates with box size.** Confident detections are tight (<2% of the
frame); unsure ones are vague blobs covering 10-20% that localise nothing. On `test.jpg`
a single 21%-of-frame blob at conf 0.29 flagged half the zone grid, including clean
water. `zone_patches.py --max-box-frac` (default 0.08) drops these. Lowering `--conf` to
chase recall amplifies the problem rather than fixing it -- at conf 0.05, 81% of the grid
lit up.

## Fine-tuning on pond frames: two failed attempts

Both attempts to fine-tune on the data1 pond frames made the held-out data2 pond
**worse**, evaluated with sliced 512/20% + ROI:

| model | AP@50 (data2) | recall | public val mAP50 |
|---|---|---|---|
| baseline `runs/trash_yolo11s` | **0.138** | **0.284** | 0.801 |
| ft-v1: 100 ep, pond 52% of mix | 0.103 | 0.234 | 0.765 |
| ft-v2: 40 ep, pond 25%, AdamW lr 5e-4 | 0.063 | 0.127 | **0.811** |

**The public validation set does not predict cross-pond generalisation.** ft-v2
scores the *best* public val of the three and the *worst* held-out pond result.
Tuning against public val will confidently make transfer worse. Judge any future
model on a held-out *scene*, never on the public split.

ft-v2 fixed the overfitting ft-v1 showed (public val 0.765 -> 0.811) and still
halved cross-pond AP, so scene imbalance was not the binding constraint. With one
training pond and one test pond there is nothing to average over; the fix is
frames from several sites, not hyperparameters.

Baseline weights remain the operational model. Tuning stopped by decision.

## Environment gotchas

**CUDA wheel for RTX 50-series.** `requirements.txt` was frozen on a CPU-only
machine, so `torch==2.13.0` from PyPI is the **CPU build**. On a Blackwell card
(RTX 5060 Ti / 5070 / 5080 / 5090, compute capability `sm_120`) install from the
`cu130` index — *not* `cu124`, which installs and imports fine but dies at the first
forward pass with `no kernel image is available for execution on the device`:

```
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130
```

Install this **before** `pip install -r requirements.txt`, with the versions pinned
explicitly. Installing requirements first downloads ~3 GB of CPU torch only to
discard it, and an unpinned `--force-reinstall` pulls whatever is newest on the
index rather than the pinned pair. Verify with a real forward pass, not just
`torch.cuda.is_available()` — a wrong CUDA build passes the availability check and
only fails once a kernel actually launches:

```
python -c "import torch,torch.nn as nn; m=nn.Conv2d(3,16,3).cuda(); print(m(torch.randn(2,3,64,64,device='cuda')).shape)"
```

**Python 3.14 vs. the frozen pins.** `requirements.txt` was frozen on an older
interpreter, and three pins had to move to install on 3.14 (they are already updated
in the file — this records *why*):

| Package | Was | Now | Reason |
|---|---|---|---|
| `inference-sdk` | 1.5.0 | *removed* | No 3.14 build (requires `<3.13`); only the Roboflow scripts used it, and they no longer do |
| `opencv-python` | 4.12.0.88 | 4.14.0.94 | 4.12 pins `numpy<2.3.0`, but numpy only gained cp314 wheels at 2.3+. Do **not** use 4.13.0.90 — `ultralytics` explicitly excludes that exact build |
| `numpy` | 2.2.6 | 2.5.2 | No cp314 wheel below 2.3; 2.2.6 falls back to a source build and dies for lack of MSVC |

The failure mode is confusing: pip reports a numpy compiler error, but the actual
constraint is opencv's numpy ceiling. Bumping numpy alone does not help.

**Ultralytics nests a relative `project` path.** In `cfg/__init__.py`, a non-absolute
`project` is joined as `runs_dir/task/project`, producing `runs/detect/runs/detect/<name>`.
`train.py` therefore passes an absolute path. Don't "simplify" it back to a relative one.
Note the final `model.val()` call still writes to `runs/detect/val`, since it does not
take the same `project` argument.

**Label files have no trailing newline.** Counting boxes with `cat *.txt | wc -l` or
`grep -c` silently merges the last line of one file with the first of the next and
undercounts. Parse per-file.

**"Slow image access detected" is a false alarm.** Ultralytics' own benchmark output
is garbled (`ping: 0.10.0 ms`, `read: 36.847.5 MB/s`). The dataset is on local disk
and caches to RAM in under a second. Ignore it.

**SAHI takes an ndarray as RGB, but OpenCV gives BGR.** Passing a `cv2` frame
straight to `get_sliced_prediction` silently swaps red and blue: on a test frame
that cut detections from 11 to 5. File paths are unaffected (SAHI reads those
through PIL), so the bug hides during file-based testing and only bites the live
camera path. `detect_sliced.as_sahi_input()` converts explicitly.

**Ultralytics auto-installs missing extras mid-run.** The first sliced call
pulled `pi-heif` from PyPI on its own. It is pinned in `requirements.txt` now,
but be aware the library will reach for the network during a run.

**`cache='ram'` makes runs non-deterministic** despite `seed=42`. Use
`--cache disk` if you need reproducible numbers for a writeup.

## Dataset

`dataset_merged/` is derived and gitignored — regenerate with `python build_dataset.py`.

| Source | Images | Boxes | Original classes |
|---|---|---|---|
| Floating Trash Detection v1i | 382 | 747 | 1 (`garbage`) |
| AquaTrash | 369 | 469 | 4 (glass/metal/paper/plastic), CSV abs-xyxy |
| **Merged** | **751** | **1216** | **1 (`trash`)** |

Splits: 525 train / 150 valid / 76 test. Floating Trash's original splits are
preserved as-is; AquaTrash had none and is split 70/20/10 with seed 42.

If `build_dataset.py` does not report 751 / 1216, something is wrong with `datasets/` —
investigate before training.

Verified at merge time: no duplicate images within or across sources (dHash), no
cross-split leakage, every image paired with a label, all class ids 0, all coords in
[0,1], AquaTrash pixel->YOLO round-trip accurate to 0.002 px. Re-verify if `datasets/`
changes.

## Conventions

- Derived artifacts stay out of git: `dataset_merged/`, `runs/`, `*.pt`, `venv/`.
- The dataset is small (525 training images), so pretrained weights are essential —
  never train from random init.
- No secrets are needed any more — inference runs on local weights. `python-dotenv`
  is still in `requirements.txt` but nothing imports it. If secrets return, keep them
  in `.env` (gitignored) and never inline them in a script.

## Commands

```
python build_dataset.py              # regenerate merged dataset
python train.py                      # fine-tune, auto-detects GPU
python train.py --batch 32           # 16 GB VRAM comfortably handles larger batches
python detect_test_image.py          # test.jpg -> test_output.jpg
python detect_test_image.py --source river.jpg --conf 0.15
python webcam_monitor.py             # capture every 5 min, print region counts
python webcam_monitor.py --interval 60 --save-dir captures
python zone_patches.py               # test.jpg -> test_zones.jpg, grid of trash zones
python zone_patches.py --grid 12x8 --conf 0.10 --json zones.json
python detect_sliced.py --dir frames/ --roi water_roi_data2.json   # tiled inference
python coverage_monitor.py --camera 0 --roi water_roi_data2.json --broker mqtt.host --dry-run
```

`coverage_monitor.py` reports water coverage over MQTT. Coverage is the union
area of detections inside the ROI over the ROI area, from a pixel mask so
overlapping boxes count once. Each cycle takes a burst of frames and reports the
**median** -- on one static scene the per-frame spread was 0.007-0.114, so a
single frame is not a reading. It publishes every cycle even at coverage 0.0:
silence must mean "process died", not "water is clean".

Treat the number as a **relative trend for one fixed camera**, never an absolute
quantity of rubbish -- recall on held-out water is 0.284, so most trash is missed.

It also subscribes to `trash/dock/servicing` and skips the entire cycle -- the
inference too -- while the dock has this camera marked servicing, because the
collection boat is then in frame and the detector reads a boat as trash.

Broker credentials come from `MQTT_USERNAME` / `MQTT_PASSWORD` in the
environment, never from argv, where they would land in shell history.

**Driving the ../Dock dock station.** Run it end to end with
`--topic trash/camera/report` and explicit `--lat`/`--long` (they default to
`None`, and a report with no position is nowhere to send the boat); see
`../Dock/README.md`. The scale is the trap: the dock was tuned for a simulated
camera reporting 0-1, but real coverage on the data2 pond measures
**0.006-0.217, median 0.072** (20 frames, conf 0.25, sliced 512/20% + ROI), so
the dock's floor and its "did the service help" threshold both have to come
down to match. The canal video is not usable as a source -- coverage there
reads 0.000-0.004, which is the recall problem, not clean water.

Both inference scripts default to `runs/trash_yolo11s/weights/best.pt` and accept
`--weights`, `--conf`, and `--device`.
