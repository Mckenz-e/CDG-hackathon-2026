# Floating Trash Detector

Detects floating trash in a river/canal from a camera feed, using a YOLO11s model
fine-tuned on a merged, single-class dataset.

## Setup on a new machine

```bash
py -3 -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/macOS
pip install -r requirements.txt
```

On Windows, prefer the `py` launcher over `python`. A bare `python` often resolves to
the Microsoft Store *app execution alias* -- a stub that prints "Python was not found"
and opens the Store, even when Python is installed. (You can disable the stub under
Settings > Apps > Advanced app settings > App execution aliases.) Note that `&&` does
not chain commands in PowerShell 5.1; use `;` or one command per line.

If you have an NVIDIA GPU, read the next section **before** running the requirements
install.

### If this machine has an NVIDIA GPU — read this

`requirements.txt` was frozen on a CPU-only machine, so `torch==2.13.0` from PyPI
installs the **CPU build**. Owning a GPU is not enough: PyTorch itself must be
compiled with CUDA kernels, or it never touches the card. Install the CUDA build
**first**, with the versions pinned explicitly:

```bash
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130
```

Then run `pip install -r requirements.txt`, which will see the pins already satisfied
and leave them alone. Doing it the other way round downloads ~3 GB of CPU torch just to
throw it away, and an unpinned `--force-reinstall` grabs whatever is newest on the index
rather than the pinned pair.

**Use `cu130`, not an older index.** RTX 50-series cards (5060 Ti, 5070, 5080, 5090)
are Blackwell, compute capability `sm_120`, which needs CUDA 12.8+. An older wheel such
as `cu124` installs and imports fine, then fails at the first forward pass with:

```
CUDA error: no kernel image is available for execution on the device
```

`cu130` also carries `torch==2.13.0` / `torchvision==0.28.0`, matching the pinned
versions exactly. (The `cu128` index stops at torch 2.11.)

Verify it worked -- with a real forward pass, not just the availability check. A wrong
CUDA build passes `is_available()` and only fails once a kernel actually launches:

```bash
python -c "import torch,torch.nn as nn; print(torch.cuda.is_available(), torch.cuda.get_device_name(0)); m=nn.Conv2d(3,16,3).cuda(); print(m(torch.randn(2,3,64,64,device='cuda')).shape)"
```

This must print `True`, your GPU name, and a tensor shape. If `is_available()` prints
`False`, training falls back to CPU and `train.py` will warn you.

### Python 3.14

Three pins had to move for the install to work on Python 3.14, and are already updated
in `requirements.txt`: `inference-sdk` removed (no 3.14 build), `opencv-python` bumped
to 4.14.0.94, and `numpy` to 2.5.2. The trap is that opencv 4.12 pins `numpy<2.3.0`
while numpy only gained 3.14 wheels at 2.3+, so pip reports a *numpy* compiler error
whose real cause is opencv. See `CLAUDE.md` for the full table.

No API key is needed -- inference runs on local weights.

## Building the dataset

The merged dataset is derived, not stored in git. Regenerate it from `datasets/`:

```bash
python build_dataset.py
```

This merges two sources into `dataset_merged/` as a single `trash` class:

| Source | Images | Boxes | Original classes |
|---|---|---|---|
| Floating Trash Detection v1i | 382 | 747 | 1 (`garbage`) |
| AquaTrash | 369 | 469 | 4 (glass/metal/paper/plastic) |
| **Merged** | **751** | **1216** | **1 (`trash`)** |

Splits: 525 train / 150 valid / 76 test. The Floating Trash dataset's original splits
are preserved; AquaTrash is split 70/20/10 with a fixed seed (42) for reproducibility.
Verified to have no duplicate images within or across sources and no cross-split leakage.

## Training

```bash
python train.py                                  # 100 epochs @ 640px, auto-detects GPU
python train.py --epochs 60 --imgsz 512          # faster
python train.py --batch 32                       # larger batch if you have VRAM
```

Starts from COCO-pretrained `yolo11s.pt` (downloads automatically). Evaluates on the
held-out test split at the end. Best weights land in
`runs/<name>/weights/best.pt`.

Rough timings for the full 100-epoch run at 640px:

| Hardware | Per epoch | Total |
|---|---|---|
| CPU (16 cores) | ~9 min | ~15 h |
| RTX 5060 Ti | ~5-8 s | ~10-15 min |

With 16 GB VRAM on a 5060 Ti you can raise the batch size well above the default 16;
`--batch 32` or `--batch 64` will cut wall-clock time further.

### Measured result

`python train.py --batch 32` on an RTX 5060 Ti: 100 epochs in **10.9 minutes**.
Held-out test split (76 images, 126 boxes):

| mAP50 | mAP50-95 | precision | recall |
|---|---|---|---|
| 0.734 | 0.515 | 0.839 | 0.667 |

Validation reached mAP50 0.801; the gap to test is the usual small-set variance.
Inference runs at ~5 ms/image. Note that `train.py` uses `cache='ram'`, which
Ultralytics warns is non-deterministic despite the fixed seed -- use `--cache disk`
if you need reproducible numbers.

## Inference

Both scripts run the fine-tuned weights locally and default to
`runs/trash_yolo11s/weights/best.pt`. Train first, or pass `--weights`.

Single image:

```bash
python detect_test_image.py                            # test.jpg -> test_output.jpg
python detect_test_image.py --source river.jpg --conf 0.15
```

Webcam monitoring loop (captures every 5 minutes, prints detection counts):

```bash
python webcam_monitor.py
python webcam_monitor.py --interval 60 --save-dir captures
```

Both also accept `--device` (`0` for the first GPU, `cpu` to force CPU).

### Zone patches

Marks *where* the trash is, rather than counting it -- a grid over the frame, each cell
flagged by how much of it the detections cover:

```bash
python zone_patches.py                                  # test.jpg -> test_zones.jpg
python zone_patches.py --source river.jpg --grid 12x8
python zone_patches.py --conf 0.10 --json zones.json    # machine-readable report
```

Coverage comes from a pixel mask of the union of all boxes, so overlaps are not
double-counted. `--min-coverage` sets how much of a cell must be covered to flag it.

`--max-box-frac` (default 0.08) drops oversized detections. This matters: on this model
confidence *anti-correlates* with box size -- confident detections are tight (<2% of the
frame) while unsure ones are vague blobs covering 10-20%, which flood the grid without
localising anything. Without the filter a single 21%-of-frame blob flagged half the grid
including clean water. Lowering `--conf` alone makes this worse, not better.

Known gap: the detector misses dense piles (see below), so the zone map **under-reports
the dirtiest areas**. Treat a flagged zone as reliable and an unflagged one as unproven.

### What the counts mean

**The model detects regions of trash, not individual items.** The training labels are
sparse -- a median of 1 box per image, even for scenes with dozens of visible bottles
and bags -- so the model emits a handful of boxes, some covering large areas, and does
not enumerate objects. Report the number as "detections", not "pieces of trash".
Lowering `--conf` widens coverage but does not resolve dense piles into items; that
would need re-annotation rather than more training.

## Files

| File | Purpose |
|---|---|
| `build_dataset.py` | Merge + collapse both source datasets into `dataset_merged/` |
| `train.py` | Fine-tune YOLO11s from pretrained weights |
| `detect_test_image.py` | Run local weights on an image, draw boxes, save output |
| `webcam_monitor.py` | Capture from webcam every 5 min, print detection counts |
| `zone_patches.py` | Grid the frame and flag which zones contain trash |
