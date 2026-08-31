# Autonomous trash-collection for canals and ponds

A shore camera watches the water, a detector measures how much of it is covered
in rubbish, and a dock station decides when to send a collection boat out to
clean it.

## See it work

```bash
python demo.py
```

That starts an MQTT broker, the camera server and the dock, streams all three
into one timeline, and prints a summary. It takes about two and a half minutes
and tears itself down afterwards. `python demo.py --check` runs the preflight
checks alone.

## What is real and what is simulated

Being precise about this matters, because the boundary is the interesting part.

**Real:** the detector (YOLO11s fine-tuned on a merged 751-image dataset, run
over SAHI tiles), the coverage measurement, the MQTT messaging, the dock's
dispatch logic, and the boat's state machine and safety rules.

**Simulated:** the water, the trash floating in it, and the boat's motors, GPS,
battery and fill sensor. There is no hardware.

The separation is deliberate. `dock.py` and `boat.py` never read the simulated
world directly - only through something standing in for a sensor - so the
control logic sees exactly what a real deployment would give it. That is what
lets the decision-making be developed and tested against the detector's real
output before anything gets wet.

## The two halves

| | |
| --- | --- |
| [`main-cam/`](main-cam/) | The eye. Training, detection, and `coverage_monitor.py`, which measures water coverage and publishes it over MQTT. See [`main-cam/CLAUDE.md`](main-cam/CLAUDE.md) for what the model can and cannot do. |
| [`Dock/`](Dock/) | The brain and the body. The dock station, the boat, the messaging layer and the simulated world. See [`Dock/README.md`](Dock/README.md). |
| `demo.py` | Runs both halves together. |

They talk only over MQTT topics, so either one can be replaced without the other
noticing.

## How much trash is there, really?

The coverage figure is a **relative trend for one fixed camera**, not a count of
rubbish. The model detects regions rather than individual items, and its recall
on held-out water is 0.284, so most trash is missed. A high reading is
trustworthy; a low one is not evidence of clean water. `main-cam/CLAUDE.md`
records the measurements behind that, including the reflection false positives
and two fine-tuning attempts that made cross-scene transfer worse.

## Not in this repository

Rebuild or fetch these separately:

* `main-cam/venv/`, `venv/` - environments, rebuild per machine. Install the
  CUDA wheel *before* `requirements.txt`; see `main-cam/CLAUDE.md`.
* `main-cam/data1/canal_trash.mp4` (676 MB) and `video_3.mp4` (372 MB) - over
  GitHub's 100 MB file limit.
* `dataset_merged/`, `dataset_finetune*/` - derived; rebuild with
  `python build_dataset.py`.
* `.env` - held a Roboflow API key. Nothing imports it any more; inference runs
  on local weights.

The one trained model that matters, `main-cam/runs/trash_yolo11s/weights/best.pt`,
*is* included, so a clone can run inference without retraining.
