"""Capture from a webcam at a fixed interval and count floating trash.

Runs the fine-tuned YOLO11s weights locally -- no hosted API, no API key. The
model loads once and is reused for every frame.

Examples:
    python webcam_monitor.py                       # every 5 min, print counts
    python webcam_monitor.py --interval 60 --save-dir captures
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

BASE = Path(__file__).parent
WEIGHTS = BASE / "runs" / "trash_yolo11s" / "weights" / "best.pt"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default=str(WEIGHTS), help="trained checkpoint")
    p.add_argument("--camera", type=int, default=0, help="cv2 camera index")
    p.add_argument("--interval", type=int, default=5 * 60, help="seconds between captures")
    p.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    p.add_argument("--device", default=None, help="'0' for first CUDA GPU, 'cpu' to force CPU")
    p.add_argument("--save-dir", default=None,
                   help="if set, write an annotated frame per capture into this directory")
    return p.parse_args()


def capture_frame(cap):
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Failed to capture frame from webcam")
    return frame


def main():
    args = parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(
            f"{weights} not found -- run 'python train.py' first."
        )

    device = args.device or ("0" if torch.cuda.is_available() else "cpu")
    model = YOLO(str(weights))

    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam at index {args.camera}")

    print(f"Monitoring started. Capturing every {args.interval} seconds. Press Ctrl+C to stop.")

    try:
        while True:
            frame = capture_frame(cap)
            now = datetime.now()
            timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

            result = model.predict(frame, conf=args.conf, device=device, verbose=False)[0]
            count = len(result.boxes)

            print(f"[{timestamp}] Trash items detected: {count}")

            if save_dir:
                out = save_dir / f"{now.strftime('%Y%m%d-%H%M%S')}_{count}.jpg"
                cv2.imwrite(str(out), result.plot())

            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        cap.release()


if __name__ == "__main__":
    main()
