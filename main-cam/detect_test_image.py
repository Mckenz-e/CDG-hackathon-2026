"""Annotate a single image with the locally-trained trash detector.

Runs the fine-tuned YOLO11s weights directly -- no hosted API, no API key.

Examples:
    python detect_test_image.py                          # test.jpg -> test_output.jpg
    python detect_test_image.py --source river.jpg --conf 0.15
"""

import argparse
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

BASE = Path(__file__).parent
WEIGHTS = BASE / "runs" / "trash_yolo11s" / "weights" / "best.pt"

BOX_COLOR = (0, 255, 0)
TEXT_COLOR = (0, 0, 0)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default=str(WEIGHTS), help="trained checkpoint")
    p.add_argument("--source", default="test.jpg")
    p.add_argument("--output", default="test_output.jpg")
    # recall is the weaker half of this model (~0.67 on the test split), so a
    # threshold below Ultralytics' 0.25 default surfaces more real trash
    p.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    p.add_argument("--device", default=None, help="'0' for first CUDA GPU, 'cpu' to force CPU")
    return p.parse_args()


def main():
    args = parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(
            f"{weights} not found -- run 'python train.py' first."
        )

    image = cv2.imread(args.source)
    if image is None:
        raise FileNotFoundError(f"Could not read image at '{args.source}'")

    device = args.device or ("0" if torch.cuda.is_available() else "cpu")

    model = YOLO(str(weights))
    result = model.predict(args.source, conf=args.conf, device=device, verbose=False)[0]

    boxes = result.boxes
    print(f"Found {len(boxes)} detection(s)")

    for box in boxes:
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
        label = f"{model.names[int(box.cls[0])]} {float(box.conf[0]):.2f}"

        cv2.rectangle(image, (x1, y1), (x2, y2), BOX_COLOR, 2)

        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            image, (x1, y1 - text_h - baseline), (x1 + text_w, y1), BOX_COLOR, -1
        )
        cv2.putText(
            image,
            label,
            (x1, y1 - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(args.output, image)
    print(f"Saved annotated image to '{args.output}'")


if __name__ == "__main__":
    main()
