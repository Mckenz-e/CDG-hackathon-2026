"""Mark which zones of a frame contain floating trash.

Divides the frame into a grid and flags each cell by how much of it the model's
detections cover. This suits the detector's actual behaviour: it finds regions of
trash rather than individual items, so "which parts of the water are dirty" is a
question it can answer, unlike "how many bottles are there".

Coverage is measured from a pixel mask of the union of all detection boxes, so
overlapping boxes are not double-counted.

Examples:
    python zone_patches.py                                  # test.jpg -> test_zones.jpg
    python zone_patches.py --source river.jpg --grid 12x8
    python zone_patches.py --conf 0.10 --min-coverage 0.05  # sweep wider
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

BASE = Path(__file__).parent
WEIGHTS = BASE / "runs" / "trash_yolo11s" / "weights" / "best.pt"

GRID_COLOR = (255, 255, 255)
HOT_COLOR = (0, 0, 255)      # BGR: red fill for flagged zones
TEXT_COLOR = (255, 255, 255)


def parse_grid(text):
    try:
        cols, rows = (int(v) for v in text.lower().split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"--grid expects COLSxROWS, got '{text}'")
    if cols < 1 or rows < 1:
        raise argparse.ArgumentTypeError("--grid dimensions must be >= 1")
    return cols, rows


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default=str(WEIGHTS), help="trained checkpoint")
    p.add_argument("--source", default="test.jpg")
    p.add_argument("--output", default="test_zones.jpg")
    p.add_argument("--grid", type=parse_grid, default="8x6", help="COLSxROWS, default 8x6")
    # zone marking wants coverage over precision: a slightly over-eager box still
    # lands in the right zone, whereas a missed one leaves a dirty zone unflagged
    p.add_argument("--conf", type=float, default=0.15, help="detection confidence threshold")
    p.add_argument("--min-coverage", type=float, default=0.10,
                   help="fraction of a cell that must be covered to flag it")
    # this model's confidence anti-correlates with box size: its sure detections are
    # tight (<2% of frame) while its unsure ones are vague blobs covering 10-20%, which
    # flood the grid without saying where the trash actually is. Drop those.
    p.add_argument("--max-box-frac", type=float, default=0.08,
                   help="ignore detections covering more than this fraction of the frame")
    p.add_argument("--device", default=None, help="'0' for first CUDA GPU, 'cpu' to force CPU")
    p.add_argument("--json", default=None, help="also write the zone report to this JSON file")
    return p.parse_args()


def cell_name(col, row):
    """Spreadsheet-style label: column letter + 1-based row, e.g. 'C4'."""
    return f"{chr(ord('A') + col)}{row + 1}"


def zone_report(frame, boxes, grid, min_coverage):
    """Coverage per grid cell, from the union mask of all detection boxes."""
    h, w = frame.shape[:2]
    cols, rows = grid

    mask = np.zeros((h, w), dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1

    # integer edges so cells tile the frame exactly, even when h/w don't divide evenly
    xs = [round(c * w / cols) for c in range(cols + 1)]
    ys = [round(r * h / rows) for r in range(rows + 1)]

    zones = []
    for row in range(rows):
        for col in range(cols):
            x1, x2 = xs[col], xs[col + 1]
            y1, y2 = ys[row], ys[row + 1]
            area = (x2 - x1) * (y2 - y1)
            coverage = float(mask[y1:y2, x1:x2].sum()) / area if area else 0.0
            zones.append({
                "name": cell_name(col, row),
                "col": col, "row": row,
                "bbox": [x1, y1, x2, y2],
                "coverage": round(coverage, 4),
                "hot": coverage >= min_coverage,
            })
    return zones


def draw_zones(frame, zones, grid):
    out = frame.copy()
    overlay = frame.copy()

    for z in zones:
        if not z["hot"]:
            continue
        x1, y1, x2, y2 = z["bbox"]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), HOT_COLOR, -1)

    # translucent fill, so the water underneath stays readable
    cv2.addWeighted(overlay, 0.35, out, 0.65, 0, out)

    for z in zones:
        x1, y1, x2, y2 = z["bbox"]
        cv2.rectangle(out, (x1, y1), (x2, y2), GRID_COLOR, 1)
        if z["hot"]:
            label = f"{z['name']} {z['coverage'] * 100:.0f}%"
            cv2.putText(out, label, (x1 + 4, y1 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_COLOR, 1, cv2.LINE_AA)
    return out


def main():
    args = parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"{weights} not found -- run 'python train.py' first.")

    frame = cv2.imread(args.source)
    if frame is None:
        raise FileNotFoundError(f"Could not read image at '{args.source}'")

    device = args.device or ("0" if torch.cuda.is_available() else "cpu")
    model = YOLO(str(weights))
    result = model.predict(args.source, conf=args.conf, device=device, verbose=False)[0]

    h, w = frame.shape[:2]
    all_boxes = [tuple(int(v) for v in b.xyxy[0]) for b in result.boxes]
    boxes = [b for b in all_boxes
             if (b[2] - b[0]) * (b[3] - b[1]) <= args.max_box_frac * w * h]
    dropped = len(all_boxes) - len(boxes)
    zones = zone_report(frame, boxes, args.grid, args.min_coverage)

    cols, rows = args.grid
    hot = [z for z in zones if z["hot"]]
    print(f"{len(boxes)} detection(s) at conf>={args.conf}"
          + (f" ({dropped} oversized blob(s) dropped)" if dropped else ""))
    print(f"{len(hot)}/{len(zones)} zones flagged "
          f"({100 * len(hot) / len(zones):.0f}% of the {cols}x{rows} grid)")
    for z in sorted(hot, key=lambda z: -z["coverage"]):
        print(f"  {z['name']:>4}  {z['coverage'] * 100:5.1f}%")

    cv2.imwrite(args.output, draw_zones(frame, zones, args.grid))
    print(f"Saved zone map to '{args.output}'")

    if args.json:
        payload = {
            "source": args.source,
            "conf": args.conf,
            "min_coverage": args.min_coverage,
            "grid": {"cols": cols, "rows": rows},
            "max_box_frac": args.max_box_frac,
            "detections": len(boxes),
            "dropped_oversized": dropped,
            "zones": zones,
        }
        Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved zone report to '{args.json}'")


if __name__ == "__main__":
    main()
