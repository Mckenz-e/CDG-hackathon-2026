"""Merge the source datasets into one single-class YOLO dataset.

Sources:
  - Floating Trash Detection.v1i.yolov11 : already YOLO format, 1 class, pre-split.
  - AquaTrash-master                     : CSV with absolute xyxy boxes, 4 classes, unsplit.

Every class collapses to a single class 0 = "trash". The Floating Trash splits are
preserved as-is; AquaTrash is split with a fixed seed to the same 70/20/10 ratio.
"""

import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import yaml

BASE = Path(__file__).parent
SRC = BASE / "datasets"
FTD = SRC / "Floating Trash Detection.v1i.yolov11"
AQUA = SRC / "AquaTrash-master"
OUT = BASE / "dataset_merged"

SPLITS = ["train", "valid", "test"]
SPLIT_RATIO = (0.70, 0.20, 0.10)
SEED = 42
CLASS_NAME = "trash"


def reset_output():
    if OUT.exists():
        shutil.rmtree(OUT)
    for split in SPLITS:
        (OUT / split / "images").mkdir(parents=True)
        (OUT / split / "labels").mkdir(parents=True)


def copy_ftd():
    """Copy Floating Trash across, forcing every class id to 0."""
    counts = defaultdict(lambda: [0, 0])
    for split in SPLITS:
        for img in sorted((FTD / split / "images").iterdir()):
            lbl = FTD / split / "labels" / f"{img.stem}.txt"
            if not lbl.exists():
                continue

            out_name = f"ftd_{img.name}"
            shutil.copy2(img, OUT / split / "images" / out_name)

            lines = []
            for line in lbl.read_text().splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                # collapse whatever the class id was onto 0
                lines.append(" ".join(["0"] + parts[1:]))

            out_lbl = OUT / split / "labels" / f"ftd_{img.stem}.txt"
            out_lbl.write_text("\n".join(lines) + "\n" if lines else "")
            counts[split][0] += 1
            counts[split][1] += len(lines)
    return counts


def load_aqua_annotations():
    rows = defaultdict(list)
    with open(AQUA / "annotations.csv", newline="") as f:
        for r in csv.DictReader(f):
            rows[r["image_name"]].append(r)
    return rows


def split_aqua(names):
    """Deterministic 70/20/10 split."""
    names = sorted(names)
    random.Random(SEED).shuffle(names)
    n = len(names)
    n_train = int(n * SPLIT_RATIO[0])
    n_valid = int(n * SPLIT_RATIO[1])
    return {
        "train": names[:n_train],
        "valid": names[n_train:n_train + n_valid],
        "test": names[n_train + n_valid:],
    }


def copy_aqua(rows):
    """Convert AquaTrash CSV boxes to normalised YOLO format, all as class 0."""
    assignment = split_aqua(rows.keys())
    counts = defaultdict(lambda: [0, 0])
    skipped = 0

    for split, names in assignment.items():
        for name in names:
            src_img = AQUA / "Images" / name
            img = cv2.imread(str(src_img))
            if img is None:
                skipped += 1
                continue
            H, W = img.shape[:2]

            lines = []
            for r in rows[name]:
                x1, y1 = float(r["x_min"]), float(r["y_min"])
                x2, y2 = float(r["x_max"]), float(r["y_max"])
                # clamp to image, guard against degenerate boxes
                x1, x2 = max(0.0, min(x1, W)), max(0.0, min(x2, W))
                y1, y2 = max(0.0, min(y1, H)), max(0.0, min(y2, H))
                if x2 - x1 < 1 or y2 - y1 < 1:
                    continue
                cx = ((x1 + x2) / 2) / W
                cy = ((y1 + y2) / 2) / H
                bw = (x2 - x1) / W
                bh = (y2 - y1) / H
                lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            stem = Path(name).stem
            shutil.copy2(src_img, OUT / split / "images" / f"aqua_{name}")
            (OUT / split / "labels" / f"aqua_{stem}.txt").write_text(
                "\n".join(lines) + "\n" if lines else ""
            )
            counts[split][0] += 1
            counts[split][1] += len(lines)

    return counts, skipped


def write_yaml():
    data = {
        "path": str(OUT.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 1,
        "names": [CLASS_NAME],
    }
    with open(OUT / "data.yaml", "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def main():
    reset_output()

    ftd_counts = copy_ftd()
    aqua_rows = load_aqua_annotations()
    aqua_counts, skipped = copy_aqua(aqua_rows)

    write_yaml()

    print(f"{'split':<8}{'FTD img':>9}{'FTD box':>9}{'Aqua img':>10}{'Aqua box':>10}{'tot img':>9}{'tot box':>9}")
    t_img = t_box = 0
    for split in SPLITS:
        fi, fb = ftd_counts[split]
        ai, ab = aqua_counts[split]
        print(f"{split:<8}{fi:>9}{fb:>9}{ai:>10}{ab:>10}{fi + ai:>9}{fb + ab:>9}")
        t_img += fi + ai
        t_box += fb + ab
    print(f"{'TOTAL':<8}{'':>9}{'':>9}{'':>10}{'':>10}{t_img:>9}{t_box:>9}")

    if skipped:
        print(f"\nWARNING: skipped {skipped} unreadable AquaTrash image(s)")
    print(f"\nWrote {OUT / 'data.yaml'}")


if __name__ == "__main__":
    main()
