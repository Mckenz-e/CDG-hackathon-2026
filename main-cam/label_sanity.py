"""Sanity-check a YOLO-format label set before training on it.

Catches the failure modes that silently waste a training run: boxes far larger
than any real object, stacks of near-duplicate boxes, coordinates out of range,
and -- when a region of interest is supplied -- boxes sitting outside the area
where the subject can physically be (e.g. trash outside the water).

With --roi it also reports how much of the labelled area is vegetation inside
that region, which is how you check whether a weed mat got labelled as trash.

Examples:
    python label_sanity.py --dataset data1/export/train
    python label_sanity.py --dataset data1/export/train --roi water_roi.json
    python label_sanity.py --dataset dataset_merged/train --max-area-frac 0.25
"""

import argparse
import json
import statistics
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True,
                   help="directory containing images/ and labels/ subdirectories")
    p.add_argument("--roi", default=None,
                   help="JSON holding a 'polygon' of [x,y] pairs in normalised "
                        "coords, marking where the subject can physically appear")
    p.add_argument("--max-area-frac", type=float, default=0.10,
                   help="flag boxes covering more than this fraction of the frame")
    p.add_argument("--overlap-iou", type=float, default=0.50,
                   help="flag box pairs in one image above this IoU")
    p.add_argument("--outside-frac", type=float, default=0.50,
                   help="flag boxes with more than this fraction of area outside the ROI")
    p.add_argument("--names", default=None,
                   help="comma-separated class names, in class-id order. Read from a "
                        "sibling data.yaml when not given")
    p.add_argument("--json", default=None, help="write the full report here")
    return p.parse_args()


def find_names(root, override):
    """Class names in id order, from --names or a nearby data.yaml."""
    if override:
        return [s.strip() for s in override.split(",")]
    for cand in (root / "data.yaml", root.parent / "data.yaml",
                 root.parent.parent / "data.yaml"):
        if not cand.is_file():
            continue
        for line in cand.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("names:") and "[" in line:
                inner = line.split("[", 1)[1].rsplit("]", 1)[0]
                return [s.strip().strip("'\"") for s in inner.split(",") if s.strip()]
    return None


def load_boxes(path):
    """YOLO label file -> list of (cls, xc, yc, w, h).

    Handles both YOLO formats, which are distinguished only by column count:
      detection    : cls xc yc w h                    (5 columns)
      segmentation : cls x1 y1 x2 y2 x3 y3 ...        (odd, >= 7 columns)
    A segmentation row is reduced to the bounding box of its polygon. Reading a
    polygon row as if it were "cls xc yc w h" silently produces enormous bogus
    boxes -- the first two vertices get read as a centre and a size -- so the
    column count must be checked, not assumed.

    Parsed per-file on purpose: these files often lack a trailing newline, so
    concatenating them merges the last line of one with the first of the next.
    """
    out = []
    for ln, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        try:
            cls = int(float(parts[0]))
            vals = [float(v) for v in parts[1:]]
        except (ValueError, IndexError):
            out.append(("BAD", ln, line[:40]))
            continue

        if len(vals) == 4:
            out.append((cls, *vals))
        elif len(vals) >= 6 and len(vals) % 2 == 0:
            xs, ys = vals[0::2], vals[1::2]
            x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
            out.append((cls, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
        else:
            out.append(("BAD", ln, f"{len(vals)} coord columns: {line[:30]}"))
    return out


def to_xyxy(box, W, H):
    _, xc, yc, w, h = box
    return (int((xc - w / 2) * W), int((yc - h / 2) * H),
            int((xc + w / 2) * W), int((yc + h / 2) * H))


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if not inter:
        return 0.0
    union = ((a[2] - a[0]) * (a[3] - a[1])
             + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / union if union else 0.0


def percentile(vals, q):
    if not vals:
        return 0.0
    s = sorted(vals)
    return s[min(len(s) - 1, int(q * len(s)))]


def roi_mask_for(poly, W, H):
    pts = np.array([[int(x * W), int(y * H)] for x, y in poly], np.int32)
    mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def report(title, items, fmt=str, limit=5):
    print(f"\n{title}: {len(items)}")
    for it in items[:limit]:
        print(f"  {fmt(it)}")
    if len(items) > limit:
        print(f"  ... and {len(items) - limit} more")


def main():
    args = parse_args()
    root = Path(args.dataset)
    imgdir, labdir = root / "images", root / "labels"
    if not imgdir.is_dir() or not labdir.is_dir():
        raise SystemExit(f"{root} must contain images/ and labels/ subdirectories")

    names = find_names(root, args.names)

    roi_poly = None
    if args.roi:
        roi_poly = json.loads(Path(args.roi).read_text())["polygon"]

    images = sorted(p for p in imgdir.iterdir() if p.suffix.lower() in IMG_EXT)
    if not images:
        raise SystemExit(f"no images found in {imgdir}")

    areas, counts = [], []
    per_class = {}
    malformed, out_of_range, missing_label = [], [], []
    oversized, overlaps, outside_roi = [], [], []
    veg_in_box = veg_total = box_px_total = 0
    roi_mask = None

    for ip in images:
        lp = labdir / (ip.stem + ".txt")
        if not lp.exists():
            missing_label.append(ip.name)
            counts.append(0)
            continue

        img = cv2.imread(str(ip))
        if img is None:
            malformed.append((ip.name, 0, "unreadable image"))
            continue
        H, W = img.shape[:2]

        if roi_poly is not None and (roi_mask is None or roi_mask.shape != (H, W)):
            roi_mask = roi_mask_for(roi_poly, W, H)

        good = []
        for b in load_boxes(lp):
            if b[0] == "BAD":
                malformed.append((ip.name, b[1], b[2]))
                continue
            _, xc, yc, w, h = b
            if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < w <= 1 and 0 < h <= 1):
                out_of_range.append(
                    (ip.name, f"xc={xc:.3f} yc={yc:.3f} w={w:.3f} h={h:.3f}"))
                continue
            good.append(b)
        counts.append(len(good))

        rects = [to_xyxy(b, W, H) for b in good]
        for b in good:
            area = b[3] * b[4]
            areas.append(area)
            per_class.setdefault(b[0], []).append(area)
            if area > args.max_area_frac:
                oversized.append((ip.name, area, b[0]))

        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                v = iou(rects[i], rects[j])
                if v >= args.overlap_iou:
                    overlaps.append((ip.name, i, j, round(v, 3)))

        if roi_mask is None:
            continue

        # vegetation = strong green *inside* the ROI; outside it, green is bank grass
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        green = cv2.inRange(hsv, (25, 60, 60), (95, 255, 255))
        veg = cv2.bitwise_and(green, green, mask=roi_mask) > 0
        veg_total += int(veg.sum())

        union = np.zeros((H, W), bool)
        for (x1, y1, x2, y2) in rects:
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if x2 > x1 and y2 > y1:
                union[y1:y2, x1:x2] = True
        veg_in_box += int((veg & union).sum())
        box_px_total += int(union.sum())

        for (x1, y1, x2, y2) in rects:
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            sub = roi_mask[y1:y2, x1:x2]
            frac_out = 1.0 - float((sub > 0).sum()) / sub.size
            if frac_out > args.outside_frac:
                outside_roi.append((ip.name, round(frac_out, 3)))

    total = sum(counts)
    print(f"=== {root} ===")
    print(f"images={len(images)}  labels={total}  "
          f"mean={statistics.mean(counts):.2f}  median={statistics.median(counts):.0f}  "
          f"min={min(counts)}  max={max(counts)}")
    print(f"empty/background files: {sum(1 for c in counts if c == 0)}")

    if areas:
        print("\nbox area (% of frame)")
        for q, lbl in ((0.05, "p5"), (0.25, "p25"), (0.50, "median"),
                       (0.75, "p75"), (0.95, "p95")):
            print(f"  {lbl:>6}: {percentile(areas, q) * 100:7.3f}%")
        print(f"  {'max':>6}: {max(areas) * 100:7.3f}%")

    if len(per_class) > 1 or names:
        print("\nper class")
        print(f"  {'id':>3} {'name':>10} {'count':>6} {'share':>7} "
              f"{'median':>9} {'p95':>9} {'max':>9}")
        for cid in sorted(per_class):
            a = sorted(per_class[cid])
            nm = names[cid] if names and cid < len(names) else "?"
            print(f"  {cid:>3} {nm:>10} {len(a):>6} {100 * len(a) / total:>6.1f}% "
                  f"{statistics.median(a) * 100:>8.3f}% {a[int(0.95 * len(a))] * 100:>8.3f}% "
                  f"{a[-1] * 100:>8.3f}%")

    report(f"OVERSIZED boxes (>{args.max_area_frac:.0%} of frame)", oversized,
           lambda t: f"{t[0][:38]}  {t[1] * 100:.1f}%  cls={t[2]}"
                     f"{'(' + names[t[2]] + ')' if names and t[2] < len(names) else ''}")
    report(f"HIGH-OVERLAP pairs (IoU>={args.overlap_iou})", overlaps,
           lambda t: f"{t[0][:34]}  box{t[1]}/box{t[2]}  IoU={t[3]}")
    if roi_poly is not None:
        report(f"boxes >{args.outside_frac:.0%} OUTSIDE the ROI", outside_roi,
               lambda t: f"{t[0][:44]}  {t[1] * 100:.0f}% outside")
    report("MALFORMED lines", malformed,
           lambda t: f"{t[0][:34]} line {t[1]}: {t[2]}")
    report("OUT-OF-RANGE coords", out_of_range, lambda t: f"{t[0][:34]}  {t[1]}")
    report("images with NO label file", missing_label)

    if roi_poly is not None and veg_total:
        print("\nvegetation inside the ROI (green pixels on the water)")
        print(f"  vegetation covered by labels    : {100 * veg_in_box / veg_total:5.1f}%")
        if box_px_total:
            print(f"  labelled area that is vegetation: {100 * veg_in_box / box_px_total:5.1f}%")

    if args.json:
        Path(args.json).write_text(json.dumps({
            "dataset": str(root), "names": names,
            "images": len(images),
            "labels": total,
            "counts": counts,
            "areas": areas,
            "oversized": oversized,
            "overlaps": overlaps,
            "outside_roi": outside_roi,
            "malformed": malformed,
            "out_of_range": out_of_range,
            "missing_label": missing_label,
            "veg_covered_frac": (veg_in_box / veg_total) if veg_total else None,
            "veg_frac_of_labels": (veg_in_box / box_px_total) if box_px_total else None,
        }, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
