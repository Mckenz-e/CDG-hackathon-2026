"""Sliced (tiled) inference with SAHI, optionally restricted to a water ROI.

Small objects vanish under whole-frame inference: a 29x17 px bottle in a
1920x1080 frame is under 10x6 px once the frame is resized to 640, at or below
the detector's smallest stride. SAHI runs the same weights over overlapping
tiles at native resolution, so the object stays 29 px inside a 512 px tile, then
merges the tile detections back into frame coordinates.

The cost is one forward pass per tile rather than per frame.

--roi drops detections whose centre falls outside a polygon, which removes the
whole class of false positives on land (hoses, grass, bank clutter) without
retraining. It cannot add detections, so it trades nothing for precision.

Examples:
    python detect_sliced.py                                   # test.jpg -> test_sliced.jpg
    python detect_sliced.py --source frame.jpg --slice 512 --overlap 0.2
    python detect_sliced.py --dir frames/ --roi water_roi.json   # -> detections/
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).parent
WEIGHTS = BASE / "runs" / "trash_yolo11s" / "weights" / "best.pt"

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BOX_COLOR = (0, 0, 255)
TEXT_COLOR = (255, 255, 255)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default=str(WEIGHTS))
    p.add_argument("--source", default="test.jpg", help="single image")
    p.add_argument("--dir", default=None,
                   help="run over every image in this directory instead of --source")
    p.add_argument("--output", default="test_sliced.jpg",
                   help="annotated output for a single --source")
    p.add_argument("--output-dir", default="detections",
                   help="directory collecting annotated images in --dir mode. "
                        "Kept separate from the inputs so a second run does not "
                        "re-process its own outputs")
    p.add_argument("--slice", type=int, default=512, help="tile size in pixels")
    p.add_argument("--overlap", type=float, default=0.20, help="tile overlap ratio")
    p.add_argument("--conf", type=float, default=0.10, help="confidence threshold")
    p.add_argument("--device", default=None, help="'cuda:0' or 'cpu'")
    p.add_argument("--roi", default=None,
                   help="JSON holding a 'polygon' of normalised [x,y] pairs; "
                        "detections centred outside it are dropped")
    p.add_argument("--whole-frame", action="store_true",
                   help="plain whole-frame inference, for comparison")
    p.add_argument("--save-labels", default=None,
                   help="write predictions as 'cls xc yc w h conf' text files here")
    p.add_argument("--no-image", action="store_true")
    return p.parse_args()


def pick_device(device):
    if device:
        return device
    import torch
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def build_model(weights, conf, device):
    from sahi import AutoDetectionModel
    return AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=str(weights),
        confidence_threshold=conf,
        device=device,
    )


def roi_mask_for(poly, W, H):
    pts = np.array([[int(x * W), int(y * H)] for x, y in poly], np.int32)
    mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def as_sahi_input(image):
    """Path -> str; OpenCV frame -> RGB array.

    SAHI reads a file through PIL (RGB) but takes an ndarray as RGB already,
    while cv2 hands us BGR. Passing a raw cv2 frame therefore swaps red and
    blue and quietly degrades detection, so convert explicitly.
    """
    if isinstance(image, np.ndarray):
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return str(image)


def predict_sliced(model, image, slice_px, overlap):
    from sahi.predict import get_sliced_prediction
    r = get_sliced_prediction(
        as_sahi_input(image), model,
        slice_height=slice_px, slice_width=slice_px,
        overlap_height_ratio=overlap, overlap_width_ratio=overlap,
        verbose=0,
    )
    return [(*(int(v) for v in o.bbox.to_xyxy()), float(o.score.value))
            for o in r.object_prediction_list]


def predict_whole(model, image):
    from sahi.predict import get_prediction
    r = get_prediction(as_sahi_input(image), model)
    return [(*(int(v) for v in o.bbox.to_xyxy()), float(o.score.value))
            for o in r.object_prediction_list]


def filter_roi(dets, mask):
    """Keep detections whose centre lies inside the mask."""
    if mask is None:
        return dets
    H, W = mask.shape
    kept = []
    for (x1, y1, x2, y2, s) in dets:
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if 0 <= cx < W and 0 <= cy < H and mask[cy, cx] > 0:
            kept.append((x1, y1, x2, y2, s))
    return kept


def annotate(img, dets):
    for x1, y1, x2, y2, s in dets:
        cv2.rectangle(img, (x1, y1), (x2, y2), BOX_COLOR, 2)
        label = f"{s:.2f}"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - base), (x1 + tw, y1), BOX_COLOR, -1)
        cv2.putText(img, label, (x1, y1 - base),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)
    return img


def write_labels(path, dets, W, H):
    lines = [
        f"0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} "
        f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f} {s:.4f}"
        for x1, y1, x2, y2, s in dets
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main():
    args = parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"{weights} not found -- run 'python train.py' first.")

    device = pick_device(args.device)
    model = build_model(weights, args.conf, device)

    roi_poly = None
    if args.roi:
        roi_poly = json.loads(Path(args.roi).read_text())["polygon"]

    out_dir = None
    if args.dir:
        sources = sorted(p for p in Path(args.dir).iterdir()
                         if p.suffix.lower() in IMG_EXT
                         and not p.stem.endswith("_sliced"))
        if not sources:
            raise SystemExit(f"no images in {args.dir}")
        out_dir = Path(args.output_dir)
        if not args.no_image:
            out_dir.mkdir(parents=True, exist_ok=True)
    else:
        sources = [Path(args.source)]

    label_dir = Path(args.save_labels) if args.save_labels else None
    if label_dir:
        label_dir.mkdir(parents=True, exist_ok=True)

    mask = None
    total = dropped = 0
    for src in sources:
        img = cv2.imread(str(src))
        if img is None:
            raise FileNotFoundError(f"Could not read image at '{src}'")
        H, W = img.shape[:2]
        if roi_poly is not None and (mask is None or mask.shape != (H, W)):
            mask = roi_mask_for(roi_poly, W, H)

        dets = (predict_whole(model, src) if args.whole_frame
                else predict_sliced(model, src, args.slice, args.overlap))
        before = len(dets)
        dets = filter_roi(dets, mask)
        dropped += before - len(dets)
        total += len(dets)

        if out_dir is None:
            print(f"{len(dets)} detection(s)"
                  + (f"  ({before - len(dets)} dropped outside ROI)" if roi_poly else ""))
        if label_dir:
            write_labels(label_dir / (src.stem + ".txt"), dets, W, H)
        if not args.no_image:
            out = (Path(args.output) if out_dir is None
                   else out_dir / f"{src.stem}_sliced.jpg")
            cv2.imwrite(str(out), annotate(img, dets))
            if out_dir is None:
                print(f"Saved annotated image to '{out}'")

    if out_dir is not None:
        if not args.no_image:
            print(f"Annotated images -> {out_dir}/")
        mode = "whole-frame" if args.whole_frame else f"slice={args.slice}px/{args.overlap:.0%}"
        print(f"{total} detections over {len(sources)} images "
              f"({total / len(sources):.2f}/image), {mode}, conf>={args.conf}"
              + (f", {dropped} dropped outside ROI" if roi_poly else ""))


if __name__ == "__main__":
    main()
