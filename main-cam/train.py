"""Fine-tune YOLO11s on the merged single-class trash dataset.

Starts from COCO-pretrained yolo11s.pt (transfer learning), not from scratch --
the merged set is only ~525 training images, far too small to train from random init.

Examples:
    python train.py                          # full run with defaults
    python train.py --epochs 2 --name probe  # quick timing probe
"""

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

BASE = Path(__file__).parent
DATA = BASE / "dataset_merged" / "data.yaml"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="yolo11s.pt", help="pretrained checkpoint to start from")
    p.add_argument("--data", default=str(DATA),
                   help="dataset yaml (default: the merged public set)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default=None,
                   help="'0' for first CUDA GPU, 'cpu' to force CPU. "
                        "Default: auto-detect (GPU if available)")
    p.add_argument("--patience", type=int, default=25, help="early-stop patience in epochs")
    # 'auto' picks the optimizer AND the lr, silently ignoring --lr0; name an
    # optimizer explicitly for --lr0 to take effect
    p.add_argument("--optimizer", default="auto",
                   choices=["auto", "SGD", "Adam", "AdamW", "RMSProp"])
    p.add_argument("--lr0", type=float, default=None,
                   help="initial lr; requires --optimizer other than 'auto'")
    p.add_argument("--cache", default="ram", choices=["ram", "disk", "False"],
                   help="cache images to speed up CPU dataloading")
    p.add_argument("--name", default="trash_yolo11s")
    # absolute: a relative project would be nested under Ultralytics' own
    # runs_dir/task, giving runs/detect/runs/detect/<name>
    p.add_argument("--project", default=str(BASE / "runs"))
    p.add_argument("--resume", action="store_true", help="resume the last interrupted run")
    return p.parse_args()


def main():
    args = parse_args()

    if args.lr0 is not None and args.optimizer == "auto":
        raise SystemExit("--lr0 is ignored when --optimizer is 'auto'; "
                         "pass e.g. --optimizer AdamW")

    data = Path(args.data)
    if not data.exists():
        raise FileNotFoundError(
            f"{data} not found -- run 'python build_dataset.py' first."
        )

    device = args.device
    if device is None:
        device = "0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: training on CPU -- expect ~9 min/epoch at imgsz=640.\n"
              "         If this machine has an NVIDIA GPU, torch is probably the CPU-only\n"
              "         build; reinstall with the CUDA index URL (see README).")
    else:
        print(f"Training on GPU: {torch.cuda.get_device_name(int(device))}")

    model = YOLO(args.model)

    model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=device,
        patience=args.patience,
        cache=(False if args.cache == "False" else args.cache),
        project=args.project,
        name=args.name,
        resume=args.resume,
        exist_ok=True,
        seed=42,
        val=True,
        plots=True,
        # single small-object class: keep the default aug recipe but lean on
        # scale/mosaic, and close mosaic near the end so the model sees clean images
        close_mosaic=10,
        pretrained=True,
        optimizer=args.optimizer,
        **({"lr0": args.lr0} if args.lr0 is not None else {}),
    )

    # final evaluation on the held-out test split
    metrics = model.val(data=str(data), split="test", imgsz=args.imgsz, device=device)
    print("\n=== held-out test split ===")
    print(f"mAP50    : {metrics.box.map50:.4f}")
    print(f"mAP50-95 : {metrics.box.map:.4f}")
    print(f"precision: {metrics.box.mp:.4f}")
    print(f"recall   : {metrics.box.mr:.4f}")

    best = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"\nBest weights: {best.resolve()}")


if __name__ == "__main__":
    main()
