"""Estimate how much of the water is covered by trash, and report it over MQTT.

Each cycle grabs a burst of frames, runs sliced inference restricted to the water
ROI, and reduces the burst to a median coverage figure. Coverage is the union
area of the detection boxes inside the ROI as a fraction of the ROI area --
computed from a pixel mask, so overlapping boxes are counted once, not twice.

The burst median is deliberate: a single frame can be spoiled by a ripple, sun
glint or a passing bird, and the median of several frames throws that away while
a mean would not.

A report is published every cycle, including when nothing is detected. A steady
stream of "coverage 0.0" is the signal that the camera is alive and the water is
clean; silence is indistinguishable from a crashed process.

The one exception is while the dock has this camera marked "servicing": the
collection boat is then sitting in the frame, and to the detector a boat is a
large piece of trash. Reporting through that would tell the dock the water got
dirtier the moment it sent the boat. So the cycle is skipped entirely -- with a
line saying so, which keeps the "silence means dead" rule intact for the log
even though nothing goes on the wire.

The reading is a coverage proxy, not a quantity of rubbish: the model detects
regions, and its recall on held-out water is well under half, so treat the number
as a relative trend for one fixed camera rather than an absolute measure.

Broker credentials come from the environment (MQTT_USERNAME / MQTT_PASSWORD),
never from the command line, where they would land in shell history.

Examples:
    python coverage_monitor.py --dir frames/ --roi water_roi_data2.json --dry-run
    python coverage_monitor.py --camera 0 --roi water_roi_data2.json \
        --broker mqtt.example.org --lat 13.7563 --long 100.5018
"""

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from detect_sliced import (build_model, filter_roi, pick_device, predict_sliced,
                           roi_mask_for)

BASE = Path(__file__).parent
WEIGHTS = BASE / "runs" / "trash_yolo11s" / "weights" / "best.pt"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_argument_group("frame source")
    src.add_argument("--camera", type=int, default=None, help="cv2 camera index")
    src.add_argument("--video", default=None, help="video file")
    src.add_argument("--dir", default=None, help="directory of images (testing)")

    det = p.add_argument_group("detection")
    det.add_argument("--weights", default=str(WEIGHTS))
    det.add_argument("--roi", required=True,
                     help="JSON holding a 'polygon' of normalised [x,y] pairs")
    det.add_argument("--slice", type=int, default=512)
    det.add_argument("--overlap", type=float, default=0.20)
    det.add_argument("--conf", type=float, default=0.25)
    det.add_argument("--device", default=None)

    b = p.add_argument_group("burst")
    b.add_argument("--burst", type=int, default=5, help="frames per cycle")
    b.add_argument("--burst-gap", type=float, default=1.0,
                   help="seconds between frames within a burst")
    b.add_argument("--interval", type=int, default=300, help="seconds between cycles")
    b.add_argument("--once", action="store_true", help="run one cycle and exit")

    m = p.add_argument_group("mqtt")
    m.add_argument("--camera-id", default="cam_1")
    m.add_argument("--lat", type=float, default=None)
    m.add_argument("--long", type=float, default=None)
    m.add_argument("--broker", default=None)
    m.add_argument("--port", type=int, default=1883)
    m.add_argument("--topic", default="water/trash/coverage")
    m.add_argument("--servicing-topic", default="trash/dock/servicing",
                   help="topic the dock marks cameras servicing/clear on; while "
                        "this camera is servicing the boat is in frame and looks "
                        "like trash, so reporting is suppressed. Empty to disable")
    m.add_argument("--qos", type=int, default=1, choices=[0, 1, 2])
    m.add_argument("--retain", action="store_true")
    m.add_argument("--tls", action="store_true", help="enable TLS (port 8883 typical)")
    m.add_argument("--dry-run", action="store_true",
                   help="print the payload instead of publishing")
    return p.parse_args()


def coverage_of(dets, roi_mask):
    """Union area of detections inside the ROI, as a fraction of the ROI area.

    Rasterised into a mask so overlapping boxes contribute their union once.
    Summing box areas would double-count every overlap and can exceed 1.0.
    """
    roi_px = int((roi_mask > 0).sum())
    if roi_px == 0:
        return 0.0
    H, W = roi_mask.shape
    union = np.zeros((H, W), bool)
    for (x1, y1, x2, y2, _score) in dets:
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(W, int(x2)), min(H, int(y2))
        if x2 > x1 and y2 > y1:
            union[y1:y2, x1:x2] = True
    inside = int((union & (roi_mask > 0)).sum())
    return inside / roi_px


class FrameSource:
    """Yields frames from a camera, a video file, or a directory of images."""

    def __init__(self, args):
        self.kind = ("dir" if args.dir else "video" if args.video else "camera")
        self.cap = None
        if self.kind == "dir":
            self.paths = sorted(p for p in Path(args.dir).iterdir()
                                if p.suffix.lower() in IMG_EXT)
            if not self.paths:
                raise SystemExit(f"no images in {args.dir}")
            self.i = 0
        elif self.kind == "video":
            self.cap = cv2.VideoCapture(args.video)
            if not self.cap.isOpened():
                raise SystemExit(f"could not open video {args.video}")
        else:
            idx = 0 if args.camera is None else args.camera
            self.cap = cv2.VideoCapture(idx)
            if not self.cap.isOpened():
                raise SystemExit(f"could not open camera {idx}")

    def read(self):
        if self.kind == "dir":
            if not self.paths:
                return None
            p = self.paths[self.i % len(self.paths)]
            self.i += 1
            return cv2.imread(str(p))
        ok, frame = self.cap.read()
        return frame if ok else None

    def close(self):
        if self.cap is not None:
            self.cap.release()


class Reporter:
    """MQTT publisher that never takes the monitoring loop down with it.

    It also listens on the dock's servicing topic, because the suppression rule
    is a two-way contract: the dock says "the boat is in your frame now", and
    this end has to stop reporting until it hears "clear".
    """

    def __init__(self, args):
        self.args = args
        self.client = None
        self.servicing = False
        if args.dry_run or not args.broker:
            return
        import paho.mqtt.client as mqtt
        self.client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=f"trash-{args.camera_id}",
        )
        user = os.environ.get("MQTT_USERNAME")
        if user:
            self.client.username_pw_set(user, os.environ.get("MQTT_PASSWORD"))
        if args.tls:
            self.client.tls_set()
        # keep the payload flowing across broker restarts without manual retries
        self.client.reconnect_delay_set(min_delay=1, max_delay=60)
        # Subscribing from on_connect, not once after connect(): a reconnect
        # gives us a fresh session, and a subscription made only at startup
        # would be silently lost - we would go on reporting the boat as trash.
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_servicing
        try:
            self.client.connect(args.broker, args.port, keepalive=60)
            self.client.loop_start()
            print(f"MQTT connected to {args.broker}:{args.port} topic '{args.topic}'")
        except Exception as e:                       # noqa: BLE001 - never fatal
            print(f"MQTT connect failed ({e}); will retry on publish", file=sys.stderr)

    def _on_connect(self, client, _userdata, _flags, reason, _props=None):
        if self.args.servicing_topic:
            client.subscribe(self.args.servicing_topic, qos=1)
            print(f"listening for servicing flags on '{self.args.servicing_topic}'")

    def _on_servicing(self, _client, _userdata, msg):
        """Dock -> camera: {"camera": "cam_1", "status": "servicing"|"clear"}."""
        try:
            payload = json.loads(msg.payload.decode())
        except (ValueError, UnicodeDecodeError):
            return                                   # not ours to interpret
        if payload.get("camera") != self.args.camera_id:
            return                                   # another camera's flag
        status = payload.get("status")
        if status == "servicing" and not self.servicing:
            self.servicing = True
            print("dock marked this camera SERVICING - the boat is in frame and "
                  "reads as trash, so reporting is suppressed until it clears")
        elif status == "clear" and self.servicing:
            self.servicing = False
            print("dock marked this camera CLEAR - reporting resumes")

    def publish(self, payload):
        line = json.dumps(payload, separators=(",", ":"))
        if self.client is None:
            print(f"[dry-run] {self.args.topic} {line}")
            return True
        try:
            info = self.client.publish(self.args.topic, line,
                                       qos=self.args.qos, retain=self.args.retain)
            info.wait_for_publish(timeout=10)
            ok = info.is_published()
            print(f"[{'sent' if ok else 'queued'}] {line}")
            return ok
        except Exception as e:                       # noqa: BLE001 - never fatal
            print(f"MQTT publish failed ({e}); continuing", file=sys.stderr)
            return False

    def close(self):
        if self.client is not None:
            self.client.loop_stop()
            self.client.disconnect()


def run_cycle(args, model, source, roi_poly, state):
    values, used = [], 0
    for i in range(args.burst):
        frame = source.read()
        if frame is None:
            continue
        H, W = frame.shape[:2]
        if state.get("shape") != (H, W):
            state["shape"] = (H, W)
            state["mask"] = roi_mask_for(roi_poly, W, H)
        dets = filter_roi(predict_sliced(model, frame, args.slice, args.overlap),
                          state["mask"])
        values.append(coverage_of(dets, state["mask"]))
        used += 1
        if i < args.burst - 1 and args.burst_gap > 0 and source.kind != "dir":
            time.sleep(args.burst_gap)

    if not used:
        return None, values
    return statistics.median(values), values


def main():
    args = parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"{weights} not found -- run 'python train.py' first.")
    roi_poly = json.loads(Path(args.roi).read_text())["polygon"]

    model = build_model(weights, args.conf, pick_device(args.device))
    source = FrameSource(args)
    reporter = Reporter(args)
    state = {}

    print(f"camera={args.camera_id} burst={args.burst} interval={args.interval}s "
          f"slice={args.slice}px/{args.overlap:.0%} conf>={args.conf}")
    try:
        while True:
            if reporter.servicing:
                # Skip the inference too, not just the publish: the frames would
                # be of a boat, and a burst of them costs a second of GPU each.
                print("cycle skipped: camera is being serviced")
                if args.once:
                    break
                time.sleep(args.interval)
                continue
            median, values = run_cycle(args, model, source, roi_poly, state)
            if median is None:
                print("no frames captured this cycle; skipping report",
                      file=sys.stderr)
            else:
                payload = {
                    "camera": args.camera_id,
                    "coverage": round(median, 4),
                    "lat": args.lat,
                    "long": args.long,
                    "timestamp": datetime.now(timezone.utc)
                                 .isoformat(timespec="seconds")
                                 .replace("+00:00", "Z"),
                }
                spread = f"{min(values):.4f}-{max(values):.4f}" if values else "-"
                print(f"burst n={len(values)} median={median:.4f} range={spread}")
                reporter.publish(payload)

            if args.once:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        source.close()
        reporter.close()


if __name__ == "__main__":
    main()
