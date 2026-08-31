#!/usr/bin/env python3
"""Run the entire system end to end with one command.

    python demo.py

Starts an MQTT broker, the real camera server (YOLO11s + SAHI over real pond
frames) and the dock station with its boat, then streams all three into one
narrated timeline and prints a summary of what happened.

Nothing here is mocked out. The coverage figures come from running the detector
on real images, the dispatch decisions come from the real dock logic, and the
messages really cross a real broker. What is simulated is the water and the
boat - there is no hardware.

The whole story - detection, threshold crossing, dispatch, a complete mission,
suppression while the boat sits in frame, cooldown, re-dispatch - fits in about
two minutes, because config.TIME_SCALE compresses the simulated clock.

    python demo.py --duration 240      # let it run longer
    python demo.py --no-vision         # skip the detector and use the built-in
                                       # simulated camera instead (insurance, if
                                       # torch or CUDA is broken on this machine)
    python demo.py --check             # run the preflight checks and change nothing

If a check fails it says exactly what to install or fix and stops, rather than
hanging halfway through in front of an audience.
"""
import argparse
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from queue import Empty, Queue

ROOT = Path(__file__).resolve().parent
DOCK = ROOT / "Dock"
VISION = ROOT / "main-cam"

sys.path.insert(0, str(DOCK))
import config                                   # noqa: E402  (needs DOCK on path)

WEIGHTS = VISION / "runs" / "trash_yolo11s" / "weights" / "best.pt"
ROI = VISION / "water_roi_data2.json"
FRAMES = VISION / "data2" / "train" / "images"
BROKER_CFG = DOCK / "broker.yaml"
RUN_LOG = ROOT / "demo_run.log"

CAMERA_ID = "cam_1"
BURST = 5
WINDOWS = os.name == "nt"


def interpreter(venv_dir):
    """The python inside a venv, or None if that venv is not there."""
    exe = venv_dir / ("Scripts/python.exe" if WINDOWS else "bin/python")
    return exe if exe.exists() else None


# The broker and the detector live in different environments on this machine:
# amqtt in the top-level venv, torch/ultralytics/paho in main-cam's. Falling
# back to whatever is running this script keeps a single-venv setup working too.
BROKER_PY = interpreter(ROOT / "venv") or Path(sys.executable)
VISION_PY = interpreter(VISION / "venv") or Path(sys.executable)


# ============================== presentation ================================
ESC = chr(27)


class Ink:
    def __init__(self, enabled):
        self.on = enabled
        if enabled and WINDOWS:
            os.system("")                       # enable ANSI on Windows 10+

    def __call__(self, code, text):
        if not self.on:
            return text
        return "%s[%sm%s%s[0m" % (ESC, code, text, ESC)


ink = Ink(False)

TAGS = {                                        # source -> (label, colour)
    "broker": ("broker", "90"),
    "dock": ("dock", "36"),
    "camera": ("camera", "33"),
}


def banner(text):
    print()
    print(ink("1;37", "  " + text))
    print(ink("90", "  " + "-" * (len(text) + 2)))


def say(text):
    print(ink("90", "  . ") + text)


# ============================== preflight ===================================
def port_open(host, port, timeout=0.3):
    with socket.socket() as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


def module_missing(python, module):
    """None if `python` can import `module`, else a short reason why not."""
    result = subprocess.run([str(python), "-c", "import " + module],
                            capture_output=True, text=True)
    if result.returncode == 0:
        return None
    tail = (result.stderr.strip().splitlines() or ["import failed"])[-1]
    return tail[:110]


def preflight(use_vision):
    """Every reason this demo could fail, checked before anything starts."""
    problems = []

    if not BROKER_CFG.exists():
        problems.append("missing %s" % BROKER_CFG)
    why = module_missing(BROKER_PY, "amqtt")
    if why:
        problems.append("no MQTT broker: %s cannot import amqtt (%s)\n"
                        "      fix: \"%s\" -m pip install amqtt"
                        % (BROKER_PY.name, why, BROKER_PY))

    why = module_missing(VISION_PY, "paho.mqtt.client")
    if why:
        problems.append("the sim needs paho-mqtt to reach the broker (%s)\n"
                        "      fix: \"%s\" -m pip install paho-mqtt"
                        % (why, VISION_PY))

    if port_open("127.0.0.1", config.MQTT_PORT):
        problems.append("something is already listening on port %d - stop it, or "
                        "it will answer instead of our broker" % config.MQTT_PORT)

    if use_vision:
        if not WEIGHTS.exists():
            problems.append("no model weights at %s\n"
                            "      fix: run 'python train.py' in main-cam, or pass "
                            "--no-vision" % WEIGHTS)
        if not ROI.exists():
            problems.append("no water ROI at %s" % ROI)
        if not FRAMES.is_dir() or not any(FRAMES.iterdir()):
            problems.append("no camera frames in %s" % FRAMES)
        for mod in ("torch", "ultralytics", "sahi", "cv2"):
            why = module_missing(VISION_PY, mod)
            if why:
                problems.append("the detector cannot import %s (%s)\n"
                                "      fix: install main-cam/requirements.txt, or "
                                "pass --no-vision to run without the detector"
                                % (mod, why))
                break
    return problems


# ============================== running =====================================
def kill_tree(proc):
    """Kill a child and everything it started.

    A console script is a launcher holding a real interpreter underneath;
    terminating only the launcher leaves a broker sitting on the port, and the
    next run then fails its own preflight for no visible reason.
    """
    if proc is None or proc.poll() is not None:
        return
    if WINDOWS:
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                       capture_output=True)
    else:
        proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


# Per-tile inference chatter from ultralytics. Useful when debugging the model,
# pure noise when the point is the control logic.
NOISE = ("0: ", "Speed:", "image 1/1", "WARNING", "Ultralytics", "YOLO11",
         "Fusing", "Adding AMP", "AMP:", "Performing", "requirements:")


def interesting(source, line):
    line = line.strip()
    if not line:
        return False
    if source == "broker":
        return False                            # only its failure to start matters
    if source == "camera" and line.startswith(NOISE):
        return False
    return True


class Show:
    """Runs the processes and merges their output into one timeline."""

    def __init__(self):
        self.procs = []
        self.queue = Queue()

    def start(self, source, cmd, cwd):
        proc = subprocess.Popen(
            [str(c) for c in cmd], cwd=str(cwd),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            env=dict(os.environ, PYTHONUNBUFFERED="1"))
        self.procs.append((source, proc))
        threading.Thread(target=self._pump, args=(source, proc),
                         daemon=True).start()
        return proc

    def _pump(self, source, proc):
        for line in proc.stdout:
            self.queue.put((source, line.rstrip("\n")))
        self.queue.put((source, None))          # this stream ended

    def drain(self, until_source_ends, timeout):
        """Print merged output until that stream ends, or `timeout` passes."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                source, line = self.queue.get(timeout=0.2)
            except Empty:
                for src, proc in self.procs:    # did it die without closing?
                    if src == until_source_ends and proc.poll() is not None:
                        return "ended"
                continue
            if line is None:
                if source == until_source_ends:
                    return "ended"
                continue
            if interesting(source, line):
                label, colour = TAGS[source]
                print("  " + ink(colour, "%-7s" % label) + " " + line)
        return "timeout"

    def stop(self):
        for _, proc in reversed(self.procs):
            kill_tree(proc)


# ============================== summary =====================================
def summarise(log_path, use_vision):
    """Read the shared run log back and say what the audience just watched."""
    if not log_path.exists():
        return
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()

    def count(needle):
        return sum(1 for line in lines if needle in line)

    dispatches = count("dispatching to")
    # "ended (" and not "mission for": the dock logs "boat acknowledged mission
    # for cam_1" too, which is a mission starting, not one finishing.
    missions = count("ended (")
    collected = count("collected item")
    serviced = count("marked SERVICING")
    improved = count("improved after service")
    unimproved = count("did NOT improve")
    faults = count("-> fault")
    attention = count("NEEDS ATTENTION")

    def n(value):
        return ink("1;37", "%3d" % value)

    banner("What just happened")
    say("%s  camera reports the dock accepted" % n(count("report queued")))
    say("%s  dispatches, each one a %s crossing the threshold"
        % (n(dispatches),
           "real detection" if use_vision else "simulated camera reading"))
    say("%s  times the camera was told to stop reporting, boat in frame"
        % n(serviced))
    say("%s  missions completed, %d items collected" % (n(missions), collected))
    if dispatches > missions:
        say("     (%d still under way when the run ended - give it --duration %d "
            "to see them finish)" % (dispatches - missions, 240))
    if improved or unimproved:
        say("%s  services judged after cooldown - %d improved, %d did not"
            % (n(improved + unimproved), improved, unimproved))
    if attention:
        say("%s  camera(s) flagged NEEDS ATTENTION and dropped from rotation"
            % n(attention))
    if faults:
        say(ink("31", "%3d  faults" % faults))
    say("full timeline: %s" % ink("1;37", str(log_path)))


# ============================== main ========================================
def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--duration", type=float, default=150.0,
                        help="how long to run, in wall-clock seconds (default 150)")
    parser.add_argument("--time-scale", type=float, default=12.0,
                        help="how hard to compress the simulated clock (default 12)")
    parser.add_argument("--interval", type=int, default=4,
                        help="seconds between detector cycles (default 4)")
    parser.add_argument("--min-coverage", type=float, default=0.05,
                        help="coverage a report must reach to earn a mission")
    parser.add_argument("--no-vision", action="store_true",
                        help="skip the detector, use the built-in simulated camera")
    parser.add_argument("--check", action="store_true",
                        help="run the preflight checks and exit")
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args()

    # Line-buffer our own output. Python block-buffers stdout when it is a pipe,
    # so `python demo.py | tee demo.txt` would show nothing for two minutes and
    # then everything at once - exactly the wrong behaviour for a live demo.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        pass

    global ink
    ink = Ink(not args.no_color and sys.stdout.isatty())
    use_vision = not args.no_vision

    banner("Autonomous trash collection - the whole system, end to end")
    say("broker       %s" % BROKER_PY)
    say("sim + model  %s" % VISION_PY)

    problems = preflight(use_vision)
    if problems:
        print()
        print(ink("31", "  Cannot run:"))
        for problem in problems:
            print(ink("31", "    - ") + problem)
        print()
        return 1
    say(ink("32", "preflight passed"))
    if args.check:
        print()
        return 0

    cam = config.CAMERAS[CAMERA_ID]
    show = Show()
    try:
        banner("1. Starting the MQTT broker")
        show.start("broker", [BROKER_PY, "-m", "amqtt.scripts.broker_script",
                              "-c", BROKER_CFG], cwd=ROOT)
        for _ in range(50):
            if port_open("127.0.0.1", config.MQTT_PORT):
                break
            time.sleep(0.2)
        else:
            print(ink("31", "  the broker never opened port %d"
                      % config.MQTT_PORT))
            return 1
        say("listening on 127.0.0.1:%d" % config.MQTT_PORT)

        banner("2. Starting the dock and its boat")
        say("simulated clock compressed %gx, so a %.0f-minute mission limit "
            "becomes ~%.0fs" % (args.time_scale, config.MISSION_TIME_LIMIT_S / 60.0,
                                config.MISSION_TIME_LIMIT_S / args.time_scale))
        dock_cmd = [VISION_PY, "sim.py", "--mqtt",
                    "--host", "127.0.0.1", "--port", config.MQTT_PORT,
                    # --realtime 1 pins one virtual second to one real second,
                    # so --duration is already in wall-clock seconds. TIME_SCALE
                    # is what buys the speed-up: it shrinks the durations the sim
                    # measures, it does not stretch the clock.
                    "--realtime", 1,
                    "--time-scale", args.time_scale,
                    "--duration", args.duration,
                    "--min-coverage", args.min_coverage,
                    "--coverage-drop-min", 0.03,
                    "--log-file", RUN_LOG]
        if use_vision:
            dock_cmd.append("--no-cameras")     # the real camera server feeds it
        show.start("dock", dock_cmd, cwd=DOCK)

        if use_vision:
            banner("3. Starting the real camera server")
            say("YOLO11s over SAHI tiles on real pond frames, %d-frame bursts, "
                "median of each burst" % BURST)
            say("the first cycle loads the model onto the GPU - give it a moment")
            show.start("camera", [
                VISION_PY, "coverage_monitor.py",
                "--dir", FRAMES, "--roi", ROI,
                "--broker", "127.0.0.1", "--port", config.MQTT_PORT,
                "--topic", config.TOPIC_CAMERA_REPORT,
                "--camera-id", CAMERA_ID,
                "--lat", cam["lat"], "--long", cam["long"],
                "--burst", BURST, "--interval", args.interval], cwd=VISION)
        else:
            banner("3. Using the built-in simulated camera server")
            say("no detector in this run (--no-vision)")

        banner("Live: detect -> report -> dispatch -> mission -> suppress -> repeat")
        show.drain(until_source_ends="dock", timeout=args.duration + 120)
    except KeyboardInterrupt:
        print()
        say("stopped early")
    finally:
        show.stop()

    summarise(RUN_LOG, use_vision)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
