"""Simulation runner - wires everything together and drives the clock.

Run it:      python sim.py
Options:     python sim.py --duration 1800 --realtime 0.05

The loop is deliberately single-threaded: each tick advances a virtual clock,
gives every component a chance to act, then delivers all messages. That makes
runs reproducible - the same seed always produces the same log.
"""
import argparse
import time

import config
from boat import Boat
from bus import make_bus
from camera_server import CameraServer
from dock import DockStation
from log import RunLog
from world import World


class Clock:
    """Virtual time. Nothing in the sim calls time.time() for decisions.

    Time is counted in whole ticks and multiplied out, never accumulated by
    repeated addition. `t += 0.05` seven thousand times drifts, because 0.05 has
    no exact binary form; `n * 0.05` rounds once. The drift is tiny, but every
    interval in the sim is an `if now - last >= interval` against a value that
    lands exactly on a tick boundary, so a few ulps decide whether a report
    fires on tick 40 or tick 41 - and TIME_SCALE makes small ticks the norm.
    """

    def __init__(self):
        self.t = 0.0
        self._ticks = 0

    def now(self):
        return self.t

    def advance(self, dt):
        self._ticks += 1
        self.t = self._ticks * dt


def run(duration, realtime, enable_cameras=True, log_path=None):
    clock = Clock()
    logs = RunLog(clock, path=log_path)
    log = logs.component("sim")

    log("=== autonomous trash-collection simulation ===")
    if log_path:
        log("logging every component to %s" % log_path)
    if config.TIME_SCALE != 1.0:
        log("TIME_SCALE %g: every duration divided and every rate multiplied by it, "
            "so the run is compressed, not altered (tick %.3fs, mission limit %.0fs)"
            % (config.TIME_SCALE, config.TICK_S, config.MISSION_TIME_LIMIT_S))
    bus = make_bus(log=logs.component("bus"))
    world = World(log=logs.component("world"))
    cameras = CameraServer(bus, world, clock, log=logs.component("camera_server"))
    dock = DockStation(bus, clock, log=logs.component("dock"))
    boat = Boat(bus, world, log=logs.component("boat"))

    log("dock at (%.6f, %.6f); %d cameras; %d trash items in the water"
        % (config.DOCK_LAT, config.DOCK_LONG, len(config.CAMERAS), len(world.trash)))
    if not enable_cameras:
        log("built-in camera server DISABLED - the dock will only act on reports "
            "published from outside (see mqtt_pub_test.py)")
    log("dispatch floor: coverage >= %.4f; a service counts as an improvement at a "
        "drop of %.4f" % (config.DISPATCH_MIN_COVERAGE, config.COVERAGE_DROP_MIN))

    dt = config.TICK_S
    while clock.now() < duration:
        now = clock.now()
        world.tick(now)
        if enable_cameras:
            cameras.tick(now)
        dock.tick(now)
        boat.tick(now, dt)
        bus.poll()                 # deliver everything published this tick
        clock.advance(dt)
        if realtime > 0:
            time.sleep(dt * realtime)

    log("=== simulation finished ===")
    print(dock.summary())
    print("--- world ---")
    for cam_id in config.CAMERAS:
        print("  %s: %d items still in view (coverage %.2f)"
              % (cam_id, world.count_near(cam_id), world.coverage(cam_id)))
    print("  boat: state=%s battery=%.1f%% fill=%.2f%s"
          % (boat.state, boat.battery, boat.fill_level(),
             " fault=" + boat.fault_reason if boat.fault_reason else ""))
    bus.close()
    logs.close()
    return dock, boat, world


def main():
    p = argparse.ArgumentParser(description="Autonomous trash-collection sim")
    p.add_argument("--duration", type=float, default=config.SIM_DURATION_S,
                   help="virtual seconds to simulate")
    p.add_argument("--realtime", type=float, default=config.REAL_TIME_FACTOR,
                   help="0 = as fast as possible, 1 = real time")
    p.add_argument("--mqtt", action="store_true",
                   help="use a real MQTT broker instead of the in-process bus")
    p.add_argument("--host", default=config.MQTT_HOST)
    p.add_argument("--port", type=int, default=config.MQTT_PORT)
    p.add_argument("--no-cameras", action="store_true",
                   help="do not run the built-in camera server, so the dock acts "
                        "only on reports published from outside")
    p.add_argument("--log-file", default="run.log",
                   help="single file every component logs to, with wall-clock "
                        "time, virtual time and component name on each line. "
                        "'-' or '' writes to the console only")
    p.add_argument("--time-scale", type=float, default=None,
                   help="divide every simulated duration by this and multiply "
                        "every per-second rate by it, compressing the run without "
                        "changing what happens (config.TIME_SCALE). 10 turns a "
                        "15-minute mission into 90 seconds of virtual time")
    p.add_argument("--min-coverage", type=float, default=None,
                   help="dock-side floor on a report worth dispatching to "
                        "(config.DISPATCH_MIN_COVERAGE). Set this when feeding the "
                        "dock from a real camera server, which reports every cycle "
                        "including coverage 0.0")
    p.add_argument("--coverage-drop-min", type=float, default=None,
                   help="how far coverage must fall for a service to count as an "
                        "improvement (config.COVERAGE_DROP_MIN). The default is on "
                        "the simulated camera's scale; a real detector's coverage is "
                        "much smaller, so lower it to match")
    args = p.parse_args()

    # Before anything else reads a duration: this rewrites most of config.
    if args.time_scale is not None:
        config.apply_time_scale(args.time_scale)
        if args.duration == p.get_default("duration"):
            args.duration = config.SIM_DURATION_S    # follow the rescaled default

    if args.mqtt:
        config.USE_REAL_MQTT = True
    config.MQTT_HOST = args.host
    config.MQTT_PORT = args.port
    if args.min_coverage is not None:
        config.DISPATCH_MIN_COVERAGE = args.min_coverage
    if args.coverage_drop_min is not None:
        config.COVERAGE_DROP_MIN = args.coverage_drop_min
    if args.mqtt and args.realtime == 0:
        # Over a real broker, messages arrive a network round trip late. With
        # virtual time running flat out that lag can span many virtual seconds.
        print("note: --mqtt with --realtime 0 lets virtual time outrun the "
              "network; using --realtime 0.05 instead")
        args.realtime = 0.05

    run(args.duration, args.realtime, enable_cameras=not args.no_cameras,
        log_path=None if args.log_file in ("", "-") else args.log_file)


if __name__ == "__main__":
    main()
