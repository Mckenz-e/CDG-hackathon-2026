"""Scenario tests.

A normal run only exercises the happy path. These tests force the awkward
cases: a flat battery, a stuck camera, a state that hangs. Plain asserts, no
test framework needed:

    python tests.py
"""
import config
from boat import Boat
from bus import InProcessBus
from camera_server import CameraServer
from dock import DockStation
from log import RunLog
from world import World


class Harness:
    """Builds a complete miniature system with temporary config overrides."""

    def __init__(self, quiet=True, **overrides):
        self.saved = {k: getattr(config, k) for k in overrides}
        for k, v in overrides.items():
            setattr(config, k, v)

        self.lines = []
        self.clock = _Clock()
        # Same RunLog the real run uses, so a test exercises the logging path
        # too - but to the console only, and only when asked.
        self.logs = RunLog(self.clock, path=None, echo=not quiet)

        def component(name):
            tagged = self.logs.component(name)

            def log(msg):
                self.lines.append(msg)     # bare message, for logged()
                tagged(msg)
            return log

        self.log = component("test")
        self.bus = InProcessBus(log=component("bus"))
        self.world = World(log=component("world"))
        self.cameras = CameraServer(self.bus, self.world, self.clock,
                                    log=component("camera_server"))
        self.dock = DockStation(self.bus, self.clock, log=component("dock"))
        self.boat = Boat(self.bus, self.world, log=component("boat"))

    def run(self, seconds):
        dt = config.TICK_S
        end = self.clock.now() + seconds
        while self.clock.now() < end:
            now = self.clock.now()
            self.world.tick(now)
            self.cameras.tick(now)
            self.dock.tick(now)
            self.boat.tick(now, dt)
            self.bus.poll()
            self.clock.advance(dt)

    def logged(self, needle):
        return any(needle in line for line in self.lines)

    def close(self):
        self.logs.close()
        for k, v in self.saved.items():
            setattr(config, k, v)


class _Clock:
    # Same whole-tick arithmetic as sim.Clock - see the note there.
    def __init__(self):
        self.t = 0.0
        self._ticks = 0

    def now(self):
        return self.t

    def advance(self, dt):
        self._ticks += 1
        self.t = self._ticks * dt


# ============================ the tests ====================================

def test_refuses_dispatch_when_battery_low():
    h = Harness()
    try:
        h.boat.battery = config.DISPATCH_MIN_BATTERY_PCT - 5.0
        h.bus.publish(config.TOPIC_BOAT_CMD, {
            "action": "dispatch", "camera": "cam_1",
            "lat": config.CAMERAS["cam_1"]["lat"],
            "long": config.CAMERAS["cam_1"]["long"], "timestamp": 0.0})
        h.bus.poll()
        assert h.boat.state == "idle", h.boat.state
        assert h.boat.mission_camera is None
        assert h.logged("REFUSED dispatch"), "boat should log a refusal"
    finally:
        h.close()


def test_dock_releases_camera_after_refusal():
    h = Harness()
    try:
        h.boat.battery = 10.0
        h.dock.boat["battery"] = 100.0        # dock's stale view says it is fine
        h.dock.queue = [{"camera": "cam_1", "coverage": 0.9,
                         "lat": config.CAMERAS["cam_1"]["lat"],
                         "long": config.CAMERAS["cam_1"]["long"], "timestamp": 0.0}]
        h.dock._maybe_dispatch(0.0)
        h.bus.poll()                          # deliver dispatch, then the refusal
        h.bus.poll()
        assert h.dock.active_camera is None, "camera must be released after a refusal"
        assert h.logged("boat refused")
    finally:
        h.close()


def test_reports_during_servicing_are_ignored():
    """A camera server that keeps reporting while the boat is in its frame must
    not be able to steer the dock.

    This is the failure the servicing flag exists to prevent, tested from the
    dock's side: the dock must be correct even when the camera server ignores
    the flag. The mid-mission report is of water with a boat parked in it, and
    without this the dock takes the first one as the post-service evaluation -
    before the boat has so much as arrived.
    """
    h = Harness()
    try:
        cam = config.CAMERAS["cam_1"]
        h.dock.queue = [{"camera": "cam_1", "coverage": 0.80, "lat": cam["lat"],
                         "long": cam["long"], "timestamp": 0.0}]
        h.dock._maybe_dispatch(0.0)
        h.bus.poll()
        assert h.dock.active_camera == "cam_1"
        rec = h.dock.records["cam_1"]
        assert rec.awaiting_evaluation and rec.baseline_coverage == 0.80

        # A non-compliant camera server reports anyway, mid-mission.
        h.bus.publish(config.TOPIC_CAMERA_REPORT, {
            "camera": "cam_1", "coverage": 0.05, "lat": cam["lat"],
            "long": cam["long"], "timestamp": 1.0})
        h.bus.poll()

        assert rec.awaiting_evaluation, "service must not be evaluated mid-mission"
        assert not h.logged("improved after service")
        assert not any(r["camera"] == "cam_1" for r in h.dock.queue),             "a boat-in-frame reading must not sit in the dispatch queue"
        assert h.logged("not honouring the servicing flag")
    finally:
        h.close()


def test_aborts_mission_when_battery_drops():
    # Drain fast enough that the boat cannot finish the trip.
    h = Harness(BATTERY_DRAIN_MOVING_PCT_S=0.8, BATTERY_START_PCT=60.0)
    try:
        h.boat.battery = 60.0
        h.run(200)
        assert h.logged("abort threshold"), "boat should abort on low battery"
        assert h.boat.state in ("returning", "charging", "idle", "dumping"), h.boat.state
    finally:
        h.close()


def test_state_timeout_causes_fault():
    timeouts = dict(config.STATE_TIMEOUTS)
    timeouts["en_route"] = 5.0
    h = Harness(STATE_TIMEOUTS=timeouts, FAULT_AUTO_RECOVER_S=None)
    try:
        h.run(120)
        assert h.logged("FAULT"), "a hung state must fault"
        assert h.boat.state == "fault", h.boat.state
        assert "en_route" in (h.boat.fault_reason or ""), h.boat.fault_reason
    finally:
        h.close()


def test_fault_recovery_returns_to_dock():
    timeouts = dict(config.STATE_TIMEOUTS)
    timeouts["en_route"] = 5.0
    h = Harness(STATE_TIMEOUTS=timeouts, FAULT_AUTO_RECOVER_S=10.0)
    try:
        h.run(400)
        assert h.logged("attempting recovery from fault")
        assert h.boat.state != "fault", "boat should not be stuck in fault"
    finally:
        h.close()


def test_camera_server_suppresses_servicing_camera():
    h = Harness()
    try:
        h.bus.publish(config.TOPIC_DOCK_SERVICING, {"camera": "cam_1", "status": "servicing"})
        h.bus.poll()
        assert "cam_1" in h.cameras.servicing
        before = len(h.dock.queue)
        h.run(config.CAMERA_REPORT_INTERVAL_S * 3)
        assert not any(r["camera"] == "cam_1" for r in h.dock.queue), \
            "no cam_1 reports may be queued while it is servicing"

        h.bus.publish(config.TOPIC_DOCK_SERVICING, {"camera": "cam_1", "status": "clear"})
        h.bus.poll()
        assert "cam_1" not in h.cameras.servicing
        _ = before
    finally:
        h.close()


def test_reports_expire():
    h = Harness(REPORT_EXPIRY_S=30.0)
    try:
        h.dock.queue.append({"camera": "cam_1", "coverage": 0.9, "lat": 0.0,
                             "long": 0.0, "timestamp": 0.0})
        h.dock._expire_reports(100.0)
        assert h.dock.queue == [], "stale reports must be discarded"
    finally:
        h.close()


def test_highest_coverage_wins():
    h = Harness()
    try:
        h.dock.queue = [
            {"camera": "cam_1", "coverage": 0.30, "lat": 1.0, "long": 1.0, "timestamp": 0.0},
            {"camera": "cam_2", "coverage": 0.90, "lat": 2.0, "long": 2.0, "timestamp": 0.0},
            {"camera": "cam_3", "coverage": 0.55, "lat": 3.0, "long": 3.0, "timestamp": 0.0},
        ]
        assert h.dock._best_report(0.0)["camera"] == "cam_2"
    finally:
        h.close()


def test_cooldown_blocks_immediate_redispatch():
    h = Harness()
    try:
        rec = h.dock.records["cam_1"]
        rec.last_serviced_at = 100.0
        h.dock.queue = [{"camera": "cam_1", "coverage": 0.99, "lat": 1.0,
                         "long": 1.0, "timestamp": 100.0}]
        assert h.dock._best_report(100.0 + config.CAMERA_COOLDOWN_S / 2) is None
        assert h.dock._best_report(100.0 + config.CAMERA_COOLDOWN_S + 1) is not None
    finally:
        h.close()


def test_phantom_camera_flagged_after_retry_limit():
    # Only the phantom camera exists, so it gets every dispatch. Its coverage
    # never falls, so the dock must eventually give up on it.
    only_phantom = {"cam_3": dict(config.CAMERAS["cam_3"])}
    h = Harness(CAMERAS=only_phantom, CAMERA_COOLDOWN_S=20.0,
                MISSION_TIME_LIMIT_S=120.0, RETRY_LIMIT=3)
    try:
        h.run(4000)
        rec = h.dock.records["cam_3"]
        assert rec.needs_attention, "phantom camera should be flagged"
        assert h.logged("NEEDS ATTENTION")
    finally:
        h.close()


def test_round_trip_when_full_and_time_remains():
    # A tiny bin fills after two items, leaving plenty of mission time.
    h = Harness(FILL_PER_ITEM=0.5, MISSION_TIME_LIMIT_S=2000.0,
                MAX_ROUND_TRIPS=3, TRASH_INITIAL_PER_CAMERA=12)
    try:
        h.run(2200)
        assert h.logged("bin full"), "boat should fill up"
        assert h.logged("round trip"), "boat should go back out after dumping"
    finally:
        h.close()


def test_fill_sensor_tracks_true_fill():
    h = Harness()
    try:
        h.boat.fill = 0.0
        assert h.boat.fill_level() < 0.1
        h.boat.fill = 1.0
        assert h.boat.fill_level() > 0.9
        h.boat.fill = 0.5
        assert 0.4 < h.boat.fill_level() < 0.6
    finally:
        h.close()


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for t in tests:
        try:
            t()
            print("PASS  %s" % t.__name__)
        except AssertionError as e:
            failures += 1
            print("FAIL  %s: %s" % (t.__name__, e))
        except Exception as e:            # noqa: BLE001 - report and continue
            failures += 1
            print("ERROR %s: %s: %s" % (t.__name__, type(e).__name__, e))
    print("\n%d passed, %d failed" % (len(tests) - failures, failures))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
