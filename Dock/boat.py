"""The boat: a state machine with simulated GPS, fill sensor and battery.

States:
    idle -> en_route -> searching -> collecting -> returning -> dumping
         -> charging -> idle          (plus `fault`, reachable from anywhere)

Rules the boat enforces for itself, regardless of what the dock asks:
  * refuse a dispatch below DISPATCH_MIN_BATTERY_PCT
  * abort and come home below BATTERY_ABORT_PCT
  * come home when the bin is full or the mission time limit expires
  * every state has a timeout; exceeding it means `fault`
"""
import math
import random

import config
import geo
from vision import find_trash_direction

MOVING_STATES = ("en_route", "searching", "collecting", "returning")


class Boat:
    def __init__(self, bus, world, log=print):
        self.bus = bus
        self.world = world
        self.log = log
        self.rng = random.Random(config.WORLD_SEED + 1)

        self.dock_pos = (config.DOCK_LAT, config.DOCK_LONG)
        self.pos = self.dock_pos
        self.heading = 0.0
        self.battery = config.BATTERY_START_PCT
        self.fill = 0.0                      # true fill 0..1 (sensor adds noise)

        self.state = "idle"
        self.state_since = 0.0
        self.fault_reason = None

        # mission bookkeeping
        self.mission_camera = None
        self.mission_target = None
        self.mission_start = 0.0
        self.trips = 0
        self.return_reason = None

        # sweep / collection bookkeeping
        self.sweep_points = []
        self.sweep_index = 0
        self.collect_target = None           # a world trash item, or a point
        self.collect_until = None
        self.collected_this_mission = 0

        self._last_status = -1e9
        self.bus.subscribe(config.TOPIC_BOAT_CMD, self._on_command)

    # ================= sensors (what the boat can actually measure) ========
    def gps(self):
        """Reported position = true position + jitter."""
        n = config.GPS_NOISE_M
        return geo.offset(self.pos, self.rng.uniform(-n, n), self.rng.uniform(-n, n))

    def ultrasonic_cm(self):
        """Distance from the sensor down to the top of the rubbish pile."""
        span = config.BIN_EMPTY_DISTANCE_CM - config.BIN_FULL_DISTANCE_CM
        d = config.BIN_EMPTY_DISTANCE_CM - self.fill * span
        return d + self.rng.uniform(-config.ULTRASONIC_NOISE_CM, config.ULTRASONIC_NOISE_CM)

    def fill_level(self):
        """Convert the ultrasonic reading back into a 0..1 fill fraction."""
        span = config.BIN_EMPTY_DISTANCE_CM - config.BIN_FULL_DISTANCE_CM
        frac = (config.BIN_EMPTY_DISTANCE_CM - self.ultrasonic_cm()) / span
        return max(0.0, min(1.0, frac))

    # ================= state machine plumbing ==============================
    def _set_state(self, new_state, now, reason=""):
        if new_state == self.state:
            return
        because = " (%s)" % reason if reason else ""
        self.log("%-10s -> %-10s%s  | batt %.1f%%  fill %.2f"
                 % (self.state, new_state, because, self.battery, self.fill_level()))
        self.state = new_state
        self.state_since = now

    def _fault(self, now, reason):
        self.fault_reason = reason
        self.log("!! FAULT: %s" % reason)
        self._set_state("fault", now, reason)
        self._end_mission(now, "fault")

    def _end_mission(self, now, why):
        if self.mission_camera:
            self.log("mission for %s ended (%s): %d items collected, %d trip(s)"
                     % (self.mission_camera, why, self.collected_this_mission, self.trips + 1))
        self.mission_camera = None
        self.mission_target = None
        self.return_reason = None
        self.collect_target = None
        self.sweep_points, self.sweep_index = [], 0
        self._publish_status(now, force=True)

    # ================= commands from the dock ==============================
    def _on_command(self, _topic, payload):
        if payload.get("action") != "dispatch":
            return
        cam = payload.get("camera")
        now = payload.get("timestamp", self.state_since)

        if self.state not in ("idle", "charging"):
            self._reject(now, cam, "busy in state %s" % self.state)
            return
        if self.battery < config.DISPATCH_MIN_BATTERY_PCT:
            self._reject(now, cam, "battery %.1f%% below dispatch minimum %.1f%%"
                         % (self.battery, config.DISPATCH_MIN_BATTERY_PCT))
            return

        self.mission_camera = cam
        self.mission_target = (payload["lat"], payload["long"])
        self.mission_start = now
        self.trips = 0
        self.collected_this_mission = 0
        self.return_reason = None
        self.log("accepted dispatch to %s at (%.6f, %.6f)"
                 % (cam, self.mission_target[0], self.mission_target[1]))
        self._set_state("en_route", now, "dispatched to %s" % cam)

    def _reject(self, now, cam, why):
        self.log("REFUSED dispatch to %s: %s" % (cam, why))
        self.bus.publish(config.TOPIC_BOAT_STATUS,
                         self._status_payload(now, event="dispatch_refused",
                                              refused_camera=cam, reason=why))

    # ================= main tick ===========================================
    def tick(self, now, dt):
        self._update_battery(dt)

        if self.state != "fault":
            timeout = config.STATE_TIMEOUTS.get(self.state)
            if timeout is not None and (now - self.state_since) > timeout:
                self._fault(now, "state '%s' exceeded timeout of %.0fs" % (self.state, timeout))

        handler = getattr(self, "_do_" + self.state)
        handler(now, dt)
        self._publish_status(now)

    def _update_battery(self, dt):
        if self.state == "charging":
            self.battery = min(100.0, self.battery + config.BATTERY_CHARGE_PCT_S * dt)
        elif self.state in MOVING_STATES:
            self.battery = max(0.0, self.battery - config.BATTERY_DRAIN_MOVING_PCT_S * dt)
        else:
            self.battery = max(0.0, self.battery - config.BATTERY_DRAIN_IDLE_PCT_S * dt)

    def _move_toward(self, target, speed, dt):
        """Advance toward a point, updating heading. True when arrived."""
        d = geo.distance_m(self.pos, target)
        if d <= config.BOAT_ARRIVE_RADIUS_M:
            return True
        self.heading = geo.bearing_rad(self.pos, target)
        self.pos = geo.step_toward(self.pos, target, speed * dt)
        return geo.distance_m(self.pos, target) <= config.BOAT_ARRIVE_RADIUS_M

    def _mission_guards(self, now):
        """Checks that apply in every away-from-dock state. True = heading home."""
        if self.battery <= config.BATTERY_ABORT_PCT:
            self.return_reason = "battery"
            self._set_state("returning", now, "battery %.1f%% <= abort threshold" % self.battery)
            return True
        if now - self.mission_start >= config.MISSION_TIME_LIMIT_S:
            self.return_reason = "time_limit"
            self._set_state("returning", now, "mission time limit reached")
            return True
        if self.fill_level() >= config.FILL_FULL_THRESHOLD:
            self.return_reason = "full"
            self._set_state("returning", now, "bin full")
            return True
        return False

    # ================= per-state behaviour =================================
    def _do_idle(self, now, dt):
        # Hysteresis: charge back up only after a real drop, otherwise the
        # boat would bounce between idle and charging every tick.
        if self.battery < config.BATTERY_RECHARGE_TRIGGER_PCT:
            self._set_state("charging", now, "topping up at dock")

    def _do_en_route(self, now, dt):
        if self._mission_guards(now):
            return
        if self._move_toward(self.mission_target, config.BOAT_SPEED_MPS, dt):
            self._plan_sweep()
            self._set_state("searching", now, "arrived at target area")

    def _do_searching(self, now, dt):
        if self._mission_guards(now):
            return

        # 1. Ask the onboard camera model first.
        frame = self.world.capture_frame(self.pos, self.heading)
        detection = find_trash_direction(frame)
        if detection is not None:
            angle, dist, conf = detection
            self.log("vision detection: angle %.2f rad, %.1f m, conf %.2f"
                     % (angle, dist, conf))
            h = self.heading + angle
            self.collect_target = {"point": geo.offset(self.pos, math.cos(h) * dist,
                                                       math.sin(h) * dist)}
            self.collect_until = None
            self._set_state("collecting", now, "steering to vision detection")
            return

        # 2. No model output -> lawnmower sweep, scooping whatever the boat
        #    comes across along the way.
        item, dist = self.world.nearest_trash(self.pos, config.TRASH_DETECT_RADIUS_M)
        if item is not None:
            self.collect_target = {"item": item}
            self.collect_until = None
            self._set_state("collecting", now, "trash detected %.0f m away" % dist)
            return

        if self.sweep_index >= len(self.sweep_points):
            self.return_reason = "area_swept"
            self._set_state("returning", now, "sweep finished, nothing left to collect")
            return

        wp = self.sweep_points[self.sweep_index]
        if self._move_toward(wp, config.BOAT_SEARCH_SPEED_MPS, dt):
            self.sweep_index += 1

    def _do_collecting(self, now, dt):
        if self._mission_guards(now):
            return
        tgt = self.collect_target
        if tgt is None:
            self._set_state("searching", now, "nothing to collect")
            return

        if "item" in tgt:
            item = tgt["item"]
            if item not in self.world.trash:            # drifted away / already taken
                self.collect_target = None
                self._set_state("searching", now, "target gone")
                return
            point = (item["lat"], item["long"])
        else:
            point = tgt["point"]

        if self.collect_until is None:
            arrived = self._move_toward(point, config.BOAT_SEARCH_SPEED_MPS, dt)
            if arrived or geo.distance_m(self.pos, point) <= config.TRASH_CAPTURE_RADIUS_M:
                self.collect_until = now + config.COLLECT_TIME_S
            return

        if now >= self.collect_until:                   # scoop complete
            if "item" in tgt:
                self.world.remove_trash(tgt["item"])
                self.fill = min(1.0, self.fill + config.FILL_PER_ITEM)
                self.collected_this_mission += 1
                self.log("collected item #%d - fill now %.2f"
                         % (tgt["item"]["id"], self.fill_level()))
            else:
                self.log("reached vision waypoint, nothing scooped")
            self.collect_target, self.collect_until = None, None
            self._set_state("searching", now, "resuming search")

    def _do_returning(self, now, dt):
        if self._move_toward(self.dock_pos, config.BOAT_SPEED_MPS, dt):
            if self.fill > 0.01:
                self._set_state("dumping", now, "at dock, emptying bin")
            else:
                self._finish_at_dock(now)

    def _do_dumping(self, now, dt):
        if (now - self.state_since) < config.DUMP_TIME_S:
            return
        self.fill = 0.0
        self.log("bin emptied")

        time_left = config.MISSION_TIME_LIMIT_S - (now - self.mission_start)
        can_repeat = (self.return_reason == "full"
                      and self.mission_camera is not None
                      and time_left > 0
                      and self.trips + 1 < config.MAX_ROUND_TRIPS
                      and self.battery >= config.DISPATCH_MIN_BATTERY_PCT)
        if can_repeat:
            self.trips += 1
            self.return_reason = None
            self._set_state("en_route", now,
                            "round trip %d/%d back to %s, %.0fs left"
                            % (self.trips + 1, config.MAX_ROUND_TRIPS,
                               self.mission_camera, time_left))
        else:
            self._finish_at_dock(now)

    def _finish_at_dock(self, now):
        self._end_mission(now, self.return_reason or "complete")
        if self.battery < config.BATTERY_CHARGED_PCT:
            self._set_state("charging", now, "recharging")
        else:
            self._set_state("idle", now, "ready")

    def _do_charging(self, now, dt):
        if self.battery >= config.BATTERY_CHARGED_PCT:
            self._set_state("idle", now, "charged to %.1f%%" % self.battery)

    def _do_fault(self, now, dt):
        # A fault stops normal work. If auto-recovery is configured the boat
        # tries to get itself back to the dock; otherwise it waits for a human.
        if config.FAULT_AUTO_RECOVER_S is None:
            return
        if (now - self.state_since) < config.FAULT_AUTO_RECOVER_S:
            return
        self.log("attempting recovery from fault: %s" % self.fault_reason)
        self.fault_reason = None
        self.return_reason = "fault_recovery"
        if geo.distance_m(self.pos, self.dock_pos) <= config.BOAT_ARRIVE_RADIUS_M:
            self._finish_at_dock(now)
        else:
            self._set_state("returning", now, "recovering to dock")

    # ================= sweep planning ======================================
    def _plan_sweep(self):
        """Serpentine ('lawnmower') waypoints over a square box centred on the
        mission target - the fallback when the camera model gives no direction."""
        half = config.SWEEP_AREA_M / 2.0
        spacing = config.SWEEP_LANE_SPACING_M
        lanes = int(config.SWEEP_AREA_M // spacing) + 1
        pts = []
        for i in range(lanes):
            x = -half + i * spacing
            ys = [-half, half] if i % 2 == 0 else [half, -half]
            for y in ys:
                pts.append(geo.offset(self.mission_target, x, y))
        self.sweep_points, self.sweep_index = pts, 0
        path_m = lanes * config.SWEEP_AREA_M + (lanes - 1) * spacing
        est_s = path_m / config.BOAT_SEARCH_SPEED_MPS
        self.log("no vision model - planned lawnmower sweep: %d waypoints over a "
                 "%.0fm box (~%.0fm, ~%.0fs at search speed)"
                 % (len(pts), config.SWEEP_AREA_M, path_m, est_s))
        limit = config.STATE_TIMEOUTS.get("searching")
        if limit is not None and est_s > limit:
            self.log("WARNING: sweep needs ~%.0fs but the 'searching' timeout is "
                     "%.0fs - the boat will fault mid-sweep. Raise it in config."
                     % (est_s, limit))

    # ================= telemetry ===========================================
    def _status_payload(self, now, **extra):
        lat, lon = self.gps()
        p = {
            "state": self.state,
            "lat": round(lat, 6),
            "long": round(lon, 6),
            "fill": round(self.fill_level(), 3),
            "battery": round(self.battery, 1),
            "mission": self.mission_camera,
            "trip": self.trips + 1 if self.mission_camera else 0,
            "timestamp": round(now, 2),
        }
        if self.fault_reason:
            p["fault_reason"] = self.fault_reason
        p.update(extra)
        return p

    def _publish_status(self, now, force=False):
        if not force and (now - self._last_status) < config.BOAT_STATUS_INTERVAL_S:
            return
        self._last_status = now
        self.bus.publish(config.TOPIC_BOAT_STATUS, self._status_payload(now))
