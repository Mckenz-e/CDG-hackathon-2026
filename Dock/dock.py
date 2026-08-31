"""The dock station: the brain that decides where the boat goes.

Responsibilities:
  * keep a queue of camera reports, dropping anything older than REPORT_EXPIRY_S
  * when the boat is idle and charged, dispatch it to the highest-coverage
    pending report
  * mark that camera "servicing" so the camera server ignores its own
    detections while the boat is in frame, then "clear" when the boat is home
  * enforce a per-camera cooldown after servicing
  * count retries: if a camera has been serviced RETRY_LIMIT times without its
    coverage dropping by COVERAGE_DROP_MIN, stop dispatching and flag it
"""
import config


class CameraRecord:
    """Everything the dock remembers about one camera."""

    def __init__(self, cam_id):
        self.cam_id = cam_id
        self.last_serviced_at = None
        self.service_count = 0          # consecutive services with no improvement
        self.baseline_coverage = None   # coverage at the moment of dispatch
        self.awaiting_evaluation = False
        self.needs_attention = False

    def in_cooldown(self, now):
        return (self.last_serviced_at is not None
                and now - self.last_serviced_at < config.CAMERA_COOLDOWN_S)


class DockStation:
    def __init__(self, bus, clock, log=print):
        self.bus = bus
        self.clock = clock
        self.log = log

        self.queue = []                                     # list of report dicts
        self.records = {c: CameraRecord(c) for c in config.CAMERAS}
        self.active_camera = None                           # currently being serviced
        self.dispatched_at = None
        self.mission_confirmed = False                      # boat has acknowledged
        self._suppressed = 0        # reports arriving for the camera being serviced
        self.boat = {"state": "idle", "battery": config.BATTERY_START_PCT, "mission": None}
        self._last_decision = -1e9

        self.bus.subscribe(config.TOPIC_CAMERA_REPORT, self._on_report)
        self.bus.subscribe(config.TOPIC_BOAT_STATUS, self._on_boat_status)

    # ================= inbound messages ====================================
    def _sanitise(self, report):
        """Turn one wire payload into a report the dock can act on, or None.

        The dock does not control who publishes to TOPIC_CAMERA_REPORT. A real
        camera server is a separate process on a separate machine, so every
        field here is untrusted input, not an internal call.

        Two things are normalised rather than trusted:

        * `timestamp` is the *publisher's* clock. A real camera server stamps
          wall-clock UTC (an ISO-8601 string); the sim's own camera server
          stamps virtual seconds since zero. Neither is comparable with this
          dock's clock, and subtracting an ISO string from a float is a crash.
          So the publisher's value is kept as `reported_at` for the log only,
          and `timestamp` is overwritten with the arrival time on the dock's own
          clock - which is what REPORT_EXPIRY_S actually wants to measure.
        * `lat`/`long` must be real numbers. They become the boat's waypoint, and
          `coverage_monitor.py` leaves them None unless --lat/--long are given.
          A report we cannot navigate to is not a report.
        """
        cam_id = report.get("camera")
        if cam_id not in self.records:
            self.log("ignoring report from unknown camera %r" % cam_id)
            return None

        coverage = report.get("coverage")
        if not isinstance(coverage, (int, float)) or isinstance(coverage, bool):
            self.log("ignoring report from %s: coverage %r is not a number"
                     % (cam_id, coverage))
            return None

        lat, lon = report.get("lat"), report.get("long")
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                   for v in (lat, lon)):
            self.log("ignoring report from %s: no usable position "
                     "(lat=%r long=%r) - nowhere to send the boat" % (cam_id, lat, lon))
            return None

        clean = dict(report)
        clean["coverage"] = float(coverage)
        clean["lat"], clean["long"] = float(lat), float(lon)
        clean["reported_at"] = report.get("timestamp")
        clean["timestamp"] = self.clock.now()
        return clean

    def _on_report(self, _topic, report):
        report = self._sanitise(report)
        if report is None:
            return
        cam_id = report["camera"]
        rec = self.records[cam_id]

        now = self.clock.now()

        if cam_id == self.active_camera:
            # We told this camera to stop reporting: the boat is in its frame
            # and reads as a large piece of trash. A report arriving anyway is
            # either still in flight from before the flag, or from a camera
            # server that does not honour it - and either way it describes water
            # with a boat parked in it.
            #
            # Dropping it here is what stops the dock's own logic from depending
            # on a remote process behaving. Without this, the very first report
            # of the mission is taken as the post-service evaluation - the boat
            # has not even arrived yet, so "did coverage drop?" is being asked of
            # the wrong water, and the answer decides whether the camera is
            # eventually flagged NEEDS ATTENTION.
            self._suppressed += 1
            if self._suppressed == 1:
                self.log("ignoring reports from %s while it is being "
                         "serviced - its camera server is not honouring the "
                         "servicing flag" % cam_id)
            return

        # Did the last service actually help? Judge on the first report that
        # arrives after the mission ended and the cooldown ran out.
        if rec.awaiting_evaluation and not rec.in_cooldown(now):
            self._evaluate_service(rec, report["coverage"])

        if rec.needs_attention:
            return                                          # stop queuing it entirely

        # Only the newest report per camera matters - an older one describes
        # water that has since changed. Keep the queue one-deep per camera.
        self.queue = [r for r in self.queue if r["camera"] != cam_id]
        self.queue.append(report)
        self.log("report queued: %s coverage %.2f (queue depth %d)"
                 % (cam_id, report["coverage"], len(self.queue)))

    def _evaluate_service(self, rec, coverage):
        rec.awaiting_evaluation = False
        dropped = rec.baseline_coverage - coverage
        if dropped >= config.COVERAGE_DROP_MIN:
            self.log("%s improved after service (%.2f -> %.2f), retry count reset"
                     % (rec.cam_id, rec.baseline_coverage, coverage))
            rec.service_count = 0
            return

        rec.service_count += 1
        self.log("%s did NOT improve after service (%.2f -> %.2f) - "
                 "attempt %d of %d"
                 % (rec.cam_id, rec.baseline_coverage, coverage,
                    rec.service_count, config.RETRY_LIMIT))
        if rec.service_count >= config.RETRY_LIMIT:
            rec.needs_attention = True
            self.log("** %s NEEDS ATTENTION: serviced %d times with no coverage "
                     "drop - no longer dispatching (possible false positive, stuck "
                     "debris, or obstruction)" % (rec.cam_id, rec.service_count))

    def _on_boat_status(self, _topic, status):
        self.boat["state"] = status.get("state", self.boat["state"])
        self.boat["battery"] = status.get("battery", self.boat["battery"])
        self.boat["mission"] = status.get("mission")

        if status.get("event") == "dispatch_refused":
            cam = status.get("refused_camera")
            self.log("boat refused %s (%s) - releasing servicing flag, will retry later"
                     % (cam, status.get("reason")))
            if self.active_camera == cam:
                rec = self.records[cam]
                rec.awaiting_evaluation = False
                rec.last_serviced_at = None     # a refusal is not a service attempt
                self._mark_clear(cam)
            return

        if self.active_camera is None:
            return

        # Wait for the boat to acknowledge before believing anything about the
        # mission. Status messages the boat published *before* it received the
        # dispatch are still in flight, and would otherwise read as "already
        # finished" - clearing the camera the instant we dispatched to it.
        if status.get("mission") == self.active_camera:
            if not self.mission_confirmed:
                self.log("boat acknowledged mission for %s" % self.active_camera)
            self.mission_confirmed = True
            return

        # The boat clears its `mission` field the moment a mission ends, for any
        # reason: finished, aborted on battery, timed out, or faulted.
        if self.mission_confirmed and status.get("mission") is None \
                and self.boat["state"] in ("idle", "charging", "dumping", "fault"):
            self._mark_clear(self.active_camera)

    # ================= servicing flags =====================================
    def _mark_servicing(self, cam_id):
        self.active_camera = cam_id
        self.dispatched_at = self.clock.now()
        self.mission_confirmed = False
        self._suppressed = 0
        self.bus.publish(config.TOPIC_DOCK_SERVICING, {"camera": cam_id, "status": "servicing"})
        self.log("%s marked SERVICING" % cam_id)

    def _mark_clear(self, cam_id):
        self.active_camera = None
        self.dispatched_at = None
        self.mission_confirmed = False
        self.bus.publish(config.TOPIC_DOCK_SERVICING, {"camera": cam_id, "status": "clear"})
        self.records[cam_id].last_serviced_at = self.clock.now()
        self.log("%s marked CLEAR (cooldown %.0fs starts now)%s"
                 % (cam_id, config.CAMERA_COOLDOWN_S,
                    "" if not self._suppressed else
                    " - ignored %d report(s) from it during the mission" % self._suppressed))
        self._suppressed = 0

    # ================= main tick ===========================================
    def tick(self, now):
        self._expire_reports(now)
        self._check_dispatch_ack(now)
        if now - self._last_decision < config.DOCK_DECISION_INTERVAL_S:
            return
        self._last_decision = now
        self._maybe_dispatch(now)

    def _check_dispatch_ack(self, now):
        """If the boat never answered, release the camera rather than stalling
        the whole system on a lost message."""
        if (self.active_camera is None or self.mission_confirmed
                or self.dispatched_at is None):
            return
        if now - self.dispatched_at < config.DISPATCH_ACK_TIMEOUT_S:
            return
        cam = self.active_camera
        self.log("no acknowledgement from boat for %s within %.0fs - "
                 "releasing and re-queueing" % (cam, config.DISPATCH_ACK_TIMEOUT_S))
        rec = self.records[cam]
        rec.awaiting_evaluation = False
        self._mark_clear(cam)
        rec.last_serviced_at = None          # never actually serviced

    def _expire_reports(self, now):
        fresh = [r for r in self.queue if now - r["timestamp"] <= config.REPORT_EXPIRY_S]
        dropped = len(self.queue) - len(fresh)
        if dropped:
            self.log("discarded %d report(s) older than %.0fs"
                     % (dropped, config.REPORT_EXPIRY_S))
        self.queue = fresh

    def _maybe_dispatch(self, now):
        if self.active_camera is not None:
            return                                  # one mission at a time
        if self.boat["state"] != "idle":
            return
        if self.boat["battery"] < config.DISPATCH_MIN_BATTERY_PCT:
            return

        best = self._best_report(now)
        if best is None:
            return

        cam_id = best["camera"]
        rec = self.records[cam_id]
        rec.baseline_coverage = best["coverage"]
        rec.awaiting_evaluation = True

        self.log("dispatching to %s (coverage %.2f, boat battery %.1f%%)"
                 % (cam_id, best["coverage"], self.boat["battery"]))
        self._mark_servicing(cam_id)
        self.bus.publish(config.TOPIC_BOAT_CMD, {
            "action": "dispatch",
            "camera": cam_id,
            "lat": best["lat"],
            "long": best["long"],
            "timestamp": now,
        })
        # Everything queued for this camera is now stale by definition.
        self.queue = [r for r in self.queue if r["camera"] != cam_id]

    def _best_report(self, now):
        """Highest-coverage report that is fresh, above DISPATCH_MIN_COVERAGE,
        not blacklisted and not cooling down."""
        candidates = []
        for r in self.queue:
            rec = self.records[r["camera"]]
            if rec.needs_attention or rec.in_cooldown(now):
                continue
            if r["coverage"] < config.DISPATCH_MIN_COVERAGE:
                continue        # reported, but not dirty enough to be worth a trip
            candidates.append(r)
        if not candidates:
            return None
        return max(candidates, key=lambda r: r["coverage"])

    # ================= reporting ===========================================
    def summary(self):
        lines = ["--- dock summary ---"]
        for cam_id, rec in self.records.items():
            flag = "NEEDS ATTENTION" if rec.needs_attention else "ok"
            last = "never" if rec.last_serviced_at is None else "%.0fs" % rec.last_serviced_at
            lines.append("  %s: %s | last serviced %s | unimproved services %d"
                         % (cam_id, flag, last, rec.service_count))
        lines.append("  queue depth: %d" % len(self.queue))
        return "\n".join(lines)
