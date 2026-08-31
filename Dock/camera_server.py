"""The shore-side camera server.

Job: every CAMERA_REPORT_INTERVAL_S, estimate coverage for each camera and
publish a report - EXCEPT for any camera the dock has marked "servicing".
While the boat is in frame it looks exactly like a big piece of trash, so
reporting during servicing would generate a false alarm and could make the
dock dispatch the boat to the boat.
"""
import config


class CameraServer:
    def __init__(self, bus, world, clock, log=print):
        self.bus = bus
        self.world = world
        self.clock = clock
        self.log = log
        self.servicing = set()      # cameras currently suppressed
        self._last_report = {c: -1e9 for c in config.CAMERAS}
        self.bus.subscribe(config.TOPIC_DOCK_SERVICING, self._on_servicing)

    def _on_servicing(self, _topic, payload):
        cam, status = payload.get("camera"), payload.get("status")
        if cam not in config.CAMERAS:
            return
        if status == "servicing":
            self.servicing.add(cam)
            self.log("%s -> SERVICING: suppressing its detections "
                     "(boat will be in frame)" % cam)
        elif status == "clear":
            self.servicing.discard(cam)
            self.log("%s -> CLEAR: detections re-enabled" % cam)

    def tick(self, now):
        for cam_id, cam in config.CAMERAS.items():
            if now - self._last_report[cam_id] < config.CAMERA_REPORT_INTERVAL_S:
                continue
            self._last_report[cam_id] = now

            if cam_id in self.servicing:
                continue                      # suppressed on purpose

            coverage = self.world.coverage(cam_id)
            if coverage < config.CAMERA_REPORT_THRESHOLD:
                continue                      # nothing worth reporting

            self.bus.publish(config.TOPIC_CAMERA_REPORT, {
                "camera": cam_id,
                "coverage": round(coverage, 3),
                "lat": cam["lat"],
                "long": cam["long"],
                "timestamp": round(now, 2),
            })
