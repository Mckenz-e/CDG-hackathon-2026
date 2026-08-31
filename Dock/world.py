"""The physical world the boat cannot see directly.

Nothing in boat.py or dock.py reads this module's internals except through
"sensors" - that keeps the honest separation between what is really out there
and what the system believes.
"""
import math
import random

import config
import geo


class World:
    def __init__(self, log=print):
        self.rng = random.Random(config.WORLD_SEED)
        self.log = log
        self.trash = []                 # [{id, lat, long, camera}]
        self._next_id = 1
        self._last_spawn = 0.0
        for cam_id in config.CAMERAS:
            for _ in range(config.TRASH_INITIAL_PER_CAMERA):
                self._spawn_near(cam_id)

    # --- world evolution ---------------------------------------------------
    def _spawn_near(self, cam_id):
        cam = config.CAMERAS[cam_id]
        s = config.TRASH_SPAWN_SPREAD_M
        pos = geo.offset((cam["lat"], cam["long"]),
                         self.rng.uniform(-s, s), self.rng.uniform(-s, s))
        item = {"id": self._next_id, "lat": pos[0], "long": pos[1], "camera": cam_id}
        self._next_id += 1
        self.trash.append(item)
        return item

    def tick(self, now):
        """New rubbish drifts in over time, so coverage is not static."""
        if now - self._last_spawn >= config.TRASH_SPAWN_INTERVAL_S:
            self._last_spawn = now
            cam_id = self.rng.choice(list(config.CAMERAS))
            self._spawn_near(cam_id)

    # --- what a fixed camera sees -----------------------------------------
    def coverage(self, cam_id):
        """Fraction of the camera's view occupied by trash, 0..1."""
        if config.CAMERAS[cam_id].get("phantom"):
            # A false positive: a reflection or a stuck log. Never improves,
            # no matter how many times the boat visits.
            return config.PHANTOM_COVERAGE
        cam = (config.CAMERAS[cam_id]["lat"], config.CAMERAS[cam_id]["long"])
        n = sum(1 for t in self.trash
                if geo.distance_m(cam, (t["lat"], t["long"])) <= config.CAMERA_VIEW_RADIUS_M)
        raw = n / float(config.CAMERA_FULL_COVERAGE_ITEMS)
        raw += self.rng.uniform(-config.CAMERA_NOISE, config.CAMERA_NOISE)
        return max(0.0, min(1.0, raw))

    # --- what the boat's own sensors see ----------------------------------
    def nearest_trash(self, pos, radius_m):
        best, best_d = None, radius_m
        for t in self.trash:
            d = geo.distance_m(pos, (t["lat"], t["long"]))
            if d <= best_d:
                best, best_d = t, d
        return best, best_d

    def remove_trash(self, item):
        if item in self.trash:
            self.trash.remove(item)

    def capture_frame(self, pos, heading_rad):
        """Stand-in for a camera frame. The real system would hand a numpy
        image to the model; the stub ignores it entirely."""
        return {"pos": pos, "heading": heading_rad, "shape": (480, 640, 3)}

    def count_near(self, cam_id):
        cam = (config.CAMERAS[cam_id]["lat"], config.CAMERAS[cam_id]["long"])
        return sum(1 for t in self.trash
                   if geo.distance_m(cam, (t["lat"], t["long"])) <= config.CAMERA_VIEW_RADIUS_M)
