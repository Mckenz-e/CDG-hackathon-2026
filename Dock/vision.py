"""Placeholder for the onboard camera model.

The real system will run a small detector on the boat's own camera frame.
It is not trained yet, so this stub returns None and the boat falls back to a
lawnmower sweep. When the model exists, replace the body below - the contract
is all the boat code depends on.
"""
import config


def find_trash_direction(frame):
    """Look at one camera frame and say where the nearest trash is.

    Returns:
        (steering_angle_rad, distance_m, confidence)  where steering_angle is
        relative to the boat's current heading (+ve = turn left), or
        None if nothing is detected / the model is unavailable.
    """
    if not config.VISION_MODEL_ENABLED:
        return None
    # --- real inference would go here ---
    return None
