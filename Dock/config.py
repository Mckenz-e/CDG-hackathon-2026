"""
=============================================================================
 CONFIG - the single place to tune the whole simulation.
 Every timing, speed, threshold, battery rate and camera coordinate lives
 here. No other file should contain a magic number.
=============================================================================
"""

# --- Simulation clock -------------------------------------------------------
# The sim runs on a *virtual* clock. One "tick" advances virtual time by
# TICK_S seconds. REAL_TIME_FACTOR controls how fast that maps to wall-clock:
#   0.0  = as fast as the CPU allows (a 30-minute mission finishes in seconds)
#   1.0  = real time (1 sim-second takes 1 real second)
TICK_S = 0.5
REAL_TIME_FACTOR = 0.0
SIM_DURATION_S = 3600.0          # stop the sim after this much virtual time

# TIME_SCALE compresses the *virtual* clock. Every duration below is divided by
# it and every per-second rate is multiplied by it, which is a change of units
# on the time axis, not a change of scenario: the boat still covers the same
# water on the same battery and the same number of ticks still elapse. Only the
# number of virtual seconds it takes goes down. See apply_time_scale().
#
# This is NOT REAL_TIME_FACTOR. That one maps virtual seconds onto wall-clock
# seconds and changes nothing about the run. Use it when you just want to watch
# faster. TIME_SCALE is for when something OUTSIDE the sim sets the pace and you
# cannot simply spin the virtual clock faster - a real camera server publishing
# every 5 real seconds, say. There, shortening the sim's own durations is the
# only way to fit a cooldown, a retry and a NEEDS ATTENTION verdict into a test
# that finishes while you are still watching.
TIME_SCALE = 1.0

# --- Messaging (MQTT) -------------------------------------------------------
# USE_REAL_MQTT=False uses a built-in in-process broker so the sim runs with
# zero dependencies. Flip to True (and `pip install paho-mqtt`) to talk to a
# real broker - the rest of the code is identical either way.
USE_REAL_MQTT = False
MQTT_HOST = "localhost"
MQTT_PORT = 1883
MQTT_CLIENT_PREFIX = "trashsim"

TOPIC_CAMERA_REPORT = "trash/camera/report"    # camera server -> dock
TOPIC_DOCK_SERVICING = "trash/dock/servicing"  # dock -> camera server
TOPIC_BOAT_STATUS = "trash/boat/status"        # boat -> everyone
TOPIC_BOAT_CMD = "trash/boat/cmd"              # dock -> boat

# --- Geography --------------------------------------------------------------
# The dock is the origin of the local metre grid.
DOCK_LAT = 13.756300
DOCK_LONG = 100.501800

# Fixed camera installations watching the water.
# "phantom": True means this camera's coverage never drops even after the boat
# services it (e.g. it is really looking at a shadow or a stuck log). This is
# what exercises the dock's retry limit / "needs attention" path.
CAMERAS = {
    "cam_1": {"lat": 13.757200, "long": 100.502600, "phantom": False},
    "cam_2": {"lat": 13.755500, "long": 100.503400, "phantom": False},
    "cam_3": {"lat": 13.756900, "long": 100.500700, "phantom": True},
}
CAMERA_VIEW_RADIUS_M = 60.0      # how much water one camera sees

# --- Camera server ----------------------------------------------------------
CAMERA_REPORT_INTERVAL_S = 20.0  # how often each camera publishes
CAMERA_REPORT_THRESHOLD = 0.15   # below this coverage, stay quiet
CAMERA_FULL_COVERAGE_ITEMS = 8   # this many items in view == coverage 1.0
CAMERA_NOISE = 0.02              # +/- random jitter on the coverage estimate
PHANTOM_COVERAGE = 0.55          # what a phantom camera always reports

# --- World / trash field ----------------------------------------------------
TRASH_SPAWN_INTERVAL_S = 150.0   # a new item drifts in this often
TRASH_SPAWN_SPREAD_M = 45.0      # scattered this far around a camera
TRASH_INITIAL_PER_CAMERA = 5
WORLD_SEED = 7                   # fixed seed => reproducible runs

# --- Dock station -----------------------------------------------------------
REPORT_EXPIRY_S = 180.0          # queued reports older than this are dropped
DISPATCH_MIN_COVERAGE = 0.0      # dock-side floor on a report worth a mission.
                                 # The built-in camera server already filters at
                                 # CAMERA_REPORT_THRESHOLD, so 0.0 changes nothing
                                 # for `python sim.py`. A REAL camera server
                                 # (main-cam/coverage_monitor.py) publishes every
                                 # cycle including coverage 0.0 on purpose - silence
                                 # has to mean "the process died", not "clean water"
                                 # - so when the dock is fed from one, it needs a
                                 # threshold of its own. See sim.py --min-coverage.
DISPATCH_MIN_BATTERY_PCT = 45.0  # won't dispatch below this
CAMERA_COOLDOWN_S = 120.0        # don't re-service the same camera this soon
RETRY_LIMIT = 3                  # services without improvement before giving up
COVERAGE_DROP_MIN = 0.10         # coverage must fall by this much to "count".
                                 # NOTE this is on the *simulated* camera's scale,
                                 # where coverage 1.0 == CAMERA_FULL_COVERAGE_ITEMS
                                 # items in view. A real detector's coverage is a
                                 # union-of-boxes area fraction and lives on a much
                                 # smaller scale - measured 0.006-0.217 on the data2
                                 # pond - so 0.10 there means "halve the worst frame
                                 # ever seen". Retune with sim.py --coverage-drop-min
                                 # when the reports come from a real camera server.
DOCK_DECISION_INTERVAL_S = 5.0   # how often the dock re-evaluates its queue
DISPATCH_ACK_TIMEOUT_S = 30.0    # give up waiting for the boat to acknowledge

# --- Boat: movement & power -------------------------------------------------
BOAT_SPEED_MPS = 1.4             # cruise speed
BOAT_SEARCH_SPEED_MPS = 0.8      # slower while sweeping
BOAT_ARRIVE_RADIUS_M = 3.0       # "close enough" to a waypoint
GPS_NOISE_M = 0.4                # simulated GPS jitter

BATTERY_START_PCT = 100.0
BATTERY_DRAIN_MOVING_PCT_S = 0.045   # while under way
BATTERY_DRAIN_IDLE_PCT_S = 0.004     # electronics only
BATTERY_CHARGE_PCT_S = 0.35          # on the dock charger
BATTERY_ABORT_PCT = 25.0             # abort mission and come home
BATTERY_CHARGED_PCT = 95.0           # charging is "done" here
BATTERY_RECHARGE_TRIGGER_PCT = 88.0  # ...and only restarts below here
                                     # (the gap stops idle<->charging flapping)

# --- Boat: bin & ultrasonic fill sensor -------------------------------------
# The sensor points down into the bin: an empty bin reads far away, a full bin
# reads close. We convert distance -> fill fraction.
BIN_EMPTY_DISTANCE_CM = 40.0
BIN_FULL_DISTANCE_CM = 6.0
ULTRASONIC_NOISE_CM = 0.6
FILL_PER_ITEM = 0.16             # each item collected adds this much fill
FILL_FULL_THRESHOLD = 0.90       # at/above this the boat heads home

# --- Boat: mission behaviour ------------------------------------------------
MISSION_TIME_LIMIT_S = 900.0     # total time allowed away from dock
MAX_ROUND_TRIPS = 3              # dump-and-return-to-same-spot attempts
COLLECT_TIME_S = 6.0             # time to actually scoop one item
DUMP_TIME_S = 15.0               # emptying the bin at the dock
TRASH_DETECT_RADIUS_M = 12.0     # boat notices trash within this range
TRASH_CAPTURE_RADIUS_M = 1.5     # close enough to scoop

# --- Boat: lawnmower sweep (fallback when the camera model returns None) ----
SWEEP_AREA_M = 50.0              # square sweep box centred on the target
SWEEP_LANE_SPACING_M = 10.0      # distance between mow lanes

# --- Boat: onboard vision ---------------------------------------------------
# The model is not trained yet, so the stub returns None and the boat falls
# back to the sweep. Set True once find_trash_direction() is real.
VISION_MODEL_ENABLED = False

# --- Boat: per-state timeouts (seconds) -------------------------------------
# If the boat sits in a state longer than this, it drops into `fault` with a
# logged reason. None = no timeout.
# NOTE: "searching" must exceed the time to walk the whole lawnmower sweep
# (roughly SWEEP_AREA_M * lanes / BOAT_SEARCH_SPEED_MPS) or the boat will fault
# on every mission. sim.py prints the estimate when it plans a sweep.
STATE_TIMEOUTS = {
    "idle": None,
    "en_route": 600.0,
    "searching": 700.0,
    "collecting": 90.0,
    "returning": 600.0,
    "dumping": 60.0,
    "charging": 1800.0,
    "fault": None,
}
BOAT_STATUS_INTERVAL_S = 2.0     # how often the boat publishes its status

# A fault is a stop-and-think, not necessarily a brick. After this many
# seconds the boat attempts to bring itself home and resume service.
# Set to None to make `fault` terminal until a human resets it.
FAULT_AUTO_RECOVER_S = 180.0


# =============================================================================
#  TIME_SCALE
# =============================================================================
# Two kinds of constant have time in them, and they move in opposite directions.
#
#   * A DURATION is a number of seconds. Compressing the clock divides it.
#   * A RATE is a quantity per second - battery percent, metres. Compressing the
#     clock multiplies it, so the same total is reached over the shorter time.
#
# Get one of those backwards and the run is not compressed, it is a different
# scenario: divide the durations but leave BOAT_SPEED_MPS alone and the boat
# simply never reaches the camera before its mission limit expires.
#
# Everything else - distances, radii, percentages, coordinates, noise per
# reading, item counts - has no time in it and must not be touched.
#
# TICK_S is in the duration list on purpose. Leaving it fixed while durations
# shrink would coarsen the sim: at TIME_SCALE 20, COLLECT_TIME_S 6.0 becomes
# 0.3s, which a 0.5s tick cannot even represent. Scaling it keeps the number of
# ticks per event constant, which is what keeps the outcome the same.

_DURATIONS = (
    "TICK_S", "SIM_DURATION_S",
    "CAMERA_REPORT_INTERVAL_S", "TRASH_SPAWN_INTERVAL_S",
    "REPORT_EXPIRY_S", "CAMERA_COOLDOWN_S", "DOCK_DECISION_INTERVAL_S",
    "DISPATCH_ACK_TIMEOUT_S", "MISSION_TIME_LIMIT_S", "COLLECT_TIME_S",
    "DUMP_TIME_S", "BOAT_STATUS_INTERVAL_S", "FAULT_AUTO_RECOVER_S",
)
_RATES = (
    "BOAT_SPEED_MPS", "BOAT_SEARCH_SPEED_MPS",
    "BATTERY_DRAIN_MOVING_PCT_S", "BATTERY_DRAIN_IDLE_PCT_S",
    "BATTERY_CHARGE_PCT_S",
)

_BASE = None                      # pristine values, captured before any scaling


def apply_time_scale(factor):
    """Rescale every duration and rate in this module to `factor`.

    Safe to call more than once: it always works from the values as they were
    at import, so apply_time_scale(10) after apply_time_scale(2) gives 10, not
    20. That also means it discards any hand-edit made to a scaled constant
    after import - set those afterwards, not before.
    """
    global _BASE, TIME_SCALE
    if factor <= 0:
        raise ValueError("TIME_SCALE must be positive, got %r" % (factor,))
    g = globals()
    if _BASE is None:
        _BASE = {n: g[n] for n in _DURATIONS + _RATES}
        _BASE["STATE_TIMEOUTS"] = dict(STATE_TIMEOUTS)

    for name in _DURATIONS:
        base = _BASE[name]
        g[name] = base if base is None else base / factor
    for name in _RATES:
        g[name] = _BASE[name] * factor
    g["STATE_TIMEOUTS"] = {k: (None if v is None else v / factor)
                           for k, v in _BASE["STATE_TIMEOUTS"].items()}
    TIME_SCALE = float(factor)


def _audit():
    """Fail loudly if a new `*_S` constant was added and never classified.

    Silence here would be the bad kind: the constant keeps its unscaled value
    while everything around it shrinks, and the sim misbehaves only at scales
    nobody tested.
    """
    known = set(_DURATIONS) | set(_RATES)
    missed = sorted(n for n, v in globals().items()
                    if n.endswith("_S") and not n.startswith("_")
                    and isinstance(v, (int, float, type(None)))
                    and not isinstance(v, bool) and n not in known
                    and n != "TIME_SCALE")
    if missed:
        raise RuntimeError(
            "config.py: %s end in _S but are in neither _DURATIONS nor _RATES. "
            "Add each to one of them (a duration is divided by TIME_SCALE, a "
            "per-second rate is multiplied by it)." % ", ".join(missed))


_audit()
if TIME_SCALE != 1.0:
    apply_time_scale(TIME_SCALE)
