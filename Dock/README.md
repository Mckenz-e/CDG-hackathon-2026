# Autonomous trash-collection simulation

A control-logic simulation of a trash-collecting boat and its dock station.
No machine learning, no hardware — every sensor, motor and radio is simulated,
so you can develop and test the decision-making before anything gets wet.

## Run it

For the whole system at once - broker, real detector, dock and boat, narrated
into one timeline and summarised at the end - use the demo runner one directory
up. It is the fastest way to see what this project does:

```bash
python ../demo.py
```

It runs for about two and a half minutes, preflights everything before it starts
anything (and says exactly what to install if a check fails), and tears its
processes down afterwards. `python ../demo.py --check` runs the checks alone;
`--no-vision` skips the detector and falls back to the built-in simulated
camera, which is the one to reach for if torch or CUDA is unhappy on the machine
you are presenting from.

To run just the simulation, with no broker and no detector:

```bash
python sim.py
```

```bash
python tests.py
```

Useful flags:

```bash
python sim.py --duration 1800 --realtime 0.05
```

`--duration` is virtual seconds to simulate. `--realtime 0` (the default) runs
as fast as the CPU allows; `1` runs in real time; `0.05` is 20x speed, which is
comfortable to watch.

## The log

Every component writes through one `RunLog`, so a single file holds the camera
reports, the dock's decisions and the boat's state changes interleaved in the
order they happened. It goes to `run.log` by default; `--log-file other.log`
moves it, `--log-file -` keeps it on the console only.

```
02:07:39.475  t=      5.0s  dock           dispatching to cam_1 (coverage 0.63, boat battery 100.0%)
02:07:39.475  t=      5.0s  dock           cam_1 marked SERVICING
02:07:39.476  t=      5.5s  boat           accepted dispatch to cam_1 at (13.757200, 100.502600)
```

Both clocks are on every line on purpose. `t=` is the virtual clock — the one
every timeout, cooldown and expiry in `config.py` is measured on. The wall-clock
stamp is what lets you line this file up against a process that is *not* on the
virtual clock: `main-cam/coverage_monitor.py` runs in real time on its own
machine, and when one of its reports looks wrong, the question is always which
of its bursts produced it. (The two processes deliberately write separate files
— interleaving appends from two processes loses lines.)

The component name is the logger's job. Nothing prefixes its own messages.

## Fast testing: `TIME_SCALE`

`TIME_SCALE` compresses the virtual clock. Every duration is divided by it and
every per-second rate multiplied by it, which is a change of units on the time
axis rather than a change of scenario — the boat covers the same water on the
same battery, and the same number of ticks elapse. Only the number of virtual
seconds it takes goes down.

```bash
python sim.py --time-scale 10 --duration 360
```

**This is not `--realtime`.** `--realtime` maps virtual seconds onto wall-clock
seconds and changes nothing about the run; reach for it when you just want to
watch faster. `TIME_SCALE` is for when something *outside* the sim sets the pace
and you cannot simply spin the virtual clock faster — a real camera server
publishing every 5 real seconds, say. There, shortening the sim's own durations
is the only way to fit a cooldown, a retry and a NEEDS ATTENTION verdict into a
test that finishes while you are still watching.

Adding a constant? `config.py` classifies each one as a duration (divided) or a
per-second rate (multiplied), and `_audit()` refuses to import if a new `*_S`
name is in neither list. Getting that backwards does not merely rescale the run,
it changes it: divide the durations but leave `BOAT_SPEED_MPS` alone and the
boat never reaches the camera before its mission limit expires.

**How faithful is it?** Measured, not assumed — the same 3600s scenario at
scales 1, 2, 5, 10 and 20 produces **494 events every time**, with 97.8-98.2% of
the message sequence identical and matched events landing within 1.5 virtual
seconds. Every dispatch, state transition, mission outcome and service
evaluation matches; the only residual difference is two `report queued` lines
swapping order. Treat it as behaviour-preserving, not log-identical.

That faithfulness depends on `Clock` counting whole ticks and multiplying out
rather than accumulating `t += dt` — `0.05` has no exact binary form, and every
interval in the sim is an `if now - last >= interval` against a value landing on
a tick boundary. Before that fix, scale 10 diverged into a different run
entirely: a few ulps moved one report across a tick, which reordered draws from
`World.rng` (one stream shared between trash spawning and coverage noise), and
from there every coverage figure differed. The shared stream is still a latent
sensitivity — worth splitting per purpose if the sim grows more randomness.

## The pieces

| File | What it is |
| --- | --- |
| `config.py` | **Every** tunable number: timings, speeds, thresholds, battery rates, camera coordinates. Nothing else holds a magic number. |
| `sim.py` | The runner: a virtual clock and the tick loop that drives everyone. |
| `bus.py` | The messaging layer — an in-process broker by default, real MQTT optionally. |
| `camera_server.py` | Shore cameras estimating how much trash they can see. |
| `dock.py` | The brain: queues reports, decides where to send the boat, tracks retries. |
| `boat.py` | The boat: state machine, simulated GPS / fill sensor / battery. |
| `vision.py` | Stub for the onboard camera model (returns `None` for now). |
| `log.py` | The shared log every component writes through. |
| `world.py` | The physical water and the trash in it — what the sensors are sensing. |
| `tests.py` | Scenario tests for the awkward paths: flat battery, hung state, stuck camera. |

Nothing in `dock.py` or `boat.py` reads `world.py` directly except through
something that stands in for a sensor. That separation is the point: the
control logic only ever sees what a real deployment would give it.

## Messages

Everything talks over pub/sub topics. Payloads are JSON, exactly as they would
be on the wire.

**Camera server → dock** (`trash/camera/report`)

```json
{"camera": "cam_1", "coverage": 0.23, "lat": 13.7572, "long": 100.5026, "timestamp": 120.0}
```

**Dock → camera server** (`trash/dock/servicing`)

```json
{"camera": "cam_1", "status": "servicing"}
{"camera": "cam_1", "status": "clear"}
```

While a camera is `servicing` the camera server ignores its own detections for
that camera. The boat is sitting in the frame and looks exactly like a large
piece of trash; without this, the dock would keep dispatching the boat to go
look at the boat.

**Boat → everyone** (`trash/boat/status`)

```json
{"state": "en_route", "lat": 13.7568, "long": 100.5022, "fill": 0.31,
 "battery": 88.4, "mission": "cam_1", "trip": 1, "timestamp": 300.0}
```

**Dock → boat** (`trash/boat/cmd`)

```json
{"action": "dispatch", "camera": "cam_1", "lat": 13.7572, "long": 100.5026, "timestamp": 300.0}
```

### Feeding the dock from the real camera server

`main-cam/coverage_monitor.py` is a real camera server: YOLO11s over SAHI tiles,
coverage measured as the union area of detections inside a water ROI. Point it
at this dock and the built-in simulated one steps aside:

```bash
# 1. a broker. `broker.yaml` here is a localhost-only, anonymous, pure-Python
#    one (pip install amqtt) for machines without mosquitto.
amqtt -c broker.yaml                       # or: mosquitto -v

# 2. the dock and boat, with the built-in camera server off
python sim.py --mqtt --no-cameras --realtime 0.05 \
    --min-coverage 0.05 --coverage-drop-min 0.03

# 3. the real camera server, in main-cam/
python coverage_monitor.py --dir data2/train/images --roi water_roi_data2.json \
    --broker 127.0.0.1 --topic trash/camera/report \
    --camera-id cam_1 --lat 13.757200 --long 100.502600 --interval 5
```

Four things have to line up, and three of them are not defaults:

* **Topic.** `coverage_monitor.py` publishes to `water/trash/coverage`; pass
  `--topic trash/camera/report`.
* **Position.** Its `--lat`/`--long` default to `None`. A report without
  coordinates is nowhere to send the boat, and the dock now drops it with a
  message saying so.
* **Timestamp.** It stamps wall-clock UTC as an ISO-8601 string; the sim's own
  camera server stamps virtual seconds since zero. Neither is on the dock's
  clock, so the dock no longer does arithmetic on the value: it keeps it as
  `reported_at` for the log and measures `REPORT_EXPIRY_S` from arrival instead.
  Any publisher's clock now works, including none at all.
* **Scale.** This is the one that actually needs a decision — see below.

**Coverage means something different at each end.** The simulated camera maps
`CAMERA_FULL_COVERAGE_ITEMS` items in view onto 0-1 and stays quiet below
`CAMERA_REPORT_THRESHOLD`. The real one reports an area fraction every cycle,
including `0.0` on purpose, and on the data2 pond that fraction measured
**0.006-0.217, median 0.072** — the whole real range sits below the simulated
camera's reporting threshold. So on real reports the dock needs its own floor
(`--min-coverage`), and `COVERAGE_DROP_MIN` has to come down with it, or a
service can never register as an improvement. The numbers above are for that
pond and that model; measure your own scene before trusting them.

**Suppression is now enforced at both ends.** `coverage_monitor.py` subscribes
to `trash/dock/servicing` and skips the whole cycle — inference included — while
its own camera is being serviced, then resumes on `clear`. Without it the boat,
sitting in the frame, reads as a large piece of trash.

The dock no longer *relies* on that, though: it drops any report for the camera
it is currently servicing, and says so in the log. The two are not redundant.
Suppression at the camera end saves the GPU work and keeps the wire quiet;
enforcement at the dock end is what makes the dock correct when the camera
server is an older build, a third-party one, or simply wrong. The failure it
prevents is not obvious — see step 5 below.

### Using a real broker

Set `USE_REAL_MQTT = True` in `config.py`, `pip install paho-mqtt`, and point
`MQTT_HOST`/`MQTT_PORT` at your broker. Nothing else changes — `bus.py` offers
the same interface either way, and no other file knows which one is running.

## Dock logic

1. Queue every camera report, keeping only the newest per camera and dropping
   anything older than `REPORT_EXPIRY_S`. Reports for the camera currently being
   serviced are discarded outright — the boat is in that frame.
2. When the boat is `idle` and above `DISPATCH_MIN_BATTERY_PCT`, dispatch it to
   the highest-coverage report that clears `DISPATCH_MIN_COVERAGE` and is not
   cooling down or flagged.
3. Mark that camera `servicing`; wait for the boat to acknowledge (a dispatch
   the boat never received is released after `DISPATCH_ACK_TIMEOUT_S`).
4. Mark it `clear` when the boat's mission ends, for any reason, and start a
   `CAMERA_COOLDOWN_S` cooldown.
5. On the first report after that cooldown, compare coverage against what it
   was at dispatch. A drop of at least `COVERAGE_DROP_MIN` resets the retry
   counter; no drop increments it. At `RETRY_LIMIT` the camera is logged as
   **NEEDS ATTENTION** and never dispatched to again.

Step 5 is why step 1 discards rather than merely declining to dispatch. The
evaluation fires on the first report that arrives with the cooldown expired, and
the cooldown does not start until the mission *ends* — so a report arriving
mid-mission is the first one it sees. Queue that report and the dock judges the
service on water it looked at before the boat even arrived, with a boat in the
picture. `test_reports_during_servicing_are_ignored` pins this.

That last rule is what stops the system burning its whole day on a false
positive. `cam_3` in the default config is deliberately such a camera — a
reflection or a stuck log that never improves no matter how often it is
visited — so you can watch the mechanism work.

## Boat states

```
idle -> en_route -> searching -> collecting -> returning -> dumping -> charging -> idle
                        ^____________|
```

`fault` is reachable from every state. Every transition is printed with the
reason, the battery level and the fill level.

* **idle** — at the dock, ready. Drops to `charging` below
  `BATTERY_RECHARGE_TRIGGER_PCT` (the gap to `BATTERY_CHARGED_PCT` is
  hysteresis; without it the boat flaps between the two states every tick).
* **en_route** — cruising to the target coordinate at `BOAT_SPEED_MPS`.
* **searching** — asks `find_trash_direction(frame)` first. The model is not
  trained yet, so it returns `None` and the boat falls back to a lawnmower
  sweep: serpentine lanes `SWEEP_LANE_SPACING_M` apart across a
  `SWEEP_AREA_M` box centred on the target, scooping what it comes across.
* **collecting** — closes on one item and spends `COLLECT_TIME_S` scooping it.
* **returning** — heads home. Triggered by a full bin, the mission time limit,
  a finished sweep, or a battery abort.
* **dumping** — empties the bin. If it came home *full* and mission time
  remains, it goes straight back out to the same coordinate, up to
  `MAX_ROUND_TRIPS`.
* **charging** — on the charger until `BATTERY_CHARGED_PCT`.
* **fault** — something took too long. The reason is logged. After
  `FAULT_AUTO_RECOVER_S` the boat tries to bring itself home; set that to
  `None` to make faults terminal until a human resets it.

### Safety rules the boat enforces on itself

The dock asks; the boat decides. It refuses a dispatch below
`DISPATCH_MIN_BATTERY_PCT`, aborts and comes home below `BATTERY_ABORT_PCT`,
and comes home when the bin reads full or `MISSION_TIME_LIMIT_S` expires —
regardless of what the dock wants.

### State timeouts

Every state has an entry in `STATE_TIMEOUTS`. Exceeding it means `fault` with a
logged reason. One relationship matters: `searching` must be longer than the
time to walk a whole sweep (roughly lanes × `SWEEP_AREA_M` ÷
`BOAT_SEARCH_SPEED_MPS`) or the boat will fault on every mission. The boat
prints the estimate when it plans a sweep and warns if the timeout is too
short.

## Plugging in the real camera model

`vision.py` is the only file to touch:

```python
def find_trash_direction(frame):
    """Returns (steering_angle_rad, distance_m, confidence) or None."""
```

`steering_angle` is relative to the boat's current heading, positive to the
left. Return `None` when nothing is detected — the sweep fallback stays as the
safety net. Set `VISION_MODEL_ENABLED = True` in `config.py` to switch it on;
the boat already handles a real detection (`_do_searching` steers to it), that
path is just never taken while the stub returns `None`.

## Simulated sensors

* **GPS** — true position plus `GPS_NOISE_M` of jitter. The boat navigates on
  its true position and *reports* the noisy one, so telemetry looks realistic
  without making the sim wobble.
* **Ultrasonic fill sensor** — measures distance down to the rubbish pile:
  `BIN_EMPTY_DISTANCE_CM` when empty, `BIN_FULL_DISTANCE_CM` when full, plus
  noise. `fill_level()` converts that reading back into a 0–1 fraction, which
  is what the control logic uses — the boat never reads its true fill.
* **Battery** — drains at `BATTERY_DRAIN_MOVING_PCT_S` while under way,
  `BATTERY_DRAIN_IDLE_PCT_S` otherwise, recharges at `BATTERY_CHARGE_PCT_S`.

## Determinism

`WORLD_SEED` fixes every random draw, and the tick loop is single-threaded, so
the same config always produces the same log. When you change a number and the
behaviour changes, the change is the cause.
