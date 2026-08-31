"""Broker check, publisher half.

Publishes one fake camera report every few seconds, in exactly the format the
real camera server uses. Run it in one terminal and mqtt_sub_test.py in
another; if the messages show up over there, your broker works.

    python mqtt_pub_test.py
    python mqtt_pub_test.py --host 192.168.1.50 --interval 2
"""
import argparse
import json
import sys
import time

import config

try:
    import paho.mqtt.client as mqtt
    from bus import make_client          # handles paho 1.x vs 2.x differences
except ImportError:
    sys.exit("paho-mqtt is not installed.  pip install paho-mqtt")


def main():
    p = argparse.ArgumentParser(description="Publish fake camera reports over MQTT")
    p.add_argument("--host", default=config.MQTT_HOST)
    p.add_argument("--port", type=int, default=config.MQTT_PORT)
    p.add_argument("--topic", default=config.TOPIC_CAMERA_REPORT)
    p.add_argument("--camera", default="cam_1")
    p.add_argument("--interval", type=float, default=5.0, help="seconds between reports")
    p.add_argument("--once", action="store_true", help="send a single report and exit")
    p.add_argument("--coverage", type=float, default=None,
                   help="fixed coverage value instead of a changing one")
    args = p.parse_args()

    cam = config.CAMERAS.get(args.camera, {"lat": config.DOCK_LAT, "long": config.DOCK_LONG})

    client = make_client(config.MQTT_CLIENT_PREFIX + "-pubtest")
    try:
        client.connect(args.host, args.port, keepalive=30)
    except OSError as e:
        sys.exit("could not reach a broker at %s:%d - %s\n"
                 "Is one running? (e.g. `mosquitto -v`)" % (args.host, args.port, e))
    client.loop_start()
    print("connected to %s:%d, publishing to '%s' every %.1fs (ctrl-c to stop)"
          % (args.host, args.port, args.topic, args.interval))

    n = 0
    try:
        while True:
            n += 1
            # Walk the coverage up and around so you can see values changing.
            coverage = args.coverage if args.coverage is not None                 else round(0.15 + (n % 8) * 0.1, 2)
            payload = {
                "camera": args.camera,
                "coverage": coverage,
                "lat": cam["lat"],
                "long": cam["long"],
                "timestamp": round(time.time(), 2),
            }
            info = client.publish(args.topic, json.dumps(payload))
            ok = "ok" if info.rc == mqtt.MQTT_ERR_SUCCESS else "FAILED rc=%d" % info.rc
            print("-> #%d %s  [%s]" % (n, json.dumps(payload), ok))
            if args.once:
                info.wait_for_publish(timeout=5)
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
