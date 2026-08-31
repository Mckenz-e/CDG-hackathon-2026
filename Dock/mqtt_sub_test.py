"""Broker check, subscriber half.

Subscribes to every topic the system uses and prints whatever arrives. Run it
alongside mqtt_pub_test.py - or alongside the real sim with
config.USE_REAL_MQTT = True, to watch the live traffic.

    python mqtt_sub_test.py
    python mqtt_sub_test.py --topic 'trash/camera/#'
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
    p = argparse.ArgumentParser(description="Print every MQTT message received")
    p.add_argument("--host", default=config.MQTT_HOST)
    p.add_argument("--port", type=int, default=config.MQTT_PORT)
    p.add_argument("--topic", default="trash/#", help="topic filter (# = everything below)")
    args = p.parse_args()

    count = {"n": 0}

    def on_connect(client, _userdata, _flags, rc):
        if rc == 0:
            client.subscribe(args.topic)
            print("connected to %s:%d, subscribed to '%s' (ctrl-c to stop)"
                  % (args.host, args.port, args.topic))
        else:
            print("connection refused, rc=%d" % rc)

    def on_message(_client, _userdata, msg):
        count["n"] += 1
        stamp = time.strftime("%H:%M:%S")
        try:                                   # pretty-print JSON when it is JSON
            body = json.dumps(json.loads(msg.payload.decode()), sort_keys=True)
        except (ValueError, UnicodeDecodeError):
            body = repr(msg.payload)
        print("<- #%d [%s] %s  %s" % (count["n"], stamp, msg.topic, body))

    client = make_client(config.MQTT_CLIENT_PREFIX + "-subtest")
    client.on_connect = on_connect
    client.on_message = on_message
    try:
        client.connect(args.host, args.port, keepalive=30)
    except OSError as e:
        sys.exit("could not reach a broker at %s:%d - %s\n"
                 "Is one running? (e.g. `mosquitto -v`)" % (args.host, args.port, e))

    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print("\nstopping - %d message(s) received" % count["n"])
        client.disconnect()


if __name__ == "__main__":
    main()
