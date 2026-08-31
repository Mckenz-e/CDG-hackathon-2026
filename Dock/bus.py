"""Messaging layer.

Everything - camera server, dock, boat - talks only through publish/subscribe
on named topics. Payloads are JSON, exactly as they would be on the wire.

Two implementations behind one interface:
  * InProcessBus - a miniature broker inside this program (default, no deps)
  * MqttBus      - a real MQTT broker via paho-mqtt

Because the interface is identical, nothing else in the codebase knows or
cares which one is running.
"""
import json
from collections import deque

import config


def make_client(client_id):
    """Create a paho client that works on both paho-mqtt 1.x and 2.x.

    paho 2.0 changed the callback signatures and made the API version an
    explicit constructor argument. Asking for VERSION1 keeps the older
    on_connect/on_message signatures - and keeps 1.x working unchanged.
    """
    import paho.mqtt.client as mqtt

    if hasattr(mqtt, "CallbackAPIVersion"):          # paho 2.x
        return mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, client_id=client_id)
    return mqtt.Client(client_id=client_id)          # paho 1.x


def _topic_matches(pattern: str, topic: str) -> bool:
    """MQTT wildcard matching: '+' = one level, '#' = the rest."""
    p, t = pattern.split("/"), topic.split("/")
    for i, seg in enumerate(p):
        if seg == "#":
            return True
        if i >= len(t):
            return False
        if seg != "+" and seg != t[i]:
            return False
    return len(p) == len(t)


class InProcessBus:
    """A broker in a dictionary. Messages are queued and delivered on poll(),
    so the whole simulation stays single-threaded and reproducible."""

    def __init__(self, log=print):
        self._subs = []            # list of (pattern, callback)
        self._queue = deque()
        self._log = log

    def subscribe(self, pattern, callback):
        self._subs.append((pattern, callback))

    def publish(self, topic, payload: dict):
        self._queue.append((topic, json.dumps(payload).encode()))

    def poll(self):
        """Deliver everything currently queued (including anything published
        as a side effect of delivery)."""
        while self._queue:
            topic, raw = self._queue.popleft()
            payload = json.loads(raw.decode())
            for pattern, cb in self._subs:
                if _topic_matches(pattern, topic):
                    cb(topic, payload)

    def close(self):
        pass


class MqttBus:
    """The same interface, backed by a real broker."""

    def __init__(self, client_id, log=print):
        self._log = log
        self._inbox = deque()
        self._client = make_client(client_id)        # lazily imports paho
        self._client.on_message = self._on_message
        self._client.connect(config.MQTT_HOST, config.MQTT_PORT, keepalive=30)
        self._client.loop_start()
        self._subs = []

    def _on_message(self, _client, _userdata, msg):
        self._inbox.append((msg.topic, msg.payload))

    def subscribe(self, pattern, callback):
        self._subs.append((pattern, callback))
        self._client.subscribe(pattern)

    def publish(self, topic, payload: dict):
        self._client.publish(topic, json.dumps(payload))

    def poll(self):
        while self._inbox:
            topic, raw = self._inbox.popleft()
            payload = json.loads(raw.decode())
            for pattern, cb in self._subs:
                if _topic_matches(pattern, topic):
                    cb(topic, payload)

    def close(self):
        self._client.loop_stop()
        self._client.disconnect()


def make_bus(log=print):
    if config.USE_REAL_MQTT:
        log("connecting to MQTT broker at %s:%d" % (config.MQTT_HOST, config.MQTT_PORT))
        return MqttBus(config.MQTT_CLIENT_PREFIX + "-sim", log=log)
    log("using in-process message bus (set config.USE_REAL_MQTT=True for a real broker)")
    return InProcessBus(log=log)
