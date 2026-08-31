"""One log for the whole run.

Every component writes through here, so a single file holds the camera server's
reports, the dock's decisions and the boat's state changes interleaved in the
order they actually happened. Chasing a dispatch across three separate console
scrollbacks is how you miss the message that explains it.

Each line carries both clocks:

    01:52:14.318  t=    215.0s  dock          dispatching to cam_1 (coverage 0.07)

`t=` is the simulation's virtual clock - the one every timeout, cooldown and
expiry in `config.py` is measured on. The wall-clock stamp on the left is what
lets you line this file up against a process that is NOT on the virtual clock:
`main-cam/coverage_monitor.py` runs in real time on its own machine, and when a
report looks wrong, the question is always which of its bursts produced it.

The component name is the logger's job, not the message's. Nothing written here
should prefix itself with its own name.
"""
from datetime import datetime


class RunLog:
    """A shared destination. `component(name)` hands out tagged log functions."""

    def __init__(self, clock, path=None, echo=True):
        self.clock = clock
        self.echo = echo
        self.path = path
        # Line buffering, so the file can be tailed while the run is still
        # going. A run that dies is exactly when the log matters most, and a
        # block-buffered one loses the last few thousand characters.
        self._fh = open(path, "w", encoding="utf-8", buffering=1) if path else None

    def component(self, name):
        """A log function tagged with this component's name."""
        def log(msg):
            self.write(name, msg)
        return log

    def write(self, name, msg):
        stamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        prefix = "%s  t=%9.1fs  %-14s " % (stamp, self.clock.now(), name)
        # Indent continuation lines to the message column so a multi-line
        # message stays visually attached to its own header.
        body = str(msg).replace("\n", "\n" + " " * len(prefix))
        line = prefix + body
        if self.echo:
            # flush explicitly: stdout is block-buffered when piped to a file,
            # and this log is usually being watched through a pipe.
            print(line, flush=True)
        if self._fh:
            self._fh.write(line + "\n")

    def close(self):
        if self._fh:
            self._fh.close()
            self._fh = None
