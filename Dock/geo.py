"""Tiny flat-earth geography helpers.

Over a few hundred metres we can pretend the earth is flat: convert lat/long
into a local grid measured in metres (x = east, y = north) centred on the
dock, do all the maths there, then convert back.
"""
import math

M_PER_DEG_LAT = 111320.0


def m_per_deg_long(lat_deg: float) -> float:
    """Longitude degrees get shorter as you move away from the equator."""
    return M_PER_DEG_LAT * math.cos(math.radians(lat_deg))


def to_local(lat, lon, origin_lat, origin_lon):
    """(lat, long) -> (x_east_m, y_north_m) relative to an origin."""
    x = (lon - origin_lon) * m_per_deg_long(origin_lat)
    y = (lat - origin_lat) * M_PER_DEG_LAT
    return x, y


def to_geo(x, y, origin_lat, origin_lon):
    """(x_east_m, y_north_m) -> (lat, long)."""
    lat = origin_lat + y / M_PER_DEG_LAT
    lon = origin_lon + x / m_per_deg_long(origin_lat)
    return lat, lon


def distance_m(a, b):
    """Distance in metres between two (lat, long) tuples."""
    ax, ay = to_local(a[0], a[1], a[0], a[1])          # a is its own origin
    bx, by = to_local(b[0], b[1], a[0], a[1])
    return math.hypot(bx - ax, by - ay)


def bearing_rad(frm, to):
    """Compass-free heading (radians, 0 = east, CCW) from one point to another."""
    dx = (to[1] - frm[1]) * m_per_deg_long(frm[0])
    dy = (to[0] - frm[0]) * M_PER_DEG_LAT
    return math.atan2(dy, dx)


def step_toward(frm, to, step_m):
    """Move `step_m` metres from `frm` toward `to`, never overshooting."""
    d = distance_m(frm, to)
    if d <= step_m or d == 0.0:
        return to
    h = bearing_rad(frm, to)
    return offset(frm, math.cos(h) * step_m, math.sin(h) * step_m)


def offset(point, dx_m, dy_m):
    """Shift a (lat, long) by a metre offset."""
    lat = point[0] + dy_m / M_PER_DEG_LAT
    lon = point[1] + dx_m / m_per_deg_long(point[0])
    return lat, lon
