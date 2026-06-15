from __future__ import annotations

"""Pixel-to-world georeferencing for camera detections.

Converts a pixel in a drone camera frame into a ground position so contacts
can be reported as locations, not frame indexes. Uses a flat-ground
ray-intersection model: build the pixel's ray in the camera frame, rotate it
into the world frame using vehicle attitude and gimbal pitch, and intersect
with the ground plane.

Frames follow the PX4 local NED convention: x north, y east, z down, yaw
clockwise from north. The drone's local position comes from
PX4ControllerInterface / WorldModel; ``ned_to_latlon`` converts a local offset
to geographic coordinates given the home position.

Accuracy notes: flat-ground assumption breaks on steep terrain; attitude noise
dominates error at shallow camera angles. Both are acceptable for directing an
analyst or a revisit waypoint, which is the goal here — this is evidence
localization, not weapon-grade targeting.
"""

import math
from dataclasses import dataclass


EARTH_RADIUS_M = 6_378_137.0


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width_px: int
    height_px: int

    @classmethod
    def from_fov(cls, *, width_px: int, height_px: int, horizontal_fov_deg: float) -> "CameraIntrinsics":
        """Build intrinsics from resolution and horizontal field of view."""
        if width_px <= 0 or height_px <= 0 or not 0 < horizontal_fov_deg < 180:
            raise ValueError("Invalid camera geometry.")
        fx = (width_px / 2.0) / math.tan(math.radians(horizontal_fov_deg) / 2.0)
        return cls(
            fx=fx,
            fy=fx,  # square pixels assumed
            cx=width_px / 2.0,
            cy=height_px / 2.0,
            width_px=width_px,
            height_px=height_px,
        )


@dataclass(frozen=True)
class GroundContact:
    north_m: float
    east_m: float
    slant_range_m: float
    ground_distance_m: float
    bearing_deg: float
    latitude: float | None = None
    longitude: float | None = None

    def as_dict(self) -> dict:
        return {
            "north_m": round(self.north_m, 2),
            "east_m": round(self.east_m, 2),
            "slant_range_m": round(self.slant_range_m, 2),
            "ground_distance_m": round(self.ground_distance_m, 2),
            "bearing_deg": round(self.bearing_deg, 2),
            "latitude": None if self.latitude is None else round(self.latitude, 7),
            "longitude": None if self.longitude is None else round(self.longitude, 7),
        }


def pixel_to_ground(
    *,
    pixel: tuple[float, float],
    intrinsics: CameraIntrinsics,
    drone_north_m: float,
    drone_east_m: float,
    drone_altitude_m: float,
    yaw_deg: float,
    camera_pitch_deg: float = -90.0,
    home_lat: float | None = None,
    home_lon: float | None = None,
) -> GroundContact | None:
    """Project a pixel onto the ground plane.

    ``camera_pitch_deg`` is the camera boresight elevation: -90 is nadir
    (straight down), 0 is the horizon. ``yaw_deg`` is vehicle heading,
    clockwise from north. Roll is assumed stabilized by the gimbal.
    Returns None when the ray does not hit the ground (at or above horizon).
    """
    if drone_altitude_m <= 0:
        raise ValueError("drone_altitude_m must be positive (height above ground).")
    u, v = pixel
    # Ray in camera frame: +z forward along boresight, +x right, +y down.
    x_cam = (u - intrinsics.cx) / intrinsics.fx
    y_cam = (v - intrinsics.cy) / intrinsics.fy
    z_cam = 1.0

    # Rotate by camera pitch about the right (x) axis: pitch -90 points
    # the boresight straight down.
    pitch = math.radians(camera_pitch_deg)
    # Forward/down components in the vehicle frame (x forward, y right, z down):
    forward = z_cam * math.cos(pitch) - y_cam * math.sin(pitch) * -1.0
    down = z_cam * -math.sin(pitch) + y_cam * math.cos(pitch)
    right = x_cam

    if down <= 1e-9:
        return None  # Ray points at or above the horizon.

    # Scale the ray to hit the ground plane (altitude above ground).
    scale = drone_altitude_m / down
    forward_m = forward * scale
    right_m = right * scale
    slant_range = math.sqrt(forward_m**2 + right_m**2 + drone_altitude_m**2)

    # Rotate vehicle frame into NED by yaw.
    yaw = math.radians(yaw_deg)
    north_offset = forward_m * math.cos(yaw) - right_m * math.sin(yaw)
    east_offset = forward_m * math.sin(yaw) + right_m * math.cos(yaw)

    north = drone_north_m + north_offset
    east = drone_east_m + east_offset
    ground_distance = math.hypot(north_offset, east_offset)
    bearing = (math.degrees(math.atan2(east_offset, north_offset)) + 360.0) % 360.0

    latitude = longitude = None
    if home_lat is not None and home_lon is not None:
        latitude, longitude = ned_to_latlon(home_lat, home_lon, north, east)

    return GroundContact(
        north_m=north,
        east_m=east,
        slant_range_m=slant_range,
        ground_distance_m=ground_distance,
        bearing_deg=bearing,
        latitude=latitude,
        longitude=longitude,
    )


def ned_to_latlon(home_lat: float, home_lon: float, north_m: float, east_m: float) -> tuple[float, float]:
    """Convert a local NED offset to lat/lon (equirectangular approximation).

    Accurate to well under a meter for offsets up to a few kilometers, which
    covers any mission this platform flies.
    """
    lat = home_lat + math.degrees(north_m / EARTH_RADIUS_M)
    lon = home_lon + math.degrees(east_m / (EARTH_RADIUS_M * math.cos(math.radians(home_lat))))
    return lat, lon


def georeference_candidate(
    *,
    bbox: tuple[int, int, int, int] | None,
    center_px: tuple[int, int] | None,
    intrinsics: CameraIntrinsics,
    drone_north_m: float,
    drone_east_m: float,
    drone_altitude_m: float,
    yaw_deg: float,
    camera_pitch_deg: float = -90.0,
    home_lat: float | None = None,
    home_lon: float | None = None,
) -> dict | None:
    """Georeference a candidate detection. Uses the bbox center, falling back
    to the provided center pixel. Returns a report-ready dict or None."""
    if bbox is not None:
        x, y, w, h = bbox
        pixel = (x + w / 2.0, y + h / 2.0)
    elif center_px is not None:
        pixel = (float(center_px[0]), float(center_px[1]))
    else:
        return None
    contact = pixel_to_ground(
        pixel=pixel,
        intrinsics=intrinsics,
        drone_north_m=drone_north_m,
        drone_east_m=drone_east_m,
        drone_altitude_m=drone_altitude_m,
        yaw_deg=yaw_deg,
        camera_pitch_deg=camera_pitch_deg,
        home_lat=home_lat,
        home_lon=home_lon,
    )
    return None if contact is None else contact.as_dict()
