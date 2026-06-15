from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autonomy.georeference import (
    CameraIntrinsics,
    georeference_candidate,
    ned_to_latlon,
    pixel_to_ground,
)


INTRINSICS = CameraIntrinsics.from_fov(width_px=640, height_px=480, horizontal_fov_deg=90.0)


def test_intrinsics_from_fov() -> None:
    # 90 degree horizontal FOV: fx = (w/2) / tan(45 deg) = w/2.
    assert abs(INTRINSICS.fx - 320.0) < 1e-6
    assert INTRINSICS.cx == 320.0
    assert INTRINSICS.cy == 240.0


def test_nadir_center_pixel_is_directly_below_drone() -> None:
    contact = pixel_to_ground(
        pixel=(320.0, 240.0),
        intrinsics=INTRINSICS,
        drone_north_m=10.0,
        drone_east_m=-5.0,
        drone_altitude_m=30.0,
        yaw_deg=0.0,
        camera_pitch_deg=-90.0,
    )
    assert contact is not None
    assert abs(contact.north_m - 10.0) < 1e-6
    assert abs(contact.east_m - (-5.0)) < 1e-6
    assert abs(contact.slant_range_m - 30.0) < 1e-6
    assert abs(contact.ground_distance_m) < 1e-6


def test_nadir_right_edge_pixel_lands_altitude_meters_right() -> None:
    # 90 deg HFOV at nadir: the right image edge ray is 45 deg off boresight,
    # so it strikes the ground exactly altitude meters to the vehicle's right.
    contact = pixel_to_ground(
        pixel=(640.0, 240.0),
        intrinsics=INTRINSICS,
        drone_north_m=0.0,
        drone_east_m=0.0,
        drone_altitude_m=30.0,
        yaw_deg=0.0,
        camera_pitch_deg=-90.0,
    )
    assert contact is not None
    assert abs(contact.east_m - 30.0) < 1e-6
    assert abs(contact.north_m) < 1e-6
    assert abs(contact.bearing_deg - 90.0) < 1e-6


def test_yaw_rotates_offset_into_ned() -> None:
    # Same right-edge pixel, vehicle facing east: "right" now points south.
    contact = pixel_to_ground(
        pixel=(640.0, 240.0),
        intrinsics=INTRINSICS,
        drone_north_m=0.0,
        drone_east_m=0.0,
        drone_altitude_m=30.0,
        yaw_deg=90.0,
        camera_pitch_deg=-90.0,
    )
    assert contact is not None
    assert abs(contact.north_m - (-30.0)) < 1e-6
    assert abs(contact.east_m) < 1e-6
    assert abs(contact.bearing_deg - 180.0) < 1e-6


def test_forward_camera_45_degrees_hits_ground_ahead() -> None:
    # Boresight pitched 45 deg below horizon: center pixel hits the ground
    # `altitude` meters ahead of the vehicle.
    contact = pixel_to_ground(
        pixel=(320.0, 240.0),
        intrinsics=INTRINSICS,
        drone_north_m=0.0,
        drone_east_m=0.0,
        drone_altitude_m=20.0,
        yaw_deg=0.0,
        camera_pitch_deg=-45.0,
    )
    assert contact is not None
    assert abs(contact.north_m - 20.0) < 1e-6
    assert abs(contact.east_m) < 1e-6
    assert abs(contact.slant_range_m - 20.0 * math.sqrt(2.0)) < 1e-6


def test_horizon_ray_returns_none() -> None:
    contact = pixel_to_ground(
        pixel=(320.0, 240.0),
        intrinsics=INTRINSICS,
        drone_north_m=0.0,
        drone_east_m=0.0,
        drone_altitude_m=20.0,
        yaw_deg=0.0,
        camera_pitch_deg=0.0,  # boresight at the horizon
    )
    assert contact is None


def test_invalid_altitude_raises() -> None:
    try:
        pixel_to_ground(
            pixel=(320.0, 240.0),
            intrinsics=INTRINSICS,
            drone_north_m=0.0,
            drone_east_m=0.0,
            drone_altitude_m=0.0,
            yaw_deg=0.0,
        )
    except ValueError:
        return
    raise AssertionError("Expected ValueError for non-positive altitude")


def test_ned_to_latlon_round_trip_scale() -> None:
    home_lat, home_lon = 44.65, -63.57  # Halifax
    lat, lon = ned_to_latlon(home_lat, home_lon, 111.0, 0.0)
    # ~111 m north is about 0.001 degrees of latitude.
    assert abs((lat - home_lat) - 111.0 / 6_378_137.0 * 180.0 / math.pi) < 1e-12
    lat2, lon2 = ned_to_latlon(home_lat, home_lon, 0.0, 100.0)
    assert lat2 == home_lat
    assert lon2 > home_lon


def test_georeference_candidate_uses_bbox_center() -> None:
    report = georeference_candidate(
        bbox=(300, 220, 40, 40),  # center = image center (320, 240)
        center_px=None,
        intrinsics=INTRINSICS,
        drone_north_m=0.0,
        drone_east_m=0.0,
        drone_altitude_m=30.0,
        yaw_deg=0.0,
        home_lat=44.65,
        home_lon=-63.57,
    )
    assert report is not None
    assert report["ground_distance_m"] == 0.0
    assert report["latitude"] is not None
    assert abs(report["latitude"] - 44.65) < 1e-6


def test_georeference_candidate_without_pixel_returns_none() -> None:
    report = georeference_candidate(
        bbox=None,
        center_px=None,
        intrinsics=INTRINSICS,
        drone_north_m=0.0,
        drone_east_m=0.0,
        drone_altitude_m=30.0,
        yaw_deg=0.0,
    )
    assert report is None


if __name__ == "__main__":
    tests = [
        test_intrinsics_from_fov,
        test_nadir_center_pixel_is_directly_below_drone,
        test_nadir_right_edge_pixel_lands_altitude_meters_right,
        test_yaw_rotates_offset_into_ned,
        test_forward_camera_45_degrees_hits_ground_ahead,
        test_horizon_ray_returns_none,
        test_invalid_altitude_raises,
        test_ned_to_latlon_round_trip_scale,
        test_georeference_candidate_uses_bbox_center,
        test_georeference_candidate_without_pixel_returns_none,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
