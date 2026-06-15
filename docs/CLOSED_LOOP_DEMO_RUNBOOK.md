# Closed-Loop Demo Runbook (PX4 SITL + Gazebo)

This runbook takes the platform from "vision lab on recorded frames" to a
live closed loop: PX4 flies a simulated X500 in Gazebo, the camera feed
streams through ROS 2 into the mission intelligence layer, candidates land in
the analyst dashboard, and the mission report is written on landing.

Prerequisites are a one-time setup — follow `docs/PX4_GAZEBO_SETUP.md` first
(PX4-Autopilot cloned at `~/Documents/PX4-Autopilot`, Gazebo Harmonic, ROS 2
with `ros_gz_bridge`, and `MicroXRCEAgent` on PATH).

## The loop at a glance

```text
Gazebo world (red_block_search.sdf, simulated camera)
        │ gz topic
        ▼
ros_gz_bridge ──► /camera/image_raw (ROS 2)
        │                                ┌──────────────────────────┐
        ▼                                │ PX4 SITL ◄── MicroXRCEAgent │
autonomy.search_mission ── offboard ────►│   (real autopilot code)   │
  proposals → semantic scoring           └──────────────────────────┘
  → candidate queue → mission report
        │
        ▼
analyst dashboard (review, confirm/reject)
```

## One-command launch

```bash
./scripts/run_closed_loop_demo.sh
```

The script preflights every dependency, then starts each component in order
with logs under `closed_loop_logs/`. Stop everything with Ctrl-C. If a
dependency is missing it tells you which one and which doc section fixes it.

## Manual launch (five terminals)

Use this when debugging a single stage. Order matters.

Terminal 1 — Gazebo world (server) with the search world and camera:

```bash
./scripts/run_red_block_world.sh
# optional GUI in a sixth terminal: GZ_MODE=gui ./scripts/run_red_block_world.sh
```

Terminal 2 — PX4 SITL attached to the world:

```bash
./scripts/run_px4_gazebo.sh
```

Terminal 3 — DDS agent so ROS 2 can talk to PX4:

```bash
./scripts/run_uxrce_agent.sh
```

Terminal 4 — camera bridge (Gazebo image topic → `/camera/image_raw`):

```bash
./scripts/start_camera_bridge.sh
```

Terminal 5 — the mission itself:

```bash
./scripts/run_search_mission.sh
```

Optional terminal 6 — analyst dashboard for live review:

```bash
./scripts/run_analyst_dashboard.sh
```

## What to verify

1. `ros2 topic hz /camera/image_raw` shows frames arriving (~10+ Hz).
2. PX4 console reaches `Ready for takeoff!` before the mission starts.
3. The mission log shows state transitions: TAKEOFF → SEARCH_PATTERN →
   DETECT_TARGET → APPROACH_TARGET → MARK_LOCATION → RETURN_HOME → LAND.
4. After landing, the mission report directory contains the candidate queue
   JSON, crops, debug frames, and the world model export.
5. Candidates appear in the analyst dashboard with score, decision band, and
   explanation — confirm one and check the status persists.

## Troubleshooting

- No camera frames: confirm the Gazebo topic name with
  `./scripts/list_camera_topics.sh`, then set `GZ_CAMERA_TOPIC` for
  `start_camera_bridge.sh`.
- PX4 never reports ready: check the world loaded (`gz topic -l` non-empty)
  and that only one Gazebo server is running.
- Offboard rejected: the MicroXRCEAgent must be running before the mission
  starts; restart terminals 3 then 5.
- Sluggish sim on macOS: run the Gazebo server headless (default here) and
  skip the GUI; record the dashboard instead for the demo video.
