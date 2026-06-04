# Mission Intelligence Layer

Simulation-first mission intelligence for robotic and sensor systems.

This project started with an autonomous drone simulation, but the core idea is broader: convert high-level mission requests into structured search behavior, sensor processing, candidate scoring, analyst review, and mission reports. A PX4/Gazebo drone is the current test platform. The intelligence layer is intended to stay modular enough to support other robotic or sensor inputs later, such as fixed cameras, ground robots, acoustic sensors, telemetry feeds, or recorded data.

The repository is for simulation and software workflow testing only. It is not flight-control firmware and should not be connected directly to real motors.

## Current Capabilities

- Mission request parsing from plain-English objectives
- Search mission state machines for simulated robotic workflows
- PX4/Gazebo helper scripts for drone-based validation
- Fast dashboard simulation for command, telemetry, alerts, and logs
- Vision-only benchmark runner for images and videos
- Classical proposal detectors for color/objectness-based candidate generation
- Semantic scoring interface for open-vocabulary target review
- Analyst dashboard for reviewing candidates, metrics, and reports
- Structured logs, JSON reports, debug images, and candidate crops
- Safety-oriented mission concepts: return-home, geofence, abort, manual override, and link-loss policy

## Architecture

```text
Mission Request
   |
Mission Command + Objective Parser
   |
Search / Collection Plan
   |
Sensor Input
   |
Proposal Detection
   |
Semantic Scoring
   |
Candidate Review Dashboard
   |
Mission Report
```

PX4, Gazebo, and the drone simulator are integration targets, not the center of the system. The goal is to build a reusable mission-intelligence layer that can reason over different sensor sources while keeping low-level platform control separate.

## Quick Dashboard Demo

Run the lightweight dashboard simulation:

```bash
python3 server.py
```

Open:

```text
http://localhost:8000
```

This dashboard shows a simplified site-monitoring mission with simulated position, health, alerts, return-home, abort, and manual override controls.

## Fast Scenario Tests

Run the headless simulation tests without opening a live view:

```bash
./scripts/run_fast_sim_tests.sh
```

The runner covers takeoff/patrol, return-home, abort, manual override, high-wind return, and detection-injection scenarios. Reports are saved to:

```text
logs/sim_scenarios_<timestamp>.json
logs/sim_scenarios_<timestamp>.csv
```

## PX4/Gazebo Drone Track

The drone simulation path is still important because it gives the mission layer a realistic moving sensor platform. Use it to validate Offboard control, camera feeds, search routes, and safety behavior.

Main docs:

- `docs/PX4_GAZEBO_SETUP.md`
- `docs/ARCHITECTURE.md`
- `docs/PORTFOLIO_TEST_PLAN.md`
- `docs/AUTONOMY_STACK.md`
- `docs/REAL_SIMULATION.md`
- `docs/DOCKER_ROS2.md`

Common scripts:

```bash
./scripts/check_px4_env.sh
./scripts/run_red_block_world.sh
./scripts/run_red_block_gui.sh
./scripts/run_px4_camera_standalone.sh
./scripts/run_uxrce_agent.sh
./scripts/run_search_mission.sh
./scripts/check_ros2_env.sh
./scripts/start_camera_bridge.sh
./scripts/verify_camera_feed.sh
./scripts/debug_camera_frame.sh
```

PX4 remains responsible for low-level stabilization and flight control. Mission logic should call controller interfaces, not publish raw flight-control messages directly.

## Autonomy Stack

Core modules live in `autonomy/`:

- `mission_command.py`: mission command and operating-mode policy
- `mission_objective.py`: plain-English objective parsing
- `mission_manager.py`: waypoint mission state machine
- `search_mission.py`: search-and-detect mission flow
- `search_mission_manager.py`: mission execution and perception loop
- `px4_controller_interface.py`: ROS 2/PX4 Offboard wrapper
- `safety_monitor.py`: safety checks and mission responses
- `world_model.py`: local grid map of searched cells, candidates, and confidence
- `semantic_vision.py`: semantic scoring interface
- `vision_lab.py`: offline image/video benchmark runner

Mission configuration lives at:

```text
config/autonomy.yaml
```

Run core tests without ROS 2, Gazebo, Docker, or hardware:

```bash
python3 tests/test_autonomy_stack.py
python3 tests/test_search_mission.py
python3 tests/test_world_model.py
```

## Mission Commands

Parse a mission request:

```bash
./scripts/parse_mission_request.sh "Search the shoreline for possible signs of a missing person"
```

Plan a command without running a vehicle:

```bash
./scripts/plan_mission_command.sh --mode connected-supervised "Search the shoreline for possible signs of a missing person"
./scripts/plan_mission_command.sh --mode autonomous-return-report "Search the area for anything matching the responder description"
```

`connected-supervised` assumes a live operator can review candidates during the mission. `autonomous-return-report` assumes the platform may lose connection, so it stores candidates and returns with a report.

## Vision Benchmarks

Test perception without running the simulator:

```bash
./scripts/test_vision_only.sh "/path/to/images" \
  --mission-request "Search this image set for the responder's described target"
```

For large folders:

```bash
./scripts/test_vision_only.sh "/path/to/images" \
  --mission-request "Search this image set for red objects that could be relevant to a rescue" \
  --save-only-detections \
  --max-saved-candidates 50
```

The report is written to:

```text
logs/vision_lab/<timestamp>/vision_report.json
```

It includes per-image results, candidate crops, debug images, scores, explanations, false positives, false negatives, and a review shortlist when labels are available.

Video files use the same pipeline:

```bash
./scripts/test_vision_only.sh "/path/to/video.mp4" \
  --video \
  --sample-every-s 1.0 \
  --mission-request "Search this video for the responder's described target" \
  --save-only-detections
```

## Labeled Evaluation

Use a CSV when you want precision/recall instead of subjective inspection:

```bash
cp config/vision_labels_template.csv /Users/noah/Desktop/vision_labels.csv
```

Example:

```csv
image_path,expected_match,label,notes
target_01.jpg,true,target,clear positive
distractor_01.jpg,false,not_target,visually similar but incorrect
```

Run:

```bash
./scripts/test_vision_only.sh "/Users/noah/Desktop/vision_test_set" \
  --mission-request "Search these images for the specific target description" \
  --labels-csv "/Users/noah/Desktop/vision_labels.csv" \
  --eval-threshold 0.25 \
  --save-only-detections \
  --max-saved-candidates 50
```

## Analyst Dashboard

Run:

```bash
./scripts/run_analyst_dashboard.sh
```

Open:

```text
http://localhost:8010
```

The dashboard lists saved vision reports, shows precision/recall metrics, displays candidate images, and lets an analyst mark candidates as approved, rejected, or needing investigation. Review decisions are saved beside the report as:

```text
candidate_reviews.json
```

## Optional Vision-Language Scoring

The local semantic scorer is intentionally conservative. It ranks candidates but does not claim exact make/model or arbitrary object recognition. For stronger open-vocabulary testing, the project supports a provider-backed vision-language scorer as an optional backend.

Set up the environment:

```bash
export OPENAI_API_KEY="..."
export OPENAI_VISION_MODEL="your-vision-capable-model"

./scripts/check_openai_vision_env.sh
```

Run a deeper benchmark:

```bash
./scripts/test_vision_only.sh "/Users/noah/Desktop/vision_test_set" \
  --mission-request "Search these images for people who may need rescue" \
  --semantic-vision openai \
  --openai-detail high \
  --full-frame-semantic all \
  --labels-csv "/Users/noah/Desktop/vision_labels.csv" \
  --save-only-detections \
  --max-saved-candidates 50
```

Use `--full-frame-semantic misses` when you only want the semantic scorer to inspect frames where the cheap proposal layer found nothing. Use `--full-frame-semantic all` for more complete evaluation on a small labeled set.

The semantic scoring backend only reviews perception outputs. It does not publish PX4 commands or directly control a vehicle.

## Safety

This project is simulation-first. For any future hardware work, start with bench tests, props-off tests, tethered hover, manual flight, assisted waypoint flight, and only then controlled autonomous tests in a legal area with permission. Keep human override, logging, geofencing, and return-home behavior in the loop.
