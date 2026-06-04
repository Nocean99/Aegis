# PX4 Offboard Autonomy Stack

This layer is the simulation-only "drone brain" that sits above PX4. PX4 remains responsible for low-level stabilization and flight control. The autonomy stack owns mission policy, waypoint progression, safety decisions, and logs.

## Architecture

```text
MissionManager
  -> SafetyMonitor
  -> WaypointPlanner
  -> PX4ControllerInterface
      -> ROS 2 px4_msgs topics
      -> PX4 Offboard mode
```

High-level mission logic never publishes PX4 messages directly. It only calls `PX4ControllerInterface`.

## PX4 NED Coordinates

PX4 local position uses NED coordinates:

- `x`: north/forward
- `y`: east/right
- `z`: down
- altitude is negative `z`

So a 5 meter flight altitude is:

```yaml
z: -5.0
```

## Configuring A Mission

Edit:

```text
config/autonomy.yaml
```

Important fields:

- `takeoff_altitude_m`: positive altitude in meters
- `control_rate_hz`: Offboard setpoint publish rate
- `max_altitude_m`: safety ceiling
- `max_distance_from_home_m`: safety radius
- `waypoint_timeout_s`: return-home trigger
- `cruise_speed_mps`: interpolation speed between waypoints
- `waypoints`: PX4 NED setpoints with negative `z`

## Running The Stack

Terminal 1, Gazebo:

```bash
cd "/Users/noah/Documents/autonomous drone"
./scripts/run_gazebo_world.sh
```

Terminal 2, PX4:

```bash
cd "/Users/noah/Documents/autonomous drone"
./scripts/run_px4_standalone.sh
```

Terminal 3, uXRCE-DDS agent if your ROS 2 bridge needs it:

```bash
cd "/Users/noah/Documents/autonomous drone"
./scripts/run_uxrce_agent.sh
```

Terminal 4, autonomy:

```bash
cd "/Users/noah/Documents/autonomous drone"
./scripts/run_autonomy_node.sh
```

Optional visual debugger:

```bash
./scripts/run_visual_debugger.sh
```

## Logs

Mission logs are CSV files:

```text
logs/mission_<timestamp>.csv
```

Each row includes:

- timestamp
- mission state
- current position
- target setpoint
- active waypoint index
- safety status
- PX4 nav state
- PX4 arming state

## Tests

Run:

```bash
python3 tests/test_autonomy_stack.py
python3 tests/test_search_mission.py
```

The tests use a PX4 mock. They do not require Gazebo, ROS 2, or a physical drone.

## Current Limitations

- ROS 2 and `px4_msgs` must be installed/sourced before the real PX4 controller can run.
- The core stack is implemented and mock-tested, but live ROS 2/PX4 integration still needs to be verified in your local PX4 environment.
- Battery monitoring is a placeholder until PX4 battery telemetry is wired into the controller interface.
- Obstacle detection, search patterns, semantic mapping, and multi-drone coordination are intentionally not implemented yet.

## Next Milestone

Verify live Offboard flight in PX4/Gazebo:

1. Start PX4/Gazebo.
2. Start the uXRCE-DDS agent.
3. Start the autonomy node.
4. Confirm setpoints publish continuously at `control_rate_hz`.
5. Confirm PX4 switches to Offboard mode.
6. Confirm arm, takeoff, hover, waypoint mission, return home, land, disarm.

## Search-And-Detect Mission

The search mission adds a classical computer-vision mode for finding a red block in simulation.

State flow:

```text
TAKEOFF
SEARCH_PATTERN
DETECT_TARGET
CONFIRM_TARGET
APPROACH_TARGET
MARK_LOCATION
RETURN_HOME
LAND
```

Run with a ROS 2 camera topic:

```bash
./scripts/run_search_mission.sh --camera-topic /camera/image_raw
```

Run modes:

```bash
# Perception-only: live/synthetic/video camera + mock vehicle kinematics.
python3 -m autonomy.search_mission --run-mode perception-only --camera-source gz

# Full PX4: ROS 2 PX4 Offboard controller sends setpoints to PX4.
python3 -m autonomy.search_mission --run-mode full-px4 --camera-source ros2 --topic /camera/image_raw
```

Use `perception-only` when validating camera, detector, state machine, logs, and WorldModel without spending laptop resources on full Offboard motion. Use `full-px4` when validating real PX4 Offboard flight.

Natural-language mission requests are accepted as metadata for the intelligence layer:

```bash
python3 -m autonomy.search_mission \
  --run-mode perception-only \
  --camera-source gz \
  --operating-mode connected-supervised \
  --mission-request "Search this area for whatever the responder described"
```

Parse a request without running a mission:

```bash
./scripts/parse_mission_request.sh "Search the shoreline for a missing person signal"
```

Plan the command layer without running the drone:

```bash
./scripts/plan_mission_command.sh --mode connected-supervised "Search the shoreline for a missing person signal"
./scripts/plan_mission_command.sh --mode autonomous-return-report "Search this area for anything matching the caller description"
```

This parser is intentionally conservative. It preserves the raw request and extracts broad hints such as colors, categories, urgency, and search-area phrases. It does not claim full visual understanding by itself.

## Semantic Vision

The project now has a semantic vision scoring interface:

- low-level detectors propose candidate regions
- candidate crops are saved for review/model scoring
- the mission objective is passed into a semantic scorer
- each candidate receives a semantic score, decision, tags, and explanation
- candidates still require human confirmation for SAR use

Current implementation:

```text
LocalSemanticVisionScorer
```

This is a deterministic placeholder that uses mission text, candidate confidence, geometry, and simple visual cues such as requested color. It is not make/model recognition yet. Its value is architectural: the rest of the stack now speaks the right language for open-vocabulary visual matching.

Future implementation:

```text
VisionLanguageScorer
```

This should send saved candidate crops plus the raw mission request to a real vision-language model. That is the step that moves from "find red things" toward "find objects matching whatever the responder described." The model should return a calibrated score, explanation, and review recommendation rather than directly controlling PX4.

Vision-only testing:

```bash
./scripts/test_vision_only.sh path/to/image_or_folder \
  --mission-request "Search this image set for the responder's described target"
```

This does not start PX4, Gazebo, Docker, or ROS 2. It writes a report to:

```text
logs/vision_lab/<timestamp>/vision_report.json
```

It also saves annotated debug images and candidate crops. This is the easiest way to build a repeatable perception benchmark before spending laptop time on full simulation.

For large datasets:

```bash
./scripts/test_vision_only.sh path/to/image_or_folder \
  --mission-request "Search this image set for red objects that could be relevant to a rescue" \
  --save-only-detections \
  --max-saved-candidates 50
```

The JSON report still includes every image, but debug/crop images are only saved for accepted detections. Each result includes `red_audit` metrics:

- `red_pixel_count`
- `red_pixel_ratio`
- `largest_red_area_px`

The report summary includes `possible_misses`, which are images with meaningful red regions that failed the detector filters. This is not ground truth, but it is a practical audit tool for finding likely false negatives.

Large runs also write:

```text
review_shortlist.json
```

This is the top scored subset for human review. `--max-saved-candidates` controls how many debug/crop images are written, while `vision_report.json` still keeps the full per-image record.

Preview the generated vision plan:

```bash
./scripts/plan_vision_search.sh "Search the shoreline for a small blue boat with a white top"
```

Proposal modes:

- `mission-color`: default. Uses the mission text to choose color proposal ranges, such as blue for a blue boat or orange for an orange life jacket.
- `high-recall`: broad red-focused mode. Useful for older red-object tests and SAR marker-style searches.
- `precise`: stricter red-block detector. Useful for clean demos and simple red-block simulation.

Video-only testing:

```bash
./scripts/test_vision_only.sh "/path/to/video.mp4" \
  --video \
  --sample-every-s 1.0 \
  --mission-request "Search this video for red objects that could be relevant to a rescue" \
  --save-only-detections
```

The report includes `frame_index` and `timestamp_s` for every sampled frame. This mirrors the future live-camera flow: sample frames, propose candidates, crop them, score them, and save evidence.

## Specific-Object Evaluation

To test precision and accuracy for a specific target, use labels. Without labels, the system can only report what it found; it cannot know what it missed.

Create a CSV from:

```text
config/vision_labels_template.csv
```

Format:

```csv
image_path,expected_match,label,notes
red_vehicle_01.jpg,true,red_vehicle,clear positive
red_sign_01.jpg,false,not_target,red but not the target
```

Run:

```bash
./scripts/test_vision_only.sh "/path/to/test/images" \
  --mission-request "Search these images for the specific target description" \
  --labels-csv "/path/to/vision_labels.csv" \
  --eval-threshold 0.25 \
  --save-only-detections \
  --max-saved-candidates 50
```

The report includes:

- true positives
- false positives
- true negatives
- false negatives
- precision
- recall
- F1
- accuracy

For search and rescue, tune for high recall first. A false positive costs review time; a false negative can mean the target was missed.

Generate an HTML report viewer:

```bash
./scripts/build_vision_report_viewer.sh "logs/vision_lab/<timestamp>/vision_report.json"
```

This creates `vision_report_viewer.html` beside the JSON report. It shows benchmark metrics, the vision plan, false positives, false negatives, and the review shortlist with image previews.

Run the local analyst dashboard:

```bash
./scripts/run_analyst_dashboard.sh
```

Open:

```text
http://localhost:8010
```

The analyst dashboard is the first operator-facing mission intelligence UI. It lists saved vision reports, shows benchmark metrics, presents candidate cards, and stores review decisions:

```text
candidate_reviews.json
```

Review statuses:

- approve
- reject
- investigate

## Open-Vocabulary Scoring Hook

The semantic scorer now has two paths:

- `local`: deterministic placeholder, fast, no network/API key
- `openai`: optional vision-language scorer for crop/full-frame matching

The semantic scorer is only used for perception review. It does not control PX4 and cannot directly publish flight commands.

Example:

```bash
export OPENAI_API_KEY="..."
export OPENAI_VISION_MODEL="your-vision-capable-model"

./scripts/check_openai_vision_env.sh

./scripts/test_vision_only.sh "/path/to/test/images" \
  --mission-request "Search these images for people who may need rescue" \
  --semantic-vision openai \
  --openai-detail high \
  --full-frame-semantic misses \
  --labels-csv "/path/to/vision_labels.csv" \
  --save-only-detections \
  --max-saved-candidates 50
```

Full-frame semantic modes:

- `off`: crop/proposal scoring only
- `misses`: score the full frame when cheap proposals found nothing
- `all`: score every sampled frame as well as crops

For hybrid drone operation, `misses` is usually the best starting point: the cheap detector handles most frames, and the semantic model checks frames where the proposal layer might have missed something.

For serious benchmark runs, use `all` on a small labeled dataset first. That gives a more honest read on whether the semantic model can find targets even when the cheap proposal layer misses them. Use `--openai-detail low` for faster broad sweeps and `--openai-detail high` when targets are small, far away, partially hidden, or visually similar to distractors.

## Mission Command Layer

The project now separates "what the user asked for" from "how the drone should operate." A `MissionCommand` contains:

- the raw plain-English request
- the parsed `MissionObjective`
- an operating mode
- candidate confirmation behavior
- link-loss policy
- required report artifacts

Supported operating modes:

- `connected-supervised`: assumes a live dashboard/operator. Candidate matches should be shown for human confirmation. If link is lost, return home.
- `autonomous-return-report`: assumes the drone may not have a live link. It searches within configured safety limits, stores candidate evidence, returns home, and produces logs/snapshots for review.

Every search mission shutdown now also saves:

```text
logs/mission_command_<timestamp>.json
```

For the real Gazebo camera path, use:

```bash
./scripts/run_red_block_world.sh
./scripts/run_px4_camera_standalone.sh
./scripts/start_camera_bridge.sh
./scripts/verify_camera_feed.sh
./scripts/debug_camera_frame.sh --camera-topic /camera/image_raw
python3 -m autonomy.search_mission --camera-source ros2 --topic /camera/image_raw
```

Run with a synthetic red-block frame source for software testing:

```bash
./scripts/run_search_mission.sh --synthetic-camera
```

Configuration lives in `config/autonomy.yaml`:

- `target`: HSV thresholds, minimum area, required confirmation frames
- `search`: lawnmower, expanding square, or waypoint-list search settings
- `approach`: slow target-centering and stopping thresholds

Detection uses OpenCV HSV color thresholding, contour filtering, rectangularity checks, confidence scoring, and persistence across multiple frames. It intentionally does not use neural-network perception yet.

Search logs are written to:

```text
logs/search_mission_<timestamp>.csv
logs/target_<timestamp>.png
logs/debug_camera_<timestamp>.png
```

Every normal shutdown now flushes final artifacts:

```text
logs/world_model_<timestamp>.json
logs/world_model_<timestamp>.png
logs/debug_camera_<timestamp>.png
logs/target_<timestamp>.png
logs/candidates_<timestamp>.json
```

## Candidate Review

The project now has a generic candidate review workflow:

- `MissionObjective`: stores the user's free-text request.
- `CandidateTarget`: stores possible matches from any detector or semantic scorer.
- `CandidateManager`: manages unreviewed, confirmed, rejected, and needs-closer-look candidates.

Candidate statuses:

```text
UNREVIEWED
CONFIRMED
REJECTED
NEEDS_CLOSER_LOOK
```

This is the bridge toward open-vocabulary SAR behavior: the drone should propose likely matches and ask first responders for confirmation instead of pretending every detection is certain.

The target snapshot includes a bounding box around the confirmed red block.

## World Model

The search mission maintains an internal virtual world model. It does not control PX4 directly; it stores search knowledge for mission logic.

Tracked state:

- drone pose
- home position
- current search mission state
- searched and unsearched grid cells
- detected target candidates
- confirmed target location
- obstacle placeholders
- confidence scores
- safety zones

Each 2D grid cell tracks:

- `visited`
- `last_seen_time`
- `target_confidence`
- `obstacle_confidence`
- `risk_score`

During search, cells are marked visited as the drone flies over them. Red-block detections increase target confidence in estimated target cells. Confidence decays over time if the target is not confirmed. If the static search pattern is exhausted, the mission can request the nearest unsearched grid cell from the world model.

World-model logs:

```text
logs/world_model_updates.jsonl
logs/world_model_<timestamp>.json
logs/world_model_<timestamp>.png
```

Create a sample snapshot:

```bash
./scripts/export_world_model_demo.sh
```

Heatmap colors:

- green: searched cells
- red: target confidence
- blue: risk/obstacle placeholder confidence
