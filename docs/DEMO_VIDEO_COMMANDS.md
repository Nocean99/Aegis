# Demo Video — Exact Commands Per Segment

Companion to `docs/DEMO_VIDEO_PLAN.md`. Every command below is verified
working from the repo root. Run each once BEFORE recording so caches are warm
and nothing downloads on camera.

```bash
cd ~/Documents/"autonomous drone"
```

## Segment 1 — Hook (0:00–0:10)

Type the mission request on camera:

```bash
./scripts/parse_mission_request.sh "find a red boat near the shoreline, urgent"
```

Optional second beat — show the compiled vision plan:

```bash
./scripts/plan_vision_search.sh "find a red boat near the shoreline, urgent"
```

## Segment 2 — Pipeline on real data (0:10–0:35)

The multisensor shoreline mission (RGB + IR + hydrophone):

```bash
./scripts/run_multisensor_demo.sh \
  --mission-request "Identify possible vessel activity in a protected coastal zone" \
  --rgb-images demo_data/shoreline_v1/rgb \
  --ir-images demo_data/shoreline_v1/ir \
  --acoustic demo_data/shoreline_v1/acoustic/hydrophone_contact_001.wav \
  --output-dir logs/demo_video/multisensor
```

Then open the visuals (each run creates a timestamped subfolder):

```bash
RUN=$(ls -t logs/demo_video/multisensor | head -1)
open logs/demo_video/multisensor/$RUN/multisensor_mission_report.html   # ranked candidates
open logs/demo_video/multisensor/$RUN/acoustic/spectrograms/            # spectrogram shots
open logs/demo_video/multisensor/$RUN/rgb/                              # debug frames + crops
```

For more on-screen detection frames, a vision-only run:

```bash
python3 -m autonomy.vision_lab \
  --mission-request "find a red boat near the shoreline, urgent" \
  demo_data/shoreline_v1/rgb \
  --output-dir logs/demo_video/vision_lab
```

For the "31% → 80.8% F1" caption, show the saved acoustic report:

```bash
open docs/ACOUSTIC_BENCHMARK_TUNING_REPORT.md
```

(Re-running it live needs the SanctSound download:
`./scripts/run_acoustic_benchmark_v1.sh /Users/noah/Downloads/dataset_final`)

## Segment 3 — The analyst decides (0:35–0:55)

```bash
./scripts/run_analyst_dashboard.sh
```

Open http://localhost:8010 — load the multisensor report from Segment 2,
click into a candidate (crop, score, decision band, explanation), confirm
one, reject one. Decisions persist to `candidate_reviews.json` beside the
report — show that file briefly:

```bash
cat logs/demo_video/multisensor/$RUN/candidate_reviews.json
```

## Segment 4 — Closed loop in sim (0:55–1:20)

One command (requires the PX4/Gazebo/ROS 2 setup from
`docs/PX4_GAZEBO_SETUP.md`):

```bash
./scripts/run_closed_loop_demo.sh
```

For the Gazebo viewport on screen, run the GUI in a second terminal:

```bash
GZ_MODE=gui ./scripts/run_red_block_world.sh
```

If the one-command launcher misbehaves on recording day, fall back to the
five-terminal manual sequence in `docs/CLOSED_LOOP_DEMO_RUNBOOK.md` — same
footage, more control. Label this segment "PX4 SITL / Gazebo simulation"
on screen. Record long, speed up 2–4× in the edit.

## Segment 5 — Close (1:20–1:30)

The five-mission system benchmark and its report:

```bash
./scripts/run_system_benchmark_v1.sh
RUN=$(ls -t logs/system_benchmark_v1 | head -1)
open logs/system_benchmark_v1/$RUN
```

End card: repo link + dataset names (SARD, DroneVehicle, NOAA SanctSound).

## The 20-second LinkedIn cut

One take, four beats — no narration needed:

```bash
# beat 1: type the request
./scripts/parse_mission_request.sh "Monitor a protected shoreline for possible vessel activity"

# beat 2: run the mission (same multisensor command as Segment 2)

# beat 3: dashboard → confirm one candidate

# beat 4: open the HTML report
```

## Recording-day checklist

- [ ] Warm run of every command above (first runs write caches/weights).
- [ ] Terminal font 16pt+, dark theme, window sized to the recording frame.
- [ ] `clear` between takes; no scrolling output longer than ~3 seconds.
- [ ] OBS at 1080p/30; each segment its own clip.
- [ ] "PX4 SITL / Gazebo simulation" label on the flight segment.
