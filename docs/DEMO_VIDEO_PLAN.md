# Demo Video Plan — Aegis Mission Intelligence Layer

Target: 90–120 seconds, screen-recorded, voiceover, no music required.
Audience: hiring managers / co-op supervisors at defense and robotics
companies. The point to land in the first 15 seconds: this is a
hardware-agnostic intelligence layer that sits on top of any competent
control stack, not "another drone project."

Record at 1080p+ with QuickTime or OBS. Pre-run every command once so model
weights and caches are warm and nothing downloads on camera.

## Storyboard

### Segment 1 — The thesis (0:00–0:15)

Screen: architecture diagram (`docs/architecture_diagram.mmd` rendered, or
the ARCHITECTURE.md figure), then a slow scroll of the repo tree.

Voiceover:
> "Robots already know how to move — PX4 solved flight. What they don't know
> is what matters. Aegis is a mission intelligence layer: natural-language
> tasking, multi-sensor perception, and analyst-ready evidence, on top of any
> control stack."

### Segment 2 — Natural-language tasking (0:15–0:35)

Screen: terminal. Run:

```bash
./scripts/parse_mission_request.sh "find a red boat near the shoreline, urgent"
./scripts/plan_vision_search.sh "find a red boat near the shoreline, urgent"
```

Highlight the parsed objective (categories, colors, urgency) and the vision
plan (proposal modes, prompts, review threshold).

Voiceover:
> "A mission starts as a sentence. The planner extracts the target, colors,
> categories, and urgency, and compiles a vision plan — what to look for and
> how cautious to be."

### Segment 3 — Multi-sensor mission (0:35–1:00)

Screen: run the end-to-end demo and scroll the report:

```bash
./scripts/run_multisensor_demo.sh
```

Show RGB + IR + acoustic evidence fusing into ranked candidates.

Voiceover:
> "Here's a full mission on a simulated shoreline: RGB, infrared, and
> acoustic evidence, fused into ranked candidates. Every detection keeps its
> provenance — which sensor, which proposal layer, why it scored what it
> scored."

### Segment 4 — Analyst review (1:00–1:25)

Screen: `./scripts/run_analyst_dashboard.sh`, open the queue, click a
candidate, show crop + score + explanation, confirm one, reject one. Show a
georeferenced contact location if available.

Voiceover:
> "Nothing is auto-decided. Candidates land in an analyst queue with crops,
> scores, and plain-language explanations. The analyst confirms or rejects —
> and those decisions feed mission memory, so the system gets better at the
> places it has been."

### Segment 5 — Proof and close (1:25–1:50)

Screen: benchmark report (`docs/SYSTEM_BENCHMARK_V1_REPORT.md` or a fresh
`./scripts/run_system_benchmark_v1.sh` run), then the CI badge / test run,
ending on the README.

Voiceover:
> "It's measured, not vibes: capture metrics for the review workflow, IoU
> localization metrics for detector quality, benchmarked across RGB, IR, and
> acoustic missions. Heuristic layers run anywhere; CLIP and YOLO drop in
> when you have the compute — all offline, no cloud in the loop. The control
> layer is replaceable. The intelligence layer is the product."

## Production checklist

- [ ] Warm caches: run every on-camera command once beforehand.
- [ ] Terminal: large font (16pt+), dark theme, window sized to recording.
- [ ] Hide personal paths/notifications; use a clean browser profile for the
      dashboard.
- [ ] Record each segment as a separate clip; cut dead time between command
      and output.
- [ ] Voiceover recorded separately and laid over the cut (cleaner than live
      narration).
- [ ] Export 1080p, keep a 30-second cut for LinkedIn and the full cut for
      the portfolio/README link.

## Optional closed-loop segment

If PX4 + Gazebo are set up (see `docs/CLOSED_LOOP_DEMO_RUNBOOK.md`), insert a
10-second clip after Segment 3: `./scripts/run_closed_loop_demo.sh`, Gazebo
GUI showing the X500 flying the search pattern, mission state transitions in
the log. Voiceover: "And it closes the loop — the same pipeline flying a
simulated vehicle through PX4's real autopilot code."
