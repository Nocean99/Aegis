#!/usr/bin/env bash
# Chained launcher for the closed-loop PX4 SITL + Gazebo demo.
# Preflights dependencies, then starts: Gazebo world -> PX4 SITL ->
# MicroXRCEAgent -> camera bridge -> search mission (+ analyst dashboard).
# Logs go to closed_loop_logs/. Ctrl-C stops everything.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PX4_DIR="${PX4_DIR:-$HOME/Documents/PX4-Autopilot}"
LOG_DIR="$ROOT_DIR/closed_loop_logs/$(date +%Y%m%d_%H%M%S)"
DASHBOARD="${DASHBOARD:-1}"

PIDS=()
cleanup() {
  echo
  echo "Stopping closed-loop demo..."
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

fail() {
  echo "ERROR: $1"
  echo "See docs/CLOSED_LOOP_DEMO_RUNBOOK.md and docs/PX4_GAZEBO_SETUP.md."
  exit 1
}

# --- Preflight -------------------------------------------------------------
command -v gz >/dev/null 2>&1 || fail "Gazebo 'gz' not on PATH."
command -v ros2 >/dev/null 2>&1 || fail "ros2 not on PATH (source your ROS 2 setup first)."
ros2 pkg prefix ros_gz_bridge >/dev/null 2>&1 || fail "ros_gz_bridge not available in this ROS 2 environment."
command -v MicroXRCEAgent >/dev/null 2>&1 || fail "MicroXRCEAgent not on PATH."
[ -d "$PX4_DIR" ] || fail "PX4 repo not found at $PX4_DIR (set PX4_DIR or clone PX4-Autopilot)."

mkdir -p "$LOG_DIR"
echo "Logs: $LOG_DIR"

start() {
  local name="$1"; shift
  echo "Starting $name..."
  "$@" >"$LOG_DIR/$name.log" 2>&1 &
  PIDS+=("$!")
}

wait_for() {
  local name="$1" check="$2" timeout="${3:-60}"
  local waited=0
  until bash -c "$check" >/dev/null 2>&1; do
    sleep 2
    waited=$((waited + 2))
    if [ "$waited" -ge "$timeout" ]; then
      fail "$name did not come up within ${timeout}s (see $LOG_DIR/$name.log)."
    fi
  done
  echo "  $name is up."
}

# --- Launch chain ----------------------------------------------------------
start gazebo_world "$ROOT_DIR/scripts/run_red_block_world.sh"
wait_for gazebo_world "gz topic -l | grep -q ." 90

start px4_sitl "$ROOT_DIR/scripts/run_px4_gazebo.sh"
sleep 5

start uxrce_agent "$ROOT_DIR/scripts/run_uxrce_agent.sh"
sleep 2

start camera_bridge "$ROOT_DIR/scripts/start_camera_bridge.sh"
wait_for camera_bridge "ros2 topic list | grep -q '/camera/image_raw'" 60

if [ "$DASHBOARD" = "1" ]; then
  start analyst_dashboard "$ROOT_DIR/scripts/run_analyst_dashboard.sh"
fi

echo
echo "All supporting services are up. Starting the search mission (foreground)."
echo "Press Ctrl-C at any time to stop everything."
echo
"$ROOT_DIR/scripts/run_search_mission.sh" "$@" 2>&1 | tee "$LOG_DIR/search_mission.log"

echo
echo "Mission finished. Logs and reports:"
echo "  $LOG_DIR"
