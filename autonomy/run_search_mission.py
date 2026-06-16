from __future__ import annotations

import argparse
import logging
import time

from autonomy.camera_interface import GzCameraSubscriber, ROS2CameraSubscriber, SyntheticRedBlockSource, VideoFileFrameSource
from autonomy.config_loader import load_search_mission_config
from autonomy.mission_command import create_mission_command
from autonomy.px4_controller_interface import MockPX4Controller, PX4ControllerInterface
from autonomy.search_mission_manager import SearchMissionManager
from autonomy.types import SearchMissionState


def main() -> None:
    parser = argparse.ArgumentParser(description="Run red-block search mission in PX4 Offboard")
    parser.add_argument("--config", default="config/autonomy.yaml")
    parser.add_argument("--camera-source", choices=["ros2", "gz", "video", "synthetic"], default=None)
    parser.add_argument("--topic", default=None)
    parser.add_argument("--camera-topic", default="/camera/image_raw")
    parser.add_argument("--gz-topic", default=GzCameraSubscriber.DEFAULT_TOPIC)
    parser.add_argument("--video")
    parser.add_argument("--synthetic-camera", action="store_true")
    parser.add_argument("--image-rate-hz", type=float, default=5.0)
    parser.add_argument("--no-require-camera-topic", action="store_true")
    parser.add_argument(
        "--run-mode",
        choices=["perception-only", "full-px4"],
        default="full-px4",
        help="perception-only uses MockPX4Controller; full-px4 uses the ROS 2 PX4 Offboard interface",
    )
    parser.add_argument("--mock-controller", action="store_true",
                        help="Deprecated alias for --run-mode perception-only")
    parser.add_argument("--keep-running-after-complete", action="store_true")
    parser.add_argument("--mission-request", default="Search the area for the configured target")
    parser.add_argument(
        "--semantic-vision",
        choices=["local"],
        default="local",
        help="Semantic candidate scorer. local is deterministic visual triage; clip/openai use vision-language scoring.",
    )
    parser.add_argument(
        "--proposal-mode",
        choices=["precise", "high-recall"],
        default="high-recall",
        help="high-recall sends more possible crops to semantic review; precise is stricter.",
    )
    parser.add_argument(
        "--operating-mode",
        choices=["connected-supervised", "autonomous-return-report"],
        default="connected-supervised",
        help="connected-supervised expects a live operator; autonomous-return-report stores candidates and returns with a report",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = load_search_mission_config(args.config)
    run_mode = "perception-only" if args.mock_controller else args.run_mode
    if run_mode == "perception-only":
        logging.info("Run mode: perception-only live camera test with MockPX4Controller")
        controller = MockPX4Controller()
    else:
        logging.info("Run mode: full PX4 Offboard flight")
        controller = PX4ControllerInterface()
    camera_source = args.camera_source
    if camera_source is None:
        if args.synthetic_camera:
            camera_source = "synthetic"
        elif args.video:
            camera_source = "video"
        else:
            camera_source = "ros2"
    topic = args.topic or args.camera_topic

    if camera_source == "synthetic":
        camera = SyntheticRedBlockSource()
    elif camera_source == "video":
        if not args.video:
            raise SystemExit("--camera-source video requires --video <path>")
        camera = VideoFileFrameSource(args.video)
    elif camera_source == "gz":
        gz_topic = args.topic or args.gz_topic
        camera = GzCameraSubscriber(gz_topic, require_topic=not args.no_require_camera_topic)
    else:
        camera = ROS2CameraSubscriber(topic, require_topic=not args.no_require_camera_topic)
    mission_command = create_mission_command(args.mission_request, operating_mode=args.operating_mode)
    logging.info(
        "Mission command: mode=%s target=%s",
        mission_command.operating_mode.value,
        mission_command.objective.target_description,
    )
    manager = SearchMissionManager(
        controller=controller,
        config=config,
        camera=camera,
        image_rate_hz=args.image_rate_hz,
        mission_command=mission_command,
        proposal_mode=args.proposal_mode,
    )
    manager.start()
    period = 1.0 / config.mission.control_rate_hz

    try:
        while True:
            started = time.monotonic()
            controller.spin_once(timeout_sec=0.0)
            state = manager.tick(now_s=started, dt_s=period)
            if state == SearchMissionState.COMPLETE and not args.keep_running_after_complete:
                logging.info("Search mission complete; exiting so final artifacts are flushed")
                break
            time.sleep(max(0.0, period - (time.monotonic() - started)))
    finally:
        manager.close()


if __name__ == "__main__":
    main()
