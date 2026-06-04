from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autonomy.vision_lab import (
    build_semantic_scorer,
    collect_image_paths,
    collect_video_paths,
    evaluate_results,
    load_labels,
    run_video_vision_lab,
    run_vision_lab,
)


def red_block_image() -> np.ndarray:
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    image[:] = (30, 50, 30)
    cv2.rectangle(image, (120, 80), (200, 150), (0, 0, 255), -1)
    return image


def test_collect_image_paths_from_folder() -> None:
    with TemporaryDirectory() as tmp:
        folder = Path(tmp)
        cv2.imwrite(str(folder / "one.png"), red_block_image())
        (folder / "notes.txt").write_text("ignore me", encoding="utf-8")
        paths = collect_image_paths([str(folder)])
        assert len(paths) == 1
        assert paths[0].name == "one.png"


def test_vision_lab_writes_report_and_debug_images() -> None:
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        image_path = tmp_path / "sample.png"
        cv2.imwrite(str(image_path), red_block_image())
        report_path = run_vision_lab(
            mission_request="Search the test image for a red object",
            image_paths=[image_path],
            output_dir=tmp_path / "out",
        )
        assert report_path.exists()
        report = json.loads(report_path.read_text(encoding="utf-8"))
        result = report["results"][0]
        assert result["detected"] is True
        assert result["semantic"]["score"] > 0
        assert Path(result["debug_path"]).exists()
        assert Path(result["crop_path"]).exists()
        assert "red_audit" in result
        assert report["summary"]["detections"] == 1


def test_vision_lab_can_save_only_detections() -> None:
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        positive = tmp_path / "positive.png"
        negative = tmp_path / "negative.png"
        cv2.imwrite(str(positive), red_block_image())
        cv2.imwrite(str(negative), np.zeros((240, 320, 3), dtype=np.uint8))
        report_path = run_vision_lab(
            mission_request="Search for a red object",
            image_paths=[positive, negative],
            output_dir=tmp_path / "out",
            save_only_detections=True,
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        positive_result, negative_result = report["results"]
        assert positive_result["debug_path"] is not None
        assert Path(positive_result["debug_path"]).exists()
        assert negative_result["debug_path"] is None
        assert report["summary"]["save_only_detections"] is True
        assert report["proposal_mode"] == "mission-color"
        assert (report_path.parent / "review_shortlist.json").exists()


def test_vision_lab_caps_saved_candidates() -> None:
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        images = []
        for index in range(3):
            path = tmp_path / f"positive_{index}.png"
            cv2.imwrite(str(path), red_block_image())
            images.append(path)
        report_path = run_vision_lab(
            mission_request="Search for a red object",
            image_paths=images,
            output_dir=tmp_path / "out",
            save_only_detections=True,
            max_saved_candidates=1,
            min_shortlist_score=0.0,
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        saved_debugs = [result for result in report["results"] if result.get("debug_path")]
        assert len(saved_debugs) == 1
        assert report["summary"]["shortlist_count"] == 1


def test_collect_video_paths_from_folder() -> None:
    with TemporaryDirectory() as tmp:
        folder = Path(tmp)
        (folder / "clip.mp4").write_bytes(b"not a real video")
        (folder / "image.png").write_bytes(b"ignore")
        paths = collect_video_paths([str(folder)])
        assert len(paths) == 1
        assert paths[0].name == "clip.mp4"


def test_video_vision_lab_samples_frames() -> None:
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        video_path = tmp_path / "sample.avi"
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (320, 240))
        assert writer.isOpened()
        writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
        writer.write(red_block_image())
        writer.release()
        report_path = run_video_vision_lab(
            mission_request="Search video frames for a red object",
            video_path=video_path,
            output_dir=tmp_path / "out",
            sample_every_s=0.2,
            max_frames=2,
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["source_type"] == "video"
        assert report["summary"]["processed"] == 2
        assert any(result["frame_index"] is not None for result in report["results"])


def test_vision_lab_evaluates_labeled_images() -> None:
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        positive = tmp_path / "positive.png"
        negative = tmp_path / "negative.png"
        labels = tmp_path / "labels.csv"
        cv2.imwrite(str(positive), red_block_image())
        cv2.imwrite(str(negative), np.zeros((240, 320, 3), dtype=np.uint8))
        labels.write_text(
            "image_path,expected_match,label,notes\n"
            "positive.png,true,target,should detect\n"
            "negative.png,false,background,should ignore\n",
            encoding="utf-8",
        )
        report_path = run_vision_lab(
            mission_request="Search for a red object",
            image_paths=[positive, negative],
            output_dir=tmp_path / "out",
            labels_csv=labels,
            eval_threshold=0.1,
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["evaluation"]["labeled_count"] == 2
        assert report["evaluation"]["true_positive"] == 1
        assert report["evaluation"]["true_negative"] == 1
        assert report["evaluation"]["recall"] == 1.0


def test_label_loader_and_metric_helpers() -> None:
    with TemporaryDirectory() as tmp:
        labels = Path(tmp) / "labels.csv"
        labels.write_text("image_path,frame_index,expected_match,label\nframe.jpg,10,yes,target\n", encoding="utf-8")
        loaded = load_labels(labels)
        assert loaded[("frame.jpg", 10)]["expected_match"] is True
    metrics = evaluate_results(
        [
            {"detected": True, "semantic": {"score": 0.9, "decision": "POSSIBLE_MATCH"}, "label": {"expected_match": True}},
            {"detected": False, "semantic": {"score": 0.0}, "label": {"expected_match": True}},
        ],
        threshold=0.5,
    )
    assert metrics["true_positive"] == 1
    assert metrics["false_negative"] == 1


def test_vision_lab_records_full_frame_semantic_scan() -> None:
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        image_path = tmp_path / "negative.png"
        cv2.imwrite(str(image_path), np.zeros((240, 320, 3), dtype=np.uint8))
        report_path = run_vision_lab(
            mission_request="Search for a person",
            image_paths=[image_path],
            output_dir=tmp_path / "out",
            proposal_mode="precise",
            full_frame_semantic="misses",
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        result = report["results"][0]
        assert result["detected"] is False
        assert result["full_frame_semantic"] is not None
        assert result["final_score"] == result["full_frame_semantic"]["score"]


def test_build_openai_scorer_passes_runtime_settings() -> None:
    previous_key = os.environ.get("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = "test-key"
    try:
        scorer = build_semantic_scorer(
            "openai",
            openai_model="test-vision-model",
            openai_detail="high",
            openai_timeout_s=9.0,
        )
        assert scorer.model_name == "test-vision-model"
        assert scorer.detail == "high"
        assert scorer.timeout_s == 9.0
    finally:
        if previous_key is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = previous_key


if __name__ == "__main__":
    tests = [
        test_collect_image_paths_from_folder,
        test_vision_lab_writes_report_and_debug_images,
        test_vision_lab_can_save_only_detections,
        test_vision_lab_caps_saved_candidates,
        test_collect_video_paths_from_folder,
        test_video_vision_lab_samples_frames,
        test_vision_lab_evaluates_labeled_images,
        test_label_loader_and_metric_helpers,
        test_vision_lab_records_full_frame_semantic_scan,
        test_build_openai_scorer_passes_runtime_settings,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
