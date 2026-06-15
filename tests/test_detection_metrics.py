from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autonomy.detection_metrics import (
    evaluate_localization,
    iou,
    localization_items_from_results,
    parse_gt_boxes,
)


def test_iou_identical_boxes_is_one() -> None:
    box = (10, 10, 50, 50)
    assert abs(iou(box, box) - 1.0) < 1e-9


def test_iou_disjoint_boxes_is_zero() -> None:
    assert iou((0, 0, 10, 10), (100, 100, 10, 10)) == 0.0


def test_iou_partial_overlap() -> None:
    # Two 10x10 boxes sharing a 5x10 strip: intersection 50, union 150.
    value = iou((0, 0, 10, 10), (5, 0, 10, 10))
    assert abs(value - 50.0 / 150.0) < 1e-9


def test_parse_gt_boxes_parses_semicolon_separated_boxes() -> None:
    boxes = parse_gt_boxes("10,20,30,40;50,60,70,80")
    assert boxes == [(10, 20, 30, 40), (50, 60, 70, 80)]


def test_parse_gt_boxes_empty_means_no_targets() -> None:
    assert parse_gt_boxes("") == []
    assert parse_gt_boxes(None) == []
    assert parse_gt_boxes("   ") == []


def test_parse_gt_boxes_rejects_malformed_input() -> None:
    for bad in ("1,2,3", "a,b,c,d", "10,10,0,5"):
        try:
            parse_gt_boxes(bad)
        except ValueError:
            continue
        raise AssertionError(f"Expected ValueError for {bad!r}")


def test_perfect_localization_scores_full_marks() -> None:
    items = [{"gt_boxes": [(10, 10, 40, 40)], "pred_boxes": [((10, 10, 40, 40), 0.9)]}]
    result = evaluate_localization(items).as_dict()
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["mean_iou"] == 1.0
    assert result["average_precision"] == 1.0


def test_missed_target_lowers_recall_not_precision() -> None:
    items = [
        {
            "gt_boxes": [(10, 10, 40, 40), (200, 200, 40, 40)],
            "pred_boxes": [((10, 10, 40, 40), 0.9)],
        }
    ]
    result = evaluate_localization(items).as_dict()
    assert result["precision"] == 1.0
    assert result["recall"] == 0.5
    assert result["false_negative"] == 1


def test_false_positive_lowers_precision_not_recall() -> None:
    items = [
        {
            "gt_boxes": [(10, 10, 40, 40)],
            "pred_boxes": [((10, 10, 40, 40), 0.9), ((300, 300, 40, 40), 0.8)],
        }
    ]
    result = evaluate_localization(items).as_dict()
    assert result["precision"] == 0.5
    assert result["recall"] == 1.0
    assert result["false_positive"] == 1


def test_greedy_matching_prefers_higher_scored_prediction() -> None:
    # Both predictions overlap the single ground truth; only one may match.
    items = [
        {
            "gt_boxes": [(10, 10, 40, 40)],
            "pred_boxes": [((12, 12, 40, 40), 0.9), ((8, 8, 40, 40), 0.6)],
        }
    ]
    result = evaluate_localization(items).as_dict()
    assert result["true_positive"] == 1
    assert result["false_positive"] == 1


def test_localization_items_built_from_result_dicts() -> None:
    results = [
        {
            "detected": True,
            "bbox": [10, 10, 40, 40],
            "final_score": 0.8,
            "label": {"gt_boxes": [(10, 10, 40, 40)]},
        },
        {
            "detected": False,
            "bbox": None,
            "label": {"gt_boxes": [(50, 50, 20, 20)]},
        },
        {"detected": True, "bbox": [0, 0, 5, 5], "label": {}},  # no gt_boxes -> skipped
    ]
    items = localization_items_from_results(results)
    assert len(items) == 2
    assert items[0]["pred_boxes"] == [((10, 10, 40, 40), 0.8)]
    assert items[1]["pred_boxes"] == []  # missed frame still counts its ground truth


def test_full_frame_fallback_contributes_no_predicted_box() -> None:
    results = [
        {
            "detected": True,
            "bbox": None,  # full-frame fallback
            "final_score": 0.2,
            "label": {"gt_boxes": [(10, 10, 40, 40)]},
        }
    ]
    items = localization_items_from_results(results)
    result = evaluate_localization(items).as_dict()
    assert result["recall"] == 0.0
    assert result["predicted_boxes"] == 0


if __name__ == "__main__":
    tests = [
        test_iou_identical_boxes_is_one,
        test_iou_disjoint_boxes_is_zero,
        test_iou_partial_overlap,
        test_parse_gt_boxes_parses_semicolon_separated_boxes,
        test_parse_gt_boxes_empty_means_no_targets,
        test_parse_gt_boxes_rejects_malformed_input,
        test_perfect_localization_scores_full_marks,
        test_missed_target_lowers_recall_not_precision,
        test_false_positive_lowers_precision_not_recall,
        test_greedy_matching_prefers_higher_scored_prediction,
        test_localization_items_built_from_result_dicts,
        test_full_frame_fallback_contributes_no_predicted_box,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
