from __future__ import annotations

"""Multi-frame contact tracking for video missions.

Associates per-frame candidate detections into persistent contacts using
greedy IoU matching with a centroid-distance fallback, so a vessel seen in
forty frames becomes one reviewable contact instead of forty queue entries.
Tracking is evidence aggregation only: it never suppresses candidates, it
annotates them with a shared track id and summarizes each track for the
analyst queue and mission report.
"""

import math
from dataclasses import dataclass, field

from autonomy.detection_metrics import iou


@dataclass
class Track:
    track_id: int
    last_frame_index: int
    last_bbox: tuple[int, int, int, int]
    candidate_ids: list[str] = field(default_factory=list)
    frame_indexes: list[int] = field(default_factory=list)
    first_timestamp_s: float | None = None
    last_timestamp_s: float | None = None
    best_score: float = 0.0
    best_candidate_id: str | None = None
    best_decision: str | None = None

    def as_summary(self) -> dict:
        return {
            "track_id": self.track_id,
            "observations": len(self.candidate_ids),
            "first_frame_index": self.frame_indexes[0] if self.frame_indexes else None,
            "last_frame_index": self.frame_indexes[-1] if self.frame_indexes else None,
            "first_timestamp_s": self.first_timestamp_s,
            "last_timestamp_s": self.last_timestamp_s,
            "duration_s": (
                round(self.last_timestamp_s - self.first_timestamp_s, 3)
                if self.first_timestamp_s is not None and self.last_timestamp_s is not None
                else None
            ),
            "best_score": round(self.best_score, 4),
            "best_candidate_id": self.best_candidate_id,
            "best_decision": self.best_decision,
            "candidate_ids": self.candidate_ids[:50],
        }


class ContactTracker:
    """Greedy IoU tracker for low-frame-rate sampled video.

    Sampled mission video is often 1 frame/second, so motion between frames can
    be large. Matching uses IoU first, then a normalized centroid distance
    fallback for fast-moving small targets. Tracks expire after
    ``max_gap_frames`` sampled frames without a match.
    """

    def __init__(
        self,
        *,
        iou_threshold: float = 0.2,
        max_centroid_distance_ratio: float = 0.12,
        max_gap_frames: int = 3,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_centroid_distance_ratio = max_centroid_distance_ratio
        self.max_gap_frames = max_gap_frames
        self.tracks: list[Track] = []
        self._next_id = 1

    def update(
        self,
        *,
        frame_index: int,
        timestamp_s: float | None,
        bbox: tuple[int, int, int, int],
        candidate_id: str,
        score: float,
        decision: str | None = None,
        frame_size: tuple[int, int] | None = None,
    ) -> int:
        """Assign this detection to an existing or new track. Returns track id."""
        active = [
            track
            for track in self.tracks
            if frame_index - track.last_frame_index <= self.max_gap_frames
        ]
        best_track: Track | None = None
        best_affinity = 0.0
        for track in active:
            affinity = self._affinity(track.last_bbox, bbox, frame_size)
            if affinity > best_affinity:
                best_affinity = affinity
                best_track = track
        if best_track is None or best_affinity <= 0.0:
            best_track = Track(track_id=self._next_id, last_frame_index=frame_index, last_bbox=bbox)
            self._next_id += 1
            self.tracks.append(best_track)
        best_track.last_frame_index = frame_index
        best_track.last_bbox = bbox
        best_track.candidate_ids.append(candidate_id)
        best_track.frame_indexes.append(frame_index)
        if timestamp_s is not None:
            if best_track.first_timestamp_s is None:
                best_track.first_timestamp_s = timestamp_s
            best_track.last_timestamp_s = timestamp_s
        if score >= best_track.best_score:
            best_track.best_score = score
            best_track.best_candidate_id = candidate_id
            best_track.best_decision = decision
        return best_track.track_id

    def summaries(self) -> list[dict]:
        ordered = sorted(self.tracks, key=lambda track: (-track.best_score, -len(track.candidate_ids)))
        return [track.as_summary() for track in ordered]

    def _affinity(
        self,
        previous_bbox: tuple[int, int, int, int],
        bbox: tuple[int, int, int, int],
        frame_size: tuple[int, int] | None,
    ) -> float:
        overlap = iou(previous_bbox, bbox)
        if overlap >= self.iou_threshold:
            return overlap
        if frame_size is None:
            return 0.0
        width, height = frame_size
        diagonal = math.hypot(width, height)
        if diagonal <= 0:
            return 0.0
        ax, ay, aw, ah = previous_bbox
        bx, by, bw, bh = bbox
        distance = math.hypot((ax + aw / 2) - (bx + bw / 2), (ay + ah / 2) - (by + bh / 2))
        ratio = distance / diagonal
        if ratio <= self.max_centroid_distance_ratio:
            # Map distance to a weak affinity below any real IoU match.
            return max(0.01, self.iou_threshold * (1.0 - ratio / self.max_centroid_distance_ratio) * 0.5)
        return 0.0


def track_video_results(results: list[dict]) -> dict | None:
    """Annotate video-run results with track ids and return a track summary.

    Mutates each detected result dict (adds ``track_id``) and returns the
    summary block for the mission report, or None when nothing was tracked.
    """
    tracker = ContactTracker()
    tracked = 0
    for result in results:
        bbox = result.get("bbox")
        frame_index = result.get("frame_index")
        if not result.get("detected") or bbox is None or frame_index is None:
            continue
        track_id = tracker.update(
            frame_index=int(frame_index),
            timestamp_s=result.get("timestamp_s"),
            bbox=tuple(bbox),
            candidate_id=str(result.get("candidate_id")),
            score=float(result.get("final_score") or result.get("detector_confidence") or 0.0),
            decision=result.get("final_decision"),
        )
        result["track_id"] = track_id
        tracked += 1
    if tracked == 0:
        return None
    summaries = tracker.summaries()
    return {
        "tracked_candidates": tracked,
        "track_count": len(summaries),
        "tracks": summaries,
    }
