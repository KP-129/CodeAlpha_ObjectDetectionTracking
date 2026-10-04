from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


Box = Tuple[int, int, int, int]


def iou(a: Box, b: Box) -> float:
    """Calculate Intersection over Union between two bounding boxes."""
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])

    intersection = max(0, right - left) * max(0, bottom - top)

    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])

    return intersection / (area_a + area_b - intersection + 1e-6)


@dataclass
class Track:
    box: Box
    label: str
    track_id: int
    confidence: float
    missed: int = 0


class IoUTracker:
    """
    Lightweight SORT-style tracker using IoU matching.

    Objects are matched between consecutive frames using
    bounding-box overlap and object class.
    """

    def __init__(self, min_iou: float = 0.3, max_missed: int = 20):
        self.min_iou = min_iou
        self.max_missed = max_missed
        self.next_id = 1
        self.tracks: Dict[int, Track] = {}

    def update(
        self,
        detections: List[Tuple[Box, str, float]]
    ) -> List[Track]:

        candidates = []

        for track_id, track in self.tracks.items():
            for det_index, (box, label, confidence) in enumerate(detections):

                if track.label == label:
                    score = iou(track.box, box)
                    candidates.append(
                        (score, track_id, det_index)
                    )

        candidates.sort(reverse=True)

        matched_tracks = set()
        matched_detections = set()

        for score, track_id, det_index in candidates:

            if score < self.min_iou:
                continue

            if track_id in matched_tracks:
                continue

            if det_index in matched_detections:
                continue

            box, label, confidence = detections[det_index]

            self.tracks[track_id].box = box
            self.tracks[track_id].confidence = confidence
            self.tracks[track_id].missed = 0

            matched_tracks.add(track_id)
            matched_detections.add(det_index)

        # Increase missed count for unmatched tracks.
        for track_id, track in list(self.tracks.items()):

            if track_id not in matched_tracks:
                track.missed += 1

                if track.missed > self.max_missed:
                    del self.tracks[track_id]

        # Create new tracks for unmatched detections.
        for det_index, (box, label, confidence) in enumerate(detections):

            if det_index not in matched_detections:

                self.tracks[self.next_id] = Track(
                    box=box,
                    label=label,
                    track_id=self.next_id,
                    confidence=confidence
                )

                self.next_id += 1

        return [
            track
            for track in self.tracks.values()
            if track.missed == 0
        ]


def track_color(track_id: int) -> Tuple[int, int, int]:
    """Generate a consistent BGR color for each tracking ID."""

    return (
        (37 * track_id) % 255,
        (17 * track_id + 90) % 255,
        (29 * track_id + 170) % 255,
    )


class ObjectDetector:

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        confidence: float = 0.45
    ):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.tracker = IoUTracker()

    def process_frame(self, frame):

        result = self.model(
            frame,
            conf=self.confidence,
            verbose=False
        )[0]

        detections = []

        for box, class_id, confidence in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.cls.cpu().numpy(),
            result.boxes.conf.cpu().numpy()
        ):

            x1, y1, x2, y2 = box.astype(int)

            label = result.names[int(class_id)]

            detections.append(
                (
                    (x1, y1, x2, y2),
                    label,
                    float(confidence)
                )
            )

        tracks = self.tracker.update(detections)

        for track in tracks:

            x1, y1, x2, y2 = track.box

            color = track_color(track.track_id)

            # Bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            # Label background
            text = (
                f"{track.label} | "
                f"ID {track.track_id} | "
                f"{track.confidence:.0%}"
            )

            (text_width, text_height), _ = cv2.getTextSize(
                text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                2
            )

            label_y = max(y1, text_height + 10)

            cv2.rectangle(
                frame,
                (x1, label_y - text_height - 10),
                (x1 + text_width + 8, label_y),
                color,
                -1
            )

            cv2.putText(
                frame,
                text,
                (x1 + 4, label_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

        # Active track counter
        active_count = len(tracks)

        cv2.rectangle(
            frame,
            (15, 15),
            (190, 55),
            (25, 25, 25),
            -1
        )

        cv2.putText(
            frame,
            f"Active Objects: {active_count}",
            (25, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        return frame