"""YOLOv8 phone-use detector."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


YOLO_WEIGHTS = "yolov8n.pt"
PHONE_CLASS_NAME = "cell phone"
PHONE_CONFIDENCE_THRESHOLD = 0.25
PHONE_IOA_THRESHOLD = 0.08
PHONE_RUN_EVERY_N_FRAMES = 8
DRIVER_REGION_FACE_EXPAND_X = 1.6
DRIVER_REGION_FACE_EXPAND_Y = 2.2


@dataclass(frozen=True)
class PhoneDetection:
    bbox: tuple[int, int, int, int]
    confidence: float
    class_name: str
    in_driver_region: bool


@dataclass(frozen=True)
class PhoneDetectionResult:
    phone_present: bool
    detections: list[PhoneDetection]
    ran_this_frame: bool
    driver_region: tuple[int, int, int, int] | None


def expand_bbox(
    bbox: tuple[int, int, int, int],
    frame_shape: tuple[int, int, int],
    expand_x: float = DRIVER_REGION_FACE_EXPAND_X,
    expand_y: float = DRIVER_REGION_FACE_EXPAND_Y,
) -> tuple[int, int, int, int]:
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    new_w = box_w * expand_x
    new_h = box_h * expand_y
    return (
        int(max(0, cx - new_w / 2.0)),
        int(max(0, cy - new_h / 2.0)),
        int(min(width - 1, cx + new_w / 2.0)),
        int(min(height - 1, cy + new_h / 2.0)),
    )


def intersection_over_area(
    inner: tuple[int, int, int, int],
    outer: tuple[int, int, int, int],
) -> float:
    ix1 = max(inner[0], outer[0])
    iy1 = max(inner[1], outer[1])
    ix2 = min(inner[2], outer[2])
    iy2 = min(inner[3], outer[3])
    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    inter_area = inter_w * inter_h
    inner_area = max(1, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return inter_area / inner_area


class PhoneDetector:
    """Runs pretrained COCO YOLOv8n and filters the cell phone class."""

    def __init__(
        self,
        weights: str | Path = YOLO_WEIGHTS,
        confidence_threshold: float = PHONE_CONFIDENCE_THRESHOLD,
        ioa_threshold: float = PHONE_IOA_THRESHOLD,
        run_every_n_frames: int = PHONE_RUN_EVERY_N_FRAMES,
        device: str | None = None,
    ) -> None:
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.confidence_threshold = confidence_threshold
        self.ioa_threshold = ioa_threshold
        self.run_every_n_frames = max(1, run_every_n_frames)
        self.device = device
        self.frame_counter = 0
        self.last_result = PhoneDetectionResult(False, [], False, None)
        self.phone_class_ids = self._resolve_phone_class_ids()

    def _resolve_phone_class_ids(self) -> set[int]:
        names = getattr(self.model, "names", {})
        return {int(class_id) for class_id, name in names.items() if str(name).lower() == PHONE_CLASS_NAME}

    def detect(
        self,
        frame_bgr: np.ndarray,
        face_bbox: tuple[int, int, int, int] | None = None,
        force: bool = False,
    ) -> PhoneDetectionResult:
        self.frame_counter += 1
        driver_region = expand_bbox(face_bbox, frame_bgr.shape) if face_bbox else None
        should_run = force or self.frame_counter % self.run_every_n_frames == 0
        if not should_run:
            stale = [
                detection
                for detection in self.last_result.detections
                if driver_region is None
                or intersection_over_area(detection.bbox, driver_region) >= self.ioa_threshold
            ]
            return PhoneDetectionResult(
                phone_present=any(d.in_driver_region for d in stale),
                detections=stale,
                ran_this_frame=False,
                driver_region=driver_region,
            )

        results = self.model.predict(
            frame_bgr,
            conf=self.confidence_threshold,
            verbose=False,
            device=self.device,
        )
        detections: list[PhoneDetection] = []
        if results:
            boxes = results[0].boxes
            for box in boxes:
                class_id = int(box.cls.item())
                if self.phone_class_ids and class_id not in self.phone_class_ids:
                    continue
                confidence = float(box.conf.item())
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                bbox = (x1, y1, x2, y2)
                in_region = True
                if driver_region is not None:
                    in_region = intersection_over_area(bbox, driver_region) >= self.ioa_threshold
                detections.append(
                    PhoneDetection(
                        bbox=bbox,
                        confidence=confidence,
                        class_name=PHONE_CLASS_NAME,
                        in_driver_region=in_region,
                    )
                )

        result = PhoneDetectionResult(
            phone_present=any(d.in_driver_region for d in detections),
            detections=detections,
            ran_this_frame=True,
            driver_region=driver_region,
        )
        self.last_result = result
        return result


def draw_phone_detections(frame: np.ndarray, result: PhoneDetectionResult | None) -> None:
    if result is None:
        return

    if result.driver_region is not None:
        x1, y1, x2, y2 = result.driver_region
        cv2.rectangle(frame, (x1, y1), (x2, y2), (90, 90, 255), 1)

    for detection in result.detections:
        x1, y1, x2, y2 = detection.bbox
        color = (0, 0, 255) if detection.in_driver_region else (120, 120, 120)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"phone {detection.confidence:.2f}"
        cv2.putText(
            frame,
            label,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
