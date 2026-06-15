"""MediaPipe Face Mesh geometry for drowsiness and distraction signals."""

from __future__ import annotations

import math
import time
import urllib.request
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


# Face Mesh runtime settings
MAX_NUM_FACES = 1
REFINE_LANDMARKS = True
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
FACE_LANDMARKER_MODEL_PATH = Path("models/face_landmarker.task")
FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)

# Drowsiness thresholds
EAR_THRESHOLD = 0.22
PERCLOS_WINDOW_SECONDS = 60.0
PERCLOS_DROWSY_THRESHOLD = 0.35
MAR_THRESHOLD = 0.35
YAWN_MIN_SECONDS = 0.6

# Distraction thresholds
YAW_LOOK_AWAY_THRESHOLD_DEG = 25.0
PITCH_LOOK_AWAY_THRESHOLD_DEG = 20.0
LOOK_AWAY_MIN_SECONDS = 1.0
HEAD_POSE_CALIBRATION_FRAMES = 30

# Drawing settings
LANDMARK_POINT_RADIUS = 1
KEYPOINT_RADIUS = 2
OVERLAY_FONT_SCALE = 0.55
OVERLAY_LINE_HEIGHT = 22

# MediaPipe landmark indices used for eye aspect ratio.
LEFT_EYE_EAR = (33, 160, 158, 133, 153, 144)
RIGHT_EYE_EAR = (362, 385, 387, 263, 373, 380)

# Mouth aspect ratio landmarks. Horizontal pair plus three vertical pairs.
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
MOUTH_VERTICAL_PAIRS = ((13, 14), (81, 178), (311, 402))

# solvePnP uses a compact six-point face model.
HEAD_POSE_INDICES = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_outer": 33,
    "right_eye_outer": 263,
    "left_mouth": 61,
    "right_mouth": 291,
}

HEAD_POSE_MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),  # nose tip
        (0.0, -63.6, -12.5),  # chin
        (-43.3, 32.7, -26.0),  # left eye outer corner
        (43.3, 32.7, -26.0),  # right eye outer corner
        (-28.9, -28.9, -24.1),  # left mouth corner
        (28.9, -28.9, -24.1),  # right mouth corner
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class HeadPose:
    pitch: float
    yaw: float
    roll: float
    rotation_vector: np.ndarray
    translation_vector: np.ndarray


@dataclass(frozen=True)
class FaceGeometryResult:
    timestamp: float
    landmarks_px: np.ndarray
    face_bbox: tuple[int, int, int, int]
    left_ear: float
    right_ear: float
    ear: float
    mar: float
    perclos: float
    blink: bool
    yawn_active: bool
    looking_away_active: bool
    head_pose: HeadPose | None


def euclidean(point_a: np.ndarray, point_b: np.ndarray) -> float:
    return float(np.linalg.norm(point_a - point_b))


def eye_aspect_ratio(points: np.ndarray, indices: Iterable[int]) -> float:
    p1, p2, p3, p4, p5, p6 = [points[i] for i in indices]
    horizontal = euclidean(p1, p4)
    if horizontal <= 1e-6:
        return 0.0
    vertical = euclidean(p2, p6) + euclidean(p3, p5)
    return vertical / (2.0 * horizontal)


def mouth_aspect_ratio(points: np.ndarray) -> float:
    horizontal = euclidean(points[MOUTH_LEFT], points[MOUTH_RIGHT])
    if horizontal <= 1e-6:
        return 0.0
    vertical_sum = sum(euclidean(points[a], points[b]) for a, b in MOUTH_VERTICAL_PAIRS)
    return vertical_sum / (len(MOUTH_VERTICAL_PAIRS) * horizontal)


def normalized_to_pixel_landmarks(face_landmarks: object, width: int, height: int) -> np.ndarray:
    landmarks = face_landmarks.landmark if hasattr(face_landmarks, "landmark") else face_landmarks
    points = []
    for landmark in landmarks:
        x = min(max(landmark.x * width, 0.0), width - 1.0)
        y = min(max(landmark.y * height, 0.0), height - 1.0)
        points.append((x, y))
    return np.asarray(points, dtype=np.float64)


def ensure_face_landmarker_model(path: Path = FACE_LANDMARKER_MODEL_PATH) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MediaPipe Face Landmarker model to {path}...")
    urllib.request.urlretrieve(FACE_LANDMARKER_MODEL_URL, path)
    return path


def face_bbox_from_landmarks(
    points: np.ndarray,
    width: int,
    height: int,
    padding: float = 0.12,
) -> tuple[int, int, int, int]:
    min_xy = points.min(axis=0)
    max_xy = points.max(axis=0)
    x1, y1 = min_xy
    x2, y2 = max_xy
    box_w = x2 - x1
    box_h = y2 - y1
    pad_x = box_w * padding
    pad_y = box_h * padding
    x1 = int(max(0, math.floor(x1 - pad_x)))
    y1 = int(max(0, math.floor(y1 - pad_y)))
    x2 = int(min(width - 1, math.ceil(x2 + pad_x)))
    y2 = int(min(height - 1, math.ceil(y2 + pad_y)))
    return x1, y1, x2, y2


def crop_bbox(frame: np.ndarray, bbox: tuple[int, int, int, int], padding: float = 0.0) -> np.ndarray:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    if padding:
        box_w = x2 - x1
        box_h = y2 - y1
        x1 = int(max(0, x1 - box_w * padding))
        y1 = int(max(0, y1 - box_h * padding))
        x2 = int(min(width - 1, x2 + box_w * padding))
        y2 = int(min(height - 1, y2 + box_h * padding))
    return frame[y1 : y2 + 1, x1 : x2 + 1]


def estimate_head_pose(points: np.ndarray, frame_size: tuple[int, int]) -> HeadPose | None:
    height, width = frame_size
    image_points = np.array(
        [
            points[HEAD_POSE_INDICES["nose_tip"]],
            points[HEAD_POSE_INDICES["chin"]],
            points[HEAD_POSE_INDICES["left_eye_outer"]],
            points[HEAD_POSE_INDICES["right_eye_outer"]],
            points[HEAD_POSE_INDICES["left_mouth"]],
            points[HEAD_POSE_INDICES["right_mouth"]],
        ],
        dtype=np.float64,
    )

    focal_length = float(width)
    camera_matrix = np.array(
        [
            [focal_length, 0.0, width / 2.0],
            [0.0, focal_length, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    success, rotation_vector, translation_vector = cv2.solvePnP(
        HEAD_POSE_MODEL_POINTS,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    projection_matrix = np.hstack((rotation_matrix, translation_vector))
    _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(projection_matrix)
    pitch = float(euler[0, 0])
    yaw = float(euler[1, 0])
    roll = float(euler[2, 0])
    return HeadPose(
        pitch=pitch,
        yaw=yaw,
        roll=roll,
        rotation_vector=rotation_vector,
        translation_vector=translation_vector,
    )


class FaceGeometryTracker:
    """Stateful MediaPipe tracker that computes EAR, MAR, PERCLOS, and head pose."""

    def __init__(
        self,
        ear_threshold: float = EAR_THRESHOLD,
        mar_threshold: float = MAR_THRESHOLD,
        perclos_window_seconds: float = PERCLOS_WINDOW_SECONDS,
        yawn_min_seconds: float = YAWN_MIN_SECONDS,
        yaw_look_away_threshold_deg: float = YAW_LOOK_AWAY_THRESHOLD_DEG,
        pitch_look_away_threshold_deg: float = PITCH_LOOK_AWAY_THRESHOLD_DEG,
        look_away_min_seconds: float = LOOK_AWAY_MIN_SECONDS,
    ) -> None:
        self.ear_threshold = ear_threshold
        self.mar_threshold = mar_threshold
        self.perclos_window_seconds = perclos_window_seconds
        self.yawn_min_seconds = yawn_min_seconds
        self.yaw_look_away_threshold_deg = yaw_look_away_threshold_deg
        self.pitch_look_away_threshold_deg = pitch_look_away_threshold_deg
        self.look_away_min_seconds = look_away_min_seconds
        self.closed_eye_history: deque[tuple[float, bool]] = deque()
        self.yawn_started_at: float | None = None
        self.look_away_started_at: float | None = None
        self.head_pose_samples: list[np.ndarray] = []
        self.head_pose_baseline: np.ndarray | None = None

        import mediapipe as mp

        self.mp = mp
        self.backend = "solutions" if hasattr(mp, "solutions") else "tasks"
        self.face_mesh = None
        self.face_landmarker = None
        self._last_video_timestamp_ms = -1

        if self.backend == "solutions":
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=MAX_NUM_FACES,
                refine_landmarks=REFINE_LANDMARKS,
                min_detection_confidence=MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
            )
        else:
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.core.base_options import BaseOptions
            from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
                VisionTaskRunningMode,
            )

            model_path = ensure_face_landmarker_model()
            options = vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_path)),
                running_mode=VisionTaskRunningMode.VIDEO,
                num_faces=MAX_NUM_FACES,
                min_face_detection_confidence=MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            self.face_landmarker = vision.FaceLandmarker.create_from_options(options)

    def close(self) -> None:
        if self.face_mesh is not None:
            self.face_mesh.close()
        if self.face_landmarker is not None:
            self.face_landmarker.close()

    def process_frame(self, frame_bgr: np.ndarray, timestamp: float | None = None) -> FaceGeometryResult | None:
        timestamp = time.time() if timestamp is None else timestamp
        height, width = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        if self.backend == "solutions":
            results = self.face_mesh.process(rgb)
            if not results.multi_face_landmarks:
                self._expire_histories(timestamp)
                return None
            raw_landmarks = results.multi_face_landmarks[0]
        else:
            timestamp_ms = max(int(timestamp * 1000), self._last_video_timestamp_ms + 1)
            self._last_video_timestamp_ms = timestamp_ms
            mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
            results = self.face_landmarker.detect_for_video(mp_image, timestamp_ms)
            if not results.face_landmarks:
                self._expire_histories(timestamp)
                return None
            raw_landmarks = results.face_landmarks[0]

        landmarks_px = normalized_to_pixel_landmarks(raw_landmarks, width, height)
        left_ear = eye_aspect_ratio(landmarks_px, LEFT_EYE_EAR)
        right_ear = eye_aspect_ratio(landmarks_px, RIGHT_EYE_EAR)
        ear = (left_ear + right_ear) / 2.0
        mar = mouth_aspect_ratio(landmarks_px)
        raw_head_pose = estimate_head_pose(landmarks_px, (height, width))
        head_pose = self._apply_head_pose_calibration(raw_head_pose)

        blink = ear < self.ear_threshold
        self.closed_eye_history.append((timestamp, blink))
        self._expire_histories(timestamp)
        perclos = self._current_perclos()

        mouth_open = mar > self.mar_threshold
        if mouth_open and self.yawn_started_at is None:
            self.yawn_started_at = timestamp
        elif not mouth_open:
            self.yawn_started_at = None
        yawn_active = (
            self.yawn_started_at is not None
            and timestamp - self.yawn_started_at >= self.yawn_min_seconds
        )

        looking_away_now = False
        if head_pose is not None:
            looking_away_now = (
                abs(head_pose.yaw) > self.yaw_look_away_threshold_deg
                or abs(head_pose.pitch) > self.pitch_look_away_threshold_deg
            )
        if looking_away_now and self.look_away_started_at is None:
            self.look_away_started_at = timestamp
        elif not looking_away_now:
            self.look_away_started_at = None
        looking_away_active = (
            self.look_away_started_at is not None
            and timestamp - self.look_away_started_at >= self.look_away_min_seconds
        )

        return FaceGeometryResult(
            timestamp=timestamp,
            landmarks_px=landmarks_px,
            face_bbox=face_bbox_from_landmarks(landmarks_px, width, height),
            left_ear=left_ear,
            right_ear=right_ear,
            ear=ear,
            mar=mar,
            perclos=perclos,
            blink=blink,
            yawn_active=yawn_active,
            looking_away_active=looking_away_active,
            head_pose=head_pose,
        )

    def _expire_histories(self, timestamp: float) -> None:
        while self.closed_eye_history and timestamp - self.closed_eye_history[0][0] > self.perclos_window_seconds:
            self.closed_eye_history.popleft()

    def _current_perclos(self) -> float:
        if not self.closed_eye_history:
            return 0.0
        closed = sum(1 for _, is_closed in self.closed_eye_history if is_closed)
        return closed / len(self.closed_eye_history)

    def _apply_head_pose_calibration(self, head_pose: HeadPose | None) -> HeadPose | None:
        if head_pose is None:
            return None

        raw_angles = np.array([head_pose.pitch, head_pose.yaw, head_pose.roll], dtype=np.float64)
        if self.head_pose_baseline is None:
            self.head_pose_samples.append(raw_angles)
            current_baseline = np.median(np.asarray(self.head_pose_samples), axis=0)
            if len(self.head_pose_samples) >= HEAD_POSE_CALIBRATION_FRAMES:
                self.head_pose_baseline = current_baseline
        else:
            current_baseline = self.head_pose_baseline

        adjusted = raw_angles - current_baseline
        return HeadPose(
            pitch=float(adjusted[0]),
            yaw=float(adjusted[1]),
            roll=float(adjusted[2]),
            rotation_vector=head_pose.rotation_vector,
            translation_vector=head_pose.translation_vector,
        )


def draw_face_geometry(
    frame: np.ndarray,
    result: FaceGeometryResult | None,
    draw_all_landmarks: bool = True,
) -> np.ndarray:
    """Draw Face Mesh-derived landmarks and metrics on an OpenCV frame."""

    if result is None:
        draw_text_block(frame, ["Face: not detected"], origin=(12, 28), color=(0, 0, 255))
        return frame

    x1, y1, x2, y2 = result.face_bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 220, 120), 1)

    if draw_all_landmarks:
        for point in result.landmarks_px.astype(int):
            cv2.circle(frame, tuple(point), LANDMARK_POINT_RADIUS, (70, 130, 240), -1)

    key_indices = set(LEFT_EYE_EAR + RIGHT_EYE_EAR + (MOUTH_LEFT, MOUTH_RIGHT))
    for pair in MOUTH_VERTICAL_PAIRS:
        key_indices.update(pair)
    key_indices.update(HEAD_POSE_INDICES.values())
    for index in key_indices:
        point = tuple(result.landmarks_px[index].astype(int))
        cv2.circle(frame, point, KEYPOINT_RADIUS, (0, 255, 255), -1)

    pose = result.head_pose
    pose_text = "Head: n/a"
    if pose is not None:
        pose_text = f"Head pitch/yaw/roll: {pose.pitch:+.1f} {pose.yaw:+.1f} {pose.roll:+.1f}"

    lines = [
        f"EAR: {result.ear:.3f} (L {result.left_ear:.3f} / R {result.right_ear:.3f})",
        f"MAR: {result.mar:.3f}",
        f"PERCLOS: {result.perclos:.2%}",
        pose_text,
        f"Blink: {result.blink}  Yawn: {result.yawn_active}  Away: {result.looking_away_active}",
    ]
    draw_text_block(frame, lines, origin=(12, 28), color=(255, 255, 255))
    return frame


def draw_text_block(
    frame: np.ndarray,
    lines: list[str],
    origin: tuple[int, int],
    color: tuple[int, int, int],
    background: tuple[int, int, int] = (25, 25, 25),
) -> None:
    x, y = origin
    max_width = 0
    for line in lines:
        (width, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, OVERLAY_FONT_SCALE, 1)
        max_width = max(max_width, width)
    block_height = OVERLAY_LINE_HEIGHT * len(lines) + 10
    cv2.rectangle(
        frame,
        (x - 8, y - 20),
        (x + max_width + 8, y - 20 + block_height),
        background,
        -1,
    )
    for i, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (x, y + i * OVERLAY_LINE_HEIGHT),
            cv2.FONT_HERSHEY_SIMPLEX,
            OVERLAY_FONT_SCALE,
            color,
            1,
            cv2.LINE_AA,
        )
