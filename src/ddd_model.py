"""Shared DDD CNN model and inference helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn


DEFAULT_IMG_SIZE = 96
DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)
LANDMARK_FEATURE_NAMES = [
    "landmarks_detected",
    "left_ear",
    "right_ear",
    "ear",
    "mar",
    "pitch",
    "yaw",
    "roll",
    "eye_distance_norm",
    "mouth_width_norm",
    "face_aspect",
]
LANDMARK_FEATURE_DIM = len(LANDMARK_FEATURE_NAMES)
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
MOUTH_LEFT = 61
MOUTH_RIGHT = 291


class DrowsinessCNN(nn.Module):
    """Small CNN for binary face-crop classification.

    The model outputs one logit. Apply sigmoid(logit) to get drowsy probability.
    """

    def __init__(self, dropout: float = 0.25) -> None:
        super().__init__()
        self.features = nn.Sequential(
            self._block(3, 24),
            self._block(24, 48),
            self._block(48, 96),
            self._block(96, 128),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    @staticmethod
    def _block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x).squeeze(1)


class HybridDrowsinessCNN(nn.Module):
    """CNN image branch plus MediaPipe geometry features."""

    def __init__(self, feature_dim: int = LANDMARK_FEATURE_DIM, dropout: float = 0.25) -> None:
        super().__init__()
        self.image_backbone = DrowsinessCNN(dropout=dropout).features
        self.landmark_branch = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128 + 16, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor, landmark_features: torch.Tensor) -> torch.Tensor:
        image_features = self.image_backbone(x)
        image_features = torch.flatten(image_features, 1)
        landmark_embedding = self.landmark_branch(landmark_features)
        combined = torch.cat([image_features, landmark_embedding], dim=1)
        return self.classifier(combined).squeeze(1)


@dataclass(frozen=True)
class LoadedDrowsinessModel:
    model: nn.Module
    device: torch.device
    img_size: int
    threshold: float
    model_kind: str = "cnn"
    landmark_feature_names: tuple[str, ...] = tuple()
    landmark_feature_mean: np.ndarray | None = None
    landmark_feature_std: np.ndarray | None = None


def preprocess_face_crop(face_bgr: np.ndarray, img_size: int = DEFAULT_IMG_SIZE) -> torch.Tensor:
    """Convert a BGR face crop to a normalized CHW tensor."""

    if face_bgr is None or face_bgr.size == 0:
        raise ValueError("face crop is empty")

    resized = cv2.resize(face_bgr, (img_size, img_size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.asarray(DEFAULT_MEAN, dtype=np.float32)
    std = np.asarray(DEFAULT_STD, dtype=np.float32)
    normalized = (rgb - mean) / std
    chw = np.transpose(normalized, (2, 0, 1))
    return torch.from_numpy(chw).float()


def landmark_features_from_geometry_result(geometry_result: object | None) -> np.ndarray:
    """Build the same compact MediaPipe feature vector used by Kaggle training."""

    if geometry_result is None:
        return np.zeros(LANDMARK_FEATURE_DIM, dtype=np.float32)

    points = np.asarray(getattr(geometry_result, "landmarks_px"), dtype=np.float64)
    x1, y1, x2, y2 = getattr(geometry_result, "face_bbox")
    face_w = max(float(x2 - x1), 1.0)
    face_h = max(float(y2 - y1), 1.0)
    face_scale = max(face_w, face_h, 1.0)

    head_pose = getattr(geometry_result, "head_pose", None)
    pitch = float(getattr(head_pose, "pitch", 0.0)) if head_pose is not None else 0.0
    yaw = float(getattr(head_pose, "yaw", 0.0)) if head_pose is not None else 0.0
    roll = float(getattr(head_pose, "roll", 0.0)) if head_pose is not None else 0.0
    eye_distance_norm = float(np.linalg.norm(points[LEFT_EYE_OUTER] - points[RIGHT_EYE_OUTER]) / face_scale)
    mouth_width_norm = float(np.linalg.norm(points[MOUTH_LEFT] - points[MOUTH_RIGHT]) / face_scale)
    face_aspect = face_w / face_h

    return np.asarray(
        [
            1.0,
            float(getattr(geometry_result, "left_ear")),
            float(getattr(geometry_result, "right_ear")),
            float(getattr(geometry_result, "ear")),
            float(getattr(geometry_result, "mar")),
            pitch,
            yaw,
            roll,
            eye_distance_norm,
            mouth_width_norm,
            face_aspect,
        ],
        dtype=np.float32,
    )


def preprocess_landmark_features(
    loaded: LoadedDrowsinessModel,
    geometry_result: object | None,
) -> torch.Tensor:
    features = landmark_features_from_geometry_result(geometry_result)
    if loaded.landmark_feature_mean is not None and loaded.landmark_feature_std is not None:
        features = (features - loaded.landmark_feature_mean) / loaded.landmark_feature_std
    return torch.from_numpy(features.astype(np.float32)).float()


def load_drowsiness_model(
    checkpoint_path: str | Path,
    device: str | torch.device | None = None,
) -> LoadedDrowsinessModel:
    """Load a trained DDD checkpoint saved by train_ddd_classifier.py."""

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"DDD checkpoint not found: {checkpoint_path}")

    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint: dict[str, Any] = torch.load(checkpoint_path, map_location=resolved_device)
    model_kind = str(checkpoint.get("model_kind", "cnn"))
    feature_names = tuple(checkpoint.get("landmark_feature_names") or ())
    feature_dim = len(feature_names) or LANDMARK_FEATURE_DIM
    if model_kind == "hybrid":
        model = HybridDrowsinessCNN(feature_dim=feature_dim, dropout=float(checkpoint.get("dropout", 0.25)))
    else:
        model = DrowsinessCNN(dropout=float(checkpoint.get("dropout", 0.25)))
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(resolved_device)
    model.eval()
    feature_mean = checkpoint.get("landmark_feature_mean")
    feature_std = checkpoint.get("landmark_feature_std")
    return LoadedDrowsinessModel(
        model=model,
        device=resolved_device,
        img_size=int(checkpoint.get("img_size", DEFAULT_IMG_SIZE)),
        threshold=float(checkpoint.get("threshold", 0.5)),
        model_kind=model_kind,
        landmark_feature_names=feature_names,
        landmark_feature_mean=np.asarray(feature_mean, dtype=np.float32) if feature_mean is not None else None,
        landmark_feature_std=np.asarray(feature_std, dtype=np.float32) if feature_std is not None else None,
    )


@torch.inference_mode()
def predict_drowsiness_probability(
    loaded: LoadedDrowsinessModel,
    face_bgr: np.ndarray,
    geometry_result: object | None = None,
) -> float:
    """Return sigmoid probability that a face crop is drowsy."""

    tensor = preprocess_face_crop(face_bgr, loaded.img_size).unsqueeze(0).to(loaded.device)
    if loaded.model_kind == "hybrid":
        features = preprocess_landmark_features(loaded, geometry_result).unsqueeze(0).to(loaded.device)
        logit = loaded.model(tensor, features)
    else:
        logit = loaded.model(tensor)
    return float(torch.sigmoid(logit).item())
