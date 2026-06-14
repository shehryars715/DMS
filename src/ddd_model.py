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


@dataclass(frozen=True)
class LoadedDrowsinessModel:
    model: DrowsinessCNN
    device: torch.device
    img_size: int
    threshold: float


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
    model = DrowsinessCNN(dropout=float(checkpoint.get("dropout", 0.25)))
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(resolved_device)
    model.eval()
    return LoadedDrowsinessModel(
        model=model,
        device=resolved_device,
        img_size=int(checkpoint.get("img_size", DEFAULT_IMG_SIZE)),
        threshold=float(checkpoint.get("threshold", 0.5)),
    )


@torch.inference_mode()
def predict_drowsiness_probability(
    loaded: LoadedDrowsinessModel,
    face_bgr: np.ndarray,
) -> float:
    """Return sigmoid probability that a face crop is drowsy."""

    tensor = preprocess_face_crop(face_bgr, loaded.img_size).unsqueeze(0).to(loaded.device)
    logit = loaded.model(tensor)
    return float(torch.sigmoid(logit).item())
