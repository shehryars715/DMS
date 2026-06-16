"""Kaggle-compatible CNN training file for the DMS drowsiness classifier.

How to use on Kaggle:
1. Create a Kaggle Notebook.
2. Turn on GPU and Internet in notebook settings.
3. Paste this file into a cell or upload it as a script and run it.
4. Keep DATA_SOURCE = "kagglehub" for the default DDD Kaggle dataset, or switch
   to "kaggle_input", "hf_snapshot", or "hf_datasets" below.
5. Download /kaggle/working/dms_cnn_model_bundle.zip from the Kaggle output pane.

The default split is grouped by filename/session prefix to reduce video-frame
leakage. An image-level split can produce very high validation scores on DDD,
but those numbers are usually inflated by near-duplicate frames.

For grouped runs, AUTO_THRESHOLD defaults to off because threshold tuning on a
small number of validation groups can overfit badly. Use threshold_sweep.csv as
a diagnostic artifact instead of trusting one extreme validation threshold.

Default dataset:
    https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd

Expected labels:
    Drowsy -> 1
    Non Drowsy / Awake / Alert / Normal -> 0
"""

# %% Configuration

from __future__ import annotations

import importlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


# Choose one:
# - "kagglehub": download a Kaggle dataset by slug.
# - "kaggle_input": use a dataset already added to the Kaggle Notebook inputs.
# - "hf_snapshot": download a Hugging Face dataset repo snapshot.
# - "hf_datasets": load a Hugging Face dataset through the datasets library.
DATA_SOURCE = os.environ.get("DATA_SOURCE", "kagglehub")

# Kaggle dataset download. This is the Driver Drowsiness Dataset (DDD).
KAGGLE_DATASET_SLUG = os.environ.get(
    "KAGGLE_DATASET_SLUG",
    "ismailnasri20/driver-drowsiness-dataset-ddd",
)

# If you add the dataset manually with Kaggle "+ Add Input", set:
# DATA_SOURCE = "kaggle_input"
# KAGGLE_INPUT_DIR = "/kaggle/input/driver-drowsiness-dataset-ddd"
KAGGLE_INPUT_DIR = os.environ.get(
    "KAGGLE_INPUT_DIR",
    "/kaggle/input/driver-drowsiness-dataset-ddd",
)

# Hugging Face options. For public repos this can stay token-free.
# Example:
# DATA_SOURCE = "hf_snapshot"
# HF_DATASET_ID = "username/dataset-name"
HF_DATASET_ID = os.environ.get("HF_DATASET_ID", "")
HF_DATASET_CONFIG = os.environ.get("HF_DATASET_CONFIG", "")
HF_DATASET_SPLIT = os.environ.get("HF_DATASET_SPLIT", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# If an HF dataset has numeric labels and no ClassLabel names, these are used.
HF_LABEL_NAMES = ["Non Drowsy", "Drowsy"]

# Training settings.
USE_MEDIAPIPE_FEATURES = os.environ.get("USE_MEDIAPIPE_FEATURES", "1").lower() in {"1", "true", "yes", "y"}
EPOCHS = int(os.environ.get("EPOCHS", "12"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "0.001"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "0.001"))
IMG_SIZE = int(os.environ.get("IMG_SIZE", "96"))
DROPOUT = float(os.environ.get("DROPOUT", "0.4"))
SEED = int(os.environ.get("SEED", "42"))
VAL_SIZE = float(os.environ.get("VAL_SIZE", "0.20"))
TEST_SIZE = float(os.environ.get("TEST_SIZE", "0.20"))
THRESHOLD = float(os.environ.get("THRESHOLD", "0.5"))
AUTO_THRESHOLD = os.environ.get("AUTO_THRESHOLD", "0").lower() in {"1", "true", "yes", "y"}
USE_CLASS_WEIGHTS = os.environ.get("USE_CLASS_WEIGHTS", "1").lower() in {"1", "true", "yes", "y"}
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "2"))
SPLIT_MODE = os.environ.get("SPLIT_MODE", "group")  # "group" or leaky/demo-only "image"
ALLOW_IMAGE_FALLBACK = os.environ.get("ALLOW_IMAGE_FALLBACK", "0").lower() in {"1", "true", "yes", "y"}
GROUP_SPLIT_TRIALS = int(os.environ.get("GROUP_SPLIT_TRIALS", "5000"))
MAX_IMAGES_PER_GROUP_CLASS = int(os.environ.get("MAX_IMAGES_PER_GROUP_CLASS", "0"))

# Artifact locations.
WORK_DIR = Path(os.environ.get("WORK_DIR", "/kaggle/working"))
DATA_WORK_DIR = WORK_DIR / "dms_dataset"
ARTIFACT_DIR = WORK_DIR / "dms_cnn_artifacts"
LANDMARK_FEATURE_CACHE = ARTIFACT_DIR / "landmark_features.csv"
MODEL_PATH = ARTIFACT_DIR / "ddd_cnn.pt"
SCRIPTED_MODEL_PATH = ARTIFACT_DIR / "ddd_cnn_torchscript.pt"
METRICS_PATH = ARTIFACT_DIR / "ddd_metrics.json"
HISTORY_CSV_PATH = ARTIFACT_DIR / "training_history.csv"
VAL_PREDICTIONS_PATH = ARTIFACT_DIR / "validation_predictions.csv"
TEST_PREDICTIONS_PATH = ARTIFACT_DIR / "test_predictions.csv"
THRESHOLD_SWEEP_PATH = ARTIFACT_DIR / "threshold_sweep.csv"
GROUP_METRICS_PATH = ARTIFACT_DIR / "group_metrics.csv"
CURVES_PATH = ARTIFACT_DIR / "training_curves.png"
CONFUSION_MATRIX_PATH = ARTIFACT_DIR / "confusion_matrix.png"
BUNDLE_BASE_PATH = WORK_DIR / "dms_cnn_model_bundle"


# %% Dependency setup

def ensure_packages() -> None:
    """Install only the small packages that are missing in the Kaggle runtime."""

    required_imports = {
        "cv2": "opencv-python-headless",
        "matplotlib": "matplotlib",
        "mediapipe": "mediapipe",
        "numpy": "numpy",
        "pandas": "pandas",
        "PIL": "Pillow",
        "sklearn": "scikit-learn",
        "torch": "torch",
        "tqdm": "tqdm",
    }
    source_imports = {
        "kagglehub": "kagglehub",
        "huggingface_hub": "huggingface-hub",
        "datasets": "datasets",
    }

    needed = dict(required_imports)
    if not USE_MEDIAPIPE_FEATURES:
        needed.pop("mediapipe", None)
    if DATA_SOURCE.lower() == "kagglehub":
        needed["kagglehub"] = source_imports["kagglehub"]
    if DATA_SOURCE.lower() in {"hf_snapshot", "hf_datasets"}:
        needed["huggingface_hub"] = source_imports["huggingface_hub"]
    if DATA_SOURCE.lower() == "hf_datasets":
        needed["datasets"] = source_imports["datasets"]

    missing: list[str] = []
    for import_name, package_name in needed.items():
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing.append(package_name)

    if not missing:
        return

    print("Installing missing packages:", ", ".join(sorted(set(missing))))
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", *sorted(set(missing))]
    )


ensure_packages()

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


# %% Reproducibility and constants

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
POSITIVE_CLASS = "Drowsy"
NEGATIVE_CLASS = "Non Drowsy"
CLASS_NAMES = [NEGATIVE_CLASS, POSITIVE_CLASS]
CLASS_TO_INDEX = {NEGATIVE_CLASS: 0, POSITIVE_CLASS: 1}
DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)
LEFT_EYE_EAR = (33, 160, 158, 133, 153, 144)
RIGHT_EYE_EAR = (362, 385, 387, 263, 373, 380)
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
MOUTH_VERTICAL_PAIRS = ((13, 14), (81, 178), (311, 402))
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
        (0.0, 0.0, 0.0),
        (0.0, -63.6, -12.5),
        (-43.3, 32.7, -26.0),
        (43.3, 32.7, -26.0),
        (-28.9, -28.9, -24.1),
        (28.9, -28.9, -24.1),
    ],
    dtype=np.float64,
)
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


set_seed(SEED)


# %% MediaPipe landmark features

def euclidean(point_a: np.ndarray, point_b: np.ndarray) -> float:
    return float(np.linalg.norm(point_a - point_b))


def eye_aspect_ratio(points: np.ndarray, indices: tuple[int, ...]) -> float:
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


def normalized_to_pixel_points(face_landmarks: object, width: int, height: int) -> np.ndarray:
    landmarks = face_landmarks.landmark if hasattr(face_landmarks, "landmark") else face_landmarks
    points = []
    for landmark in landmarks:
        x = min(max(float(landmark.x) * width, 0.0), width - 1.0)
        y = min(max(float(landmark.y) * height, 0.0), height - 1.0)
        points.append((x, y))
    return np.asarray(points, dtype=np.float64)


def face_bbox_from_points(points: np.ndarray) -> tuple[float, float, float, float]:
    min_xy = points.min(axis=0)
    max_xy = points.max(axis=0)
    return float(min_xy[0]), float(min_xy[1]), float(max_xy[0]), float(max_xy[1])


def estimate_head_pose_angles(points: np.ndarray, frame_size: tuple[int, int]) -> tuple[float, float, float]:
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
        return 0.0, 0.0, 0.0

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    projection_matrix = np.hstack((rotation_matrix, translation_vector))
    _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(projection_matrix)
    return float(euler[0, 0]), float(euler[1, 0]), float(euler[2, 0])


def landmark_feature_vector_from_points(points: np.ndarray, frame_size: tuple[int, int]) -> np.ndarray:
    left_ear = eye_aspect_ratio(points, LEFT_EYE_EAR)
    right_ear = eye_aspect_ratio(points, RIGHT_EYE_EAR)
    ear = (left_ear + right_ear) / 2.0
    mar = mouth_aspect_ratio(points)
    pitch, yaw, roll = estimate_head_pose_angles(points, frame_size)
    x1, y1, x2, y2 = face_bbox_from_points(points)
    face_w = max(x2 - x1, 1.0)
    face_h = max(y2 - y1, 1.0)
    face_scale = max(face_w, face_h, 1.0)
    eye_distance_norm = euclidean(points[33], points[263]) / face_scale
    mouth_width_norm = euclidean(points[MOUTH_LEFT], points[MOUTH_RIGHT]) / face_scale
    face_aspect = face_w / face_h
    return np.asarray(
        [
            1.0,
            left_ear,
            right_ear,
            ear,
            mar,
            pitch,
            yaw,
            roll,
            eye_distance_norm,
            mouth_width_norm,
            face_aspect,
        ],
        dtype=np.float32,
    )


def zero_landmark_features() -> np.ndarray:
    return np.zeros(LANDMARK_FEATURE_DIM, dtype=np.float32)


def extract_landmark_features_for_image(path: Path, face_mesh: Any) -> np.ndarray:
    image_bgr = cv2.imread(str(path))
    if image_bgr is None:
        return zero_landmark_features()

    height, width = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return zero_landmark_features()

    points = normalized_to_pixel_points(results.multi_face_landmarks[0], width, height)
    return landmark_feature_vector_from_points(points, (height, width))


def load_or_extract_landmark_features(paths: list[Path]) -> pd.DataFrame:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    path_strings = [str(path) for path in paths]
    required_paths = set(path_strings)

    if LANDMARK_FEATURE_CACHE.exists():
        cached = pd.read_csv(LANDMARK_FEATURE_CACHE)
        if "path" in cached.columns and required_paths.issubset(set(cached["path"].astype(str))):
            print(f"Using cached MediaPipe features: {LANDMARK_FEATURE_CACHE}")
            return cached[cached["path"].astype(str).isin(required_paths)].copy()

    import mediapipe as mp

    print(f"Extracting MediaPipe features for {len(paths)} images...")
    rows: list[dict[str, float | str]] = []
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )
    try:
        for path in tqdm(paths, desc="MediaPipe features"):
            features = extract_landmark_features_for_image(path, face_mesh)
            row: dict[str, float | str] = {"path": str(path)}
            row.update({name: float(value) for name, value in zip(LANDMARK_FEATURE_NAMES, features)})
            rows.append(row)
    finally:
        face_mesh.close()

    frame = pd.DataFrame(rows)
    frame.to_csv(LANDMARK_FEATURE_CACHE, index=False)
    detected_rate = float(frame["landmarks_detected"].mean()) if not frame.empty else 0.0
    print(f"Saved MediaPipe feature cache to {LANDMARK_FEATURE_CACHE}")
    print(f"MediaPipe detection rate: {detected_rate:.2%}")
    return frame


def features_for_paths(feature_frame: pd.DataFrame, paths: list[Path]) -> np.ndarray:
    indexed = feature_frame.set_index("path")
    rows = []
    for path in paths:
        key = str(path)
        if key in indexed.index:
            rows.append(indexed.loc[key, LANDMARK_FEATURE_NAMES].to_numpy(dtype=np.float32))
        else:
            rows.append(zero_landmark_features())
    return np.vstack(rows).astype(np.float32)


def fit_feature_scaler(train_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_features.mean(axis=0).astype(np.float32)
    std = train_features.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def standardize_features(
    features: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    return ((features - mean) / std).astype(np.float32)


# %% Dataset download and preparation

def download_with_kagglehub() -> Path:
    import kagglehub

    print(f"Downloading Kaggle dataset: {KAGGLE_DATASET_SLUG}")
    path = Path(kagglehub.dataset_download(KAGGLE_DATASET_SLUG))
    print(f"Kaggle dataset path: {path}")
    return path


def use_kaggle_input() -> Path:
    path = Path(KAGGLE_INPUT_DIR)
    if not path.exists():
        raise FileNotFoundError(
            f"Kaggle input path was not found: {path}. "
            "Use Kaggle '+ Add Input' or switch DATA_SOURCE to 'kagglehub'."
        )
    print(f"Using Kaggle input path: {path}")
    return path


def download_hf_snapshot() -> Path:
    if not HF_DATASET_ID:
        raise ValueError("Set HF_DATASET_ID before using DATA_SOURCE='hf_snapshot'.")

    from huggingface_hub import snapshot_download

    print(f"Downloading Hugging Face dataset snapshot: {HF_DATASET_ID}")
    kwargs: dict[str, Any] = {
        "repo_id": HF_DATASET_ID,
        "repo_type": "dataset",
        "local_dir": str(DATA_WORK_DIR / "hf_snapshot"),
    }
    if HF_TOKEN:
        kwargs["token"] = HF_TOKEN
    path = Path(snapshot_download(**kwargs))
    print(f"Hugging Face snapshot path: {path}")
    return path


def normalize_label_name(value: str) -> str:
    value = str(value).lower()
    value = value.replace("_", " ").replace("-", " ")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def canonical_label_name(value: str | int) -> str | None:
    normalized = normalize_label_name(str(value))
    negative_names = {
        "0",
        "non drowsy",
        "nondrowsy",
        "not drowsy",
        "awake",
        "alert",
        "normal",
        "open",
        "non sleepy",
    }
    positive_names = {
        "1",
        "drowsy",
        "sleepy",
        "closed",
        "fatigue",
        "fatigued",
    }
    if normalized in negative_names:
        return NEGATIVE_CLASS
    if normalized in positive_names:
        return POSITIVE_CLASS
    return None


def label_from_path(path: Path, data_dir: Path) -> str | None:
    for part in path.relative_to(data_dir).parts[:-1]:
        label_name = canonical_label_name(part)
        if label_name is not None:
            return label_name
    return None


def discover_images(data_dir: Path) -> tuple[list[Path], list[int]]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {data_dir}")

    paths: list[Path] = []
    labels: list[int] = []
    skipped = 0

    for path in sorted(data_dir.rglob("*")):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label_name = label_from_path(path, data_dir)
        if label_name is None:
            skipped += 1
            continue
        paths.append(path)
        labels.append(CLASS_TO_INDEX[label_name])

    counts = {
        NEGATIVE_CLASS: int(labels.count(0)),
        POSITIVE_CLASS: int(labels.count(1)),
    }
    print(f"Discovered {len(paths)} labeled images in {data_dir}")
    print("Class counts:", counts)
    if skipped:
        print(f"Skipped {skipped} images whose folder names did not reveal a class.")
    if not paths or min(counts.values()) == 0:
        raise RuntimeError(
            "Both classes are required. Expected folders named like "
            "'Drowsy' and 'Non Drowsy', 'Awake', 'Alert', or 'Normal'."
        )
    return paths, labels


def score_dataset_root(path: Path) -> int:
    """Return how many classes can be inferred from image paths under path."""

    try:
        _, labels = discover_images(path)
    except Exception:
        return 0
    return len(set(labels))


def find_best_dataset_root(root: Path) -> Path:
    """Pick a dataset root that exposes both binary classes."""

    candidates = [root]
    for child in root.iterdir() if root.exists() else []:
        if child.is_dir():
            candidates.append(child)

    best_path = root
    best_score = -1
    for candidate in candidates:
        try:
            paths, labels = discover_images(candidate)
        except Exception:
            continue
        score = len(set(labels)) * 1_000_000 + len(paths)
        if score > best_score:
            best_path = candidate
            best_score = score

    if best_score < 0:
        raise RuntimeError(f"Could not find labeled image folders under {root}")
    print(f"Selected dataset root: {best_path}")
    return best_path


def export_hf_datasets_to_image_folders() -> Path:
    if not HF_DATASET_ID:
        raise ValueError("Set HF_DATASET_ID before using DATA_SOURCE='hf_datasets'.")

    from datasets import ClassLabel, Image as HFImage, load_dataset

    print(f"Loading Hugging Face dataset with datasets: {HF_DATASET_ID}")
    config = HF_DATASET_CONFIG or None
    if HF_DATASET_SPLIT:
        raw = load_dataset(HF_DATASET_ID, config, split=HF_DATASET_SPLIT)
        raw_splits = {HF_DATASET_SPLIT: raw}
    else:
        raw_splits = load_dataset(HF_DATASET_ID, config)

    if not hasattr(raw_splits, "items"):
        raw_splits = {"train": raw_splits}

    first_split = next(iter(raw_splits.values()))
    features = first_split.features

    image_col = None
    label_col = None
    for name, feature in features.items():
        if isinstance(feature, HFImage) or name.lower() in {"image", "img", "picture"}:
            image_col = name
            break
    for name, feature in features.items():
        if isinstance(feature, ClassLabel) or name.lower() in {"label", "class", "target"}:
            label_col = name
            break

    if image_col is None or label_col is None:
        raise RuntimeError(
            f"Could not infer image and label columns. Features: {features}"
        )

    label_feature = features[label_col]
    out_dir = DATA_WORK_DIR / "hf_datasets_export"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def to_label_name(value: Any) -> str:
        if isinstance(label_feature, ClassLabel):
            raw_name = label_feature.int2str(int(value))
        elif isinstance(value, (int, np.integer)) and 0 <= int(value) < len(HF_LABEL_NAMES):
            raw_name = HF_LABEL_NAMES[int(value)]
        else:
            raw_name = str(value)

        canonical = canonical_label_name(raw_name)
        if canonical is None:
            raise ValueError(
                f"Could not map HF label '{raw_name}' to Drowsy/Non Drowsy. "
                "Edit HF_LABEL_NAMES or canonical_label_name()."
            )
        return canonical

    saved = 0
    for split_name, dataset in raw_splits.items():
        print(f"Exporting split '{split_name}' with {len(dataset)} rows")
        for idx, row in enumerate(tqdm(dataset, desc=f"HF {split_name}")):
            label_name = to_label_name(row[label_col])
            class_dir = out_dir / label_name
            class_dir.mkdir(parents=True, exist_ok=True)

            image_value = row[image_col]
            if isinstance(image_value, Image.Image):
                image = image_value.convert("RGB")
            elif isinstance(image_value, dict) and image_value.get("path"):
                image = Image.open(image_value["path"]).convert("RGB")
            elif isinstance(image_value, (str, Path)):
                image = Image.open(image_value).convert("RGB")
            else:
                image = Image.fromarray(np.asarray(image_value)).convert("RGB")

            image.save(class_dir / f"{split_name}_{idx:07d}.jpg", quality=95)
            saved += 1

    print(f"Exported {saved} images to {out_dir}")
    return out_dir


def prepare_dataset() -> Path:
    source = DATA_SOURCE.lower().strip()
    if source == "kagglehub":
        raw_root = download_with_kagglehub()
    elif source == "kaggle_input":
        raw_root = use_kaggle_input()
    elif source == "hf_snapshot":
        raw_root = download_hf_snapshot()
    elif source == "hf_datasets":
        raw_root = export_hf_datasets_to_image_folders()
    else:
        raise ValueError(
            "DATA_SOURCE must be one of: kagglehub, kaggle_input, "
            "hf_snapshot, hf_datasets."
        )
    return find_best_dataset_root(raw_root)


# %% Splitting

def group_id_from_path(path: Path) -> str:
    """Infer coarse DDD sequence/session groups from filename prefixes."""

    match = re.match(r"([A-Za-z]+)", path.stem)
    return match.group(1).lower() if match else path.stem.lower()


def maybe_limit_images_per_group_class(
    paths: list[Path],
    labels: list[int],
) -> tuple[list[Path], list[int]]:
    """Optionally reduce near-duplicate dominance within each group/class bucket."""

    if MAX_IMAGES_PER_GROUP_CLASS <= 0:
        return paths, labels

    rng = random.Random(SEED)
    buckets: dict[tuple[str, int], list[Path]] = {}
    for path, label in zip(paths, labels):
        buckets.setdefault((group_id_from_path(path), label), []).append(path)

    kept_pairs: list[tuple[Path, int]] = []
    for (group, label), bucket_paths in sorted(buckets.items()):
        bucket_paths = sorted(bucket_paths)
        if len(bucket_paths) > MAX_IMAGES_PER_GROUP_CLASS:
            bucket_paths = sorted(rng.sample(bucket_paths, MAX_IMAGES_PER_GROUP_CLASS))
        kept_pairs.extend((path, label) for path in bucket_paths)

    kept_pairs.sort(key=lambda pair: str(pair[0]))
    limited_paths = [path for path, _ in kept_pairs]
    limited_labels = [label for _, label in kept_pairs]
    print(
        f"Limited images per group/class to {MAX_IMAGES_PER_GROUP_CLASS}: "
        f"{len(paths)} -> {len(limited_paths)}"
    )
    return limited_paths, limited_labels


def stratified_image_split(
    paths: list[Path],
    labels: list[int],
) -> tuple[list[Path], list[int], list[Path], list[int], list[Path], list[int]]:
    train_size = 1.0 - VAL_SIZE - TEST_SIZE
    if train_size <= 0:
        raise ValueError("VAL_SIZE + TEST_SIZE must be less than 1.0")

    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths,
        labels,
        test_size=VAL_SIZE + TEST_SIZE,
        stratify=labels,
        random_state=SEED,
    )
    relative_test = TEST_SIZE / (VAL_SIZE + TEST_SIZE)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths,
        temp_labels,
        test_size=relative_test,
        stratify=temp_labels,
        random_state=SEED,
    )
    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels


def grouped_split(
    paths: list[Path],
    labels: list[int],
) -> tuple[list[Path], list[int], list[Path], list[int], list[Path], list[int], dict[str, list[str]]]:
    train_size = 1.0 - VAL_SIZE - TEST_SIZE
    if train_size <= 0:
        raise ValueError("VAL_SIZE + TEST_SIZE must be less than 1.0")

    group_stats: dict[str, np.ndarray] = {}
    for path, label in zip(paths, labels):
        group = group_id_from_path(path)
        group_stats.setdefault(group, np.zeros(2, dtype=np.int64))
        group_stats[group][label] += 1

    groups = sorted(group_stats)
    if len(groups) < 3:
        raise RuntimeError("Grouped split needs at least three filename groups.")

    totals = np.array([labels.count(0), labels.count(1)], dtype=np.float64)
    total_count = float(totals.sum())
    overall_positive_rate = totals[1] / total_count
    target_fracs = np.array([train_size, VAL_SIZE, TEST_SIZE], dtype=np.float64)
    assignment_probs = target_fracs / target_fracs.sum()
    rng = np.random.default_rng(SEED)

    best_assignment: dict[str, int] | None = None
    best_cost = float("inf")

    for _ in range(max(1, GROUP_SPLIT_TRIALS)):
        assignment = {group: int(rng.choice(3, p=assignment_probs)) for group in groups}
        split_counts = np.zeros((3, 2), dtype=np.float64)
        split_group_counts = np.zeros(3, dtype=np.int64)
        for group, split_id in assignment.items():
            split_counts[split_id] += group_stats[group]
            split_group_counts[split_id] += 1

        if np.any(split_group_counts == 0) or np.any(split_counts == 0):
            continue

        split_totals = split_counts.sum(axis=1)
        split_fracs = split_totals / total_count
        positive_rates = split_counts[:, 1] / split_totals
        size_cost = float(np.sum((split_fracs - target_fracs) ** 2))
        class_cost = float(np.sum((positive_rates - overall_positive_rate) ** 2))
        cost = size_cost + 0.25 * class_cost

        if cost < best_cost:
            best_cost = cost
            best_assignment = assignment

    if best_assignment is None:
        raise RuntimeError("Could not create a valid grouped split.")

    split_paths: list[list[Path]] = [[], [], []]
    split_labels: list[list[int]] = [[], [], []]
    for path, label in zip(paths, labels):
        split_id = best_assignment[group_id_from_path(path)]
        split_paths[split_id].append(path)
        split_labels[split_id].append(label)

    split_groups = {
        "train": sorted(group for group, split_id in best_assignment.items() if split_id == 0),
        "val": sorted(group for group, split_id in best_assignment.items() if split_id == 1),
        "test": sorted(group for group, split_id in best_assignment.items() if split_id == 2),
    }
    return (
        split_paths[0],
        split_labels[0],
        split_paths[1],
        split_labels[1],
        split_paths[2],
        split_labels[2],
        split_groups,
    )


def make_splits(
    paths: list[Path],
    labels: list[int],
) -> tuple[list[Path], list[int], list[Path], list[int], list[Path], list[int], dict[str, Any]]:
    if SPLIT_MODE.lower() == "group":
        try:
            split = grouped_split(paths, labels)
            train_paths, train_labels, val_paths, val_labels, test_paths, test_labels, split_groups = split
            split_info = {"mode": "group", "groups": split_groups}
        except Exception as exc:
            if not ALLOW_IMAGE_FALLBACK:
                raise RuntimeError(
                    f"Grouped split failed: {exc}. This dataset may not expose "
                    "session-like filename groups. Set ALLOW_IMAGE_FALLBACK=1 "
                    "only if you accept image-level split leakage risk."
                ) from exc
            print(
                f"WARNING: grouped split failed ({exc}); falling back to "
                "stratified image split. Metrics may be inflated by near-duplicate leakage."
            )
            train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = stratified_image_split(
                paths,
                labels,
            )
            split_info = {"mode": "image_fallback", "groups": None}
    else:
        print(
            "WARNING: using image-level split. On frame-based drowsiness datasets "
            "this can leak near-duplicate frames across train/val/test."
        )
        train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = stratified_image_split(
            paths,
            labels,
        )
        split_info = {"mode": "image", "groups": None}

    print(
        "Split sizes:",
        {
            "train": len(train_paths),
            "val": len(val_paths),
            "test": len(test_paths),
        },
    )
    print(
        "Split class counts:",
        {
            "train": {NEGATIVE_CLASS: train_labels.count(0), POSITIVE_CLASS: train_labels.count(1)},
            "val": {NEGATIVE_CLASS: val_labels.count(0), POSITIVE_CLASS: val_labels.count(1)},
            "test": {NEGATIVE_CLASS: test_labels.count(0), POSITIVE_CLASS: test_labels.count(1)},
        },
    )
    if split_info.get("groups"):
        print(
            "Split group counts:",
            {name: len(groups) for name, groups in split_info["groups"].items()},
        )
    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels, split_info


# %% Model and dataloaders

class DrowsinessCNN(nn.Module):
    """Small binary CNN matching the local DMS project checkpoint style."""

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
    """CNN image branch plus a small MediaPipe geometry branch."""

    def __init__(self, feature_dim: int, dropout: float = 0.25) -> None:
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


class ImageFolderBinaryDataset(Dataset):
    def __init__(
        self,
        paths: list[Path],
        labels: list[int],
        img_size: int,
        augment: bool,
        landmark_features: np.ndarray | None = None,
    ) -> None:
        self.paths = paths
        self.labels = labels
        self.img_size = img_size
        self.augment = augment
        self.landmark_features = landmark_features
        self.mean = np.asarray(DEFAULT_MEAN, dtype=np.float32)
        self.std = np.asarray(DEFAULT_STD, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        path = self.paths[index]
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"Failed to read image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.augment:
            image = self._augment(image)

        image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        image = np.transpose(image, (2, 0, 1))
        label = np.float32(self.labels[index])
        image_tensor = torch.from_numpy(image).float()
        label_tensor = torch.tensor(label, dtype=torch.float32)
        if self.landmark_features is None:
            return image_tensor, label_tensor
        feature_tensor = torch.from_numpy(self.landmark_features[index]).float()
        return image_tensor, feature_tensor, label_tensor

    @staticmethod
    def _augment(image: np.ndarray) -> np.ndarray:
        if random.random() < 0.5:
            image = cv2.flip(image, 1)

        height, width = image.shape[:2]
        if random.random() < 0.65:
            crop_scale = random.uniform(0.84, 1.0)
            crop_h = max(8, int(height * crop_scale))
            crop_w = max(8, int(width * crop_scale))
            top = random.randint(0, max(height - crop_h, 0))
            left = random.randint(0, max(width - crop_w, 0))
            image = image[top : top + crop_h, left : left + crop_w]
            height, width = image.shape[:2]

        if random.random() < 0.75:
            angle = random.uniform(-10.0, 10.0)
            matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
            image = cv2.warpAffine(
                image,
                matrix,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )

        if random.random() < 0.75:
            factor = random.uniform(0.75, 1.25)
            image = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        if random.random() < 0.45:
            contrast = random.uniform(0.75, 1.35)
            brightness = random.uniform(-18.0, 18.0)
            image = np.clip(image.astype(np.float32) * contrast + brightness, 0, 255).astype(np.uint8)

        if random.random() < 0.25:
            kernel_size = random.choice([3, 5])
            image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

        if random.random() < 0.20:
            noise = np.random.normal(0.0, random.uniform(2.0, 8.0), image.shape)
            image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        return image


@dataclass(frozen=True)
class RunConfig:
    data_source: str
    use_mediapipe_features: bool
    kaggle_dataset_slug: str
    kaggle_input_dir: str
    hf_dataset_id: str
    hf_dataset_config: str
    hf_dataset_split: str
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    img_size: int
    dropout: float
    seed: int
    val_size: float
    test_size: float
    threshold: float
    auto_threshold: bool
    use_class_weights: bool
    split_mode: str
    allow_image_fallback: bool
    max_images_per_group_class: int


def make_loader(
    paths: list[Path],
    labels: list[int],
    augment: bool,
    shuffle: bool,
    landmark_features: np.ndarray | None = None,
) -> DataLoader:
    dataset = ImageFolderBinaryDataset(
        paths,
        labels,
        IMG_SIZE,
        augment=augment,
        landmark_features=landmark_features,
    )
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=NUM_WORKERS > 0,
    )


# %% Training and evaluation

def compute_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, Any]:
    preds = (probs >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": confusion_matrix(labels, preds, labels=[0, 1]).tolist(),
        "classes": CLASS_NAMES,
        "threshold": threshold,
    }


def find_best_threshold(labels: np.ndarray, probs: np.ndarray) -> tuple[float, dict[str, Any]]:
    """Choose the validation threshold that maximizes binary F1."""

    best_threshold = THRESHOLD
    best_metrics = compute_metrics(labels, probs, THRESHOLD)
    best_score = (float(best_metrics["f1"]), float(best_metrics["accuracy"]))

    for threshold in np.linspace(0.05, 0.95, 181):
        metrics = compute_metrics(labels, probs, float(threshold))
        score = (float(metrics["f1"]), float(metrics["accuracy"]))
        if score > best_score:
            best_threshold = float(threshold)
            best_metrics = metrics
            best_score = score

    return best_threshold, best_metrics


def make_grad_scaler(enabled: bool) -> Any:
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(device: torch.device, enabled: bool) -> Any:
    try:
        return torch.amp.autocast(device_type=device.type, enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(enabled=enabled)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    all_probs: list[float] = []
    all_labels: list[int] = []
    use_amp = device.type == "cuda"

    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch in tqdm(loader, leave=False):
            if len(batch) == 3:
                images, landmark_features, labels = batch
                landmark_features = landmark_features.to(device, non_blocking=True)
            else:
                images, labels = batch
                landmark_features = None
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            with autocast_context(device, enabled=use_amp):
                if landmark_features is None:
                    logits = model(images)
                else:
                    logits = model(images, landmark_features)
                loss = criterion(logits, labels)

            if training:
                if scaler is not None and use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item() * images.size(0)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.detach().cpu().numpy().astype(int).tolist())

    mean_loss = total_loss / max(len(loader.dataset), 1)
    return mean_loss, np.asarray(all_probs), np.asarray(all_labels)


def save_checkpoint(
    model: nn.Module,
    best_val_f1: float,
    split_info: dict[str, Any],
    selected_threshold: float,
    best_epoch: int,
    pos_weight_value: float,
    model_kind: str,
    feature_mean: np.ndarray | None,
    feature_std: np.ndarray | None,
) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    run_config = RunConfig(
        data_source=DATA_SOURCE,
        use_mediapipe_features=USE_MEDIAPIPE_FEATURES,
        kaggle_dataset_slug=KAGGLE_DATASET_SLUG,
        kaggle_input_dir=KAGGLE_INPUT_DIR,
        hf_dataset_id=HF_DATASET_ID,
        hf_dataset_config=HF_DATASET_CONFIG,
        hf_dataset_split=HF_DATASET_SPLIT,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        img_size=IMG_SIZE,
        dropout=DROPOUT,
        seed=SEED,
        val_size=VAL_SIZE,
        test_size=TEST_SIZE,
        threshold=THRESHOLD,
        auto_threshold=AUTO_THRESHOLD,
        use_class_weights=USE_CLASS_WEIGHTS,
        split_mode=SPLIT_MODE,
        allow_image_fallback=ALLOW_IMAGE_FALLBACK,
        max_images_per_group_class=MAX_IMAGES_PER_GROUP_CLASS,
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_kind": model_kind,
            "img_size": IMG_SIZE,
            "threshold": float(selected_threshold),
            "dropout": DROPOUT,
            "classes": CLASS_NAMES,
            "landmark_feature_names": LANDMARK_FEATURE_NAMES if USE_MEDIAPIPE_FEATURES else [],
            "landmark_feature_mean": feature_mean.tolist() if feature_mean is not None else None,
            "landmark_feature_std": feature_std.tolist() if feature_std is not None else None,
            "config": asdict(run_config),
            "split_info": split_info,
            "best_val_f1": float(best_val_f1),
            "best_epoch": int(best_epoch),
            "pos_weight": float(pos_weight_value),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        MODEL_PATH,
    )


def plot_training_curves(history: list[dict[str, float]]) -> None:
    if not history:
        return

    frame = pd.DataFrame(history)
    frame.to_csv(HISTORY_CSV_PATH, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=150)
    axes[0].plot(frame["epoch"], frame["train_loss"], label="Train")
    axes[0].plot(frame["epoch"], frame["val_loss"], label="Validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(frame["epoch"], frame["train_f1"], label="Train F1")
    axes[1].plot(frame["epoch"], frame["val_f1"], label="Validation F1")
    axes[1].plot(frame["epoch"], frame["val_accuracy"], label="Validation accuracy")
    axes[1].set_title("Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(CURVES_PATH)
    plt.close(fig)


def plot_confusion_matrix(matrix: list[list[int]]) -> None:
    cm = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(4.5, 4), dpi=150)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Test confusion matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1], CLASS_NAMES, rotation=20, ha="right")
    ax.set_yticks([0, 1], CLASS_NAMES)

    max_value = cm.max() if cm.size else 0
    threshold = max_value / 2 if max_value else math.inf
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            color = "white" if cm[row, col] > threshold else "black"
            ax.text(col, row, str(cm[row, col]), ha="center", va="center", color=color)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(CONFUSION_MATRIX_PATH)
    plt.close(fig)


def flatten_metrics(metrics: dict[str, Any]) -> dict[str, float | int]:
    cm = np.asarray(metrics["confusion_matrix"], dtype=int)
    return {
        "accuracy": float(metrics["accuracy"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "threshold": float(metrics["threshold"]),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


def save_prediction_frame(
    split_name: str,
    paths: list[Path],
    labels: list[int],
    probs: np.ndarray,
    output_path: Path,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "split": split_name,
            "path": [str(path) for path in paths],
            "group": [group_id_from_path(path) for path in paths],
            "label": labels,
            "label_name": [CLASS_NAMES[label] for label in labels],
            "prob_drowsy": probs.astype(float),
        }
    )
    frame.to_csv(output_path, index=False)
    return frame


def save_threshold_sweep(
    val_labels: np.ndarray,
    val_probs: np.ndarray,
    test_labels: np.ndarray,
    test_probs: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for split_name, labels, probs in (
        ("validation", val_labels, val_probs),
        ("test", test_labels, test_probs),
    ):
        for threshold in np.linspace(0.05, 0.95, 181):
            metrics = compute_metrics(labels, probs, float(threshold))
            rows.append({"split": split_name, **flatten_metrics(metrics)})

    frame = pd.DataFrame(rows)
    frame.to_csv(THRESHOLD_SWEEP_PATH, index=False)
    return frame


def save_group_metrics(
    prediction_frames: list[pd.DataFrame],
    threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for frame in prediction_frames:
        split_name = str(frame["split"].iloc[0])
        for group, group_frame in frame.groupby("group"):
            labels = group_frame["label"].to_numpy(dtype=int)
            probs = group_frame["prob_drowsy"].to_numpy(dtype=float)
            metrics = compute_metrics(labels, probs, threshold)
            rows.append(
                {
                    "split": split_name,
                    "group": str(group),
                    "count": int(len(group_frame)),
                    "non_drowsy": int((labels == 0).sum()),
                    "drowsy": int((labels == 1).sum()),
                    **flatten_metrics(metrics),
                }
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(GROUP_METRICS_PATH, index=False)
    return frame


def train_model() -> dict[str, Any]:
    dataset_root = prepare_dataset()
    paths, labels = discover_images(dataset_root)
    paths, labels = maybe_limit_images_per_group_class(paths, labels)
    split = make_splits(paths, labels)
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels, split_info = split

    feature_mean: np.ndarray | None = None
    feature_std: np.ndarray | None = None
    train_features: np.ndarray | None = None
    val_features: np.ndarray | None = None
    test_features: np.ndarray | None = None
    model_kind = "cnn"
    landmark_detection_rate: float | None = None

    if USE_MEDIAPIPE_FEATURES:
        feature_frame = load_or_extract_landmark_features(paths)
        all_features = features_for_paths(feature_frame, paths)
        landmark_detection_rate = float(all_features[:, 0].mean()) if len(all_features) else 0.0
        raw_train_features = features_for_paths(feature_frame, train_paths)
        raw_val_features = features_for_paths(feature_frame, val_paths)
        raw_test_features = features_for_paths(feature_frame, test_paths)
        feature_mean, feature_std = fit_feature_scaler(raw_train_features)
        train_features = standardize_features(raw_train_features, feature_mean, feature_std)
        val_features = standardize_features(raw_val_features, feature_mean, feature_std)
        test_features = standardize_features(raw_test_features, feature_mean, feature_std)
        model_kind = "hybrid"
        print("Training hybrid CNN + MediaPipe feature model.")
        print(f"MediaPipe landmark detection rate: {landmark_detection_rate:.2%}")
    else:
        print("Training image-only CNN model.")

    train_loader = make_loader(
        train_paths,
        train_labels,
        augment=True,
        shuffle=True,
        landmark_features=train_features,
    )
    val_loader = make_loader(
        val_paths,
        val_labels,
        augment=False,
        shuffle=False,
        landmark_features=val_features,
    )
    test_loader = make_loader(
        test_paths,
        test_labels,
        augment=False,
        shuffle=False,
        landmark_features=test_features,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if model_kind == "hybrid":
        model = HybridDrowsinessCNN(LANDMARK_FEATURE_DIM, dropout=DROPOUT).to(device)
    else:
        model = DrowsinessCNN(dropout=DROPOUT).to(device)
    neg_count = max(train_labels.count(0), 1)
    pos_count = max(train_labels.count(1), 1)
    pos_weight_value = neg_count / pos_count
    if USE_CLASS_WEIGHTS:
        pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"Using class-weighted BCEWithLogitsLoss: pos_weight={pos_weight_value:.4f}")
    else:
        criterion = nn.BCEWithLogitsLoss()
        print("Using unweighted BCEWithLogitsLoss")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = make_grad_scaler(enabled=device.type == "cuda")

    best_val_f1 = -1.0
    best_threshold = THRESHOLD
    best_epoch = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        train_loss, train_probs, train_y = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
        )
        val_loss, val_probs, val_y = run_epoch(model, val_loader, criterion, device)
        if AUTO_THRESHOLD:
            val_threshold, val_metrics = find_best_threshold(val_y, val_probs)
        else:
            val_threshold = THRESHOLD
            val_metrics = compute_metrics(val_y, val_probs, THRESHOLD)
        train_metrics = compute_metrics(train_y, train_probs, val_threshold)

        row = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "train_f1": float(train_metrics["f1"]),
            "val_f1": float(val_metrics["f1"]),
            "val_accuracy": float(val_metrics["accuracy"]),
            "threshold": float(val_threshold),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)

        print(
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['f1']:.4f} "
            f"threshold={val_threshold:.3f}"
        )

        if float(val_metrics["f1"]) > best_val_f1:
            best_val_f1 = float(val_metrics["f1"])
            best_threshold = float(val_threshold)
            best_epoch = epoch
            save_checkpoint(
                model,
                best_val_f1,
                split_info,
                best_threshold,
                best_epoch,
                pos_weight_value,
                model_kind,
                feature_mean,
                feature_std,
            )
            print(f"Saved best checkpoint to {MODEL_PATH}")

        plot_training_curves(history)

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_loss, val_probs, val_y = run_epoch(model, val_loader, criterion, device)
    test_loss, test_probs, test_y = run_epoch(model, test_loader, criterion, device)
    checkpoint_threshold = float(checkpoint.get("threshold", best_threshold))
    val_metrics = compute_metrics(val_y, val_probs, checkpoint_threshold)
    test_metrics = compute_metrics(test_y, test_probs, checkpoint_threshold)
    fixed_val_metrics = compute_metrics(val_y, val_probs, THRESHOLD)
    fixed_test_metrics = compute_metrics(test_y, test_probs, THRESHOLD)

    val_prediction_frame = save_prediction_frame(
        "validation",
        val_paths,
        val_labels,
        val_probs,
        VAL_PREDICTIONS_PATH,
    )
    test_prediction_frame = save_prediction_frame(
        "test",
        test_paths,
        test_labels,
        test_probs,
        TEST_PREDICTIONS_PATH,
    )
    save_threshold_sweep(val_y, val_probs, test_y, test_probs)
    save_group_metrics([val_prediction_frame, test_prediction_frame], checkpoint_threshold)

    example = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)
    if model_kind == "hybrid":
        example_features = torch.zeros(1, LANDMARK_FEATURE_DIM, device=device)
        scripted = torch.jit.trace(model.eval(), (example, example_features))
    else:
        scripted = torch.jit.trace(model.eval(), example)
    scripted.save(str(SCRIPTED_MODEL_PATH))

    results = {
        "dataset_root": str(dataset_root),
        "model_kind": model_kind,
        "use_mediapipe_features": USE_MEDIAPIPE_FEATURES,
        "landmark_feature_names": LANDMARK_FEATURE_NAMES if USE_MEDIAPIPE_FEATURES else [],
        "landmark_feature_cache": str(LANDMARK_FEATURE_CACHE) if USE_MEDIAPIPE_FEATURES else None,
        "landmark_detection_rate": landmark_detection_rate,
        "val_loss": float(val_loss),
        "test_loss": float(test_loss),
        "validation": val_metrics,
        "test": test_metrics,
        "fixed_threshold": THRESHOLD,
        "fixed_threshold_validation": fixed_val_metrics,
        "fixed_threshold_test": fixed_test_metrics,
        "best_val_f1": float(best_val_f1),
        "best_epoch": int(best_epoch),
        "best_threshold": float(checkpoint_threshold),
        "pos_weight": float(pos_weight_value),
        "history": history,
        "split_sizes": {
            "train": len(train_paths),
            "val": len(val_paths),
            "test": len(test_paths),
        },
        "split_info": split_info,
        "model_path": str(MODEL_PATH),
        "scripted_model_path": str(SCRIPTED_MODEL_PATH),
        "validation_predictions_path": str(VAL_PREDICTIONS_PATH),
        "test_predictions_path": str(TEST_PREDICTIONS_PATH),
        "threshold_sweep_path": str(THRESHOLD_SWEEP_PATH),
        "group_metrics_path": str(GROUP_METRICS_PATH),
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    plot_confusion_matrix(test_metrics["confusion_matrix"])

    print("\nTest metrics:")
    print(json.dumps(test_metrics, indent=2))
    print("\nFixed-threshold test metrics:")
    print(json.dumps(fixed_test_metrics, indent=2))
    print(f"Saved metrics to {METRICS_PATH}")
    print(f"Saved prediction CSVs to {VAL_PREDICTIONS_PATH} and {TEST_PREDICTIONS_PATH}")
    print(f"Saved threshold sweep to {THRESHOLD_SWEEP_PATH}")
    print(f"Saved group metrics to {GROUP_METRICS_PATH}")
    print(f"Saved Torch checkpoint to {MODEL_PATH}")
    print(f"Saved TorchScript model to {SCRIPTED_MODEL_PATH}")
    return results


# %% Package outputs for download

def package_outputs() -> Path:
    archive_path = Path(shutil.make_archive(str(BUNDLE_BASE_PATH), "zip", ARTIFACT_DIR))
    print(f"\nPackaged model bundle: {archive_path}")
    print("Kaggle download path:")
    print(str(archive_path))

    try:
        from IPython.display import FileLink, display

        display(FileLink(str(archive_path)))
    except Exception:
        pass

    return archive_path


# %% Run all

if __name__ == "__main__":
    results = train_model()
    bundle_path = package_outputs()
    print("\nDone.")
    print(f"Model kind: {results['model_kind']}")
    print(f"Best validation F1: {results['best_val_f1']:.4f}")
    print(f"Saved decision threshold: {results['best_threshold']:.3f}")
    print(f"Model bundle: {bundle_path}")
