"""Evaluate geometric, CNN-only, and fused driver-state predictions on labeled clips."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

try:
    from .ddd_model import LoadedDrowsinessModel, load_drowsiness_model, predict_drowsiness_probability
    from .fusion import DriverState, FusionConfig, FusionStateMachine, SignalSnapshot, collapse_state
    from .landmarks import FaceGeometryTracker, crop_bbox
    from .phone_detector import PhoneDetector
except ImportError:  # pragma: no cover
    from ddd_model import LoadedDrowsinessModel, load_drowsiness_model, predict_drowsiness_probability
    from fusion import DriverState, FusionConfig, FusionStateMachine, SignalSnapshot, collapse_state
    from landmarks import FaceGeometryTracker, crop_bbox
    from phone_detector import PhoneDetector


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
BROAD_LABEL_ORDER = ["ALERT", "DISTRACTED", "DROWSY", "CRITICAL"]


@dataclass(frozen=True)
class LabelInterval:
    video: str
    start: float
    end: float
    state: str


def find_column(columns: list[str], aliases: set[str]) -> str | None:
    normalized = {column.lower().strip().replace(" ", "_"): column for column in columns}
    for alias in aliases:
        if alias in normalized:
            return normalized[alias]
    return None


def parse_seconds(value: object) -> float:
    if isinstance(value, (int, float)) and not math.isnan(float(value)):
        return float(value)
    text = str(value).strip()
    if ":" not in text:
        return float(text)
    parts = [float(part) for part in text.split(":")]
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return hours * 3600.0 + minutes * 60.0 + seconds
    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60.0 + seconds
    raise ValueError(f"Could not parse timestamp: {value}")


def normalize_state_label(value: object, keep_levels: bool = False) -> str:
    text = str(value).strip().upper().replace(" ", "_").replace("-", "_")
    if keep_levels:
        if text in {state.value for state in DriverState}:
            return text
        if text == "DROWSY":
            return DriverState.DROWSY_MEDIUM.value
        if text == "DISTRACTED":
            return DriverState.DISTRACTED_MEDIUM.value
    if text.startswith("DROWSY"):
        return "DROWSY"
    if text.startswith("DISTRACTED") or text in {"PHONE", "LOOKING_AWAY"}:
        return "DISTRACTED"
    if text == "CRITICAL":
        return "CRITICAL"
    return "ALERT"


def discover_videos(data_dir: Path) -> list[Path]:
    return sorted(path for path in data_dir.iterdir() if path.suffix.lower() in VIDEO_EXTENSIONS)


def load_label_intervals(labels_csv: Path, data_dir: Path, keep_levels: bool = False) -> dict[str, list[LabelInterval]]:
    if not labels_csv.exists():
        raise FileNotFoundError(f"Missing labels CSV: {labels_csv}")
    df = pd.read_csv(labels_csv)
    if df.empty:
        raise RuntimeError(f"Labels CSV is empty: {labels_csv}")

    video_col = find_column(list(df.columns), {"video", "clip", "file", "filename", "video_file"})
    state_col = find_column(list(df.columns), {"state", "label", "ground_truth", "gt_state"})
    start_col = find_column(list(df.columns), {"start", "start_time", "t_start", "begin"})
    end_col = find_column(list(df.columns), {"end", "end_time", "t_end", "stop"})
    timestamp_col = find_column(list(df.columns), {"timestamp", "time", "seconds", "t"})

    if state_col is None:
        raise RuntimeError("labels.csv needs a state/label column.")

    videos = discover_videos(data_dir)
    if video_col is None:
        if len(videos) != 1:
            raise RuntimeError("labels.csv needs a video column when multiple clips are present.")
        df = df.copy()
        video_col = "video"
        df[video_col] = videos[0].name

    intervals_by_video: dict[str, list[LabelInterval]] = {}
    if start_col is not None and end_col is not None:
        for _, row in df.iterrows():
            video = str(row[video_col])
            interval = LabelInterval(
                video=video,
                start=parse_seconds(row[start_col]),
                end=parse_seconds(row[end_col]),
                state=normalize_state_label(row[state_col], keep_levels=keep_levels),
            )
            intervals_by_video.setdefault(video, []).append(interval)
    elif timestamp_col is not None:
        df = df.sort_values([video_col, timestamp_col])
        for video, group in df.groupby(video_col):
            rows = list(group.iterrows())
            intervals: list[LabelInterval] = []
            for index, (_, row) in enumerate(rows):
                start = parse_seconds(row[timestamp_col])
                end = parse_seconds(rows[index + 1][1][timestamp_col]) if index + 1 < len(rows) else float("inf")
                intervals.append(
                    LabelInterval(
                        video=str(video),
                        start=start,
                        end=end,
                        state=normalize_state_label(row[state_col], keep_levels=keep_levels),
                    )
                )
            intervals_by_video[str(video)] = intervals
    else:
        raise RuntimeError("labels.csv needs either start/end interval columns or a timestamp column.")

    for intervals in intervals_by_video.values():
        intervals.sort(key=lambda item: item.start)
    return intervals_by_video


def lookup_label(intervals: list[LabelInterval], timestamp: float) -> str | None:
    for interval in intervals:
        if interval.start <= timestamp < interval.end:
            return interval.state
    return None


def optional_cnn(checkpoint: Path) -> LoadedDrowsinessModel | None:
    try:
        loaded = load_drowsiness_model(checkpoint)
        print(f"Loaded CNN checkpoint: {checkpoint}")
        return loaded
    except FileNotFoundError:
        print(f"CNN checkpoint not found at {checkpoint}; CNN-only mode will predict ALERT.")
    except Exception as exc:
        print(f"Could not load CNN checkpoint ({exc}); CNN-only mode will predict ALERT.")
    return None


def optional_phone_detector(args: argparse.Namespace) -> PhoneDetector | None:
    if not args.enable_phone:
        return None
    try:
        return PhoneDetector(
            weights=args.yolo_weights,
            confidence_threshold=args.phone_confidence,
            run_every_n_frames=args.phone_every,
            device=args.yolo_device,
        )
    except Exception as exc:
        print(f"Phone detector unavailable ({exc}); fused evaluation will omit phone detections.")
        return None


def prediction_label(state: DriverState, keep_levels: bool) -> str:
    return state.value if keep_levels else collapse_state(state)


def evaluate_video(
    video_path: Path,
    intervals: list[LabelInterval],
    args: argparse.Namespace,
    cnn_model: LoadedDrowsinessModel | None,
) -> tuple[list[str], dict[str, list[str]]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    geometry = FaceGeometryTracker()
    phone_detector = optional_phone_detector(args)
    config = FusionConfig(min_consecutive_frames=args.hysteresis_frames)
    geo_fusion = FusionStateMachine(config)
    cnn_fusion = FusionStateMachine(config)
    fused_fusion = FusionStateMachine(config)
    y_true: list[str] = []
    y_pred = {"geometric": [], "cnn": [], "fused": []}
    last_cnn_probability: float | None = None
    frame_index = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1
            if frame_index % max(args.sample_every, 1) != 0:
                continue

            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            label = lookup_label(intervals, timestamp)
            if label is None:
                continue

            geometry_result = geometry.process_frame(frame, timestamp=timestamp)
            head_pose = geometry_result.head_pose if geometry_result is not None else None

            if (
                cnn_model is not None
                and geometry_result is not None
                and frame_index % max(args.cnn_every, 1) == 0
            ):
                face_crop = crop_bbox(frame, geometry_result.face_bbox, padding=0.08)
                try:
                    last_cnn_probability = predict_drowsiness_probability(
                        cnn_model,
                        face_crop,
                        geometry_result=geometry_result,
                    )
                except Exception:
                    last_cnn_probability = None

            phone_present = False
            if phone_detector is not None:
                face_bbox = geometry_result.face_bbox if geometry_result is not None else None
                phone_present = phone_detector.detect(frame, face_bbox=face_bbox).phone_present

            geo_signal = SignalSnapshot(
                timestamp=timestamp,
                perclos=geometry_result.perclos if geometry_result is not None else 0.0,
                yawn_active=geometry_result.yawn_active if geometry_result is not None else False,
                head_yaw=head_pose.yaw if head_pose is not None else None,
                head_pitch=head_pose.pitch if head_pose is not None else None,
                looking_away_active=geometry_result.looking_away_active if geometry_result is not None else False,
            )
            cnn_signal = SignalSnapshot(
                timestamp=timestamp,
                cnn_drowsy_probability=last_cnn_probability,
            )
            fused_signal = SignalSnapshot(
                timestamp=timestamp,
                perclos=geo_signal.perclos,
                yawn_active=geo_signal.yawn_active,
                cnn_drowsy_probability=last_cnn_probability,
                head_yaw=geo_signal.head_yaw,
                head_pitch=geo_signal.head_pitch,
                looking_away_active=geo_signal.looking_away_active,
                phone_present=phone_present,
            )

            geo_state, _, _ = geo_fusion.update(geo_signal)
            cnn_state, _, _ = cnn_fusion.update(cnn_signal)
            fused_state, _, _ = fused_fusion.update(fused_signal)

            y_true.append(label)
            y_pred["geometric"].append(prediction_label(geo_state, args.keep_levels))
            y_pred["cnn"].append(prediction_label(cnn_state, args.keep_levels))
            y_pred["fused"].append(prediction_label(fused_state, args.keep_levels))
    finally:
        geometry.close()
        cap.release()

    return y_true, y_pred


def summarize_predictions(
    y_true: list[str],
    predictions: dict[str, list[str]],
    output_dir: Path,
    keep_levels: bool,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {"sample_count": len(y_true), "modes": {}}

    if keep_levels:
        label_order = [state.value for state in DriverState]
        labels = [label for label in label_order if label in set(y_true).union(*[set(v) for v in predictions.values()])]
    else:
        labels = [label for label in BROAD_LABEL_ORDER if label in set(y_true).union(*[set(v) for v in predictions.values()])]

    for mode, y_pred in predictions.items():
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        report = classification_report(y_true, y_pred, labels=labels, zero_division=0, output_dict=True)
        pd.DataFrame(cm, index=labels, columns=labels).to_csv(output_dir / f"{mode}_confusion_matrix.csv")
        summary["modes"][mode] = {
            "labels": labels,
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
        }

    (output_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/eval"))
    parser.add_argument("--labels", type=Path, default=Path("data/eval/labels.csv"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/ddd_cnn.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/eval"))
    parser.add_argument("--sample-every", type=int, default=5, help="Evaluate every Nth frame.")
    parser.add_argument("--cnn-every", type=int, default=5, help="Run CNN every N source frames.")
    parser.add_argument("--hysteresis-frames", type=int, default=4)
    parser.add_argument("--keep-levels", action="store_true", help="Evaluate low/medium/high states separately.")
    parser.add_argument("--enable-phone", action="store_true", help="Include YOLO phone detection in fused mode.")
    parser.add_argument("--phone-every", type=int, default=8)
    parser.add_argument("--phone-confidence", type=float, default=0.25)
    parser.add_argument("--yolo-weights", type=str, default="yolov8n.pt")
    parser.add_argument("--yolo-device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    intervals_by_video = load_label_intervals(args.labels, args.data_dir, keep_levels=args.keep_levels)
    cnn_model = optional_cnn(args.checkpoint)

    all_true: list[str] = []
    all_pred = {"geometric": [], "cnn": [], "fused": []}
    videos = {path.name: path for path in discover_videos(args.data_dir)}

    for video_name, intervals in intervals_by_video.items():
        video_path = videos.get(video_name) or args.data_dir / video_name
        if not video_path.exists():
            raise FileNotFoundError(f"Label references missing video: {video_name}")
        print(f"Evaluating {video_path.name}...")
        y_true, y_pred = evaluate_video(video_path, intervals, args, cnn_model)
        all_true.extend(y_true)
        for mode in all_pred:
            all_pred[mode].extend(y_pred[mode])

    if not all_true:
        raise RuntimeError("No labeled frames were evaluated. Check timestamps in labels.csv.")

    summary = summarize_predictions(all_true, all_pred, args.output_dir, keep_levels=args.keep_levels)
    print(json.dumps({"sample_count": summary["sample_count"], "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
