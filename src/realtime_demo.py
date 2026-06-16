"""Live webcam driver monitoring demo."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

try:
    from .ddd_model import LoadedDrowsinessModel, load_drowsiness_model, predict_drowsiness_probability
    from .fusion import DriverState, FusionConfig, FusionScores, FusionStateMachine, SignalSnapshot, TransitionLogger
    from .landmarks import (
        EAR_THRESHOLD,
        LOOK_AWAY_MIN_SECONDS,
        MAR_THRESHOLD,
        PITCH_LOOK_AWAY_THRESHOLD_DEG,
        YAWN_MIN_SECONDS,
        YAW_LOOK_AWAY_THRESHOLD_DEG,
        FaceGeometryTracker,
        crop_bbox,
        draw_face_geometry,
        draw_text_block,
    )
    from .phone_detector import PhoneDetectionResult, PhoneDetector, draw_phone_detections
except ImportError:  # pragma: no cover - allows `python src/realtime_demo.py`
    from ddd_model import LoadedDrowsinessModel, load_drowsiness_model, predict_drowsiness_probability
    from fusion import DriverState, FusionConfig, FusionScores, FusionStateMachine, SignalSnapshot, TransitionLogger
    from landmarks import (
        EAR_THRESHOLD,
        LOOK_AWAY_MIN_SECONDS,
        MAR_THRESHOLD,
        PITCH_LOOK_AWAY_THRESHOLD_DEG,
        YAWN_MIN_SECONDS,
        YAW_LOOK_AWAY_THRESHOLD_DEG,
        FaceGeometryTracker,
        crop_bbox,
        draw_face_geometry,
        draw_text_block,
    )
    from phone_detector import PhoneDetectionResult, PhoneDetector, draw_phone_detections


WINDOW_NAME = "Driver Monitoring System"
STATE_COLORS = {
    DriverState.ALERT: (40, 180, 60),
    DriverState.DISTRACTED_LOW: (0, 210, 255),
    DriverState.DISTRACTED_MEDIUM: (0, 160, 255),
    DriverState.DISTRACTED_HIGH: (0, 90, 255),
    DriverState.DROWSY_LOW: (0, 210, 255),
    DriverState.DROWSY_MEDIUM: (0, 160, 255),
    DriverState.DROWSY_HIGH: (0, 90, 255),
    DriverState.CRITICAL: (0, 0, 255),
}


def load_optional_cnn(checkpoint: Path, disabled: bool) -> LoadedDrowsinessModel | None:
    if disabled:
        print("CNN drowsiness classifier disabled.")
        return None
    try:
        loaded = load_drowsiness_model(checkpoint)
        print(f"Loaded DDD CNN checkpoint from {checkpoint} on {loaded.device}.")
        return loaded
    except FileNotFoundError:
        print(f"DDD checkpoint not found at {checkpoint}; continuing without CNN drowsiness.")
    except Exception as exc:
        print(f"Could not load DDD CNN checkpoint ({exc}); continuing without CNN drowsiness.")
    return None


def load_optional_phone_detector(args: argparse.Namespace) -> PhoneDetector | None:
    if args.no_phone:
        print("Phone detector disabled.")
        return None
    try:
        detector = PhoneDetector(
            weights=args.yolo_weights,
            confidence_threshold=args.phone_confidence,
            run_every_n_frames=args.phone_every,
            device=args.yolo_device,
        )
        print(f"Loaded YOLO phone detector from {args.yolo_weights}.")
        return detector
    except Exception as exc:
        print(f"Could not start YOLO phone detector ({exc}); continuing without phone detection.")
        return None


def draw_state_overlay(
    frame,
    state: DriverState,
    scores: FusionScores,
    cnn_probability: float | None,
    phone_result: PhoneDetectionResult | None,
    fps: float,
) -> None:
    color = STATE_COLORS[state]
    phone_text = "yes" if phone_result and phone_result.phone_present else "no"
    cnn_text = "n/a" if cnn_probability is None else f"{cnn_probability:.3f}"
    lines = [
        f"State: {state.value}",
        f"Drowsy score: {scores.drowsiness:.3f}",
        f"Distraction score: {scores.distraction:.3f}",
        f"CNN drowsy p: {cnn_text}",
        f"Phone: {phone_text}",
        f"FPS: {fps:.1f}",
    ]
    height, width = frame.shape[:2]
    x = max(12, width - 300)
    draw_text_block(frame, lines, origin=(x, 28), color=(255, 255, 255), background=(25, 25, 25))

    if state != DriverState.ALERT:
        banner = f"ALERT: {state.value.replace('_', ' ')}"
        (text_w, text_h), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        bx1 = max(0, (width - text_w) // 2 - 20)
        by1 = height - 70
        bx2 = min(width - 1, bx1 + text_w + 40)
        by2 = min(height - 1, by1 + text_h + 28)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, -1)
        cv2.putText(
            frame,
            banner,
            (bx1 + 20, by2 - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument("--width", type=int, default=1280, help="Requested capture width.")
    parser.add_argument("--height", type=int, default=720, help="Requested capture height.")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/ddd_cnn.pt"))
    parser.add_argument("--cnn-every", type=int, default=5, help="Run DDD CNN every N frames.")
    parser.add_argument("--no-cnn", action="store_true", help="Disable DDD CNN even if a checkpoint exists.")
    parser.add_argument("--no-phone", action="store_true", help="Disable YOLO phone detection.")
    parser.add_argument("--phone-every", type=int, default=8, help="Run YOLO phone detector every N frames.")
    parser.add_argument("--phone-confidence", type=float, default=0.25)
    parser.add_argument("--yolo-weights", type=str, default="yolov8n.pt")
    parser.add_argument("--yolo-device", type=str, default=None)
    parser.add_argument("--log", type=Path, default=Path("logs/state_transitions.csv"))
    parser.add_argument("--hide-landmarks", action="store_true", help="Only show key landmarks and metrics.")
    parser.add_argument("--ear-threshold", type=float, default=EAR_THRESHOLD)
    parser.add_argument("--mar-threshold", type=float, default=MAR_THRESHOLD)
    parser.add_argument("--yawn-seconds", type=float, default=YAWN_MIN_SECONDS)
    parser.add_argument("--yaw-threshold", type=float, default=YAW_LOOK_AWAY_THRESHOLD_DEG)
    parser.add_argument("--pitch-threshold", type=float, default=PITCH_LOOK_AWAY_THRESHOLD_DEG)
    parser.add_argument("--look-away-seconds", type=float, default=LOOK_AWAY_MIN_SECONDS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cnn_model = load_optional_cnn(args.checkpoint, disabled=args.no_cnn)
    phone_detector = load_optional_phone_detector(args)
    geometry = FaceGeometryTracker(
        ear_threshold=args.ear_threshold,
        mar_threshold=args.mar_threshold,
        yawn_min_seconds=args.yawn_seconds,
        yaw_look_away_threshold_deg=args.yaw_threshold,
        pitch_look_away_threshold_deg=args.pitch_threshold,
        look_away_min_seconds=args.look_away_seconds,
    )
    fusion = FusionStateMachine(FusionConfig())
    transition_logger = TransitionLogger(args.log)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    frame_index = 0
    last_cnn_probability: float | None = None
    last_phone_result: PhoneDetectionResult | None = None
    last_frame_at = time.time()
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera frame read failed; stopping.")
                break

            frame_index += 1
            now = time.time()
            dt = max(now - last_frame_at, 1e-6)
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt
            last_frame_at = now

            geometry_result = geometry.process_frame(frame, timestamp=now)

            if cnn_model is not None and geometry_result is not None and frame_index % max(args.cnn_every, 1) == 0:
                face_crop = crop_bbox(frame, geometry_result.face_bbox, padding=0.08)
                try:
                    last_cnn_probability = predict_drowsiness_probability(
                        cnn_model,
                        face_crop,
                        geometry_result=geometry_result,
                    )
                except Exception as exc:
                    print(f"CNN inference failed once ({exc}); disabling CNN.")
                    cnn_model = None
                    last_cnn_probability = None

            if phone_detector is not None:
                face_bbox = geometry_result.face_bbox if geometry_result is not None else None
                try:
                    last_phone_result = phone_detector.detect(frame, face_bbox=face_bbox)
                except Exception as exc:
                    print(f"Phone detector failed once ({exc}); disabling phone detection.")
                    phone_detector = None
                    last_phone_result = None

            head_pose = geometry_result.head_pose if geometry_result is not None else None
            signal = SignalSnapshot(
                timestamp=now,
                perclos=geometry_result.perclos if geometry_result is not None else 0.0,
                yawn_active=geometry_result.yawn_active if geometry_result is not None else False,
                cnn_drowsy_probability=last_cnn_probability,
                head_yaw=head_pose.yaw if head_pose is not None else None,
                head_pitch=head_pose.pitch if head_pose is not None else None,
                looking_away_active=geometry_result.looking_away_active if geometry_result is not None else False,
                phone_present=last_phone_result.phone_present if last_phone_result is not None else False,
            )
            state, scores, transition = fusion.update(signal)
            transition_logger.log(transition)

            draw_face_geometry(frame, geometry_result, draw_all_landmarks=not args.hide_landmarks)
            draw_phone_detections(frame, last_phone_result)
            draw_state_overlay(frame, state, scores, last_cnn_probability, last_phone_result, fps)

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        geometry.close()
        cap.release()
        cv2.destroyAllWindows()
        print(f"Transition log: {args.log}")


if __name__ == "__main__":
    main()
