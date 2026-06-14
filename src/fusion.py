"""Signal fusion, state machine, hysteresis, and transition logging."""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class DriverState(str, Enum):
    ALERT = "ALERT"
    DISTRACTED_LOW = "DISTRACTED_LOW"
    DISTRACTED_MEDIUM = "DISTRACTED_MEDIUM"
    DISTRACTED_HIGH = "DISTRACTED_HIGH"
    DROWSY_LOW = "DROWSY_LOW"
    DROWSY_MEDIUM = "DROWSY_MEDIUM"
    DROWSY_HIGH = "DROWSY_HIGH"
    CRITICAL = "CRITICAL"


STATE_SEVERITY = {
    DriverState.ALERT: 0,
    DriverState.DISTRACTED_LOW: 1,
    DriverState.DROWSY_LOW: 1,
    DriverState.DISTRACTED_MEDIUM: 2,
    DriverState.DROWSY_MEDIUM: 2,
    DriverState.DISTRACTED_HIGH: 3,
    DriverState.DROWSY_HIGH: 3,
    DriverState.CRITICAL: 4,
}


@dataclass(frozen=True)
class FusionConfig:
    # PERCLOS is divided by this value to become a 0..1 score. A value around
    # 0.70 keeps the common 0.30-0.40 drowsiness threshold below CRITICAL.
    perclos_reference: float = 0.70
    yawn_score: float = 0.75
    cnn_weight: float = 0.45
    geometry_weight: float = 0.55
    require_cnn_geo_agreement_for_high: bool = False
    drowsy_low_threshold: float = 0.35
    drowsy_medium_threshold: float = 0.55
    drowsy_high_threshold: float = 0.72
    distraction_low_threshold: float = 0.35
    distraction_medium_threshold: float = 0.55
    distraction_high_threshold: float = 0.72
    critical_threshold: float = 0.88
    yaw_reference_deg: float = 35.0
    pitch_reference_deg: float = 30.0
    phone_score: float = 0.8
    min_consecutive_frames: int = 8
    cooldown_seconds: float = 1.25


@dataclass(frozen=True)
class SignalSnapshot:
    timestamp: float
    perclos: float = 0.0
    yawn_active: bool = False
    cnn_drowsy_probability: float | None = None
    head_yaw: float | None = None
    head_pitch: float | None = None
    looking_away_active: bool = False
    phone_present: bool = False


@dataclass(frozen=True)
class FusionScores:
    geometry_drowsiness: float
    cnn_drowsiness: float | None
    drowsiness: float
    head_distraction: float
    phone_distraction: float
    distraction: float


@dataclass(frozen=True)
class StateTransition:
    timestamp: float
    previous_state: DriverState
    new_state: DriverState
    drowsiness_score: float
    distraction_score: float


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def collapse_state(state: DriverState | str) -> str:
    value = state.value if isinstance(state, DriverState) else str(state).upper()
    if value.startswith("DROWSY"):
        return "DROWSY"
    if value.startswith("DISTRACTED"):
        return "DISTRACTED"
    if value == "CRITICAL":
        return "CRITICAL"
    return "ALERT"


class FusionStateMachine:
    """Combines drowsiness and distraction signals with hysteresis."""

    def __init__(self, config: FusionConfig | None = None) -> None:
        self.config = config or FusionConfig()
        self.current_state = DriverState.ALERT
        self.last_transition_at = 0.0
        self.pending_state: DriverState | None = None
        self.pending_count = 0
        self.last_scores = FusionScores(0.0, None, 0.0, 0.0, 0.0, 0.0)

    def compute_scores(self, signal: SignalSnapshot) -> FusionScores:
        cfg = self.config
        perclos_score = clamp01(signal.perclos / max(cfg.perclos_reference, 1e-6))
        geometry_score = max(perclos_score, cfg.yawn_score if signal.yawn_active else 0.0)

        if signal.cnn_drowsy_probability is None:
            cnn_score = None
            drowsiness = geometry_score
        else:
            cnn_score = clamp01(signal.cnn_drowsy_probability)
            drowsiness = clamp01(cfg.geometry_weight * geometry_score + cfg.cnn_weight * cnn_score)
            if cfg.require_cnn_geo_agreement_for_high and (geometry_score < cfg.drowsy_medium_threshold or cnn_score < cfg.drowsy_medium_threshold):
                drowsiness = min(drowsiness, cfg.drowsy_high_threshold - 0.01)

        yaw_score = 0.0
        pitch_score = 0.0
        if signal.head_yaw is not None:
            yaw_score = clamp01(abs(signal.head_yaw) / max(cfg.yaw_reference_deg, 1e-6))
        if signal.head_pitch is not None:
            pitch_score = clamp01(abs(signal.head_pitch) / max(cfg.pitch_reference_deg, 1e-6))
        head_score = max(yaw_score, pitch_score)
        if signal.looking_away_active:
            head_score = max(head_score, cfg.distraction_medium_threshold)

        phone_score = cfg.phone_score if signal.phone_present else 0.0
        distraction = max(head_score, phone_score)

        scores = FusionScores(
            geometry_drowsiness=geometry_score,
            cnn_drowsiness=cnn_score,
            drowsiness=drowsiness,
            head_distraction=head_score,
            phone_distraction=phone_score,
            distraction=distraction,
        )
        self.last_scores = scores
        return scores

    def classify(self, scores: FusionScores) -> DriverState:
        cfg = self.config
        drowsy = scores.drowsiness
        distracted = scores.distraction

        if (
            drowsy >= cfg.critical_threshold
            or distracted >= cfg.critical_threshold
            or (drowsy >= cfg.drowsy_high_threshold and distracted >= cfg.distraction_medium_threshold)
        ):
            return DriverState.CRITICAL

        drowsy_state = self._level_state(
            drowsy,
            cfg.drowsy_low_threshold,
            cfg.drowsy_medium_threshold,
            cfg.drowsy_high_threshold,
            DriverState.DROWSY_LOW,
            DriverState.DROWSY_MEDIUM,
            DriverState.DROWSY_HIGH,
        )
        distracted_state = self._level_state(
            distracted,
            cfg.distraction_low_threshold,
            cfg.distraction_medium_threshold,
            cfg.distraction_high_threshold,
            DriverState.DISTRACTED_LOW,
            DriverState.DISTRACTED_MEDIUM,
            DriverState.DISTRACTED_HIGH,
        )

        if drowsy_state == DriverState.ALERT:
            return distracted_state
        if distracted_state == DriverState.ALERT:
            return drowsy_state
        return drowsy_state if drowsy >= distracted else distracted_state

    @staticmethod
    def _level_state(
        score: float,
        low: float,
        medium: float,
        high: float,
        low_state: DriverState,
        medium_state: DriverState,
        high_state: DriverState,
    ) -> DriverState:
        if score >= high:
            return high_state
        if score >= medium:
            return medium_state
        if score >= low:
            return low_state
        return DriverState.ALERT

    def update(self, signal: SignalSnapshot) -> tuple[DriverState, FusionScores, StateTransition | None]:
        scores = self.compute_scores(signal)
        desired = self.classify(scores)
        now = signal.timestamp

        if desired == self.current_state:
            self.pending_state = None
            self.pending_count = 0
            return self.current_state, scores, None

        current_severity = STATE_SEVERITY[self.current_state]
        desired_severity = STATE_SEVERITY[desired]
        cooling_down = now - self.last_transition_at < self.config.cooldown_seconds
        if cooling_down and desired_severity < current_severity:
            return self.current_state, scores, None

        if desired != self.pending_state:
            self.pending_state = desired
            self.pending_count = 1
        else:
            self.pending_count += 1

        if self.pending_count < self.config.min_consecutive_frames:
            return self.current_state, scores, None

        previous = self.current_state
        self.current_state = desired
        self.last_transition_at = now
        self.pending_state = None
        self.pending_count = 0
        transition = StateTransition(
            timestamp=now,
            previous_state=previous,
            new_state=desired,
            drowsiness_score=scores.drowsiness,
            distraction_score=scores.distraction,
        )
        return self.current_state, scores, transition


class TransitionLogger:
    """Append state transitions to a CSV file."""

    FIELDNAMES = [
        "timestamp_epoch",
        "timestamp_iso",
        "previous_state",
        "new_state",
        "drowsiness_score",
        "distraction_score",
    ]

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=self.FIELDNAMES)
                writer.writeheader()

    def log(self, transition: StateTransition | None) -> None:
        if transition is None:
            return
        with self.path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=self.FIELDNAMES)
            writer.writerow(
                {
                    "timestamp_epoch": f"{transition.timestamp:.3f}",
                    "timestamp_iso": time.strftime(
                        "%Y-%m-%dT%H:%M:%S",
                        time.localtime(transition.timestamp),
                    ),
                    "previous_state": transition.previous_state.value,
                    "new_state": transition.new_state.value,
                    "drowsiness_score": f"{transition.drowsiness_score:.4f}",
                    "distraction_score": f"{transition.distraction_score:.4f}",
                }
            )
