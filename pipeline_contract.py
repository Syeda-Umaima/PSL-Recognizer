"""Shared PSL feature contract for offline extraction and webcam inference."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

SEQUENCE_LENGTH, FEATURE_DIM = 30, 225
RIGHT_HAND, LEFT_HAND, POSE = slice(0, 63), slice(63, 126), slice(126, 225)
SLICES = (("Right hand", RIGHT_HAND), ("Left hand", LEFT_HAND), ("Pose", POSE))
TRAINING_ASPECT_RATIO = 16.0 / 9.0


def prepare_mediapipe_frame(frame: np.ndarray, target_ratio: float = TRAINING_ASPECT_RATIO) -> np.ndarray:
    """Center-crop to the 16:9 training geometry, preserving 16:9 frames unchanged."""
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Expected BGR frame (height, width, 3)")
    h, w = frame.shape[:2]
    ratio = w / h
    if ratio > target_ratio:
        new_w = max(1, int(round(h * target_ratio)))
        left = (w - new_w) // 2
        frame = frame[:, left:left + new_w]
    elif ratio < target_ratio:
        new_h = max(1, int(round(w / target_ratio)))
        top = (h - new_h) // 2
        frame = frame[top:top + new_h, :]
    return np.ascontiguousarray(frame)


def mediapipe_rgb(frame: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(cv2.cvtColor(prepare_mediapipe_frame(frame), cv2.COLOR_BGR2RGB))


def extract_vector_from_result(result: Any) -> tuple[np.ndarray, dict[str, int]]:
    """Encode Tasks Holistic output as RH(63), LH(63), Pose(99), with zero padding."""
    values: list[float] = []
    counts = {"rh": 0, "lh": 0, "pose": 0, "face": 0}
    for attr, key, expected in (("right_hand_landmarks", "rh", 21), ("left_hand_landmarks", "lh", 21), ("pose_landmarks", "pose", 33)):
        landmarks = getattr(result, attr, None) or []
        counts[key] = len(landmarks)
        if len(landmarks) == expected:
            for point in landmarks:
                values.extend((point.x, point.y, point.z))
        else:
            values.extend((0.0,) * (expected * 3))
    counts["face"] = len(getattr(result, "face_landmarks", None) or [])
    vector = np.asarray(values, dtype=np.float32)
    if vector.shape != (FEATURE_DIM,):
        raise ValueError(f"Expected ({FEATURE_DIM},), got {vector.shape}")
    return normalize_feature_vector(vector), counts


def normalize_feature_vector(vector: np.ndarray) -> np.ndarray:
    """Remove camera translation and person scale while preserving sign shape.

    Hands are represented relative to their wrist and normalized by hand span.
    Pose is represented relative to the shoulder midpoint and normalized by
    shoulder width. Missing landmark groups remain exactly zero.
    """
    output = np.asarray(vector, dtype=np.float32).copy()

    for part in (RIGHT_HAND, LEFT_HAND):
        points = output[part].reshape(21, 3)
        if not np.count_nonzero(points):
            continue
        wrist = points[0].copy()
        points -= wrist
        scale = float(np.max(np.linalg.norm(points[:, :2], axis=1)))
        if scale < 1e-4:
            scale = 1.0
        output[part] = (points / scale).reshape(-1)

    pose = output[POSE].reshape(33, 3)
    if np.count_nonzero(pose):
        # MediaPipe pose landmark indices 11 and 12 are left/right shoulders.
        shoulders = pose[[11, 12]]
        if np.count_nonzero(shoulders):
            origin = shoulders.mean(axis=0)
            scale = float(np.linalg.norm(shoulders[0, :2] - shoulders[1, :2]))
            pose -= origin
            pose /= max(scale, 1e-4)
            output[POSE] = pose.reshape(-1)
    return output.astype(np.float32)


def first_active_frame(sequence: np.ndarray, fallback_index: int = 15) -> tuple[int, np.ndarray]:
    array = np.asarray(sequence, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != FEATURE_DIM:
        raise ValueError(f"Expected (frames, {FEATURE_DIM}), got {array.shape}")
    active = np.flatnonzero(np.count_nonzero(array, axis=1))
    index = int(active[0]) if active.size else min(max(fallback_index, 0), len(array) - 1)
    return index, array[index]


@dataclass
class FeatureClipper:
    lower: np.ndarray
    upper: np.ndarray

    @classmethod
    def from_reference(cls, sequence: np.ndarray) -> "FeatureClipper":
        array = np.asarray(sequence, dtype=np.float32)
        valid = array[np.count_nonzero(array, axis=1) > 0]
        if not len(valid):
            valid = array
        mean, std = valid.mean(axis=0), valid.std(axis=0)
        lower = np.minimum(valid.min(axis=0), mean - 3 * std)
        upper = np.maximum(valid.max(axis=0), mean + 3 * std)
        # A single reference sign can legitimately contain no hand landmarks. Do not
        # turn that absence into a zero-only bound for other live signs.
        unseen = np.ptp(valid, axis=0) < 1e-6
        lower[unseen], upper[unseen] = -3.0, 3.0
        return cls(lower.astype(np.float32), upper.astype(np.float32))

    def clip(self, vector: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(vector, dtype=np.float32), self.lower, self.upper).astype(np.float32)


class HandLandmarkImputer:
    """Hold valid hand tracks for five dropped frames, then decay them toward zero."""
    def __init__(self, max_hold_frames: int = 5, decay: float = 0.65) -> None:
        self.max_hold_frames, self.decay = max_hold_frames, decay
        self.last = {"rh": None, "lh": None}
        self.missing = {"rh": 0, "lh": 0}

    def apply(self, vector: np.ndarray, counts: dict[str, int]) -> np.ndarray:
        output = np.asarray(vector, dtype=np.float32).copy()
        for key, part in (("rh", RIGHT_HAND), ("lh", LEFT_HAND)):
            valid = counts.get(key, 0) == 21 and np.count_nonzero(output[part]) > 0
            if valid:
                self.last[key], self.missing[key] = output[part].copy(), 0
            else:
                self.missing[key] += 1
                if self.last[key] is not None:
                    exponent = max(0, self.missing[key] - self.max_hold_frames)
                    output[part] = self.last[key] * self.decay ** exponent
        return output


def _stats(vector: np.ndarray) -> tuple[float, float, float, float, int]:
    vector = np.asarray(vector, dtype=np.float32)
    return float(vector.mean()), float(vector.max()), float(vector.min()), float(vector.std()), int(np.count_nonzero(vector))


def print_statistics_table(offline: np.ndarray, live: np.ndarray) -> None:
    print("\nFeature statistics: active offline frame vs processed live frame")
    print(f"{'Slice':<14} {'Source':<8} {'Mean':>10} {'Max':>10} {'Min':>10} {'Std':>10} {'Non-zero':>10}")
    print("-" * 82)
    for name, part in SLICES:
        for source, vector in (("offline", offline[part]), ("live", live[part])):
            mean, maximum, minimum, std, nonzero = _stats(vector)
            print(f"{name:<14} {source:<8} {mean:>10.5f} {maximum:>10.5f} {minimum:>10.5f} {std:>10.5f} {nonzero:>10}")


def run_pipeline_verification(holistic: Any, webcam: cv2.VideoCapture, reference_path: Path, seconds: float = 10.0) -> bool:
    """Run a bounded live/offline contract audit without loading the classifier."""
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference feature file not found: {reference_path}")
    reference = np.load(reference_path).astype(np.float32)
    offline_index, offline = first_active_frame(reference)
    clipper, imputer, live_frames = FeatureClipper.from_reference(reference), HandLandmarkImputer(), []
    from mediapipe.tasks.python.vision.core import image as mp_image
    start = cv2.getTickCount() / cv2.getTickFrequency()
    while cv2.getTickCount() / cv2.getTickFrequency() - start < seconds:
        ok, frame = webcam.read()
        if not ok:
            continue
        result = holistic.detect(mp_image.Image(mp_image.ImageFormat.SRGB, mediapipe_rgb(frame)))
        vector, counts = extract_vector_from_result(result)
        live_frames.append(clipper.clip(imputer.apply(vector, counts)))
    if not live_frames:
        raise RuntimeError("No webcam frames captured during verification")
    live = next((x for x in live_frames if np.count_nonzero(x)), live_frames[-1])
    active = (offline != 0) & (live != 0)
    mse = float(np.mean((offline[active] - live[active]) ** 2)) if np.any(active) else float("inf")
    cosine = float(np.dot(offline[active], live[active]) / (np.linalg.norm(offline[active]) * np.linalg.norm(live[active]) + 1e-8)) if np.any(active) else 0.0
    print(f"\nReference: {reference_path.name}; selected offline frame: {offline_index}")
    print_statistics_table(offline, live)
    checks = {
        "Shape compatibility": live.shape == offline.shape == (FEATURE_DIM,),
        "Finite feature values": bool(np.isfinite(live).all()),
        "Slice alignment": (live[RIGHT_HAND].size, live[LEFT_HAND].size, live[POSE].size) == (63, 63, 99),
        "Active-channel overlap": bool(np.any(active)),
        "Distribution MSE <= 0.75": mse <= 0.75,
        "Cosine similarity >= 0.10": cosine >= 0.10,
    }
    print("\nPipeline audit matrix")
    for label, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {label}")
    print(f"  INFO  active-channel MSE={mse:.6f}; cosine={cosine:.6f}; live frames={len(live_frames)}")
    return all(checks.values())
