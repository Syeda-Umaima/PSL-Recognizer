"""Free CPU deployment for the isolated PSL webcam recognizer."""
from __future__ import annotations

import json
import threading
from pathlib import Path

import av
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from scipy.interpolate import interp1d
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from mediapipe.tasks.python.core import base_options
from mediapipe.tasks.python.vision import holistic_landmarker
from mediapipe.tasks.python.vision.core import image as mp_image
from mediapipe.tasks.python.vision.core import vision_task_running_mode

from pipeline_contract import extract_vector_from_result, mediapipe_rgb, HandLandmarkImputer

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "champion_model.keras"
HOLISTIC_MODEL = ROOT / "holistic_landmarker.task"
LABELS_PATH = ROOT / "labels.json"
TARGET_FRAMES, FEATURE_DIM = 30, 225
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)
POSE_CONNECTIONS = ((11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28))


def draw_landmarks(frame: np.ndarray, result) -> np.ndarray:
    output = np.ascontiguousarray(frame.copy())
    height, width = output.shape[:2]
    for attr, links, color, radius in (
        ("right_hand_landmarks", HAND_CONNECTIONS, (0, 255, 0), 5),
        ("left_hand_landmarks", HAND_CONNECTIONS, (255, 80, 0), 5),
        ("pose_landmarks", POSE_CONNECTIONS, (0, 0, 255), 4),
    ):
        points = getattr(result, attr, None) or []
        pixels = [(int(np.clip(p.x, 0, 1) * (width - 1)), int(np.clip(p.y, 0, 1) * (height - 1))) for p in points]
        for first, second in links:
            if first < len(pixels) and second < len(pixels):
                cv2.line(output, pixels[first], pixels[second], color, 2, cv2.LINE_AA)
        for point in pixels:
            cv2.circle(output, point, radius, color, -1, cv2.LINE_AA)
            cv2.circle(output, point, radius + 1, (255, 255, 255), 1, cv2.LINE_AA)
    return output


class CaptureController:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.recording = False
        self.frames: list[np.ndarray] = []
        self.detector = None
        self.imputer = HandLandmarkImputer()

    def _get_detector(self):
        if self.detector is None:
            options = holistic_landmarker.HolisticLandmarkerOptions(
                base_options=base_options.BaseOptions(model_asset_path=str(HOLISTIC_MODEL)),
                running_mode=vision_task_running_mode.VisionTaskRunningMode.IMAGE,
                min_face_landmarks_confidence=0.4,
                min_pose_landmarks_confidence=0.4,
                min_hand_landmarks_confidence=0.4,
            )
            self.detector = holistic_landmarker.HolisticLandmarker.create_from_options(options)
        return self.detector

    def start(self) -> None:
        with self.lock:
            self.frames.clear()
            self.imputer = HandLandmarkImputer()
            self.recording = True

    def stop(self) -> np.ndarray:
        with self.lock:
            self.recording = False
            return np.asarray(self.frames, dtype=np.float32)

    def process(self, frame: av.VideoFrame) -> av.VideoFrame:
        bgr = frame.to_ndarray(format="bgr24")
        prepared = cv2.resize(mediapipe_rgb(bgr), (720, 405), interpolation=cv2.INTER_AREA)
        prepared = np.ascontiguousarray(prepared)
        result = self._get_detector().detect(mp_image.Image(mp_image.ImageFormat.SRGB, prepared))
        vector, counts = extract_vector_from_result(result)
        vector = self.imputer.apply(vector, counts)
        with self.lock:
            if self.recording:
                self.frames.append(vector.copy())
        annotated = draw_landmarks(cv2.cvtColor(prepared, cv2.COLOR_RGB2BGR), result)
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


def resample_sequence(sequence: np.ndarray) -> np.ndarray:
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.ndim != 2 or sequence.shape[1] != FEATURE_DIM or not len(sequence):
        raise ValueError(f"Expected captured frames shaped (N, {FEATURE_DIM}); got {sequence.shape}")
    if len(sequence) == TARGET_FRAMES:
        return sequence
    x = np.linspace(0.0, 1.0, len(sequence))
    y = np.linspace(0.0, 1.0, TARGET_FRAMES)
    return np.asarray(interp1d(x, sequence, axis=0, kind="linear")(y), dtype=np.float32)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


@st.cache_data
def load_labels():
    return np.asarray(json.loads(LABELS_PATH.read_text(encoding="utf-8-sig")))


def predict(frames: np.ndarray) -> str:
    sequence = resample_sequence(frames)
    probabilities = np.asarray(load_model().predict(sequence[None, ...], verbose=0))[0]
    labels = load_labels()
    top = np.argsort(probabilities)[::-1][:5]
    lines = [f"{rank}. {str(labels[index]).split('::', 1)[-1]} ({probabilities[index] * 100:.2f}%)" for rank, index in enumerate(top, 1)]
    return "\n".join(lines) + f"\n\nProcessed {len(frames)} captured frames and resampled to {TARGET_FRAMES}."


st.set_page_config(page_title="PSL Isolated Recognition", page_icon="🤟", layout="wide")
st.title("Pakistani Sign Language: Isolated Sign Recognition")
st.caption("Use Start Recording, perform one isolated sign, then press Stop and Predict.")

controller = st.session_state.setdefault("capture_controller", CaptureController())
if "prediction" not in st.session_state:
    st.session_state.prediction = "No prediction yet."

webrtc_streamer(
    key="psl-isolated-camera",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=controller.process,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

left, right = st.columns(2)
with left:
    if st.button("Start Recording", type="primary", use_container_width=True):
        controller.start()
        st.session_state.prediction = "Recording started. Perform one sign, then stop."
        st.rerun()
with right:
    if st.button("Stop and Predict", type="secondary", use_container_width=True):
        frames = controller.stop()
        if len(frames) < 2:
            st.session_state.prediction = "Record at least two frames before stopping."
        else:
            try:
                st.session_state.prediction = predict(frames)
            except Exception as exc:
                st.session_state.prediction = f"Prediction error: {type(exc).__name__}: {exc}"
        st.rerun()

status = "RECORDING" if controller.recording else "IDLE"
st.info(f"Status: {status} | Captured frames: {len(controller.frames)}")
st.text_area("Prediction", value=st.session_state.prediction, height=170, disabled=True)
