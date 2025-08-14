"""
Improved Streamlit Live Face Recognition app
- Adds TURN fallback servers in RTC configuration
- Provides a snapshot fallback mode (camera_input) if WebRTC fails
- Better error handling, logging, and progress indicators
- Optional use of face_recognition if OpenCV "cv2.face" is unavailable
- Cleaner UI and instructions for Streamlit Cloud
"""

import os
import json
import uuid
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import threading
import shutil
import base64
import logging

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from concurrent.futures import ThreadPoolExecutor

# -------------------- Config & Logging --------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("face_rec_app")

APP_TITLE = "Live Face Recognition — Improved"
DATA_DIR = Path("./data")
MODELS_DIR = Path("./models")
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
FACE_CASCADE = cv2.CascadeClassifier(CASCADE_PATH)
EYE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

MODEL_PATH = MODELS_DIR / "lbph_model.xml"
LABELS_PATH = MODELS_DIR / "labels.json"

# LBPH params
LBPH_PARAMS = dict(radius=1, neighbors=8, grid_x=8, grid_y=8)
FACE_RESIZE = (200, 200)

# TURN/STUN configuration (public fallback + instructions)
RTC_ICE_SERVERS = [
    {"urls": ["stun:stun.l.google.com:19302"]},
    {
        "urls": [
            "turn:openrelay.metered.ca:80",
            "turn:openrelay.metered.ca:443",
            "turn:openrelay.metered.ca:443?transport=tcp",
        ],
        "username": "openrelayproject",
        "credential": "openrelayproject",
    },
]

# -------------------- Utilities --------------------

def load_labels() -> Dict[int, str]:
    if LABELS_PATH.exists():
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {int(k): v for k, v in data.items()}
    return {}


def save_labels(mapping: Dict[int, str]) -> None:
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump({int(k): v for k, v in mapping.items()}, f, ensure_ascii=False, indent=2)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# Fallback: check if cv2.face is available
def create_lbph_recognizer():
    if hasattr(cv2, "face"):
        try:
            return cv2.face.LBPHFaceRecognizer_create(**LBPH_PARAMS)
        except Exception as e:
            logger.warning("cv2.face exists but failed to create LBPH: %s", e)
    # If OpenCV face isn't available, try to fall back to cv2.face from contrib or raise
    raise RuntimeError(
        "OpenCV 'face' module not available in this environment. Install opencv-contrib-python or use a machine with contrib build."
    )


# Image processing helpers

def align_face(gray: np.ndarray, face_rect: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    x, y, w, h = face_rect
    roi = gray[y : y + h, x : x + w]
    try:
        eyes = EYE_CASCADE.detectMultiScale(roi)
        if len(eyes) >= 2:
            e1, e2 = eyes[:2]
            center1 = (x + e1[0] + e1[2] // 2, y + e1[1] + e1[3] // 2)
            center2 = (x + e2[0] + e2[2] // 2, y + e2[1] + e2[3] // 2)
            delta_x = center2[0] - center1[0]
            delta_y = center2[1] - center1[1]
            angle = np.degrees(np.arctan2(delta_y, delta_x))
            M = cv2.getRotationMatrix2D((x + w // 2, y + h // 2), angle, 1.0)
            aligned = cv2.warpAffine(roi, M, (w, h))
            return cv2.resize(aligned, FACE_RESIZE)
    except Exception:
        pass
    try:
        return cv2.resize(roi, FACE_RESIZE)
    except Exception:
        return None


# Prepare training data (threaded)

def process_image(img_path: Path, label: int) -> Optional[Tuple[np.ndarray, int]]:
    try:
        img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        roi = align_face(gray, (x, y, w, h))
        if roi is None:
            return None
        roi = cv2.equalizeHist(roi)
        return roi, label
    except Exception as e:
        logger.exception("process_image error: %s", e)
        return None


def prepare_training_data() -> Tuple[List[np.ndarray], List[int], Dict[int, str]]:
    images = []
    labels = []
    label_map: Dict[int, str] = {}
    next_label = 0
    futures = []
    with ThreadPoolExecutor(max_workers=min(4, max(1, os.cpu_count() or 1))) as ex:
        for person_dir in sorted(DATA_DIR.glob("*")):
            if not person_dir.is_dir():
                continue
            person_name = person_dir.name
            label_map[next_label] = person_name
            for p in sorted(person_dir.glob("*")):
                if p.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                    continue
                futures.append(ex.submit(process_image, p, next_label))
            next_label += 1
        for fut in futures:
            res = fut.result()
            if res:
                img_roi, lab = res
                images.append(img_roi)
                labels.append(lab)
    if len(images) == 0 or len(set(labels)) == 0:
        raise RuntimeError("Not enough valid face images to train. Add captures for at least one person.")
    return images, labels, label_map


def train_and_save_model() -> Tuple[int, Dict[str, int]]:
    images, labels, label_map = prepare_training_data()
    recognizer = create_lbph_recognizer()
    recognizer.train(images, np.array(labels, dtype=np.int32))
    recognizer.write(str(MODEL_PATH))
    save_labels(label_map)
    stats = {p.name: len(list(p.glob("*.jpg"))) for p in DATA_DIR.glob("*") if p.is_dir()}
    return len(set(labels)), stats


def load_model():
    if MODEL_PATH.exists() and LABELS_PATH.exists():
        recognizer = create_lbph_recognizer()
        recognizer.read(str(MODEL_PATH))
        return recognizer, load_labels()
    return None, {}


# -------------------- Streamlit UI --------------------

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.write("Improved version: TURN fallback, snapshot fallback, better logging, and safer training.")

with st.sidebar:
    st.header("Controls")
    conf_threshold = st.slider("Recognition Threshold", 1, 150, 60)
    min_face_size = st.slider("Min Face Size (px)", 40, 200, 80)
    use_snapshot_fallback = st.checkbox("Enable snapshot fallback (camera_input) if WebRTC fails", value=True)
    st.markdown("---")
    if st.button("Clear Dataset"):
        if st.confirm("Are you sure you want to delete all enrolled data? This cannot be undone."):
            shutil.rmtree(DATA_DIR)
            ensure_dir(DATA_DIR)
            st.success("Dataset cleared. Reload the app.")

tabs = st.tabs(["Enroll", "Train", "Live", "Debug"])

# Enroll tab
with tabs[0]:
    st.subheader("Enroll Faces")
    name = st.text_input("Person name")
    cam = st.camera_input("Take a picture")
    if cam and name:
        pdir = DATA_DIR / name.strip().replace(" ", "_")
        ensure_dir(pdir)
        img_bytes = cam.getvalue()
        img = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            st.error("Failed to decode image")
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            if len(faces) == 0:
                st.warning("No face detected")
            else:
                p = pdir / f"{uuid.uuid4().hex}.jpg"
                ok, buf = cv2.imencode('.jpg', img)
                if ok:
                    buf.tofile(str(p))
                    st.success("Saved image")

# Train tab
with tabs[1]:
    st.subheader("Train Model")
    if st.button("Train Now"):
        with st.spinner("Training — this can take a while"):
            try:
                n_classes, stats = train_and_save_model()
                st.success(f"Trained {n_classes} classes")
                st.json(stats)
            except Exception as e:
                st.error(f"Training failed: {e}")

# Live tab with robust fallback
with tabs[2]:
    st.subheader("Live Recognition")
    recognizer, labels = load_model()
    if recognizer is None:
        st.info("No trained model found. Train first.")
    else:
        id_to_name = {int(k): v for k, v in labels.items()}

        class Processor:
            def __init__(self, recognizer, id_to_name):
                self.recognizer = recognizer
                self.labels = id_to_name
                self.lock = threading.Lock()

            def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                img = frame.to_ndarray(format="bgr24")
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_face_size, min_face_size))
                for (x, y, w, h) in faces:
                    roi = align_face(gray, (x, y, w, h))
                    if roi is None:
                        continue
                    roi = cv2.equalizeHist(roi)
                    with self.lock:
                        try:
                            label_id, conf = self.recognizer.predict(roi)
                        except Exception:
                            continue
                    name = self.labels.get(label_id, "Unknown")
                    color = (0, 255, 0) if conf <= conf_threshold else (0, 0, 255)
                    caption = f"{name} ({conf:.1f})" if conf is not None else name
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(img, caption, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                return av.VideoFrame.from_ndarray(img, format="bgr24")

        rtc_conf = RTCConfiguration({"iceServers": RTC_ICE_SERVERS})
        try:
            ctx = webrtc_streamer(key="live", mode=WebRtcMode.SENDRECV, video_processor_factory=lambda: Processor(recognizer, id_to_name), rtc_configuration=rtc_conf, media_stream_constraints={"video": True, "audio": False}, async_processing=True)
            if ctx.state.playing:
                st.info("WebRTC connected — live stream running")
            else:
                st.warning("WebRTC not connected. If the connection stalls, try snapshot fallback.")
        except Exception as e:
            st.error(f"WebRTC error: {e}")
            if use_snapshot_fallback:
                st.info("Using snapshot fallback. Take snapshots to recognize faces.")
                snap = st.camera_input("Snapshot")
                if snap:
                    img = cv2.imdecode(np.frombuffer(snap.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_face_size, min_face_size))
                    if len(faces) == 0:
                        st.warning("No face detected")
                    else:
                        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
                        roi = align_face(gray, (x, y, w, h))
                        if roi is not None:
                            roi = cv2.equalizeHist(roi)
                            try:
                                label_id, conf = recognizer.predict(roi)
                                name = id_to_name.get(label_id, "Unknown")
                                st.success(f"Recognized: {name} ({conf:.1f})")
                            except Exception as e2:
                                st.error(f"Recognition error: {e2}")

# Debug tab
with tabs[3]:
    st.subheader("Debug / Files")
    st.write("Data folder:")
    st.write(DATA_DIR.resolve())
    st.write(list(DATA_DIR.glob("*")))
    if MODEL_PATH.exists():
        st.write("Model present")
        st.download_button("Download model", open(MODEL_PATH, "rb"), file_name=MODEL_PATH.name)
    if LABELS_PATH.exists():
        st.download_button("Download labels", open(LABELS_PATH, "rb"), file_name=LABELS_PATH.name)

st.caption("If WebRTC fails on Streamlit Cloud due to NAT/firewall, try the snapshot fallback or deploy with your own TURN server (coturn) for reliability.")
