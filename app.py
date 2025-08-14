
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

APP_TITLE = "Live Face Recognition (Streamlit + WebRTC, OpenCV LBPH)"
DATA_DIR = Path("data")
MODELS_DIR = Path("models")
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
FACE_CASCADE = cv2.CascadeClassifier(CASCADE_PATH)

MODEL_PATH = MODELS_DIR / "lbph_model.xml"
LABELS_PATH = MODELS_DIR / "labels.json"

def load_labels() -> Dict[int, str]:
    if LABELS_PATH.exists():
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {int(k): v for k, v in data.items()}
    return {}

def save_labels(mapping: Dict[int, str]) -> None:
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump({int(k): v for k, v in mapping.items()}, f, ensure_ascii=False, indent=2)

def collect_dataset_stats() -> Dict[str, int]:
    stats = {}
    for person_dir in DATA_DIR.glob("*"):
        if person_dir.is_dir():
            stats[person_dir.name] = len(list(person_dir.glob("*.jpg"))) + len(list(person_dir.glob("*.png")))
    return stats

def prepare_training_data() -> Tuple[List[np.ndarray], List[int], Dict[int, str]]:
    images = []
    labels = []
    label_map: Dict[int, str] = {}
    next_label = 0

    for person_dir in sorted(DATA_DIR.glob("*")):
        if not person_dir.is_dir(): 
            continue
        person_name = person_dir.name
        label_map[next_label] = person_name

        for img_path in person_dir.glob("*"):
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                continue
            img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))
            for (x, y, w, h) in faces:
                roi = gray[y : y + h, x : x + w]
                roi = cv2.resize(roi, (200, 200))
                images.append(roi)
                labels.append(next_label)
        next_label += 1

    return images, labels, label_map

def train_and_save_model() -> Tuple[int, Dict[str, int]]:
    images, labels, label_map = prepare_training_data()
    if len(images) == 0 or len(set(labels)) == 0:
        raise RuntimeError("Not enough data to train. Please add images for at least one person.")

    recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
    recognizer.train(images, np.array(labels, dtype=np.int32))
    recognizer.write(str(MODEL_PATH))
    save_labels({k: v for k, v in label_map.items()})
    stats = collect_dataset_stats()
    return len(set(labels)), stats

def load_model():
    if MODEL_PATH.exists():
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(str(MODEL_PATH))
        labels = load_labels()
        return recognizer, labels
    return None, {}

def draw_label(img: np.ndarray, text: str, x: int, y: int) -> None:
    cv2.rectangle(img, (x, y - 25), (x + max(120, 8 * len(text)), y), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 5, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

# ---------- Streamlit UI ----------
st.set_page_config(page_title=APP_TITLE, page_icon="🎥", layout="wide")
st.title(APP_TITLE)
st.caption("No Pi? No problem. Enroll faces, train LBPH, and run LIVE recognition in your browser.")

with st.sidebar:
    st.header("Controls")
    conf_threshold = st.slider("Recognition threshold (LBPH, lower=more strict)", min_value=1, max_value=150, value=60, step=1)
    min_face_size = st.slider("Min face size (px)", 40, 200, 80, 10)
    st.markdown("---")
    st.subheader("Project Folders")
    st.code(f"DATA_DIR: {DATA_DIR.resolve()}\nMODELS_DIR: {MODELS_DIR.resolve()}", language="bash")

tabs = st.tabs(["📇 Enroll", "🧠 Train", "🔴 Live Recognition"])

# --------- ENROLL TAB ---------
with tabs[0]:
    st.subheader("Add images for a person")
    person_name = st.text_input("Person name", placeholder="e.g., Abhinav")
    st.write("Capture from your webcam (recommended) or upload existing images.")
    col1, col2 = st.columns(2)

    with col1:
        cam_img = st.camera_input("Capture image")
        if cam_img and person_name.strip():
            # Save captured image
            person_dir = DATA_DIR / person_name.strip().replace(" ", "_")
            ensure_dir(person_dir)
            img_bytes = cam_img.getvalue()
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            file_path = person_dir / f"{uuid.uuid4().hex}.jpg"
            # Use imencode to support unicode paths on some systems
            ok, buf = cv2.imencode(".jpg", img)
            if ok:
                buf.tofile(str(file_path))
                st.success(f"Saved capture for {person_name} → {file_path.name}")
        elif cam_img and not person_name.strip():
            st.warning("Please enter a person name before capturing.")

    with col2:
        uploads = st.file_uploader("Or upload image(s)", type=["jpg", "jpeg", "png", "bmp", "webp"], accept_multiple_files=True)
        if uploads and person_name.strip():
            person_dir = DATA_DIR / person_name.strip().replace(" ", "_")
            ensure_dir(person_dir)
            saved = 0
            for up in uploads:
                img_bytes = up.getvalue()
                img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                file_path = person_dir / f"{uuid.uuid4().hex}.jpg"
                ok, buf = cv2.imencode(".jpg", img)
                if ok:
                    buf.tofile(str(file_path))
                    saved += 1
            st.success(f"Saved {saved} image(s) for {person_name}.")

    st.markdown("### Current dataset")
    stats = collect_dataset_stats()
    if stats:
        st.json(stats)
    else:
        st.info("No images yet. Add some captures for at least one person.")

# --------- TRAIN TAB ---------
with tabs[1]:
    st.subheader("Train LBPH model")
    st.write("After enrolling images, click **Train** to build/update the model.")
    if st.button("Train / Retrain"):
        try:
            n_classes, stats = train_and_save_model()
            st.success(f"Training complete. Classes: {n_classes}. Model saved to `{MODEL_PATH}`.")
            st.json(stats)
        except Exception as e:
            st.error(str(e))

    # Show existing labels if any
    if LABELS_PATH.exists():
        st.markdown("#### Current labels")
        st.json(load_labels())

# --------- LIVE RECOGNITION TAB ---------
with tabs[2]:
    st.subheader("Start live webcam recognition")
    st.write("Click **Start** below and allow camera access.")

    recognizer, id_to_name = load_model()
    if recognizer is None or not id_to_name:
        st.warning("No trained model found. Please train the model first in the **Train** tab.")
    else:
        # Reverse mapping for display
        id_to_name = {int(k): v for k, v in id_to_name.items()}

        class VideoProcessor:
            def __init__(self):
                self.recognizer = recognizer
                self.labels = id_to_name

            def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                img = frame.to_ndarray(format="bgr24")
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = FACE_CASCADE.detectMultiScale(
                    gray,
                    scaleFactor=1.2,
                    minNeighbors=5,
                    minSize=(st.session_state.get("min_face_size", 80), st.session_state.get("min_face_size", 80)),
                )
                for (x, y, w, h) in faces:
                    roi = gray[y : y + h, x : x + w]
                    try:
                        roi = cv2.resize(roi, (200, 200))
                    except Exception:
                        continue
                    label_id, confidence = self.recognizer.predict(roi)
                    # LBPH: lower confidence means better match
                    if confidence <= st.session_state.get("conf_threshold", 60):
                        name = self.labels.get(label_id, "Unknown")
                        color = (0, 255, 0)
                        caption = f"{name} ({confidence:.1f})"
                    else:
                        name = "Unknown"
                        color = (0, 0, 255)
                        caption = f"{name} ({confidence:.1f})"

                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    draw_label(img, caption, x, y)

                return av.VideoFrame.from_ndarray(img, format="bgr24")

        # Keep sidebar sliders in session_state for use inside VideoProcessor
        st.session_state["conf_threshold"] = conf_threshold
        st.session_state["min_face_size"] = min_face_size

        rtc_config = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )

        webrtc_streamer(
            key="live",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=VideoProcessor,
            rtc_configuration=rtc_config,
            media_stream_constraints={"video": True, "audio": False},
        )

st.markdown("---")
st.caption("Tip: Capture at least 5–10 varied images per person (different lighting/angles) before training.")
