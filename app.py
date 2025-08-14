# app.py
"""
Polished Streamlit Live Face Recognition App
Features:
- Enroll/register faces (camera upload or file upload)
- Train LBPH model with threaded preprocessing
- Live recognition via WebRTC with TURN fallback and snapshot fallback
- Multiple-face detection, alignment, confidence -> similarity %
- Recognition history panel and CSV export
- Per-person dataset management, model/labels export
- Sleek UI with CSS, light/dark friendly
"""

import os
import json
import uuid
import shutil
import base64
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
import threading
import io
import time
import pandas as pd

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# ---------- Config & Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("streamlit_face_rec_polished")

APP_TITLE = "FaceX — Polished Live Face Recognition"
DATA_DIR = Path("./data")
MODELS_DIR = Path("./models")
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
FACE_CASCADE = cv2.CascadeClassifier(CASCADE_PATH)
EYE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

MODEL_PATH = MODELS_DIR / "lbph_model.xml"
LABELS_PATH = MODELS_DIR / "labels.json"

LBPH_PARAMS = dict(radius=1, neighbors=8, grid_x=8, grid_y=8)
FACE_RESIZE = (200, 200)

# RTC fallback servers (STUN + free TURN)
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

# ---------- Utility functions ----------
def sanitize_name(name: str) -> str:
    return "".join(c for c in name.strip().replace(" ", "_") if c.isalnum() or c in ["_", "-"])

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_labels() -> Dict[int, str]:
    if LABELS_PATH.exists():
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {int(k): v for k, v in data.items()}
    return {}

def save_labels(mapping: Dict[int, str]):
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump({int(k): v for k, v in mapping.items()}, f, ensure_ascii=False, indent=2)

def list_persons() -> List[str]:
    return sorted([p.name for p in DATA_DIR.glob("*") if p.is_dir()])

def get_image_paths(person_dir: Path) -> List[Path]:
    return sorted([p for p in person_dir.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]])

# ---------- Face processing ----------
def align_face(gray: np.ndarray, face_rect: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    x, y, w, h = face_rect
    roi = gray[y:y+h, x:x+w]
    if roi.size == 0:
        return None
    try:
        eyes = EYE_CASCADE.detectMultiScale(roi)
        if len(eyes) >= 2:
            e1, e2 = eyes[:2]
            center1 = (x + e1[0] + e1[2] // 2, y + e1[1] + e1[3] // 2)
            center2 = (x + e2[0] + e2[2] // 2, y + e2[1] + e2[3] // 2)
            dx = center2[0] - center1[0]
            dy = center2[1] - center1[1]
            angle = np.degrees(np.arctan2(dy, dx))
            M = cv2.getRotationMatrix2D((x + w // 2, y + h // 2), angle, 1.0)
            aligned = cv2.warpAffine(roi, M, (w, h))
            return cv2.resize(aligned, FACE_RESIZE)
    except Exception:
        pass
    try:
        return cv2.resize(roi, FACE_RESIZE)
    except Exception:
        return None

def process_image_file(img_path: Path, label: int) -> Optional[Tuple[np.ndarray, int]]:
    try:
        img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60,60))
        if len(faces) == 0:
            return None
        face_rect = max(faces, key=lambda r: r[2] * r[3])
        roi = align_face(gray, face_rect)
        if roi is None:
            return None
        roi = cv2.equalizeHist(roi)
        return roi, label
    except Exception as e:
        logger.exception("Error processing image %s: %s", img_path, e)
        return None

# ---------- Recognizer creation ----------
def create_lbph_recognizer():
    if hasattr(cv2, "face"):
        try:
            return cv2.face.LBPHFaceRecognizer_create(**LBPH_PARAMS)
        except Exception as e:
            logger.warning("cv2.face exists but create failed: %s", e)
    raise RuntimeError("OpenCV 'face' module not available. Install opencv-contrib-python.")

# ---------- Training pipeline ----------
def prepare_training_data(max_workers: int = 4):
    images = []
    labels = []
    label_map = {}
    next_label = 0
    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for person_dir in sorted(DATA_DIR.glob("*")):
            if not person_dir.is_dir():
                continue
            label_map[next_label] = person_dir.name
            for p in get_image_paths(person_dir):
                futures.append(ex.submit(process_image_file, p, next_label))
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

def train_and_save_model():
    images, labels, label_map = prepare_training_data(max_workers=min(8, max(1, os.cpu_count() or 1)))
    recognizer = create_lbph_recognizer()
    recognizer.train(images, np.array(labels, dtype=np.int32))
    recognizer.write(str(MODEL_PATH))
    save_labels(label_map)
    stats = {p.name: len(get_image_paths(p)) for p in DATA_DIR.glob("*") if p.is_dir()}
    return len(set(labels)), stats

def load_model():
    if MODEL_PATH.exists() and LABELS_PATH.exists():
        recognizer = create_lbph_recognizer()
        recognizer.read(str(MODEL_PATH))
        labels = load_labels()
        return recognizer, labels
    return None, {}

# ---------- Similarity mapping ----------
def confidence_to_similarity(confidence: float, max_conf: float) -> float:
    # Map LBPH confidence (lower=better). We'll clamp and invert into 0-100%
    # similarity = max(0, min(100, 100 * (1 - confidence / max_conf)))
    if confidence is None or np.isnan(confidence):
        return 0.0
    val = 100.0 * (1.0 - (confidence / float(max_conf)))
    if val < 0.0:
        val = 0.0
    if val > 100.0:
        val = 100.0
    return float(val)

# ---------- Session init ----------
if "history" not in st.session_state:
    st.session_state["history"] = []  # list of dicts: timestamp, name, similarity

if "last_train_stats" not in st.session_state:
    st.session_state["last_train_stats"] = {}

# ---------- CSS (polished) ----------
CUSTOM_CSS = """
<style>
.block-container{padding:1rem 2rem;}
.stApp { font-family: Inter, "Segoe UI", Roboto, system-ui, -apple-system, "Helvetica Neue", Arial; }
.title { font-weight:700; }
.card { background: linear-gradient(180deg, rgba(255,255,255,0.85), rgba(245,245,245,0.6)); padding:12px; border-radius:12px; box-shadow: 0 6px 18px rgba(0,0,0,0.06); }
.small-muted { color: #666; font-size: 0.9rem; }
</style>
"""
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------- Header ----------
st.markdown(f"## {APP_TITLE}  ✨")
st.markdown("A polished demo: register faces, train LBPH, and run live or snapshot recognition. Works on Streamlit Cloud (use snapshot fallback if WebRTC stalls).")

# ---------- Sidebar controls ----------
with st.sidebar:
    st.header("Controls")
    conf_threshold = st.slider("Recognition threshold (lower=stricter)", 1, 150, 60, 1)
    max_confidence = st.slider("Max confidence (for similarity mapping)", 50, 300, 150, 10, help="Adjust how LBPH confidence maps to similarity %")
    min_face_size = st.slider("Min face size (px)", 40, 300, 80, 10)
    scale_factor = st.slider("Cascade scaleFactor", 1.05, 1.5, 1.1, 0.01)
    min_neighbors = st.slider("Cascade minNeighbors", 3, 12, 5, 1)
    use_snapshot_fallback = st.checkbox("Enable snapshot fallback (camera input) if WebRTC fails", value=True)
    st.markdown("---")
    st.subheader("Dataset")
    if st.button("Clear All Data"):
        if st.confirm("Delete all enrolled images and models? This cannot be undone."):
            try:
                shutil.rmtree(DATA_DIR)
                shutil.rmtree(MODELS_DIR)
            except Exception:
                pass
            DATA_DIR.mkdir(exist_ok=True)
            MODELS_DIR.mkdir(exist_ok=True)
            st.success("All data cleared. Reload the app.")
    persons = list_persons()
    if persons:
        sel_person = st.selectbox("Manage person", [""] + persons)
        if sel_person:
            if st.button("Delete person dataset"):
                shutil.rmtree(DATA_DIR / sel_person)
                st.success(f"Deleted {sel_person}.")
    st.markdown("---")
    st.subheader("Export")
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            st.download_button("Download model (XML)", f, file_name="lbph_model.xml")
    if LABELS_PATH.exists():
        with open(LABELS_PATH, "rb") as f:
            st.download_button("Download labels (JSON)", f, file_name="labels.json")
    if st.button("Export recognition history CSV"):
        hist = st.session_state.get("history", [])
        if hist:
            df = pd.DataFrame(hist)
            csv = df.to_csv(index=False).encode()
            st.download_button("Download CSV", csv, file_name="recognitions.csv", mime="text/csv")
        else:
            st.info("No history yet.")

# ---------- Main layout: Tabs ----------
tabs = st.tabs(["Register ▶", "Train ▶", "Live ▶", "History & Stats ▶"])

# ---------- Register tab ----------
with tabs[0]:
    st.subheader("Register / Enroll Faces")
    col1, col2 = st.columns([2, 1])
    with col1:
        name = st.text_input("Person name", placeholder="Full name or username")
        st.markdown("Capture images via webcam or upload multiple images. Good variety (angles, lighting) improves recognition.")
        cam = st.camera_input("Take snapshot (click to capture)", key="reg_cam")
        uploads = st.file_uploader("Or upload images (multiple allowed)", type=["jpg","jpeg","png","bmp","webp"], accept_multiple_files=True)
        if st.button("Save enrollment"):
            if not name:
                st.warning("Enter a person name first.")
            else:
                sanitized = sanitize_name(name)
                person_dir = DATA_DIR / sanitized
                ensure_dir(person_dir)
                saved = 0
                if cam:
                    img_bytes = cam.getvalue()
                    img = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if img is not None:
                        p = person_dir / f"{uuid.uuid4().hex}.jpg"
                        ok, buf = cv2.imencode(".jpg", img)
                        if ok:
                            buf.tofile(str(p))
                            saved += 1
                for up in uploads:
                    img_bytes = up.getvalue()
                    img = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if img is None:
                        continue
                    p = person_dir / f"{uuid.uuid4().hex}.jpg"
                    ok, buf = cv2.imencode(".jpg", img)
                    if ok:
                        buf.tofile(str(p))
                        saved += 1
                if saved > 0:
                    st.success(f"Saved {saved} image(s) for {name}")
                else:
                    st.warning("No valid images captured/uploaded.")
    with col2:
        st.markdown("#### Dataset Preview")
        persons = list_persons()
        if persons:
            for p in persons:
                cnt = len(get_image_paths(DATA_DIR / p))
                st.markdown(f"- **{p}** — {cnt} images")
        else:
            st.info("No enrolled faces yet.")

# ---------- Train tab ----------
with tabs[1]:
    st.subheader("Train Model")
    st.markdown("Train LBPH on enrolled faces. Training runs on the server and stores model & labels.")
    if st.button("Train / Retrain Model"):
        start = time.time()
        with st.spinner("Training... This may take a while depending on dataset size."):
            try:
                n_classes, stats = train_and_save_model()
                st.success(f"Training completed — {n_classes} classes.")
                st.session_state["last_train_stats"] = stats
                st.json(stats)
                st.write(f"Time: {time.time() - start:.1f}s")
            except Exception as e:
                st.error(f"Training failed: {e}")

    if st.session_state.get("last_train_stats"):
        st.markdown("#### Last training stats")
        st.json(st.session_state["last_train_stats"])

# ---------- Live tab ----------
with tabs[2]:
    st.subheader("Live Recognition")
    st.markdown("Use live webcam (WebRTC) or snapshot fallback. Multiple faces supported. Similarity % shown.")

    recognizer, labels = load_model()
    if recognizer is None or not labels:
        st.warning("No trained model found. Train first on the Train tab.")
        # still offer snapshot to test face detection
        if use_snapshot_fallback:
            snap = st.camera_input("Take a snapshot to test face detection")
            if snap:
                img = cv2.imdecode(np.frombuffer(snap.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=(min_face_size, min_face_size))
                if faces is None or len(faces) == 0:
                    st.warning("No faces detected.")
                else:
                    st.success(f"Detected {len(faces)} face(s).")
    else:
        id_to_name = {int(k): v for k, v in labels.items()}

        class VideoProcessor:
            def __init__(self):
                self.recognizer = recognizer
                self.labels = id_to_name
                self.lock = threading.Lock()

            def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                img = frame.to_ndarray(format="bgr24")
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)
                faces = FACE_CASCADE.detectMultiScale(
                    gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=(min_face_size, min_face_size)
                )
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
                    # Determine match & similarity
                    sim = confidence_to_similarity(conf, max_confidence)
                    name = self.labels.get(label_id, "Unknown")
                    is_match = (conf <= conf_threshold)
                    color = (0, 200, 0) if is_match else (0, 0, 200)
                    caption = f"{name} — {sim:.1f}%"
                    # Draw
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(img, caption, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
                    # Save to history (non-blocking)
                    st.session_state["history"].insert(0, {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "name": name if is_match else "Unknown", "similarity": round(sim, 1)})
                    # Limit history size
                    if len(st.session_state["history"]) > 200:
                        st.session_state["history"] = st.session_state["history"][:200]
                return av.VideoFrame.from_ndarray(img, format="bgr24")

        rtc_conf = RTCConfiguration({"iceServers": RTC_ICE_SERVERS})
        try:
            ctx = webrtc_streamer(
                key="live_stream",
                mode=WebRtcMode.SENDRECV,
                video_processor_factory=VideoProcessor,
                rtc_configuration=rtc_conf,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )
            if ctx.state.playing:
                st.success("Live stream connected.")
            else:
                st.info("WebRTC not connected yet. If it stalls, use snapshot fallback below.")
        except Exception as e:
            st.error(f"WebRTC error: {e}")
            if use_snapshot_fallback:
                st.info("Using snapshot fallback. Take a snapshot to recognize.")
                snap = st.camera_input("Snapshot for recognition")
                if snap:
                    img = cv2.imdecode(np.frombuffer(snap.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=(min_face_size, min_face_size))
                    if faces is None or len(faces) == 0:
                        st.warning("No face detected.")
                    else:
                        results = []
                        for (x, y, w, h) in faces:
                            roi = align_face(gray, (x, y, w, h))
                            if roi is None:
                                continue
                            roi = cv2.equalizeHist(roi)
                            try:
                                label_id, conf = recognizer.predict(roi)
                            except Exception as e2:
                                st.error(f"Recognition error: {e2}")
                                continue
                            sim = confidence_to_similarity(conf, max_confidence)
                            name = id_to_name.get(label_id, "Unknown")
                            is_match = (conf <= conf_threshold)
                            results.append((name if is_match else "Unknown", sim, conf))
                            st.session_state["history"].insert(0, {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "name": name if is_match else "Unknown", "similarity": round(sim, 1)})
                        # Show annotated preview
                        for (x, y, w, h), (n, sim, conf) in zip(faces, results):
                            color = (0, 200, 0) if (conf <= conf_threshold) else (0, 0, 200)
                            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                            cv2.putText(img, f"{n} {sim:.1f}%", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
                        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_column_width=True)

# ---------- History & Stats tab ----------
with tabs[3]:
    st.subheader("Recognition History & Stats")
    hist = st.session_state.get("history", [])
    if hist:
        df = pd.DataFrame(hist)
        st.dataframe(df.head(200))
        st.markdown("#### Quick stats")
        names = df['name'].value_counts().to_dict()
        st.json(names)
        if st.button("Clear history"):
            st.session_state["history"] = []
            st.success("Cleared history.")
    else:
        st.info("No recognitions yet. Do a live test to generate history.")

st.markdown("---")
st.caption("Tip: Add 10-20 diverse images per person for robust recognition. On Streamlit Cloud use snapshot fallback if live WebRTC has connectivity issues.")
