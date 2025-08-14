import os
import json
from pathlib import Path
from typing import Dict, List, Tuple
import uuid
import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from concurrent.futures import ThreadPoolExecutor
import threading
import pandas as pd
import base64
import shutil
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

APP_TITLE = "Live Face Recognition (Streamlit + WebRTC, OpenCV LBPH)"
DATA_DIR = Path("./data")  # Use relative path for local testing
MODELS_DIR = Path("./models")
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
FACE_CASCADE = cv2.CascadeClassifier(CASCADE_PATH)
EYE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")  # For face alignment

MODEL_PATH = MODELS_DIR / "lbph_model.xml"
LABELS_PATH = MODELS_DIR / "labels.json"

LBPH_RADIUS = 1
LBPH_NEIGHBORS = 8
LBPH_GRID_X = 8
LBPH_GRID_Y = 8
FACE_RESIZE = (200, 200)

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

def align_face(gray: np.ndarray, face_rect: Tuple[int, int, int, int]) -> np.ndarray:
    """Align face using eye detection for better recognition."""
    x, y, w, h = face_rect
    roi = gray[y:y+h, x:x+w]
    eyes = EYE_CASCADE.detectMultiScale(roi)
    if len(eyes) >= 2:
        eye_1 = eyes[0]
        eye_2 = eyes[1]
        eye_center_1 = (x + eye_1[0] + eye_1[2]//2, y + eye_1[1] + eye_1[3]//2)
        eye_center_2 = (x + eye_2[0] + eye_2[2]//2, y + eye_2[1] + eye_2[3]//2)
        delta_x = eye_center_2[0] - eye_center_1[0]
        delta_y = eye_center_2[1] - eye_center_1[1]
        angle = np.degrees(np.arctan2(delta_y, delta_x))
        M = cv2.getRotationMatrix2D((x + w//2, y + h//2), angle, 1)
        roi = cv2.warpAffine(roi, M, (w, h))
        return cv2.resize(roi, FACE_RESIZE, interpolation=cv2.INTER_LINEAR)
    return cv2.resize(roi, FACE_RESIZE, interpolation=cv2.INTER_LINEAR)

def prepare_training_data() -> Tuple[List[np.ndarray], List[int], Dict[int, str]]:
    images = []
    labels = []
    label_map: Dict[int, str] = {}
    next_label = 0

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for person_dir in sorted(DATA_DIR.glob("*")):
            if not person_dir.is_dir():
                continue
            person_name = person_dir.name
            label_map[next_label] = person_name

            for img_path in person_dir.glob("*"):
                if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                    continue
                futures.append(executor.submit(process_image, img_path, next_label))

            next_label += 1

        for future in futures:
            result = future.result()
            if result:
                img_roi, label = result
                images.append(img_roi)
                labels.append(label)

    if len(images) == 0 or len(set(labels)) == 0:
        raise RuntimeError("Not enough data to train. Please add images for at least one person.")

    return images, labels, label_map

def process_image(img_path: Path, label: int) -> Tuple[np.ndarray, int] | None:
    img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        logger.warning(f"Failed to load image: {img_path}")
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        logger.warning(f"No face detected in: {img_path}")
        return None
    x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
    roi = align_face(gray, (x, y, w, h))
    if roi is None:
        return None
    roi = cv2.equalizeHist(roi)
    return roi, label

def train_and_save_model() -> Tuple[int, Dict[str, int]]:
    images, labels, label_map = prepare_training_data()
    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=LBPH_RADIUS, neighbors=LBPH_NEIGHBORS, grid_x=LBPH_GRID_X, grid_y=LBPH_GRID_Y
    )
    recognizer.train(images, np.array(labels, dtype=np.int32))
    recognizer.write(str(MODEL_PATH))
    save_labels(label_map)
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

def sanitize_name(name: str) -> str:
    return "".join(c for c in name.strip().replace(" ", "_") if c.isalnum() or c in ["_", "-"])

def get_image_paths(person_dir: Path) -> List[Path]:
    return sorted([p for p in person_dir.glob("*") if p.suffix.lower() in [".jpg", ".png"]])

# Custom CSS for enhanced beauty
CUSTOM_CSS = """
<style>
    .stApp {
        background: linear-gradient(135deg, #e0f7fa, #ffffff);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stButton > button {
        background-color: #1976d2;
        color: white;
        border-radius: 8px;
        padding: 8px 16px;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #1565c0;
        transform: scale(1.05);
    }
    .stSlider > div > div > div {
        background-color: #1976d2;
        border-radius: 5px;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(135deg, #bbdefb, #e3f2fd);
        border-right: 1px solid #bbdefb;
    }
    .block-container {
        padding: 1.5rem;
        border-radius: 15px;
        background: rgba(255, 255, 255, 0.9);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
    }
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 1px solid #bbdefb;
    }
    .stCameraInput, .stFileUploader {
        border: 2px dashed #bbdefb;
        border-radius: 10px;
        padding: 1rem;
        background: #f5fbfd;
    }
    .stExpander {
        border: 1px solid #bbdefb;
        border-radius: 10px;
        background: #ffffff;
    }
</style>
"""

# ---------- Streamlit UI ----------
st.set_page_config(page_title=APP_TITLE, page_icon="🎥", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.title(f"{APP_TITLE} ✨")
st.caption("No Pi? No problem. Enroll faces, train LBPH, and run LIVE recognition with enhanced features.")

with st.sidebar:
    st.header("Controls 🎛️")
    conf_threshold = st.slider("Recognition Threshold (LBPH)", 1, 150, 60, 1, help="Lower values for stricter matches.")
    min_face_size = st.slider("Min Face Size (px)", 40, 200, 80, 10, help="Detects smaller faces with lower values.")
    scale_factor = st.slider("Scale Factor", 1.05, 1.5, 1.1, 0.05, help="Lower for more sensitivity.")
    min_neighbors = st.slider("Min Neighbors", 3, 10, 5, 1, help="Higher reduces false positives.")
    st.markdown("---")
    st.subheader("Project Folders 📁")
    st.code(f"DATA_DIR: {DATA_DIR.resolve()}\nMODELS_DIR: {MODELS_DIR.resolve()}", language="bash")
    st.markdown("---")
    st.subheader("Export Dataset 📦")
    if st.button("Download Dataset as ZIP"):
        zip_path = shutil.make_archive("dataset", "zip", DATA_DIR)
        with open(zip_path, "rb") as f:
            bytes_data = f.read()
            b64 = base64.b64encode(bytes_data).decode()
            href = f'<a href="data:file/zip;base64,{b64}" download="dataset.zip">Download ZIP</a>'
            st.markdown(href, unsafe_allow_html=True)
        os.remove(zip_path)

tabs = st.tabs(["📇 Enroll", "🧠 Train", "🔴 Live Recognition", "📊 Stats"])

# --------- ENROLL TAB ---------
with tabs[0]:
    st.subheader("Add Images for a Person 📸")
    person_name = st.text_input("Person Name", placeholder="e.g., Abhinav", key="enroll_name")
    sanitized_name = sanitize_name(person_name) if person_name else ""
    st.write("Capture via webcam or upload images. Auto-detects faces for quality.")

    col1, col2 = st.columns(2)

    with col1:
        cam_img = st.camera_input("Capture Image", help="Click to take a photo.")
        if cam_img and sanitized_name:
            person_dir = DATA_DIR / sanitized_name
            ensure_dir(person_dir)
            img_bytes = cam_img.getvalue()
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            if len(faces) == 0:
                st.warning("No face detected. Try adjusting your position or lighting.")
            else:
                file_path = person_dir / f"{uuid.uuid4().hex}.jpg"
                ok, buf = cv2.imencode(".jpg", img)
                if ok:
                    buf.tofile(str(file_path))
                    st.success(f"Saved for {person_name} → {file_path.name}")
                    st.image(img_bytes, caption="Captured Image", use_column_width=True)
        elif cam_img and not sanitized_name:
            st.warning("Please enter a name.")

    with col2:
        uploads = st.file_uploader("Upload Image(s)", type=["jpg", "jpeg", "png", "bmp", "webp"], accept_multiple_files=True, help="Drag or browse files (max 200MB each).")
        if uploads and sanitized_name:
            person_dir = DATA_DIR / sanitized_name
            ensure_dir(person_dir)
            saved = 0
            previews = []
            for up in uploads:
                img_bytes = up.getvalue()
                img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if img is None:
                    st.warning(f"Invalid image: {up.name}")
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
                if len(faces) == 0:
                    st.warning(f"No face in {up.name}. Skipping.")
                    continue
                file_path = person_dir / f"{uuid.uuid4().hex}.jpg"
                ok, buf = cv2.imencode(".jpg", img)
                if ok:
                    buf.tofile(str(file_path))
                    saved += 1
                    previews.append(img_bytes)
            if saved > 0:
                st.success(f"Saved {saved} image(s) for {person_name}")
                for prev in previews[:4]:  # Limit to 4 previews
                    st.image(prev, caption="Uploaded Image", use_column_width=True)

    st.markdown("### Current Dataset 🌟")
    stats = collect_dataset_stats()
    if stats:
        st.json(stats)
    else:
        st.info("No images yet. Start capturing!")

    if stats:
        st.markdown("### Dataset Preview & Management 🛠️")
        for person in stats.keys():
            with st.expander(f"{person} ({stats[person]} images)"):
                person_dir = DATA_DIR / person
                image_paths = get_image_paths(person_dir)
                cols = st.columns(min(4, len(image_paths)))  # Dynamic columns
                for i, img_path in enumerate(image_paths):
                    with cols[i % len(cols)]:
                        img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
                        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_column_width=True, caption=img_path.name)
                        if st.button("🗑️ Delete", key=f"del_{person}_{img_path.name}"):
                            os.remove(img_path)
                            st.success(f"Deleted {img_path.name}")
                            st.rerun()

        delete_person = st.selectbox("Delete Person's Dataset", [""] + list(stats.keys()))
        if delete_person and st.button("Confirm Delete", type="primary"):
            shutil.rmtree(DATA_DIR / delete_person)
            st.success(f"Deleted {delete_person}'s dataset")
            st.rerun()

# --------- TRAIN TAB ---------
with tabs[1]:
    st.subheader("Train LBPH Model 🚀")
    st.write("Train or retrain after enrolling images.")
    auto_train = st.checkbox("Auto-retrain on changes", value=False)
    if auto_train and stats and not MODEL_PATH.exists():
        with st.spinner("Auto-training..."):
            try:
                n_classes, stats = train_and_save_model()
                st.success(f"Auto-trained. Classes: {n_classes}")
            except Exception as e:
                st.error(f"Auto-train failed: {e}")
    if st.button("Train / Retrain", type="primary"):
        with st.spinner("Training..."):
            try:
                n_classes, stats = train_and_save_model()
                st.success(f"Training complete. Classes: {n_classes}. Saved to `{MODEL_PATH}`")
                st.json(stats)
            except Exception as e:
                st.error(f"Error: {e}")

    if LABELS_PATH.exists():
        st.markdown("#### Labels 📋")
        st.json(load_labels())

    if MODEL_PATH.exists():
        st.markdown("#### Export Model 💾")
        with open(MODEL_PATH, "rb") as f:
            st.download_button("Download Model", f, file_name="lbph_model.xml")
        with open(LABELS_PATH, "rb") as f:
            st.download_button("Download Labels", f, file_name="labels.json")

# --------- LIVE RECOGNITION TAB ---------
with tabs[2]:
    st.subheader("Live Webcam Recognition 🎥")
    st.write("Start to recognize faces in real-time.")

    recognizer, id_to_name = load_model()
    if recognizer is None or not id_to_name:
        st.warning("No model trained. Please train first.")
    else:
        id_to_name = {int(k): v for k, v in id_to_name.items()}

        class VideoProcessor:
            def __init__(self):
                self.recognizer = recognizer
                self.labels = id_to_name
                self.lock = threading.Lock()
                self.detected = {}

            def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                img = frame.to_ndarray(format="bgr24")
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)
                faces = FACE_CASCADE.detectMultiScale(
                    gray,
                    scaleFactor=st.session_state.get("scale_factor", 1.1),
                    minNeighbors=st.session_state.get("min_neighbors", 5),
                    minSize=(st.session_state.get("min_face_size", 80), st.session_state.get("min_face_size", 80)),
                )
                for (x, y, w, h) in faces:
                    roi = align_face(gray, (x, y, w, h))
                    try:
                        roi = cv2.equalizeHist(roi)
                    except Exception:
                        continue
                    with self.lock:
                        label_id, confidence = self.recognizer.predict(roi)
                    name = self.labels.get(label_id, "Unknown")
                    if confidence <= st.session_state.get("conf_threshold", 60):
                        color = (0, 255, 0)
                        caption = f"{name} ({confidence:.1f})"
                        self.detected[name] = self.detected.get(name, 0) + 1
                    else:
                        color = (0, 0, 255)
                        caption = f"Unknown ({confidence:.1f})"
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    draw_label(img, caption, x, y)

                return av.VideoFrame.from_ndarray(img, format="bgr24")

            def detected_names(self):
                return self.detected

        st.session_state["conf_threshold"] = conf_threshold
        st.session_state["min_face_size"] = min_face_size
        st.session_state["scale_factor"] = scale_factor
        st.session_state["min_neighbors"] = min_neighbors

        rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
        webrtc_ctx = webrtc_streamer(
            key="live",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=VideoProcessor,
            rtc_configuration=rtc_config,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        if webrtc_ctx.state.playing:
            st.markdown("#### Detected Faces 👥")
            processor = webrtc_ctx.video_processor
            if processor:
                detected = processor.detected_names()
                if detected:
                    st.write("Recently detected:", ", ".join([f"{k} ({v} times)" for k, v in detected.items()]))

# --------- STATS TAB ---------
with tabs[3]:
    st.subheader("Dataset & Model Statistics 📉")
    stats = collect_dataset_stats()
    if stats:
        df = pd.DataFrame(list(stats.items()), columns=["Person", "Images"])
        st.bar_chart(df.set_index("Person"))
        st.dataframe(df.style.background_gradient(cmap="Blues"))
    else:
        st.info("No dataset yet.")

    if MODEL_PATH.exists():
        st.markdown("#### Model Info ℹ️")
        labels = load_labels()
        st.write(f"Number of Classes: {len(labels)}")
        st.json(labels)

st.markdown("---")
st.caption("Tip: Use 5–10 varied images per person. Enjoy the enhanced UI and features! 🌈")
