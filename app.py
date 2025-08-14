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

APP_TITLE = "Live Face Recognition with Glasses Support (Streamlit + WebRTC, OpenCV LBPH)"
DATA_DIR = Path("data")
MODELS_DIR = Path("models")
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Load Haar cascades for face and eye detection
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
EYE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
FACE_CASCADE = cv2.CascadeClassifier(CASCADE_PATH)
EYE_CASCADE = cv2.CascadeClassifier(EYE_CASCADE_PATH)

MODEL_PATH = MODELS_DIR / "lbph_model.xml"
LABELS_PATH = MODELS_DIR / "labels.json"

# Optimized LBPH parameters for glasses
LBPH_RADIUS = 2
LBPH_NEIGHBORS = 10
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

def reduce_glass_reflection(gray: np.ndarray) -> np.ndarray:
    """Reduce reflections on glasses using adaptive thresholding."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)
    return cv2.inpaint(gray, thresh, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

def process_image(img_path: Path, label: int) -> Tuple[np.ndarray, int] | None:
    """Process a single image, optimized for glasses."""
    img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # Try face detection with eye detection as fallback
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=4, minSize=(60, 60))
    if len(faces) == 0:
        # Fallback to eye detection
        eyes = EYE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))
        if len(eyes) >= 1:
            ex, ey, ew, eh = eyes[0]
            x = max(0, ex - ew)
            y = max(0, ey - eh)
            w = ew * 3
            h = eh * 4
            faces = [(x, y, w, h)]
        else:
            return None
    
    x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
    roi = gray[y : y + h, x : x + w]
    try:
        roi = reduce_glass_reflection(roi)
        roi = cv2.resize(roi, FACE_RESIZE, interpolation=cv2.INTER_LINEAR)
    except Exception:
        return None
    return roi, label

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
        try:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read(str(MODEL_PATH))
            labels = load_labels()
            return recognizer, labels
        except Exception as e:
            st.error(f"Failed to load model: {str(e)}")
            return None, {}
    return None, {}

def draw_label(img: np.ndarray, text: str, x: int, y: int) -> None:
    cv2.rectangle(img, (x, y - 25), (x + max(120, 8 * len(text)), y), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 5, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def sanitize_name(name: str) -> str:
    return "".join(c for c in name.strip().replace(" ", "_") if c.isalnum() or c in ["_", "-"])

# ---------- Streamlit UI ----------
st.set_page_config(page_title=APP_TITLE, page_icon="🎥", layout="wide")
st.title(APP_TITLE)
st.caption("Enroll faces (with and without glasses), train LBPH, and run LIVE recognition. Optimized for glasses.")

with st.sidebar:
    st.header("Controls")
    conf_threshold = st.slider("Recognition threshold (lower=more strict)", min_value=1, max_value=150, value=50, step=1)
    min_face_size = st.slider("Min face size (px)", 40, 200, 60, 10)
    scale_factor = st.slider("Detection scale factor (lower=more sensitive)", 1.05, 1.5, 1.05, 0.05)
    min_neighbors = st.slider("Min neighbors for detection", 3, 10, 4, 1)
    st.markdown("---")
    st.subheader("Project Folders")
    st.code(f"DATA_DIR: {DATA_DIR.resolve()}\nMODELS_DIR: {MODELS_DIR.resolve()}", language="bash")

tabs = st.tabs(["📇 Enroll", "🧠 Train", "🔴 Live Recognition"])

# --------- ENROLL TAB ---------
with tabs[0]:
    st.subheader("Add images for a person")
    person_name = st.text_input("Person name", placeholder="e.g., Abhinav")
    sanitized_name = sanitize_name(person_name) if person_name else ""
    st.write("Capture from your webcam or upload images. **Include images with and without glasses** for better recognition.")
    
    col1, col2 = st.columns(2)

    with col1:
        cam_img = st.camera_input("Capture image")
        if cam_img and sanitized_name:
            person_dir = DATA_DIR / sanitized_name
            ensure_dir(person_dir)
            img_bytes = cam_img.getvalue()
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=4, minSize=(60, 60))
            if len(faces) == 0:
                eyes = EYE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))
                if len(eyes) >= 1:
                    ex, ey, ew, eh = eyes[0]
                    x = max(0, ex - ew)
                    y = max(0, ey - eh)
                    w = ew * 3
                    h = eh * 4
                    faces = [(x, y, w, h)]
                else:
                    st.warning("No face or eyes detected in the capture. Try again.")
                    faces = []
            if len(faces) > 0:
                file_path = person_dir / f"{uuid.uuid4().hex}.jpg"
                ok, buf = cv2.imencode(".jpg", img)
                if ok:
                    buf.tofile(str(file_path))
                    st.success(f"Saved capture for {person_name} → {file_path.name}")
                    st.image(img_bytes, caption="Captured Image (Face Detected)", use_column_width=True)
        elif cam_img and not sanitized_name:
            st.warning("Please enter a valid person name before capturing.")

    with col2:
        uploads = st.file_uploader("Or upload image(s)", type=["jpg", "jpeg", "png", "bmp", "webp"], accept_multiple_files=True)
        if uploads and sanitized_name:
            person_dir = DATA_DIR / sanitized_name
            ensure_dir(person_dir)
            saved = 0
            previews = []
            for up in uploads:
                if len(up.getvalue()) > 10 * 1024 * 1024:
                    st.warning(f"Image {up.name} is too large. Max size: 10MB.")
                    continue
                img_bytes = up.getvalue()
                img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                gray = clahe.apply(gray)
                faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=4, minSize=(60, 60))
                if len(faces) == 0:
                    eyes = EYE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))
                    if len(eyes) >= 1:
                        ex, ey, ew, eh = eyes[0]
                        x = max(0, ex - ew)
                        y = max(0, ey - eh)
                        w = ew * 3
                        h = eh * 4
                        faces = [(x, y, w, h)]
                    else:
                        st.warning(f"No face or eyes detected in {up.name}. Skipping.")
                        continue
                file_path = person_dir / f"{uuid.uuid4().hex}.jpg"
                ok, buf = cv2.imencode(".jpg", img)
                if ok:
                    buf.tofile(str(file_path))
                    saved += 1
                    previews.append(img_bytes)
            if saved > 0:
                st.success(f"Saved {saved} image(s) for {person_name}.")
                for prev in previews:
                    st.image(prev, caption="Uploaded Image (Face Detected)", use_column_width=True)

    st.markdown("### Current dataset")
    stats = collect_dataset_stats()
    if stats:
        st.json(stats)
    else:
        st.info("No images yet. Add images (with/without glasses) for at least one person.")

    if stats:
        delete_person = st.selectbox("Delete a person's dataset", [""] + list(stats.keys()))
        if delete_person and st.button("Confirm Delete"):
            import shutil
            shutil.rmtree(DATA_DIR / delete_person)
            st.success(f"Deleted dataset for {delete_person}.")
            st.rerun()

# --------- TRAIN TAB ---------
with tabs[1]:
    st.subheader("Train LBPH model")
    st.write("Train with images including faces with and without glasses for better accuracy.")
    ifਮ: if st.button("Train / Retrain"):
        with st.spinner("Training model..."):
            try:
                n_classes, stats = train_and_save_model()
                st.success(f"Training complete. Classes: {n_classes}. Model saved to `{MODEL_PATH}`.")
                st.json(stats)
            except Exception as e:
                st.error(str(e))

    if LABELS_PATH.exists():
        st.markdown("#### Current labels")
        st.json(load_labels())

# --------- LIVE RECOGNITION TAB ---------
with tabs[2]:
    st.subheader("Start live webcam recognition")
    st.write("Recognizes faces with and without glasses. Adjust sliders for sensitivity.")

    recognizer, id_to_name = load_model()
    if recognizer is None or not id_to_name:
        st.warning("No trained model found. Please train the model first in the **Train** tab.")
    else:
        id_to_name = {int(k): v for k, v in id_to_name.items()}

        class VideoProcessor:
            def __init__(self):
                self.recognizer = recognizer
                self.labels = id_to_name
                self.lock = threading.Lock()

            def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                img = frame.to_ndarray(format="bgr24")
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                gray = clahe.apply(gray)
                faces = FACE_CASCADE.detectMultiScale(
                    gray,
                    scaleFactor=st.session_state.get("scale_factor", 1.05),
                    minNeighbors=st.session_state.get("min_neighbors", 4),
                    minSize=(st.session_state.get("min_face_size", 60), st.session_state.get("min_face_size", 60)),
                )
                if len(faces) == 0:
                    eyes = EYE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))
                    if len(eyes) >= 1:
                        ex, ey, ew, eh = eyes[0]
                        x = max(0, ex - ew)
                        y = max(0, ey - eh)
                        w = ew * 3
                        h = eh * 4
                        faces = [(x, y, w, h)]

                for (x, y, w, h) in faces:
                    roi = gray[y : y + h, x : x + w]
                    try:
                        roi = reduce_glass_reflection(roi)
                        roi = cv2.resize(roi, FACE_RESIZE, interpolation=cv2.INTER_LINEAR)
                    except Exception:
                        continue
                    with self.lock:
                        label_id, confidence = self.recognizer.predict(roi)
                    conf_threshold = st.session_state.get("conf_threshold", 50)
                    if confidence <= conf_threshold:
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

        st.session_state["conf_threshold"] = conf_threshold
        st.session_state["min_face_size"] = min_face_size
        st.session_state["scale_factor"] = scale_factor
        st.session_state["min_neighbors"] = min_neighbors

        rtc_config = RTCConfiguration(
            {"iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]}
            ]}
        )

        webrtc_streamer(
            key="live",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=VideoProcessor,
            rtc_configuration=rtc_config,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

st.markdown("---")
st.caption("Tip: Capture 5–10 images per person, including with and without glasses, in varied lighting/angles. Optimized with CLAHE and reflection reduction.")
