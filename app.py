import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import uuid
import cv2
import numpy as np
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
import random
import time
import pandas as pd
from streamlit import rerun
import imgaug.augmenters as iaa
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('face_recognition.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

APP_TITLE = "Streamlit Face Recognition (OpenCV LBPH - Enhanced Model)"
DATA_DIR = Path("data")
MODELS_DIR = Path("models")
LOGS_DIR = Path("logs")
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
FACE_CASCADE = cv2.CascadeClassifier(CASCADE_PATH)

MODEL_PATH = MODELS_DIR / "lbph_model.xml"
LABELS_PATH = MODELS_DIR / "labels.json"

# Enhanced LBPH parameters
LBPH_RADIUS = 2  # Increased for more robust feature capture
LBPH_NEIGHBORS = 8  # More neighbors for better matching
LBPH_GRID_X = 8  # Finer grid for detailed histograms
LBPH_GRID_Y = 8
FACE_RESIZE = (200, 200)

# Minimum photos required per person
MIN_PHOTOS_PER_PERSON = 1
MIN_FACE_SIZE = 100  # Minimum face size (pixels) for quality control

def load_labels() -> Dict[int, str]:
    try:
        if LABELS_PATH.exists():
            with open(LABELS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {int(k): v for k, v in data.items()}
        return {}
    except Exception as e:
        logger.error(f"Error loading labels: {e}")
        return {}

def save_labels(mapping: Dict[int, str]) -> None:
    try:
        with open(LABELS_PATH, "w", encoding="utf-8") as f:
            json.dump({int(k): v for k, v in mapping.items()}, f, ensure_ascii=False, indent=2)
        logger.info(f"Labels saved to {LABELS_PATH}")
    except Exception as e:
        logger.error(f"Error saving labels: {e}")

def collect_dataset_stats() -> Dict[str, int]:
    stats = {}
    for person_dir in DATA_DIR.glob("*"):
        if person_dir.is_dir():
            stats[person_dir.name] = len(list(person_dir.glob("*.jpg"))) + len(list(person_dir.glob("*.png")))
    logger.info(f"Dataset stats collected: {stats}")
    return stats

def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    try:
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return rotated
    except Exception as e:
        logger.error(f"Error rotating image: {e}")
        return image

def advanced_augment(roi: np.ndarray) -> List[np.ndarray]:
    """Enhanced augmentation: rotations, flips, brightness, contrast, noise, scaling, shearing, blur."""
    aug_images = [roi, cv2.flip(roi, 1)]
    angles = [-15, -10, -5, 5, 10, 15]
    for angle in angles:
        rot = rotate_image(roi, angle)
        aug_images.append(rot)
        aug_images.append(cv2.flip(rot, 1))

    # Enhanced augmentations
    seq_brightness = iaa.Sequential([iaa.AddToBrightness((-40, 40))])
    seq_contrast = iaa.Sequential([iaa.ContrastNormalization((0.7, 1.3))])
    seq_noise = iaa.Sequential([iaa.AdditiveGaussianNoise(scale=(0, 0.07 * 255))])
    seq_scale = iaa.Sequential([iaa.Affine(scale=(0.8, 1.2))])  # Zoom in/out
    seq_shear = iaa.Sequential([iaa.Affine(shear=(-10, 10))])  # Shear transformation
    seq_blur = iaa.Sequential([iaa.GaussianBlur(sigma=(0, 1.0))])  # Slight blur

    for aug in [seq_brightness, seq_contrast, seq_noise, seq_scale, seq_shear, seq_blur]:
        augmented = aug(image=roi)
        aug_images.append(augmented)
        aug_images.append(cv2.flip(augmented, 1))

    return aug_images

def process_image(img_path: Path, label: int) -> List[Tuple[np.ndarray, int]]:
    try:
        img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            logger.warning(f"Failed to decode image: {img_path}")
            return []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.01, minNeighbors=1, minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE))
        if len(faces) == 0:
            logger.info(f"No faces detected in {img_path}; using whole image as ROI")
            roi = cv2.resize(gray, FACE_RESIZE, interpolation=cv2.INTER_LINEAR)
            augmented_rois = advanced_augment(roi)
            return [(aug_roi, label) for aug_roi in augmented_rois]
        x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
        if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
            logger.warning(f"Face too small in {img_path}: {w}x{h}")
            return []
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, FACE_RESIZE, interpolation=cv2.INTER_LINEAR)
        augmented_rois = advanced_augment(roi)
        results = [(aug_roi, label) for aug_roi in augmented_rois]
        logger.info(f"Processed image {img_path} with {len(results)} augmentations")
        return results
    except Exception as e:
        logger.error(f"Error processing image {img_path}: {e}")
        return []

def prepare_training_data() -> Tuple[List[np.ndarray], List[int], Dict[int, str]]:
    images = []
    labels = []
    label_map: Dict[int, str] = {}
    next_label = 0

    stats = collect_dataset_stats()
    for person_name, count in stats.items():
        if count < MIN_PHOTOS_PER_PERSON:
            raise RuntimeError(f"Person '{person_name}' has only {count} photos. Minimum required: {MIN_PHOTOS_PER_PERSON}.")

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
            results = future.result()
            for img_roi, label in results:
                images.append(img_roi)
                labels.append(label)

    if len(images) == 0 or len(set(labels)) == 0:
        raise RuntimeError("Not enough valid faces detected for training. Ensure images contain clear, large faces.")
    
    logger.info(f"Prepared training data: {len(images)} images, {len(set(labels))} classes")
    return images, labels, label_map

def train_and_save_model() -> Tuple[int, Dict[str, int]]:
    try:
        images, labels, label_map = prepare_training_data()
        if len(images) < MIN_PHOTOS_PER_PERSON * len(label_map):
            raise RuntimeError(f"Insufficient valid faces after processing: {len(images)} images for {len(label_map)} classes.")
        recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=LBPH_RADIUS, neighbors=LBPH_NEIGHBORS, grid_x=LBPH_GRID_X, grid_y=LBPH_GRID_Y
        )
        recognizer.train(images, np.array(labels, dtype=np.int32))
        recognizer.write(str(MODEL_PATH))
        save_labels(label_map)
        stats = collect_dataset_stats()
        logger.info(f"Model trained with {len(set(labels))} classes and saved to {MODEL_PATH}")
        return len(set(labels)), stats
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

def load_model():
    try:
        if MODEL_PATH.exists():
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read(str(MODEL_PATH))
            labels = load_labels()
            logger.info("Model and labels loaded successfully")
            return recognizer, labels
        logger.warning("No model found")
        return None, {}
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None, {}

def draw_label(img: np.ndarray, text: str, x: int, y: int) -> None:
    try:
        cv2.rectangle(img, (x, y - 25), (x + max(120, 8 * len(text)), y), (0, 0, 0), -1)
        cv2.putText(img, text, (x + 5, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    except Exception as e:
        logger.error(f"Error drawing label: {e}")

def ensure_dir(path: Path):
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Error creating directory {path}: {e}")

def sanitize_name(name: str) -> str:
    return "".join(c for c in name.strip().replace(" ", "_") if c.isalnum() or c in ["_", "-"])

# ---------- Streamlit UI ----------
st.set_page_config(page_title=APP_TITLE, page_icon="🎥", layout="wide")
st.title(APP_TITLE)
st.caption(f"Enhanced LBPH model with advanced augmentation (rotations, flips, brightness, contrast, noise, scaling, shearing, blur). Capture from live video or upload images.")

with st.sidebar:
    st.header("Controls")
    conf_threshold = st.slider("Recognition threshold (lower=strict)", min_value=1, max_value=100, value=50, step=1)
    min_face_size = st.slider("Min face size (px)", 50, 200, MIN_FACE_SIZE, 10)
    scale_factor = st.slider("Detection scale factor", 1.01, 1.5, 1.1, 0.01)
    min_neighbors = st.slider("Min neighbors", 1, 10, 3, 1)
    st.markdown("---")
    st.subheader("Model Status")
    if MODEL_PATH.exists():
        labels = load_labels()
        st.write(f"Trained Classes: {len(labels)}")
        st.json(labels)
    else:
        st.info("No model trained yet.")
    st.markdown("---")
    st.subheader("Project Folders")
    st.code(f"DATA_DIR: {DATA_DIR.resolve()}\nMODELS_DIR: {MODELS_DIR.resolve()}\nLOGS_DIR: {LOGS_DIR.resolve()}")
    st.markdown("---")
    st.subheader("Download Logs")
    log_file = LOGS_DIR / "face_recognition.log"
    if log_file.exists():
        with open(log_file, "rb") as f:
            st.download_button("Download Logs", f, file_name="face_recognition.log")
    st.markdown("---")
    st.info(f"Supports training with {MIN_PHOTOS_PER_PERSON} photo(s) per person via enhanced augmentation.")

tabs = st.tabs(["📇 Enroll", "🧠 Train", "🔴 Live Recognition"])

# --------- ENROLL TAB ---------
with tabs[0]:
    st.subheader("Enroll a Person")
    person_name = st.text_input("Person name", placeholder="e.g., Abhinav")
    sanitized_name = sanitize_name(person_name) if person_name else ""
    st.write(f"Capture from live video or upload at least {MIN_PHOTOS_PER_PERSON} photo(s) for {person_name or 'a person'}. Augmentation enables single-photo training.")
    st.info("Tips: Use 'Capture Snapshot' from live video or upload clear face images. Ensure good lighting and large faces.")

    if sanitized_name:
        person_dir = DATA_DIR / sanitized_name
        ensure_dir(person_dir)
        if "captured_count" not in st.session_state:
            st.session_state.captured_count = 0
        if "snapshot_trigger" not in st.session_state:
            st.session_state.snapshot_trigger = False

        stats = collect_dataset_stats()
        current_count = stats.get(sanitized_name, 0)
        progress_bar = st.progress(current_count / MIN_PHOTOS_PER_PERSON)
        status_text = st.empty()
        status_text.write(f"Progress: {current_count}/{MIN_PHOTOS_PER_PERSON} photos captured for {person_name}")

        # File Uploader
        uploaded_files = st.file_uploader("Upload photos", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
        if uploaded_files:
            for uploaded_file in uploaded_files:
                img_bytes = uploaded_file.read()
                img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if img is not None:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    gray = cv2.equalizeHist(gray)
                    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE))
                    if len(faces) > 0:
                        x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
                        roi = img[y:y+h, x:x+w]
                        file_path = person_dir / f"{uuid.uuid4().hex}.jpg"
                        cv2.imwrite(str(file_path), roi)
                        st.success(f"Uploaded and saved {uploaded_file.name} for {person_name}")
                        logger.info(f"Saved uploaded image for {person_name}: {file_path.name}")
                    else:
                        st.warning(f"No face detected in uploaded {uploaded_file.name}. Try another image.")
                else:
                    st.error(f"Failed to decode uploaded {uploaded_file.name}.")
            rerun()

        # Live Video Capture with Snapshot
        st.subheader("Live Video Capture")
        if st.button("Capture Snapshot"):
            st.session_state.snapshot_trigger = True

        class VideoCaptureProcessor:
            def __init__(self):
                self.last_frame = None

            def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                img = frame.to_ndarray(format="bgr24")
                self.last_frame = img
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE))
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    draw_label(img, f"Face: {w}x{h}", x, y)
                return av.VideoFrame.from_ndarray(img, format="bgr24")

        ctx = webrtc_streamer(
            key="capture",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=VideoCaptureProcessor,
            rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 640},
                    "height": {"ideal": 480},
                    "frameRate": {"ideal": 15}
                },
                "audio": False
            },
            async_processing=True,
        )

        if st.session_state.snapshot_trigger and ctx.video_processor and ctx.video_processor.last_frame is not None:
            img = ctx.video_processor.last_frame
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE))
            if len(faces) == 0:
                st.warning("No face detected in snapshot. Adjust lighting or position.")
                logger.warning("No face detected in snapshot")
            else:
                x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
                if w * h < MIN_FACE_SIZE * MIN_FACE_SIZE:
                    st.warning("Face too small in snapshot. Move closer to the camera.")
                    logger.warning("Face too small in snapshot")
                else:
                    roi = img[y:y+h, x:x+w]
                    file_path = person_dir / f"{uuid.uuid4().hex}.jpg"
                    cv2.imwrite(str(file_path), roi)
                    st.session_state.captured_count += 1
                    st.success(f"Saved snapshot {st.session_state.captured_count}/{MIN_PHOTOS_PER_PERSON} for {person_name}")
                    st.image(img, caption="Captured Snapshot", use_column_width=True)
                    logger.info(f"Saved snapshot for {person_name}: {file_path.name}")
                    progress_bar.progress(st.session_state.captured_count / MIN_PHOTOS_PER_PERSON)
                    status_text.write(f"Progress: {st.session_state.captured_count}/{MIN_PHOTOS_PER_PERSON} photos captured for {person_name}")
                    if st.session_state.captured_count >= MIN_PHOTOS_PER_PERSON:
                        status_text.success(f"Completed capturing {MIN_PHOTOS_PER_PERSON} photo(s) for {person_name}! Ready to train.")
                    st.session_state.snapshot_trigger = False
                    rerun()

        if current_count >= MIN_PHOTOS_PER_PERSON:
            st.success(f"Completed capturing {MIN_PHOTOS_PER_PERSON} photo(s) for {person_name}! Ready to train.")
            progress_bar.progress(1.0)

    else:
        st.warning("Enter a person name to start capturing.")

    st.markdown("### Current Dataset")
    stats = collect_dataset_stats()
    if stats:
        st.json(stats)
    else:
        st.info("No images yet. Add captures for at least one person.")

    if stats:
        delete_person = st.selectbox("Delete a person's dataset", [""] + list(stats.keys()))
        if delete_person and st.button("Confirm Delete"):
            import shutil
            shutil.rmtree(DATA_DIR / delete_person)
            st.success(f"Deleted dataset for {delete_person}.")
            logger.info(f"Deleted dataset for {delete_person}")
            st.session_state.captured_count = 0
            rerun()

# --------- TRAIN TAB ---------
with tabs[1]:
    st.subheader("Train LBPH Model")
    st.write(f"Requires at least {MIN_PHOTOS_PER_PERSON} photo(s) per person, with enhanced augmentation for robust training.")
    if st.button("Train / Retrain"):
        with st.spinner("Training model..."):
            try:
                n_classes, stats = train_and_save_model()
                st.success(f"Training complete. Classes: {n_classes}. Model saved to `{MODEL_PATH}`.")
                st.json(stats)
            except RuntimeError as e:
                st.error(f"Training failed: {e}. Check logs for details (e.g., no valid faces detected). Try uploading clearer, larger face images.")
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}. See logs for details.")

    if LABELS_PATH.exists():
        st.markdown("#### Current Labels")
        st.json(load_labels())

# --------- LIVE RECOGNITION TAB ---------
with tabs[2]:
    st.subheader("Live Webcam Recognition")
    st.write("Click 'Start' and allow camera access.")

    recognizer, id_to_name = load_model()
    if recognizer is None or not id_to_name:
        st.warning("No trained model found. Train the model first.")
    else:
        id_to_name = {int(k): v for k, v in id_to_name.items()}
        
        class VideoProcessor:
            def __init__(self):
                self.recognizer = recognizer
                self.labels = id_to_name
                self.frame_count = 0
                self.last_faces = []
                self.skip_frames = 2
                self.prev_time = time.time()

            def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                try:
                    img = frame.to_ndarray(format="bgr24")
                    img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_LINEAR)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                    curr_time = time.time()
                    fps = 1 / (curr_time - self.prev_time) if self.prev_time else 0
                    self.prev_time = curr_time
                    draw_label(img, f"FPS: {fps:.1f}", 10, 30)

                    self.frame_count += 1
                    if self.frame_count % self.skip_frames == 0:
                        faces = FACE_CASCADE.detectMultiScale(
                            gray,
                            scaleFactor=st.session_state.get("scale_factor", 1.1),
                            minNeighbors=st.session_state.get("min_neighbors", 3),
                            minSize=(st.session_state.get("min_face_size", MIN_FACE_SIZE), st.session_state.get("min_face_size", MIN_FACE_SIZE)),
                        )
                        self.last_faces = faces
                    else:
                        faces = self.last_faces

                    st.write(f"Frame resolution: {img.shape[1]}x{img.shape[0]}, Detected faces: {len(faces)}")
                    for (x, y, w, h) in faces:
                        roi = gray[y:y+h, x:x+w]
                        try:
                            roi = cv2.resize(roi, FACE_RESIZE, interpolation=cv2.INTER_LINEAR)
                            roi = cv2.equalizeHist(roi)
                        except Exception as e:
                            logger.warning(f"Error processing face in live stream: {e}")
                            continue
                        label_id, confidence = self.recognizer.predict(roi)
                        if confidence <= st.session_state.get("conf_threshold", 50):
                            name = self.labels.get(label_id, "Unknown")
                            color = (0, 255, 0)
                            caption = f"{name} ({confidence:.1f})"
                        else:
                            name = "Unknown"
                            color = (0, 0, 255)
                            caption = f"{name} ({confidence:.1f})"

                        st.write(f"Face size: {w}x{h}, Confidence: {confidence:.1f}, Predicted: {name}")
                        logger.info(f"Live recognition: Face size {w}x{h}, Confidence {confidence:.1f}, Predicted {name}")
                        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                        draw_label(img, caption, x, y)

                    return av.VideoFrame.from_ndarray(img, format="bgr24")
                except Exception as e:
                    logger.error(f"Error in video processing: {e}")
                    return frame

        if "conf_threshold" not in st.session_state:
            st.session_state["conf_threshold"] = conf_threshold
        if "min_face_size" not in st.session_state:
            st.session_state["min_face_size"] = min_face_size
        if "scale_factor" not in st.session_state:
            st.session_state["scale_factor"] = scale_factor
        if "min_neighbors" not in st.session_state:
            st.session_state["min_neighbors"] = min_neighbors

        try:
            webrtc_streamer(
                key="live",
                mode=WebRtcMode.SENDRECV,
                video_processor_factory=VideoProcessor,
                rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
                media_stream_constraints={
                    "video": {
                        "width": {"ideal": 640},
                        "height": {"ideal": 480},
                        "frameRate": {"ideal": 15}
                    },
                    "audio": False
                },
                async_processing=True,
            )
        except Exception as e:
            st.error("WebRTC failed to initialize. Ensure camera access is granted.")
            logger.error(f"WebRTC initialization failed: {e}")

st.markdown("---")
st.caption(f"Enhancements: Improved LBPH model with advanced augmentation (rotations, flips, brightness, contrast, noise, scaling, shearing, blur). Capture via live video snapshots or file upload.")
