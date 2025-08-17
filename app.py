import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import uuid
import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from concurrent.futures import ThreadPoolExecutor
import random
import time
import pandas as pd
from datetime import datetime
import dlib

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

APP_TITLE = "Advanced Live Face Recognition (Streamlit + WebRTC, OpenCV LBPH)"
DATA_DIR = Path("data")
MODELS_DIR = Path("models")
LOGS_DIR = Path("logs")
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
FACE_CASCADE = cv2.CascadeClassifier(CASCADE_PATH)
# Load dlib's facial landmark detector
DLIB_LANDMARKS_PATH = "shape_predictor_68_face_landmarks.dat"
LANDMARK_DETECTOR = None
if os.path.exists(DLIB_LANDMARKS_PATH):
    LANDMARK_DETECTOR = dlib.shape_predictor(DLIB_LANDMARKS_PATH)
else:
    logger.warning("dlib shape predictor not found. Face alignment disabled.")

MODEL_PATH = MODELS_DIR / "lbph_model.xml"
LABELS_PATH = MODELS_DIR / "labels.json"

# Optimized LBPH parameters
LBPH_RADIUS = 1
LBPH_NEIGHBORS = 6
LBPH_GRID_X = 6
LBPH_GRID_Y = 6
FACE_RESIZE = (200, 200)

# Minimum photos required per person for training
MIN_PHOTOS_PER_PERSON = 20

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

def align_face(image: np.ndarray, landmarks: dlib.full_object_detection) -> np.ndarray:
    """Align face based on eye landmarks to improve recognition accuracy."""
    try:
        left_eye = np.mean([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)], axis=0)
        right_eye = np.mean([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)], axis=0)
        delta_x = right_eye[0] - left_eye[0]
        delta_y = right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(delta_y, delta_x)) - 180
        return rotate_image(image, angle)
    except Exception as e:
        logger.warning(f"Face alignment failed: {e}")
        return image

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

def process_image(img_path: Path, label: int) -> List[Tuple[np.ndarray, int]]:
    """Process a single image with augmentation and optional face alignment."""
    try:
        img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            logger.warning(f"Failed to decode image: {img_path}")
            return []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(50, 50))
        if len(faces) == 0:
            logger.info(f"No faces detected in {img_path}")
            return []
        x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
        roi = gray[y : y + h, x : x + w]
        
        if LANDMARK_DETECTOR:
            dlib_rect = dlib.rectangle(x, y, x + w, y + h)
            landmarks = LANDMARK_DETECTOR(img, dlib_rect)
            roi = align_face(roi, landmarks)
        
        roi = cv2.resize(roi, FACE_RESIZE, interpolation=cv2.INTER_LINEAR)
        results = [(roi, label), (cv2.flip(roi, 1), label)]
        for angle in [-10, 10]:
            rot = rotate_image(roi, angle)
            results.append((rot, label))
            results.append((cv2.flip(rot, 1), label))
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
            raise RuntimeError(f"Person '{person_name}' has only {count} photos. Minimum required: {MIN_PHOTOS_PER_PERSON}. Please capture more.")

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
        raise RuntimeError("Not enough data to train. Please add images for at least one person.")
    
    logger.info(f"Prepared training data: {len(images)} images, {len(set(labels))} classes")
    return images, labels, label_map

def train_and_save_model() -> Tuple[int, Dict[str, int]]:
    try:
        images, labels, label_map = prepare_training_data()
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
        logger.error(f"Training failed: {e}")
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
st.caption("Advanced face recognition with face alignment, logging, batch processing, enhanced evaluation, and auto-capture for required photos.")

with st.sidebar:
    st.header("Controls")
    conf_threshold = st.slider("Recognition threshold (LBPH, lower=more strict)", min_value=1, max_value=100, value=50, step=1)
    min_face_size = st.slider("Min face size (px)", 60, 200, 100, 10)
    scale_factor = st.slider("Detection scale factor (lower=more sensitive)", 1.05, 1.5, 1.2, 0.05)
    min_neighbors = st.slider("Min neighbors for detection", 3, 10, 6, 1)
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
    st.code(f"DATA_DIR: {DATA_DIR.resolve()}\nMODELS_DIR: {MODELS_DIR.resolve()}\nLOGS_DIR: {LOGS_DIR.resolve()}", language="bash")
    st.markdown("---")
    st.subheader("Download Logs")
    log_file = LOGS_DIR / "face_recognition.log"
    if log_file.exists():
        with open(log_file, "rb") as f:
            st.download_button("Download Logs", f, file_name="face_recognition.log")
    st.markdown("---")
    st.info(f"Minimum photos per person for training: {MIN_PHOTOS_PER_PERSON}")

tabs = st.tabs(["📇 Enroll", "🧠 Train", "📊 Evaluate", "🔴 Live Recognition"])

# --------- ENROLL TAB ---------
with tabs[0]:
    st.subheader("Add images for a person")
    person_name = st.text_input("Person name", placeholder="e.g., Abhinav")
    sanitized_name = sanitize_name(person_name) if person_name else ""
    st.write("Capture from webcam or upload images. Auto-detects faces with alignment for quality.")
    st.info(f"Tips: Good lighting, direct camera facing, centered face. At least {MIN_PHOTOS_PER_PERSON} images per person required for training.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Manual Capture")
        cam_img = st.camera_input("Capture image")
        if cam_img and sanitized_name:
            person_dir = DATA_DIR / sanitized_name
            ensure_dir(person_dir)
            img_bytes = cam_img.getvalue()
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                st.error("Failed to decode image. Try again.")
                logger.error("Failed to decode webcam capture")
            else:
                target_resolution = (1280, 720)
                img = cv2.resize(img, target_resolution, interpolation=cv2.INTER_LINEAR)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)
                faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(50, 50))
                st.write(f"Debug: Image resolution: {img.shape[1]}x{img.shape[0]}, Faces detected: {len(faces)}")
                if len(faces) == 0:
                    st.warning("No face detected. Try better lighting or closer camera.")
                    logger.warning("No face detected in webcam capture")
                else:
                    if LANDMARK_DETECTOR:
                        x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
                        dlib_rect = dlib.rectangle(x, y, x + w, y + h)
                        landmarks = LANDMARK_DETECTOR(img, dlib_rect)
                        img = align_face(img, landmarks)
                    file_path = person_dir / f"{uuid.uuid4().hex}.jpg"
                    cv2.imwrite(str(file_path), img)
                    st.success(f"Saved capture for {person_name} → {file_path.name}")
                    st.image(img_bytes, caption="Captured Image (Face Detected)", use_column_width=True)
                    logger.info(f"Saved webcam capture for {person_name}: {file_path.name}")
        elif cam_img and not sanitized_name:
            st.warning("Please enter a valid person name.")
            logger.warning("Webcam capture attempted without valid person name")

    with col2:
        st.subheader("Upload Images")
        uploads = st.file_uploader("Upload image(s)", type=["jpg", "jpeg", "png", "bmp", "webp"], accept_multiple_files=True)
        if uploads and sanitized_name:
            person_dir = DATA_DIR / sanitized_name
            ensure_dir(person_dir)
            saved = 0
            previews = []
            batch_size = 10
            for i in range(0, len(uploads), batch_size):
                batch = uploads[i:i + batch_size]
                with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                    futures = []
                    for up in batch:
                        futures.append(executor.submit(process_upload, up, person_dir))
                    for future in futures:
                        result = future.result()
                        if result:
                            saved += 1
                            previews.append(result)
            if saved > 0:
                st.success(f"Saved {saved} image(s) for {person_name}.")
                for prev in previews:
                    st.image(prev, caption="Uploaded Image (Face Detected)", use_column_width=True)
                logger.info(f"Saved {saved} uploaded images for {person_name}")
            else:
                logger.warning("No valid images saved from uploads")

    def process_upload(up, person_dir):
        try:
            img_bytes = up.getvalue()
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                st.warning(f"Failed to decode {up.name}. Skipping.")
                logger.warning(f"Failed to decode uploaded image: {up.name}")
                return None
            img = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_LINEAR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(50, 50))
            st.write(f"Debug: {up.name} resolution: {img.shape[1]}x{img.shape[0]}, Faces detected: {len(faces)}")
            if len(faces) == 0:
                st.warning(f"No face detected in {up.name}. Skipping.")
                logger.warning(f"No face detected in uploaded image: {up.name}")
                return None
            if LANDMARK_DETECTOR:
                x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
                dlib_rect = dlib.rectangle(x, y, x + w, y + h)
                landmarks = LANDMARK_DETECTOR(img, dlib_rect)
                img = align_face(img, landmarks)
            file_path = person_dir / f"{uuid.uuid4().hex}.jpg"
            cv2.imwrite(str(file_path), img)
            logger.info(f"Saved uploaded image for {person_dir.name}: {file_path.name}")
            return img_bytes
        except Exception as e:
            logger.error(f"Error processing upload {up.name}: {e}")
            return None

    st.markdown("### Auto-Capture for Training")
    num_photos = st.number_input("Number of photos to auto-capture", min_value=MIN_PHOTOS_PER_PERSON, max_value=100, value=MIN_PHOTOS_PER_PERSON)
    if sanitized_name:
        person_dir = DATA_DIR / sanitized_name
        ensure_dir(person_dir)
        if st.button("Start Auto-Capture"):
            st.info("Starting auto-capture. Face the camera and vary your pose slightly. Captures every 1 second when face detected.")
            class CaptureProcessor:
                def __init__(self, person_dir, num_photos):
                    self.person_dir = person_dir
                    self.num_photos = num_photos
                    self.captured = 0
                    self.last_capture_time = 0
                    self.progress = st.progress(0)
                    self.status_text = st.empty()

                def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                    try:
                        img = frame.to_ndarray(format="bgr24")
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        gray = cv2.equalizeHist(gray)
                        faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(50, 50))

                        if len(faces) > 0 and time.time() - self.last_capture_time > 1:  # Capture every 1 second if face detected
                            if LANDMARK_DETECTOR:
                                x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
                                dlib_rect = dlib.rectangle(x, y, x + w, y + h)
                                landmarks = LANDMARK_DETECTOR(img, dlib_rect)
                                img = align_face(img, landmarks)
                            file_path = self.person_dir / f"{uuid.uuid4().hex}.jpg"
                            cv2.imwrite(str(file_path), img)
                            self.captured += 1
                            self.last_capture_time = time.time()
                            self.progress.progress(self.captured / self.num_photos)
                            self.status_text.text(f"Captured {self.captured}/{self.num_photos}")
                            logger.info(f"Auto-captured image for {person_name}: {file_path.name}")

                        # Draw capture progress on frame
                        draw_label(img, f"Capturing: {self.captured}/{self.num_photos}", 10, 30)

                        if self.captured >= self.num_photos:
                            self.status_text.success("Auto-capture complete!")
                            # Note: Streamlit-webrtc doesn't auto-stop, user can stop manually

                        return av.VideoFrame.from_ndarray(img, format="bgr24")
                    except Exception as e:
                        logger.error(f"Error in auto-capture: {e}")
                        return frame

            rtc_config = RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            )

            webrtc_streamer(
                key="auto_capture",
                mode=WebRtcMode.SENDRECV,
                video_processor_factory=lambda: CaptureProcessor(person_dir, num_photos),
                rtc_configuration=rtc_config,
                media_stream_constraints={
                    "video": {
                        "width": {"ideal": 1280},
                        "height": {"ideal": 720},
                        "frameRate": {"ideal": 15}
                    },
                    "audio": False
                },
                async_processing=True,
            )
    else:
        st.warning("Enter a person name to enable auto-capture.")

    st.markdown("### Current dataset")
    stats = collect_dataset_stats()
    if stats:
        st.json(stats)
    else:
        st.info("No images yet. Add some captures for at least one person.")

    if stats:
        delete_person = st.selectbox("Delete a person's dataset", [""] + list(stats.keys()))
        if delete_person and st.button("Confirm Delete"):
            import shutil
            shutil.rmtree(DATA_DIR / delete_person)
            st.success(f"Deleted dataset for {delete_person}.")
            logger.info(f"Deleted dataset for {delete_person}")
            st.rerun()

# --------- TRAIN TAB ---------
with tabs[1]:
    st.subheader("Train LBPH model")
    st.write(f"Train or retrain the model with augmented data. Ensures at least {MIN_PHOTOS_PER_PERSON} photos per person.")
    if st.button("Train / Retrain"):
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

# --------- EVALUATE TAB ---------
with tabs[2]:
    st.subheader("Evaluate Model Performance")
    st.write("Splits dataset into 80% train / 20% test, trains a temporary model, and computes accuracy and confusion matrix.")
    if st.button("Run Evaluation"):
        with st.spinner("Evaluating model..."):
            try:
                all_rois = []
                all_labels = []
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
                        gray = cv2.equalizeHist(gray)
                        faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(50, 50))
                        if len(faces) == 0:
                            continue
                        x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
                        roi = gray[y : y + h, x : x + w]
                        if LANDMARK_DETECTOR:
                            dlib_rect = dlib.rectangle(x, y, x + w, y + h)
                            landmarks = LANDMARK_DETECTOR(img, dlib_rect)
                            roi = align_face(roi, landmarks)
                        try:
                            roi = cv2.resize(roi, FACE_RESIZE, interpolation=cv2.INTER_LINEAR)
                            all_rois.append(roi)
                            all_labels.append(next_label)
                        except Exception:
                            continue
                    next_label += 1

                if len(all_rois) < 10 or len(set(all_labels)) < 2:
                    raise RuntimeError("Not enough data for evaluation. Need at least 10 images across 2+ persons.")

                data = list(zip(all_rois, all_labels))
                random.shuffle(data)
                rois, labels = zip(*data)
                rois = list(rois)
                labels = list(labels)
                split_idx = int(0.8 * len(rois))
                train_rois = rois[:split_idx]
                train_labels = labels[:split_idx]
                test_rois = rois[split_idx:]
                test_labels = labels[split_idx:]

                eval_recognizer = cv2.face.LBPHFaceRecognizer_create(
                    radius=LBPH_RADIUS, neighbors=LBPH_NEIGHBORS, grid_x=LBPH_GRID_X, grid_y=LBPH_GRID_Y
                )
                eval_recognizer.train(train_rois, np.array(train_labels, dtype=np.int32))

                predictions = []
                confidences = []
                for roi in test_rois:
                    pred, conf = eval_recognizer.predict(roi)
                    predictions.append(pred)
                    confidences.append(conf)

                correct = sum(p == t for p, t in zip(predictions, test_labels))
                accuracy = correct / len(test_labels) * 100
                st.success(f"Test Accuracy: {accuracy:.2f}% (on {len(test_labels)} test samples)")
                logger.info(f"Evaluation complete: Accuracy {accuracy:.2f}% on {len(test_labels)} samples")

                classes = sorted(label_map.keys())
                cm = np.zeros((len(classes), len(classes)), dtype=int)
                for p, t in zip(predictions, test_labels):
                    cm[t, p] += 1
                class_names = [label_map[c] for c in classes]
                df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
                st.subheader("Confusion Matrix")
                st.dataframe(df_cm)

                # Plot confusion matrix
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots()
                cax = ax.matshow(cm, cmap='Blues')
                plt.title("Confusion Matrix")
                fig.colorbar(cax)
                ax.set_xticks(np.arange(len(class_names)))
                ax.set_yticks(np.arange(len(class_names)))
                ax.set_xticklabels(class_names, rotation=45)
                ax.set_yticklabels(class_names)
                for i in range(len(class_names)):
                    for j in range(len(class_names)):
                        ax.text(j, i, cm[i, j], ha='center', va='center', color='black')
                st.pyplot(fig)

            except Exception as e:
                st.error(str(e))
                logger.error(f"Evaluation failed: {e}")

# --------- LIVE RECOGNITION TAB ---------
with tabs[3]:
    st.subheader("Start live webcam recognition")
    st.write("Click **Start** below and allow camera access. Optimized with face alignment and FPS display.")

    recognizer, id_to_name = load_model()
    if recognizer is None or not id_to_name:
        st.warning("No trained model found. Please train the model first in the **Train** tab.")
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
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                    # FPS calculation
                    curr_time = time.time()
                    fps = 1 / (curr_time - self.prev_time) if self.prev_time else 0
                    self.prev_time = curr_time
                    draw_label(img, f"FPS: {fps:.1f}", 10, 30)

                    self.frame_count += 1
                    if self.frame_count % self.skip_frames == 0:
                        faces = FACE_CASCADE.detectMultiScale(
                            gray,
                            scaleFactor=st.session_state.get("scale_factor", 1.2),
                            minNeighbors=st.session_state.get("min_neighbors", 6),
                            minSize=(st.session_state.get("min_face_size", 100), st.session_state.get("min_face_size", 100)),
                        )
                        self.last_faces = faces
                    else:
                        faces = self.last_faces

                    st.write(f"Frame resolution: {img.shape[1]}x{img.shape[0]}, Detected faces: {len(faces)}")
                    for (x, y, w, h) in faces:
                        roi = gray[y : y + h, x : x + w]
                        try:
                            if LANDMARK_DETECTOR:
                                dlib_rect = dlib.rectangle(x, y, x + w, y + h)
                                landmarks = LANDMARK_DETECTOR(img, dlib_rect)
                                roi = align_face(roi, landmarks)
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

        if "conf_threshold" not in st.session_state or st.session_state["conf_threshold"] != conf_threshold:
            st.session_state["conf_threshold"] = conf_threshold
        if "min_face_size" not in st.session_state or st.session_state["min_face_size"] != min_face_size:
            st.session_state["min_face_size"] = min_face_size
        if "scale_factor" not in st.session_state or st.session_state["scale_factor"] != scale_factor:
            st.session_state["scale_factor"] = scale_factor
        if "min_neighbors" not in st.session_state or st.session_state["min_neighbors"] != min_neighbors:
            st.session_state["min_neighbors"] = min_neighbors

        rtc_config = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )

        webrtc_streamer(
            key="live",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=VideoProcessor,
            rtc_configuration=rtc_config,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 1280},
                    "height": {"ideal": 720},
                    "frameRate": {"ideal": 15}
                },
                "audio": False
            },
            async_processing=True,
        )

st.markdown("---")
st.caption("Enhancements: Added auto-capture feature to automatically take a specified number of photos (minimum 20) for training, with progress display and face detection check. Ensures minimum photos per person before training.")
```
