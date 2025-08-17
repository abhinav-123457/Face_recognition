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
import random
import time
import pandas as pd
import datetime
import math

APP_TITLE = "Advanced Live Face Recognition (Streamlit + WebRTC, OpenCV)"
DATA_DIR = Path("data")
MODELS_DIR = Path("models")
LOGS_DIR = Path("logs")
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
EYE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_eye.xml"
FACE_CASCADE = cv2.CascadeClassifier(CASCADE_PATH)
EYE_CASCADE = cv2.CascadeClassifier(EYE_CASCADE_PATH)

FACE_RESIZE = (200, 200)

# Attendance log
ATTENDANCE_LOG = LOGS_DIR / "attendance.csv"

def load_labels(model_type: str) -> Dict[int, str]:
    labels_path = MODELS_DIR / f"labels_{model_type}.json"
    if labels_path.exists():
        with open(labels_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {int(k): v for k, v in data.items()}
    return {}

def save_labels(mapping: Dict[int, str], model_type: str) -> None:
    labels_path = MODELS_DIR / f"labels_{model_type}.json"
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump({int(k): v for k, v in mapping.items()}, f, ensure_ascii=False, indent=2)

def collect_dataset_stats() -> Dict[str, int]:
    stats = {}
    for person_dir in DATA_DIR.glob("*"):
        if person_dir.is_dir():
            stats[person_dir.name] = len(list(person_dir.glob("*.jpg"))) + len(list(person_dir.glob("*.png")))
    return stats

def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def align_face(roi: np.ndarray) -> np.ndarray:
    eyes = EYE_CASCADE.detectMultiScale(roi, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))
    if len(eyes) >= 2:
        eyes = sorted(eyes, key=lambda e: e[0])  # Sort by x to get left and right
        eye1_center = (eyes[0][0] + eyes[0][2] // 2, eyes[0][1] + eyes[0][3] // 2)
        eye2_center = (eyes[1][0] + eyes[1][2] // 2, eyes[1][1] + eyes[1][3] // 2)
        dx = eye2_center[0] - eye1_center[0]
        dy = eye2_center[1] - eye1_center[1]
        angle = math.degrees(math.atan2(dy, dx))
        roi = rotate_image(roi, -angle)  # Rotate to make eyes horizontal
    return roi

def adjust_brightness_contrast(image: np.ndarray, alpha: float = 1.0, beta: int = 0) -> np.ndarray:
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def process_image(img_path: Path, label: int) -> List[Tuple[np.ndarray, int]]:
    """Process a single image with alignment, augmentation (flips, rotations, brightness/contrast)."""
    img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(50, 50))
    if len(faces) == 0:
        return []
    x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
    roi = gray[y : y + h, x : x + w]
    try:
        roi = cv2.resize(roi, FACE_RESIZE, interpolation=cv2.INTER_LINEAR)
        roi_aligned = align_face(roi)
        results = []
        # Original and aligned
        for r in [roi, roi_aligned]:
            results.append((r, label))
            results.append((cv2.flip(r, 1), label))
            # Rotations
            for angle in [-10, 10]:
                rot = rotate_image(r, angle)
                results.append((rot, label))
                results.append((cv2.flip(rot, 1), label))
            # Brightness/contrast variations
            for alpha, beta in [(0.8, -20), (1.2, 20)]:
                adj = adjust_brightness_contrast(r, alpha, beta)
                results.append((adj, label))
                results.append((cv2.flip(adj, 1), label))
        return results
    except Exception:
        return []

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
            results = future.result()
            for img_roi, label in results:
                images.append(img_roi)
                labels.append(label)

    if len(images) == 0 or len(set(labels)) == 0:
        raise RuntimeError("Not enough data to train. Please add images for at least one person.")

    return images, labels, label_map

def create_recognizer(model_type: str):
    if model_type == "LBPH":
        return cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=6, grid_x=6, grid_y=6)
    elif model_type == "Eigen":
        return cv2.face.EigenFaceRecognizer_create()
    elif model_type == "Fisher":
        return cv2.face.FisherFaceRecognizer_create()
    else:
        raise ValueError("Invalid model type")

def train_and_save_model(model_type: str) -> Tuple[int, Dict[str, int]]:
    images, labels, label_map = prepare_training_data()
    recognizer = create_recognizer(model_type)
    recognizer.train(images, np.array(labels, dtype=np.int32))
    model_path = MODELS_DIR / f"{model_type}_model.xml"
    recognizer.write(str(model_path))
    save_labels(label_map, model_type)
    stats = collect_dataset_stats()
    return len(set(labels)), stats

def load_model(model_type: str):
    model_path = MODELS_DIR / f"{model_type}_model.xml"
    if model_path.exists():
        recognizer = create_recognizer(model_type)
        recognizer.read(str(model_path))
        labels = load_labels(model_type)
        return recognizer, labels
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
st.caption("Advanced for final year project: Multiple recognizers (LBPH/Eigen/Fisher), face alignment via eyes, enhanced augmentation (brightness/contrast), attendance logging, detailed evaluation metrics.")

with st.sidebar:
    st.header("Controls")
    model_type = st.selectbox("Recognizer Type", ["LBPH", "Eigen", "Fisher"])
    conf_threshold = st.slider("Recognition threshold (lower=more strict)", min_value=1, max_value=100, value=50, step=1)
    min_face_size = st.slider("Min face size (px)", 60, 200, 100, 10)
    scale_factor = st.slider("Detection scale factor (lower=more sensitive)", 1.05, 1.5, 1.2, 0.05)
    min_neighbors = st.slider("Min neighbors for detection", 3, 10, 6, 1)
    st.markdown("---")
    st.subheader("Model Status")
    if (MODELS_DIR / f"{model_type}_model.xml").exists():
        labels = load_labels(model_type)
        st.write(f"Trained Classes: {len(labels)}")
        st.json(labels)
    else:
        st.info("No model trained yet.")
    st.markdown("---")
    st.subheader("Project Folders")
    st.code(f"DATA_DIR: {DATA_DIR.resolve()}\nMODELS_DIR: {MODELS_DIR.resolve()}\nLOGS_DIR: {LOGS_DIR.resolve()}", language="bash")

tabs = st.tabs(["📇 Enroll", "🧠 Train", "📊 Evaluate", "🔴 Live Recognition", "📝 Attendance"])

# --------- ENROLL TAB ---------
with tabs[0]:
    st.subheader("Add images for a person")
    person_name = st.text_input("Person name", placeholder="e.g., Abhinav")
    sanitized_name = sanitize_name(person_name) if person_name else ""
    st.write("Capture from your webcam (recommended) or upload existing images. Auto-detects faces for quality.")
    st.info("Tips for better detection: Ensure good lighting, face the camera directly, and keep your face centered and close.")

    col1, col2 = st.columns(2)

    with col1:
        cam_img = st.camera_input("Capture image")
        if cam_img and sanitized_name:
            person_dir = DATA_DIR / sanitized_name
            ensure_dir(person_dir)
            img_bytes = cam_img.getvalue()
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                st.error("Failed to decode image. Try again.")
            else:
                target_resolution = (1280, 720)
                img = cv2.resize(img, target_resolution, interpolation=cv2.INTER_LINEAR)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)
                faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(50, 50))
                st.write(f"Debug: Image resolution: {img.shape[1]}x{img.shape[0]}, Faces detected: {len(faces)}")
                if len(faces) == 0:
                    st.warning("No face detected in the capture. Try again with better lighting or closer to the camera.")
                else:
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
                img_bytes = up.getvalue()
                img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if img is None:
                    st.warning(f"Failed to decode {up.name}. Skipping.")
                    continue
                img = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_LINEAR)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)
                faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(50, 50))
                st.write(f"Debug: {up.name} resolution: {img.shape[1]}x{img.shape[0]}, Faces detected: {len(faces)}")
                if len(faces) == 0:
                    st.warning(f"No face detected in {up.name}. Skipping.")
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
        st.info("No images yet. Add some captures for at least one person.")

    if stats:
        delete_person = st.selectbox("Delete a person's dataset", [""] + list(stats.keys()))
        if delete_person and st.button("Confirm Delete"):
            import shutil
            shutil.rmtree(DATA_DIR / delete_person)
            st.success(f"Deleted dataset for {delete_person}.")
            st.rerun()

# --------- TRAIN TAB ---------
with tabs[1]:
    st.subheader("Train Model")
    st.write("Select recognizer type in sidebar. Train to build/update the model. Uses parallel processing, alignment, and advanced augmentation.")
    if st.button("Train / Retrain"):
        with st.spinner("Training model..."):
            try:
                n_classes, stats = train_and_save_model(model_type)
                st.success(f"Training complete. Classes: {n_classes}. Model saved.")
                st.json(stats)
            except Exception as e:
                st.error(str(e))

    if (MODELS_DIR / f"labels_{model_type}.json").exists():
        st.markdown("#### Current labels")
        st.json(load_labels(model_type))

# --------- EVALUATE TAB ---------
with tabs[2]:
    st.subheader("Evaluate Model Performance")
    st.write("Splits dataset into 80% train / 20% test (no augmentation), trains a temporary model, computes accuracy, precision, recall, F1, and confusion matrix.")
    if st.button("Run Evaluation"):
        with st.spinner("Evaluating model..."):
            try:
                # Prepare data without augmentation for fair evaluation
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
                        try:
                            roi = cv2.resize(roi, FACE_RESIZE, interpolation=cv2.INTER_LINEAR)
                            roi = align_face(roi)  # Align for evaluation too
                            all_rois.append(roi)
                            all_labels.append(next_label)
                        except Exception:
                            continue

                    next_label += 1

                if len(all_rois) < 10 or len(set(all_labels)) < 2:
                    raise RuntimeError("Not enough data for evaluation. Need at least 10 images across 2+ persons.")

                # Shuffle and split
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

                # Train temporary model
                eval_recognizer = create_recognizer(model_type)
                eval_recognizer.train(train_rois, np.array(train_labels, dtype=np.int32))

                # Predict on test
                predictions = []
                confidences = []
                for roi in test_rois:
                    pred, conf = eval_recognizer.predict(roi)
                    predictions.append(pred)
                    confidences.append(conf)

                # Compute metrics (ignoring threshold for pure accuracy)
                cm = np.zeros((len(label_map), len(label_map)), dtype=int)
                for p, t in zip(predictions, test_labels):
                    cm[t, p] += 1

                accuracy = np.trace(cm) / np.sum(cm) * 100 if np.sum(cm) > 0 else 0

                # Per class metrics
                class_names = [label_map[c] for c in sorted(label_map.keys())]
                precision = np.diag(cm) / np.sum(cm, axis=0) if np.sum(cm, axis=0).all() else np.zeros(len(class_names))
                recall = np.diag(cm) / np.sum(cm, axis=1) if np.sum(cm, axis=1).all() else np.zeros(len(class_names))
                f1 = 2 * precision * recall / (precision + recall) where (precision + recall) != 0 else np.zeros(len(class_names))

                metrics_df = pd.DataFrame({
                    "Class": class_names,
                    "Precision": precision,
                    "Recall": recall,
                    "F1": f1
                })

                st.success(f"Test Accuracy: {accuracy:.2f}% (on {len(test_labels)} test samples)")
                st.subheader("Per-Class Metrics")
                st.dataframe(metrics_df)

                st.subheader("Confusion Matrix")
                df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
                st.dataframe(df_cm)

            except Exception as e:
                st.error(str(e))

# --------- LIVE RECOGNITION TAB ---------
with tabs[3]:
    st.subheader("Start live webcam recognition")
    st.write("Click **Start** below and allow camera access. Features alignment, FPS, and optional attendance logging.")

    log_attendance = st.checkbox("Log Attendance", value=False)

    recognizer, id_to_name = load_model(model_type)
    if recognizer is None or not id_to_name:
        st.warning("No trained model found. Please train the model first in the **Train** tab.")
    else:
        id_to_name = {int(k): v for k, v in id_to_name.items()}

        last_logged = {}  # To debounce logging per person

        class VideoProcessor:
            def __init__(self):
                self.recognizer = recognizer
                self.labels = id_to_name
                self.frame_count = 0
                self.last_faces = []
                self.skip_frames = 2
                self.prev_time = time.time()

            def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
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
                        roi = cv2.resize(roi, FACE_RESIZE, interpolation=cv2.INTER_LINEAR)
                        roi = cv2.equalizeHist(roi)
                        roi = align_face(roi)
                    except Exception:
                        continue
                    label_id, confidence = self.recognizer.predict(roi)
                    if confidence <= st.session_state.get("conf_threshold", 50):
                        name = self.labels.get(label_id, "Unknown")
                        color = (0, 255, 0)
                        caption = f"{name} ({confidence:.1f})"
                        if log_attendance and name != "Unknown":
                            now = time.time()
                            if name not in last_logged or now - last_logged[name] > 60:  # Log every 60s
                                last_logged[name] = now
                                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                log_entry = pd.DataFrame({"Timestamp": [timestamp], "Name": [name]})
                                if ATTENDANCE_LOG.exists():
                                    log_entry.to_csv(ATTENDANCE_LOG, mode='a', header=False, index=False)
                                else:
                                    log_entry.to_csv(ATTENDANCE_LOG, index=False)
                    else:
                        name = "Unknown"
                        color = (0, 0, 255)
                        caption = f"{name} ({confidence:.1f})"

                    st.write(f"Face size: {w}x{h}, Confidence: {confidence:.1f}, Predicted: {name}")
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    draw_label(img, caption, x, y)

                return av.VideoFrame.from_ndarray(img, format="bgr24")

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

# --------- ATTENDANCE TAB ---------
with tabs[4]:
    st.subheader("Attendance Log")
    if ATTENDANCE_LOG.exists():
        df_log = pd.read_csv(ATTENDANCE_LOG)
        st.dataframe(df_log)
        if st.button("Clear Log"):
            ATTENDANCE_LOG.unlink()
            st.success("Log cleared.")
            st.rerun()
    else:
        st.info("No attendance logged yet. Enable in Live Recognition tab.")

st.markdown("---")
st.caption("Further enhancements: Multiple recognizer options, face alignment, brightness/contrast augmentation, attendance logging with debounce, detailed metrics. Tip: Use 20+ varied images per person for optimal results.")
```
