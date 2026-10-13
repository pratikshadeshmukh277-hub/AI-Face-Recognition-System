
import streamlit as st
import cv2
import os
import json
import hashlib
import numpy as np
from datetime import datetime, timezone

from firebase_service import (
    register_user,
    get_user,
    save_recognition_log
)
from firebase_config import db

# ---------------- CONFIGURATION ----------------

st.set_page_config(
    page_title="AI Face Recognition System",
    page_icon="🤖",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAINER_DIR = os.path.join(BASE_DIR, "trainer")
MODEL_PATH = os.path.join(TRAINER_DIR, "face_model.yml")
MAPPING_PATH = os.path.join(TRAINER_DIR, "label_map.json")
CASCADE_PATH = os.path.join(
    BASE_DIR, "haarcascade_frontalface_default.xml"
)

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(TRAINER_DIR, exist_ok=True)

st.title("🤖 AI Face Recognition System")
st.caption("Face recognition powered by OpenCV, LBPH and Firebase")

# ---------------- HELPER FUNCTIONS ----------------

def get_face_detector():
    detector = cv2.CascadeClassifier(CASCADE_PATH)
    if detector.empty():
        raise FileNotFoundError(
            f"Haar Cascade XML not found: {CASCADE_PATH}"
        )
    return detector


def image_to_gray(image_bytes):
    array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Could not read the image.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, gray


def save_face_sample(user_id, image_bytes):
    _, gray = image_to_gray(image_bytes)
    detector = get_face_detector()

    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(80, 80)
    )

    if len(faces) == 0:
        return False, "No face detected. Try again with better lighting."

    # Use the largest detected face
    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    face = cv2.resize(gray[y:y+h, x:x+w], (200, 200))

    user_folder = os.path.join(DATASET_DIR, user_id)
    os.makedirs(user_folder, exist_ok=True)

    existing = [
        name for name in os.listdir(user_folder)
        if name.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    image_path = os.path.join(
        user_folder, f"{len(existing) + 1}.jpg"
    )

    if not cv2.imwrite(image_path, face):
        return False, "Could not save the face image."

    return True, f"Face sample saved. Total samples: {len(existing) + 1}"


def train_model():
    if not hasattr(cv2, "face"):
        raise RuntimeError(
            "cv2.face unavailable. Install compatible "
            "opencv-contrib-python."
        )

    faces = []
    labels = []
    label_map = {}
    label = 0

    for user_id in sorted(os.listdir(DATASET_DIR)):
        folder = os.path.join(DATASET_DIR, user_id)
        if not os.path.isdir(folder):
            continue

        user_images = []

        for filename in os.listdir(folder):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            path = os.path.join(folder, filename)
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            if image is not None:
                user_images.append(image)

        if user_images:
            label_map[str(label)] = user_id
            faces.extend(user_images)
            labels.extend([label] * len(user_images))
            label += 1

    if not faces:
        raise ValueError("No face images found. Capture faces first.")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels, dtype=np.int32))
    recognizer.save(MODEL_PATH)

    with open(MAPPING_PATH, "w", encoding="utf-8") as file:
        json.dump(label_map, file, indent=4)

    return len(faces), len(label_map)


def get_user_count():
    return len(list(db.collection("users").stream()))


def get_recent_logs(limit=20):
    docs = (
        db.collection("recognition_logs")
        .order_by("timestamp", direction="DESCENDING")
        .limit(limit)
        .stream()
    )
    return [doc.to_dict() for doc in docs]


# ---------------- SIDEBAR ----------------

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Register User",
        "Capture Face",
        "Train Model",
        "Recognize Face",
        "Recognition History"
    ]
)

# ---------------- DASHBOARD ----------------

if page == "Dashboard":
    st.subheader("Dashboard")

    try:
        users = get_user_count()
        logs = get_recent_logs(limit=100)

        col1, col2, col3 = st.columns(3)
        col1.metric("Registered Users", users)
        col2.metric("Recent Logs Loaded", len(logs))
        col3.metric(
            "Trained Model",
            "Available" if os.path.isfile(MODEL_PATH) else "Not trained"
        )

        st.subheader("Recent Recognition Activity")

        if logs:
            st.dataframe(logs, use_container_width=True)
        else:
            st.info("No recognition logs found yet.")

    except Exception as error:
        st.error(f"Could not load Firebase dashboard: {error}")

# ---------------- REGISTER USER ----------------

elif page == "Register User":
    st.subheader("Register a New User")

    with st.form("registration_form"):
        user_id = st.text_input("User ID", placeholder="USER002").strip().upper()
        name = st.text_input("Full Name")
        email = st.text_input("Email Address")
        submitted = st.form_submit_button("Register User")

    if submitted:
        if not user_id or not name.strip() or not email.strip():
            st.warning("Please fill in all fields.")
        else:
            try:
                existing_user = get_user(user_id)

                if existing_user:
                    st.warning("This User ID is already registered.")
                else:
                    message = register_user(
                        user_id, name.strip(), email.strip()
                    )
                    st.success(message)
                    st.info("Next, open Capture Face to save face samples.")
            except Exception as error:
                st.error(f"Registration failed: {error}")

# ---------------- CAPTURE FACE ----------------

elif page == "Capture Face":
    st.subheader("Capture Face Samples")
    st.write("Register the user first, then capture images in good lighting.")

    user_id = st.text_input(
        "Registered User ID", placeholder="USER002"
    ).strip().upper()

    photo = st.camera_input("Take a face photo")

    if photo is not None and st.button("Save Face Sample"):
        try:
            if not user_id or not get_user(user_id):
                st.error("Please enter a registered User ID.")
            else:
                image_hash = hashlib.sha256(photo.getvalue()).hexdigest()
                saved_hashes = st.session_state.setdefault(
                    "saved_photo_hashes", set()
                )

                if image_hash in saved_hashes:
                    st.warning("This exact photo has already been saved.")
                else:
                    success, message = save_face_sample(
                        user_id, photo.getvalue()
                    )

                    if success:
                        saved_hashes.add(image_hash)
                        st.success(message)
                    else:
                        st.warning(message)

        except Exception as error:
            st.error(f"Face capture failed: {error}")

    st.caption(
        "Take and save multiple different photos, changing your head angle "
        "and expression slightly. Aim for around 20–30 samples per user."
    )

# ---------------- TRAIN MODEL ----------------

elif page == "Train Model":
    st.subheader("Train Face Recognition Model")

    st.write(
        "Training uses the face images currently stored in the dataset folder."
    )

    if st.button("Train / Retrain Model"):
        try:
            with st.spinner("Training model..."):
                image_count, user_count = train_model()

            st.success("Training completed successfully!")
            st.metric("Images Trained", image_count)
            st.metric("Users Trained", user_count)

        except Exception as error:
            st.error(f"Training failed: {error}")

# ---------------- RECOGNIZE FACE ----------------

elif page == "Recognize Face":
    st.subheader("Recognize a Face")

    if not os.path.isfile(MODEL_PATH) or not os.path.isfile(MAPPING_PATH):
        st.warning("Train the model before recognizing faces.")
    else:
        photo = st.camera_input("Take a photo for recognition")

        threshold = st.slider(
            "LBPH distance threshold",
            min_value=30,
            max_value=100,
            value=65
        )

        if photo is not None and st.button("Recognize"):
            try:
                if not hasattr(cv2, "face"):
                    raise RuntimeError("cv2.face is unavailable.")

                _, gray = image_to_gray(photo.getvalue())
                detector = get_face_detector()

                detected = detector.detectMultiScale(
                    gray,
                    scaleFactor=1.2,
                    minNeighbors=5,
                    minSize=(80, 80)
                )

                if len(detected) == 0:
                    st.warning("No face detected. Please try again.")
                else:
                    recognizer = cv2.face.LBPHFaceRecognizer_create()
                    recognizer.read(MODEL_PATH)

                    with open(MAPPING_PATH, "r", encoding="utf-8") as file:
                        label_map = json.load(file)

                    x, y, w, h = max(
                        detected, key=lambda box: box[2] * box[3]
                    )
                    face = cv2.resize(gray[y:y+h, x:x+w], (200, 200))

                    label, distance = recognizer.predict(face)
                    user_id = label_map.get(str(label))

                    if user_id is not None and distance < threshold:
                        st.success(f"Recognized User ID: {user_id}")
                        st.write(f"LBPH distance: {distance:.2f}")

                        try:
                            save_recognition_log(
                                user_id=user_id,
                                recognized=True,
                                confidence=round(float(distance), 2)
                            )
                            st.success("Recognition log saved to Firebase.")
                        except Exception as error:
                            st.error(f"Firebase logging failed: {error}")
                    else:
                        st.error("Unknown face or confidence too low.")
                        st.write(f"LBPH distance: {distance:.2f}")

                        # This log does not identify the unknown person.
                        try:
                            save_recognition_log(
                                user_id="UNKNOWN",
                                recognized=False,
                                confidence=round(float(distance), 2)
                            )
                        except Exception as error:
                            st.warning(f"Could not save unknown-face log: {error}")

            except Exception as error:
                st.error(f"Recognition failed: {error}")

# ---------------- RECOGNITION HISTORY ----------------

elif page == "Recognition History":
    st.subheader("Recognition History")

    if st.button("Refresh History"):
        st.rerun()

    try:
        logs = get_recent_logs(limit=100)

        if logs:
            st.dataframe(logs, use_container_width=True)
        else:
            st.info("No recognition history available.")
    except Exception as error:
        st.error(f"Could not retrieve history: {error}")
 