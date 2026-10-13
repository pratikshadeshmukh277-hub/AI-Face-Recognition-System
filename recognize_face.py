
import cv2
import os
import json
from datetime import datetime, timezone

from firebase_service import save_recognition_log

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "trainer", "face_model.yml")
MAPPING_PATH = os.path.join(BASE_DIR, "trainer", "label_map.json")
CASCADE_PATH = os.path.join(
    BASE_DIR, "haarcascade_frontalface_default.xml"
)

# Check required files
for path in (MODEL_PATH, MAPPING_PATH, CASCADE_PATH):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Required file not found: {path}")

# Load trained model
if not hasattr(cv2, "face"):
    raise RuntimeError(
        "cv2.face is unavailable. Install the compatible "
        "opencv-contrib-python package in this Python environment."
    )

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(MODEL_PATH)

with open(MAPPING_PATH, "r", encoding="utf-8") as file:
    label_map = json.load(file)

face_detector = cv2.CascadeClassifier(CASCADE_PATH)

if face_detector.empty():
    raise RuntimeError("Haar Cascade XML file could not be loaded.")

# LBPH distance: lower values generally indicate a closer match.
# Tune this threshold using tests with known and unknown people.
CONFIDENCE_THRESHOLD = 65

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Webcam could not be opened.")

last_logged = None

try:
    while True:
        success, frame = camera.read()
        if not success:
            print("Could not read webcam frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detected_faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(80, 80)
        )

        for (x, y, w, h) in detected_faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))

            label, distance = recognizer.predict(face)

            user_id = label_map.get(str(label))
            if user_id is None:
                user_id = label_map.get(label)

            if user_id is not None and distance < CONFIDENCE_THRESHOLD:
                display_text = f"User: {user_id} ({distance:.1f})"
                color = (0, 200, 0)

                # Avoid writing a log for every webcam frame.
                if last_logged != user_id:
                    try:
                        save_recognition_log(
                            user_id=user_id,
                            recognized=True,
                            confidence=round(float(distance), 2)
                        )
                        print(f"Recognition log saved for {user_id}")
                        last_logged = user_id
                    except Exception as error:
                        print("Firebase logging failed:", error)
            else:
                display_text = f"Unknown ({distance:.1f})"
                color = (0, 0, 255)
                last_logged = None

            cv2.rectangle(
                frame, (x, y), (x+w, y+h), color, 2
            )
            cv2.putText(
                frame,
                display_text,
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

        cv2.imshow("AI Face Recognition - Press Q to Quit", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
