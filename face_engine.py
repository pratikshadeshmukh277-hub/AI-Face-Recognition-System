import cv2
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAINER_DIR = os.path.join(BASE_DIR, "trainer")

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(TRAINER_DIR, exist_ok=True)

cascade_path = os.path.join(
    BASE_DIR,
    "haarcascade_frontalface_default.xml"
)

face_detector = cv2.CascadeClassifier(cascade_path)

if face_detector.empty():
    raise FileNotFoundError(
        f"XML file missing or invalid: {cascade_path}"
    )


def capture_faces(user_id, samples=30):
    user_folder = os.path.join(DATASET_DIR, user_id)
    os.makedirs(user_folder, exist_ok=True)

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Webcam could not be opened.")

    count = 0

    try:
        while count < samples:
            success, frame = camera.read()

            if not success:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_detector.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(80, 80)
            )

            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                face = cv2.resize(face, (200, 200))

                count += 1

                image_path = os.path.join(
                    user_folder, f"{count}.jpg"
                )

                cv2.imwrite(image_path, face)

                cv2.rectangle(
                    frame, (x, y), (x+w, y+h),
                    (0, 255, 0), 2
                )

                cv2.putText(
                    frame,
                    f"Samples: {count}/{samples}",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                break

            cv2.imshow("Face Registration", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()

    return count