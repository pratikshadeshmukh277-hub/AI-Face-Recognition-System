import cv2
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAINER_DIR = os.path.join(BASE_DIR, "trainer")

MODEL_PATH = os.path.join(TRAINER_DIR, "face_model.yml")

os.makedirs(TRAINER_DIR, exist_ok=True)

# Check LBPH recognizer availability
if not hasattr(cv2, "face"):
    raise RuntimeError(
        "cv2.face is not available. Install opencv-contrib-python."
    )

recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []

# Assign a numeric label to each user's folder
label_map = {}
current_label = 0

for user_id in os.listdir(DATASET_DIR):
    user_folder = os.path.join(DATASET_DIR, user_id)

    if not os.path.isdir(user_folder):
        continue

    label_map[current_label] = user_id

    for filename in os.listdir(user_folder):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(user_folder, filename)

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"Skipping unreadable image: {image_path}")
            continue

        faces.append(image)
        labels.append(current_label)

    current_label += 1

if not faces:
    raise RuntimeError(
        "No face images found. Capture face images first."
    )

labels = np.array(labels, dtype=np.int32)

# Train the recognizer
recognizer.train(faces, labels)

# Save trained model
recognizer.save(MODEL_PATH)

# Save user-label mapping
import json

mapping_path = os.path.join(TRAINER_DIR, "label_map.json")

with open(mapping_path, "w", encoding="utf-8") as file:
    json.dump(label_map, file, indent=4)

print("Face training completed successfully!")
print(f"Total images trained: {len(faces)}")
print(f"Total users trained: {len(label_map)}")
print(f"Model saved at: {MODEL_PATH}")
print(f"Label mapping saved at: {mapping_path}")