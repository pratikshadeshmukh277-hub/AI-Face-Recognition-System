from firebase_config import db
from datetime import datetime, timezone


# Register a new user
def register_user(user_id, name, email):

    user_data = {
        "user_id": user_id,
        "name": name,
        "email": email,
        "created_at": datetime.now(timezone.utc)
    }

    db.collection("users").document(user_id).set(user_data)

    return "User registered successfully!"


# Get user details
def get_user(user_id):

    document = (
        db.collection("users")
        .document(user_id)
        .get()
    )

    if document.exists:
        return document.to_dict()

    return None


# Save face recognition log
def save_recognition_log(user_id, recognized, confidence):

    log_data = {
        "user_id": user_id,
        "recognized": recognized,
        "confidence": confidence,
        "timestamp": datetime.now(timezone.utc)
    }

    db.collection("recognition_logs").add(log_data)

    return "Recognition log saved successfully!"