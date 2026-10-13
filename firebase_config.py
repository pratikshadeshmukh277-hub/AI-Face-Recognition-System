
import os
import firebase_admin
from firebase_admin import credentials, firestore

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KEY_PATH = os.path.join(BASE_DIR, "serviceAccountKey.json")


def initialize_firebase():
    if firebase_admin._apps:
        firebase_app = firebase_admin.get_app()
    else:
        if not os.path.isfile(KEY_PATH):
            raise FileNotFoundError(
                f"Firebase credentials not found: {KEY_PATH}\n"
                "Save serviceAccountKey.json in "
                "D:\\AI-Face-Recognition-System"
            )

        firebase_credential = credentials.Certificate(KEY_PATH)
        firebase_app = firebase_admin.initialize_app(
            firebase_credential
        )

    return firestore.client(app=firebase_app)


db = initialize_firebase()
