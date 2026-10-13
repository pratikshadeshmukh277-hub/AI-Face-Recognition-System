from firebase_config import db

try:
    collections = list(db.collections())

    print("Firebase connected successfully!")

    for collection in collections:
        print("Collection:", collection.id) 

except Exception as e:
    print("Firebase connection failed:", e)