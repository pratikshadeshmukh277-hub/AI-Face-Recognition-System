from face_engine import capture_faces

user_id = input("Enter registered user ID: ")

try:
    count = capture_faces(user_id, samples=30)

    print(f"Captured {count} face images.")

except Exception as error:
    print("Error:", error)