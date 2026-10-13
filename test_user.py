from firebase_service import register_user, get_user

# Add a sample user
result = register_user(
    user_id="USER001",
    name="Test User",
    email="test@example.com"
)

print(result)

# Retrieve user details
user = get_user("USER001")

print("User Details:")
print(user)