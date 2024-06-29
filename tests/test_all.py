from fastapi.testclient import TestClient
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.main import app

client = TestClient(app)

# Helper function to load image and image paths
def load_image(file_path):
    with open(file_path, "rb") as f:
        return f.read()
    
image_path1 = os.path.join(os.path.dirname(__file__), "test_images/tom1.jpg")
image_path2 = os.path.join(os.path.dirname(__file__), "test_images/tom2.jpg")
different_image_path = os.path.join(os.path.dirname(__file__), "test_images/devito.jpg")
fake_image_path = os.path.join(os.path.dirname(__file__), "test_images/tom_deepfake.jpg")

## Profile creation and retrieval ##
def test_retrieve_existing_profile():
    print("Testing create and get profile")
    # Create a profile first
    image_data = load_image(image_path1)
    response = client.post("/profile/create", files={"file": ("tom1.jpg", image_data, "image/jpeg")})
    profile_id = response.json()['profile_id']
    assert response.status_code == 200

    # Retrieve the created profile
    response = client.get(f"/profile/{profile_id}")
    assert response.status_code == 200
    assert response.json()["message"] == f"Retrieved profile with id {profile_id}"

def test_retrieve_non_existing_profile():
    print("Testing retrieval non-existent profile")
    response = client.get("/profile/non_existing_id")
    assert response.status_code == 404
    assert response.json()["detail"] == "Profile not found"

## Profile creation and deletion ##
def test_delete_existing_profile():
    print("Testing delete profile")
    # Create a profile first
    image_data = load_image(image_path2)
    response = client.post("/profile/create", files={"file": ("tom2.jpg", image_data, "image/jpeg")})
    profile_id = response.json()['profile_id']

    # Delete the created profile
    response = client.delete(f"/profile/delete/{profile_id}")
    assert response.status_code == 200
    assert response.json()["message"] == f"Removed profile with id {profile_id}"

def test_delete_non_existing_profile():
    print("Testing deletion of non-existent profile")
    response = client.delete("/profile/delete/non_existing_id")
    assert response.status_code == 404
    assert response.json()["detail"] == "Profile not found"

## Test deepfake detection ##
def test_verify_real_photo_with_existing_profile():
    print("Testing verification of different photos of same person")
    # Create a profile first
    image_data = load_image(image_path1)
    response = client.post("/profile/create", files={"file": ("tom1.jpg", image_data, "image/jpeg")})
    profile_id = response.json()['profile_id']

    # Verify the photo
    test_image_data = load_image(image_path2)
    response = client.post(f"/profile/verify/{profile_id}", files={"file": ("tom2.jpg", test_image_data, "image/jpeg")})
    print(response.json()['message'])
    assert response.status_code == 200
    assert response.json()["is_deepfaked"] == False

def test_verify_fake_photo_with_existing_profile():
    print("Testing verification of deepfaked photo")
    # Create a profile first
    image_data = load_image(image_path1)
    response = client.post("/profile/create", files={"file": ("tom1.jpg", image_data, "image/jpeg")})
    profile_id = response.json().get("profile_id")

    # Verify the photo
    test_image_data = load_image(fake_image_path)
    response = client.post(f"/profile/verify/{profile_id}", files={"file": ("tom_deepfake.jpg", test_image_data, "image/jpeg")})
    print(response.json()['message'])
    assert response.status_code == 200
    assert response.json()["is_deepfaked"] == True

def test_verify_real_photo_with_non_existing_profile():
    print("Testing verification on non-existent profile")
    test_image_data = load_image(image_path2)
    response = client.post("/profile/verify/9999999999", files={"file": ("tom2.jpg", test_image_data, "image/jpeg")})
    assert response.status_code == 404
    assert response.json()["detail"] == "Profile not found"

if __name__ == "__main__":
    test_retrieve_existing_profile()
    test_retrieve_non_existing_profile()
    test_delete_existing_profile()
    test_delete_non_existing_profile()
    test_verify_real_photo_with_existing_profile()
    test_verify_fake_photo_with_existing_profile()
    test_verify_real_photo_with_non_existing_profile()
    print("All tests passed!")
