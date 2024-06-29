import random
from fastapi import APIRouter, File, UploadFile, HTTPException
from app.models import ProfileResponse, VerificationResponse
from app.utils import generate_profile, compare_profiles
from app.utils.analysis_params import CONFIDENCE_THRESHOLD
from io import BytesIO
from PIL import Image

router = APIRouter()

# In-memory storage for facial profiles, pivot to database in future iterations
profile_db = {}

@router.get(
        "/profile/{profile_id}",
        response_model=ProfileResponse,
        description="Retrieves previously catalogued profile",
        summary="Retrieves profile",
        tags=["profile"],
    )
async def profile_get(profile_id: str):
    """
    Retrieve a previously uploaded facial profile

    Args:
        profile_id (str): string containing profile id

    Return:
        string: String with success message
        Profile: Profile with inputted profile_id

    Error:
        HTTPException: If profile is not found
    """
    if profile_id not in profile_db:
        raise HTTPException(status_code=404, detail="Profile not found")
    return ProfileResponse(
        message="Retrieved profile with id " + str(profile_id),
        profile=profile_db[profile_id]
    )

@router.delete(
        "/profile/delete/{profile_id}",
        response_model=dict[str,str],
        description="Deletes previously catalogued profile",
        summary="Deletes profile",
        tags=["profile"])
async def profile_delete(profile_id: str):
    """
    Deletes a previously uploaded facial profile

    Args:
        profile_id (str): string containing profile id

    Return:
        dict: Dictionary containing a success message

    Error:
        HTTPException: If profile is not found
    """
    if profile_id not in profile_db:
        raise HTTPException(status_code=404, detail="Profile not found")
    try: 
        del profile_db[profile_id]
        return {
            "message": "Removed profile with id " + profile_id,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post(
    "/profile/create", 
    response_model=dict[str,str],
    description="Builds a detailed facial profile from an image.",
    summary="Creates profile",
    tags=["profile"])
async def create_profile(file: UploadFile = File(...)):
    """
    Creates a detailed profile based on an uploaded image

    Args:
        file (File): File containing image corresponding to profile

    Return:
        dict: Dictionary containing a profile id

    Error:
        HTTPException: If file is not in correct format or if profile fails to generate
    """
    if not file.filename.endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(status_code=400, detail="Invalid Image Format")
    try:
        img = Image.open(BytesIO(await file.read()))
        profile = generate_profile(img)
        profile_id = random.randrange(10000) # Temporary measure, future iterations would use uuid generator
        profile_db[str(profile_id)] = profile
        return {"profile_id": str(profile_id)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post(
    "/profile/verify/{profile_id}",
    response_model=VerificationResponse,
    description="Uses an existing facial profile to verify legitimacy of photo",
    summary="Verifies image",
    tags=["profile"])
async def verify_photo(profile_id: str, file: UploadFile = File(...)):
    """
    Verifies the legitimacy of an uploaded photo based on a previously uploaded profile

    Args:
        profile_id (str): String containing reference profile id
        file (File): File containing image corresponding to profile

    Return:
        dict: Dictionary containing success message, deepfake status, and confidence level regarding deepfake status

    Error:
        HTTPException: If file is not in correct format, profile not found, or verification fails
    """
    if profile_id not in profile_db:
        raise HTTPException(status_code=404, detail="Profile not found")
    
    if not file.filename.endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(status_code=400, details="Invalid Image Format")
    try:
        img = Image.open(BytesIO(await file.read()))
        profile1 = profile_db[profile_id]
        profile2 = generate_profile(img)
        confidence = compare_profiles(profile1, profile2)
        is_deepfaked = False
        message = ""
        if confidence < CONFIDENCE_THRESHOLD:
            message = "Image is deepfaked with confidence of " + str(confidence)
            is_deepfaked = True
        else:
            message = "Image is not deepfaked with confidence of " + str(confidence)
            
        return VerificationResponse(
            message=message,
            is_deepfaked=is_deepfaked,
            confidence=confidence
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


