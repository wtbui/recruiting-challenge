from pydantic import BaseModel
from typing import List, Dict

# Profile Model
class Profile(BaseModel):
    landmark_distances: dict
    deep_features: list
    lbp_histogram: list

# Profile Response
class ProfileResponse(BaseModel):
    message: str
    profile: Profile

# Verification Response
class VerificationResponse(BaseModel):
    message: str
    is_deepfaked: bool
    confidence: float