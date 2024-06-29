from app.models import Profile
from .analysis_params import LANDMARK_MODEL_PATH, LM_WEIGHT, DF_WEIGHT, LBPH_WEIGHT
import numpy as np
from .deep_analysis import extract_deep_features, compare_embeddings
from .landmark_analysis import LandmarkAnalyzer, compute_distance_values, compare_distances
from .lbph_analysis import extract_lbp_histogram, compare_lbp_histograms
import cv2
import os

# Path to dlib models
relative_path = os.path.dirname(os.path.abspath(__file__))
dlib_predictor_filepath = os.path.join(relative_path, LANDMARK_MODEL_PATH)

lm_analyzer = LandmarkAnalyzer(dlib_predictor_filepath)

def generate_profile(image_file) -> Profile:
    """
    Generate a facial profile from the input image.

    Args:
        image_file (PIL.Image.Image): Input image file.

    Returns:
        Profile: Generated profile containing landmark distances, deep features, and LBP histogram.
    """
    # Convert the PIL Image to a NumPy array
    image = np.array(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Extract Features
    landmark_values = lm_analyzer.extract_landmarks(image)
    landmark_distances = compute_distance_values(landmark_values)
    deep_features = extract_deep_features(image).tolist()
    lbp_histogram = extract_lbp_histogram(image).tolist()

    # Generate profile
    profile = Profile(
        landmark_distances=landmark_distances,
        deep_features=deep_features,
        lbp_histogram=lbp_histogram,
    )

    return profile

def compute_combined_confidence(confidences: list[float], weights: list[float]) -> float:
    """
    Compute combined confidence score from individual confidence scores and weights.

    Args:
        confidences (list[float]): List of individual confidence scores.
        weights (list[float]): List of weights corresponding to the confidence scores.

    Returns:
        float: Combined confidence score.
    """
    weighted_sum = sum(c * w for c, w in zip(confidences, weights))
    total_weight = sum(weights)

    return weighted_sum / total_weight

def compare_profiles(profile1: Profile, profile2: Profile) -> float:
    """
    Compare two facial profiles and compute a confidence score.

    Args:
        profile1 (Profile): The first profile.
        profile2 (Profile): The second profile.

    Returns:
        float: Confidence score based on the similarity of the profiles.
    """
    lm_confidence = compare_distances(profile1.landmark_distances, profile2.landmark_distances)
    df_confidence = compare_embeddings(profile1.deep_features, profile2.deep_features)
    lbph_confidence = compare_lbp_histograms(profile1.lbp_histogram, profile2.lbp_histogram)
    weights = [LM_WEIGHT, DF_WEIGHT, LBPH_WEIGHT]

    return compute_combined_confidence([lm_confidence, df_confidence, lbph_confidence], weights)
