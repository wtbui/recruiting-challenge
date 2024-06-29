import cv2
import dlib
import numpy as np
from .analysis_params import LANDMARK_MAX_DIFFERENCE

class LandmarkAnalyzer:
    """
    Analyzes facial landmarks using dlib's face detector and shape predictor.

    Args:
        model_filepath (str): Path to dlib's shape predictor model file.

    Attributes:
        dlib_predictor (dlib.shape_predictor): dlib shape predictor.
        dlib_detector (dlib.get_frontal_face_detector): dlib face detector.
    """
    def __init__(self, model_filepath: str):
        # Initalize dlib face detector and predictor
        self.dlib_predictor = dlib.shape_predictor(model_filepath)
        self.dlib_detector = dlib.get_frontal_face_detector()

    def extract_landmarks(self, image):
        """
        Extract facial landmarks from the input image.

        Args:
            image (numpy.ndarray): Input image in BGR format.

        Returns:
            dlib.full_object_detection: Detected facial landmarks.

        Raises:
            TypeError: If no faces are detected within the image.
        """
        # Convert image to grayscale
        image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_grayscale = cv2.resize(image, (160, 160))

        # Detect faces in image, will use first one for analysis
        detected_faces = self.dlib_detector(image_grayscale)

        # Extract facial landmarks
        if not len(detected_faces):
            raise TypeError("No faces detected within image")
        
        landmarks = self.dlib_predictor(image_grayscale, detected_faces[0])

        return landmarks

def compute_distance_values(landmarks):
    """
    Compute euclidean distances between facial landmarks.

    Args:
        landmarks (dlib.full_object_detection): Detected facial landmarks.

    Returns:
        dict: Dictionary of calculated distances between key facial landmarks.

    Raises:
        ValueError: If the landmark array shape is unexpected.
    """
    # Convert landmarks to numpy array
    landmark_np = np.array([(p.x, p.y) for p in landmarks.parts()])

    # Ensure landmark_np has the expected shape (68, 2)
    if landmark_np.shape != (68, 2):
        raise ValueError(f"Unexpected landmark array shape: {landmark_np.shape}")

    # Calculate euclidean distances between facial landmarks
    distances = {
        "inter_eye": np.linalg.norm(landmark_np[36] - landmark_np[45]),
        "left_eye_to_left_brow": np.linalg.norm(landmark_np[36] - landmark_np[19]),
        "right_eye_to_right_brow": np.linalg.norm(landmark_np[42] - landmark_np[24]),
        "nose_to_left_eye": np.linalg.norm(landmark_np[30] - landmark_np[36]),
        "nose_to_right_eye": np.linalg.norm(landmark_np[30] - landmark_np[45]),
        "nose_width": np.linalg.norm(landmark_np[31] - landmark_np[35]),
        "mouth_width": np.linalg.norm(landmark_np[48] - landmark_np[54]),
        "upper_lip_to_lower_lip": np.linalg.norm(landmark_np[62] - landmark_np[66]),
        "chin_to_jaw_left": np.linalg.norm(landmark_np[8] - landmark_np[0]),
        "chin_to_jaw_right": np.linalg.norm(landmark_np[8] - landmark_np[16]),
        "nose_to_chin": np.linalg.norm(landmark_np[30] - landmark_np[8])
    }
    
    # Additional calculation to calculate facial symmetry
    nose_tip = landmark_np[30]

    symmetry_distances = {
        "eye_symmetry": abs(np.linalg.norm(landmark_np[36] - nose_tip) - np.linalg.norm(landmark_np[45] - nose_tip)),
        "brow_symmetry": abs(np.linalg.norm(landmark_np[19] - nose_tip) - np.linalg.norm(landmark_np[24] - nose_tip)),
        "mouth_symmetry": abs(np.linalg.norm(landmark_np[48] - nose_tip) - np.linalg.norm(landmark_np[54] - nose_tip)),
        "jaw_symmetry": abs(np.linalg.norm(landmark_np[0] - nose_tip) - np.linalg.norm(landmark_np[16] - nose_tip)),
    }
    
    distances.update(symmetry_distances)

    return distances

def compare_distances(face1_distances: dict, face2_distances: dict) -> float:
    """
    Compare distances between two sets of facial landmarks.

    Args:
        face1_distances (dict): Dictionary of distances for the first face.
        face2_distances (dict): Dictionary of distances for the second face.

    Returns:
        float: Confidence score based on the similarity of the distances.
    """
    # Calculate differences between each face's distances
    differences = {key: abs(face1_distances[key] - face2_distances[key]) for key in face1_distances}

    # Sum of differences to get a single similarity measure
    total_difference = sum(differences.values())

    # Normalize the total difference and compute confidence score
    normalized_difference = total_difference / LANDMARK_MAX_DIFFERENCE
    confidence_score = max(0, 100 * (1 - normalized_difference))

    return confidence_score
