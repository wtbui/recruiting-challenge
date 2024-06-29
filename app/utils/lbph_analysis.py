import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from .analysis_params import LBP_TEXTURE_LEVELS, LBP_MAX_DISTANCE

def extract_lbp_histogram(image, P=LBP_TEXTURE_LEVELS[0], R=LBP_TEXTURE_LEVELS[1]) -> np.array:
    """
    Extract Local Binary Pattern (LBP) histogram from the input image.

    Args:
        image (numpy.ndarray): Input image in BGR format.
        P (int): Number of circularly symmetric neighbor set points (default from LBP_TEXTURE_LEVELS).
        R (int): Radius of circle (default from LBP_TEXTURE_LEVELS).

    Returns:
        numpy.ndarray: Normalized histogram of the LBP of the image.
    """
    # Convert image to grayscale
    image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_grayscale = cv2.resize(image_grayscale, (128, 128))
    image_grayscale = cv2.equalizeHist(image_grayscale)

    # Generate LBP of image and its histogram
    image_lbp = local_binary_pattern(image_grayscale, P, R, method="uniform")
    image_lbp_hist, _ = np.histogram(image_lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))

    # Normalize the histogram
    image_lbp_hist = image_lbp_hist.astype("float")
    image_lbp_hist /= (image_lbp_hist.sum() + 1e-6)

    return image_lbp_hist

def compare_lbp_histograms(face1_histogram: list[float], face2_histogram: list[float]) -> float:
    """
    Compare two LBP histograms using the chi-square distance.

    Args:
        face1_histogram (list[float]): LBP histogram of the first face.
        face2_histogram (list[float]): LBP histogram of the second face.

    Returns:
        float: Confidence score based on the similarity of the histograms.
    """
    # Convert to numpy arrays
    face1_histogram = np.array(face1_histogram)
    face2_histogram = np.array(face2_histogram)

    chi_sq_dist = 0.5 * np.sum(((face1_histogram - face2_histogram) ** 2) / (face1_histogram + face2_histogram + 1e-6))
    confidence_score = max(0, 100 * (1 - (chi_sq_dist / LBP_MAX_DISTANCE)))

    return confidence_score