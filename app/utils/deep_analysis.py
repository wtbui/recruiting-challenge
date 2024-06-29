from facenet_pytorch import InceptionResnetV1
import torch
import cv2
import numpy as np
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity

def image_preprocess(image):
    """
    Preprocess the input image for FaceNet model.

    Args:
        image (numpy.ndarray): Input image in BGR format.

    Returns:
        torch.Tensor: Preprocessed image tensor ready for model input.
    """
    # Convert to RGB and resize
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image, (160, 160))

    # Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image_tensor = transform(image_rgb)

    # Add a batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor

def extract_deep_features(image):
    """
    Extract deep features from the input image using FaceNet model.

    Args:
        image (numpy.ndarray): Input image in BGR format.

    Returns:
        numpy.ndarray: Embeddings extracted from the image.
    """
    # Load FaceNet model
    model = InceptionResnetV1(pretrained='vggface2').eval()

    image_tensor = image_preprocess(image)

    # Calculate Embeddings
    with torch.no_grad():
        embeddings = model(image_tensor)

    embeddings_np = embeddings.numpy()
    return embeddings_np

def compare_embeddings(face1_embeddings: list[float], face2_embeddings: list[float]) -> float:
    """
    Compare two sets of facial embeddings using cosine similarity.

    Args:
        face1_embeddings (list[float]): List of embeddings for the first face.
        face2_embeddings (list[float]): List of embeddings for the second face.

    Returns:
        float: Confidence score based on the similarity of the embeddings.
    """
    # Convert to numpy arrays
    face1_embeddings = np.array(face1_embeddings)
    face2_embeddings = np.array(face2_embeddings)

    # Calculate cosine similarity between embeddings
    similarity = cosine_similarity(face1_embeddings, face2_embeddings)

    # Calculate confidence level
    confidence = (similarity[0][0] + 1) / 2 * 100

    return confidence
