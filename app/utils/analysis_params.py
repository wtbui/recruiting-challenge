# ANALYSIS CONSTANTS
LANDMARK_MAX_DIFFERENCE = 200 # Normalization factor for landmark analysis
LANDMARK_MODEL_PATH = './dlib_models/shape_predictor_68_face_landmarks_GTX.dat' # Current landmark model
LBP_TEXTURE_LEVELS = (24, 3) # Defines default parameters for lbph computation
LBP_MAX_DISTANCE = 20 # Maximum expected chi square distance

# CONFIDENCE WEIGHTS
LM_WEIGHT = 0.50
DF_WEIGHT = 0.395
LBPH_WEIGHT = 0.10
CONFIDENCE_THRESHOLD = 65 # Threshold for deepfake confidence (out of 100)