
# IdentifAI Recruiting Challenge - William Bui

Objective: Develop a FastAPI application that allows users to upload an image and generates a "facial profile" that describes key characteristics of the face which define its reality.

## Getting Started

### 1. Clone this repo
```sh
git clone https://github.com/wtbui/recruiting-challenge.git
```

### 2. Install dependencies
```sh
pip install -r requirements.txt
```

### 3. Run FastAPI server
**Note**: MUST run from project directory with `app.` before `main:app`
```sh
uvicorn app.main:app --reload
```

### 4. Try out the API!
**Note**: Data is not persistent among calls, will need database integration
```sh
http://localhost:8000/docs
```

## Testing
Tests are located within the `tests` directory and can be run with:

Running Tests:
```
python3 tests/test_all.py

Testing create and get profile
Testing retrieval non-existent profile
Testing delete profile
Testing deletion of non-existent profile
Testing verification of different photos of same person
Image is not deepfaked with confidence of 84.49363453276332
Testing verification of deepfaked photo
Image is deepfaked with confidence of 56.73082372275643
Testing verification on non-existent profile
All tests passed!
```

### Current Tests Include:
1. Creation, Retrieval, and Deletion of existing/non-existing profiles
2. Profile verification with photo of same person
3. Profile verification with photo of deepfaked (same) person
4. Profile verification with nonexistent profile

## Design Decisions

### 1. Project Structure
The current project structure separates Pydantic models within the `models` subdirectory, API endpoints within the `routers` subdirectory, and facial feature extraction/analysis logic within the `utils` subdirectory.

- **Models**: Contains all Pydantic models
- **Routers**: Contains routers to endpoints
- **Utils**: Contains logic for facial extraction/analysis as well as tunable constants for each logic module

This project structure supports the expansion of the API beyond profile generation/analysis.

### 2. Endpoints
The current iteration of this project includes four endpoints:
- `/profile/create`: To create profiles using images
- `/profile/delete`: To delete profiles
- `/profile/get`: To retrieve profiles
- `/profile/verify`: To verify an existing profile with a new image

### 3. Facial Detection Logic
The logic behind facial feature extraction and detection lies within three modules in the `utils` subdirectory. These modules generate their own similarity confidence values between different profiles, which can then be aggregated to generate an overall confidence level. **Note**: Photos are preprocessed for consistency among comparisons and calculations.

#### a) `landmark_analysis.py` - Landmark Module

i. This module generates landmark points with the dlib library's facial detection/shape predictor and calculates Euclidean distances between key points to generate a geometric profile of the face.

ii. To calculate its similarity confidence value, the module takes differences between the Euclidean distances.

iii. The current iteration uses 200 for the maximum difference between profiles and the "68 face landmarks GTX" for its dlib model, both of which can be tuned within `utils/analysis_params.py`.

#### b) `deep_analysis.py` - Deep Feature Module

i. This module extracts deep facial features using the FaceNet model, specifically the InceptionResnetV1 architecture pre-trained on the VGGFace2 dataset. The process involves preprocessing the input image (converting it to RGB, resizing, and normalizing), then passing it through the FaceNet model to generate a set of deep features or embeddings that represent the face.

ii. To calculate the similarity confidence value between two sets of facial embeddings, the module uses cosine similarity. This metric measures the cosine of the angle between two non-zero vectors, effectively comparing the directionality of the embeddings. The confidence score is derived by normalizing the cosine similarity value to a percentage scale.

#### c) `lbph_analysis.py` - Local Binary Pattern Histogram (LBPH) Module

i. This module generates texture-based features using Local Binary Patterns (LBP) from the input image. The process involves converting the image to grayscale, resizing, and applying histogram equalization. The LBP is then computed for the image, and a normalized histogram is generated to represent the texture features of the face.

ii. To calculate the similarity confidence value between two LBP histograms, the module uses the chi-square distance metric. This metric measures the difference between two histograms, with a lower chi-square distance indicating higher similarity. The confidence score is derived by normalizing the chi-square distance to a percentage scale.

iii. The current iteration uses the parameters specified in `utils/analysis_params.py` for the number of circularly symmetric neighbor set points (`LBP_TEXTURE_LEVELS`) and the radius of the circle. The maximum allowable chi-square distance (`LBP_MAX_DISTANCE`) is also defined in the same file. These parameters can be tuned for different datasets or applications within the `lbph_analysis.py` module.

#### d) `profile.py` - Confidence Level

i. Using each module's separate confidence score, an aggregate overall confidence level can be calculated using a weighted sum.

ii. The weights currently favor the landmark module, which shows the highest chance of predicting a real image, but these can be tuned within `utils/analysis_params.py`.

### Closing Remarks
In its current form, the api does not have database integration which is limiting, and can only currently analyze images with a singlular face. In future iterations, changes could be made to improve these areas.
