import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib
import mediapipe as mp

# === CONFIG ===
IMG_SIZE = 48  # Resize all cropped mouths to 48x48
DATA_PATH = 'dataset/mouth'
LABELS = ['yawn', 'no_yawn']

# === SETUP MEDIAPIPE FACE MESH ===
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,  # Analyze images one at a time
    max_num_faces=1          # Only care about one face per image
)

# Landmark indices around the mouth based on MediaPipe's 468-point face mesh
MOUTH_IDX = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324]

def extract_mouth(img):
    """
    Detects facial landmarks and extracts the mouth region as a cropped image.
    Returns None if no face/mouth is detected.
    """
    h, w, _ = img.shape
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        # Use the first face detected
        landmarks = results.multi_face_landmarks[0].landmark

        # Convert normalized landmark coords to pixel values
        x_vals = [int(landmarks[i].x * w) for i in MOUTH_IDX]
        y_vals = [int(landmarks[i].y * h) for i in MOUTH_IDX]

        # Get bounding box with padding
        x_min = max(min(x_vals) - 20, 0)
        y_min = max(min(y_vals) - 20, 0)
        x_max = min(max(x_vals) + 20, w)
        y_max = min(max(y_vals) + 20, h)

        mouth_crop = img[y_min:y_max, x_min:x_max]
        return mouth_crop if mouth_crop.size > 0 else None

    return None

# === LOAD AND PROCESS DATASET ===
X = []  # Image data
y = []  # Labels (0 = no_yawn, 1 = yawn)

print("🧠 Preprocessing dataset with mouth detection...")

# Iterate through each category folder
for label in LABELS:
    label_path = os.path.join(DATA_PATH, label)
    for file in os.listdir(label_path):
        img_path = os.path.join(label_path, file)

        # Load image in color so we can run face detection
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Crop the mouth from the image
        mouth = extract_mouth(img)
        if mouth is None:
            continue  # Skip if mouth can't be found

        # Convert mouth region to grayscale
        gray = cv2.cvtColor(mouth, cv2.COLOR_BGR2GRAY)

        # Resize to 48x48, flatten to 1D array
        resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
        X.append(resized.flatten())
        y.append(1 if label == 'yawn' else 0)

# Convert to numpy arrays for ML model training
X = np.array(X)
y = np.array(y)

# === SPLIT DATA ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === TRAIN MODEL ===
print("🎯 Training Support Vector Machine on cropped mouth images...")
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

# === EVALUATE MODEL ===
y_pred = model.predict(X_test)
print(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# === SAVE MODEL ===
joblib.dump(model, 'yawn_model.pkl')
print("💾 Trained model saved as yawn_model.pkl")

