import os
import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import mediapipe as mp
from augment_utils import augment_image

# === CONFIGURATION ===
IMG_SIZE = 48
DATA_PATH = 'dataset/mouth'
LABELS = ['yawn', 'no_yawn']
N_COMPONENTS = 100  # PCA output dimensions

# === INIT MEDIAPIPE ===
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

# === Mouth landmarks from MediaPipe mesh ===
MOUTH_IDX = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324]

def extract_mouth(img):
    """Returns cropped mouth from image or None if detection fails"""
    h, w, _ = img.shape
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark
        x_vals = [int(lm[i].x * w) for i in MOUTH_IDX]
        y_vals = [int(lm[i].y * h) for i in MOUTH_IDX]
        x_min, y_min = max(min(x_vals)-20, 0), max(min(y_vals)-20, 0)
        x_max, y_max = min(max(x_vals)+20, w), min(max(y_vals)+20, h)
        return img[y_min:y_max, x_min:x_max]
    return None

X, y = [], []

print("📦 Loading and augmenting mouth dataset...")

# === LOAD IMAGES, CROP, AUGMENT ===
for label in LABELS:
    folder = os.path.join(DATA_PATH, label)
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        img = cv2.imread(path)
        if img is None:
            continue
        mouth = extract_mouth(img)
        if mouth is None:
            continue
        gray = cv2.cvtColor(mouth, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
        augmented = augment_image(resized)
        label_val = 1 if label == 'yawn' else 0

        for aug in augmented:
            X.append(aug.flatten())
            y.append(label_val)

X, y = np.array(X), np.array(y)

# === SPLIT AND PCA ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
pca = PCA(n_components=N_COMPONENTS)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# === TRAIN LOGISTIC REGRESSION ===
print("⏳ Training mouth Logistic Regression...")
model = LogisticRegression(solver='saga', max_iter=500, verbose=1)
model.fit(X_train, y_train)

# === EVALUATE AND SAVE ===
y_pred = model.predict(X_test)
print(f"✅ Mouth Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")
joblib.dump(model, 'yawn_model.pkl')
joblib.dump(pca, 'yawn_pca.pkl')
print("💾 Saved: yawn_model.pkl + yawn_pca.pkl")
