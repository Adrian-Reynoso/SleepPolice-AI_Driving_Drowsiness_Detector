import os
import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib
from augment_utils import augment_image

# === CONFIGURATION ===
IMG_SIZE = 48  # All images are resized to 48x48
DATA_PATH = 'dataset/eyes'  # Directory for eye dataset
LABELS = ['Open', 'Closed']  # Two classes

X = []  # Feature list
y = []  # Label list

print("📦 Loading and augmenting eye dataset...")

# === LOAD IMAGES AND APPLY AUGMENTATION ===
for label in LABELS:
    folder = os.path.join(DATA_PATH, label)
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        augmented = augment_image(img)  # Apply flips, noise, etc.
        label_val = 1 if label == 'Open' else 0  # Convert label to int

        for aug in augmented:
            X.append(aug.flatten())
            y.append(label_val)

X, y = np.array(X), np.array(y)

# === TRAIN/TEST SPLIT ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# === TRAIN LOGISTIC REGRESSION ===
print("⏳ Training eye Logistic Regression...")
model = LogisticRegression(solver='saga', max_iter=500, verbose=1)
model.fit(X_train, y_train)

# === EVALUATE ===
y_pred = model.predict(X_test)
print(f"✅ Eye Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# === SAVE ===
joblib.dump(model, 'eye_state_model.pkl')
print("💾 Saved: eye_state_model.pkl")
