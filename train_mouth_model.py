import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib

# Folder structure and label names
data_path = 'dataset/mouth'
labels = ['yawn', 'no_yawn']

IMG_SIZE = 48  # Resize images to 48x48

X = []
y = []

# Load all images from each label folder
for label in labels:
    label_path = os.path.join(data_path, label)
    for file in os.listdir(label_path):
        img_path = os.path.join(label_path, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        X.append(img.flatten())
        y.append(label)

# Convert labels to numbers (1 = yawn, 0 = no_yawn)
y = np.array([1 if label == 'yawn' else 0 for label in y])
X = np.array(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the SVM
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Save the trained model
joblib.dump(model, 'yawn_model.pkl')
