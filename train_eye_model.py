import os
import cv2
import numpy as np
from sklearn.svm import SVC  # SVM classifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib  # Used to save/load model

# Define the data directory and class labels
data_path = 'dataset/eyes'
labels = ['Open', 'Closed']

IMG_SIZE = 48  # We'll resize all images to 48x48 pixels

# Lists to hold image data and corresponding labels
X = []
y = []

# Loop through each class folder ('Open', 'Closed')
for label in labels:
    label_path = os.path.join(data_path, label)
    for file in os.listdir(label_path):
        img_path = os.path.join(label_path, file)

        # Read image in grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue  # skip unreadable files

        # Resize image and flatten it into a 1D vector
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        X.append(img.flatten())
        y.append(label)  # Store string label

# Convert string labels to numeric (0 = Closed, 1 = Open)
y = np.array([0 if label == 'Closed' else 1 for label in y])
X = np.array(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create an SVM classifier with a linear kernel
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)  # Train the model

# Evaluate accuracy
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.2f}")

# Save model to disk
joblib.dump(model, 'eye_state_model.pkl')
