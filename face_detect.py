import cv2                # OpenCV for video and image processing
import joblib             # For loading saved machine learning models
import numpy as np        # Useful for image manipulation
import mediapipe as mp    # MediaPipe for face landmarks
from collections import deque # For landmark detection buffer
from statistics import mode

# Create a rolling window for the last N predictions
EYE_BUFFER = deque(maxlen=5)     # Try 5–10 for low flicker
MOUTH_BUFFER = deque(maxlen=5)

# Load the trained SVM models for eye state and yawn detection
eye_model = joblib.load('eye_state_model.pkl')
yawn_model = joblib.load('yawn_model.pkl')

# Initialize the MediaPipe Face Mesh model (detects 468 facial landmarks)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,  # Real-time detection, not single-image
    max_num_faces=1           # We only care about the first face in frame
)

# Define landmark indices around left eye, right eye, and mouth
# These are carefully chosen for tight coverage of these regions
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144, 163, 7]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380, 390, 249]
MOUTH_IDX = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324]

# All input images must be resized to this size before being passed to the models
IMG_SIZE = 48

def classify_eye(roi):
    """Classifies a region of interest (ROI) as an open or closed eye."""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)               # Convert to grayscale
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))           # Resize to match training input
    flat = resized.flatten().reshape(1, -1)                    # Flatten for the model
    pred = eye_model.predict(flat)[0]                          # Predict with trained SVM
    return "Open" if pred == 1 else "Closed"                   # Convert prediction to label

def classify_yawn(roi):
    """Classifies a region of interest (ROI) as yawning or not yawning."""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    flat = resized.flatten().reshape(1, -1)
    pred = yawn_model.predict(flat)[0]
    return "Yawning" if pred == 1 else "Not Yawning"

# def get_bounding_box(landmarks, indices, frame_shape):
#     """
#     Given facial landmarks and a list of indices, return the bounding box
#     (x_min, y_min, x_max, y_max) that encloses all points.
#     """
#     h, w, _ = frame_shape  # Get image dimensions
#     x_vals, y_vals = [], []

#     # Convert normalized landmark coordinates to pixel coordinates
#     for idx in indices:
#         x = int(landmarks[idx].x * w)
#         y = int(landmarks[idx].y * h)
#         x_vals.append(x)
#         y_vals.append(y)

#     # Add a bit of padding so the box isn't too tight
#     padding = 5
#     x_min = max(min(x_vals) - padding, 0)
#     y_min = max(min(y_vals) - padding, 0)
#     x_max = min(max(x_vals) + padding, w)
#     y_max = min(max(y_vals) + padding, h)

#     return x_min, y_min, x_max, y_max

def get_bounding_box(landmarks, indices, frame_shape, padding=5):
    """Compute a padded bounding box for a group of facial landmarks."""
    h, w, _ = frame_shape
    x_vals, y_vals = [], []

    for idx in indices:
        x = int(landmarks[idx].x * w)
        y = int(landmarks[idx].y * h)
        x_vals.append(x)
        y_vals.append(y)

    # Add customizable padding for looser box
    x_min = max(min(x_vals) - padding, 0)
    y_min = max(min(y_vals) - padding, 0)
    x_max = min(max(x_vals) + padding, w)
    y_max = min(max(y_vals) + padding, h)

    return x_min, y_min, x_max, y_max

# Start webcam video capture
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()  # Read one frame
    if not success:
        break  # If camera fails, exit

    h, w, _ = frame.shape  # Store frame size
    # Convert to RGB (MediaPipe expects RGB images)
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Default status if detection fails
    eye_status = "Unknown"
    yawn_status = "Unknown"

    # If face was detected by MediaPipe
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get bounding boxes for left eye, right eye, and mouth
            left_eye_box = get_bounding_box(face_landmarks.landmark, LEFT_EYE_IDX, frame.shape, padding=15)
            right_eye_box = get_bounding_box(face_landmarks.landmark, RIGHT_EYE_IDX, frame.shape, padding=15)
            mouth_box = get_bounding_box(face_landmarks.landmark, MOUTH_IDX, frame.shape, padding=30)


            # Crop the regions of interest from the frame using the bounding boxes
            eye_roi = frame[left_eye_box[1]:left_eye_box[3], left_eye_box[0]:left_eye_box[2]]
            mouth_roi = frame[mouth_box[1]:mouth_box[3], mouth_box[0]:mouth_box[2]]

            # Append new predictions to buffer
            if eye_roi.size:
                pred_eye = classify_eye(eye_roi)
                EYE_BUFFER.append(pred_eye)
                # Use the most common prediction in buffer
                eye_status = mode(EYE_BUFFER)

            if mouth_roi.size:
                pred_mouth = classify_yawn(mouth_roi)
                MOUTH_BUFFER.append(pred_mouth)
                yawn_status = mode(MOUTH_BUFFER)

            # Draw bounding boxes around the regions
            cv2.rectangle(frame, (left_eye_box[0], left_eye_box[1]), (left_eye_box[2], left_eye_box[3]), (0, 255, 0), 2)
            cv2.rectangle(frame, (right_eye_box[0], right_eye_box[1]), (right_eye_box[2], right_eye_box[3]), (0, 255, 0), 2)
            cv2.rectangle(frame, (mouth_box[0], mouth_box[1]), (mouth_box[2], mouth_box[3]), (0, 0, 255), 2)

            break  # Only process the first face detected

    # Show the eye and yawn prediction status as text
    cv2.putText(frame, f"Eyes: {eye_status}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(frame, f"Yawn: {yawn_status}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Show the final video output
    cv2.imshow("AI Monitor", frame)

    # Exit loop if user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up camera and window when done
cap.release()
cv2.destroyAllWindows()
