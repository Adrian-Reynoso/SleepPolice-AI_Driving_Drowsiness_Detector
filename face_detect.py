import cv2
import numpy as np
import mediapipe as mp
import joblib
from collections import deque
from statistics import mode
import time  # Make sure this is imported at the top if not already

# === LOAD MODELS ===
eye_model = joblib.load('eye_state_model.pkl')
yawn_model = joblib.load('yawn_model.pkl')
pca_yawn = joblib.load('yawn_pca.pkl')

# === CONFIGURATION ===
IMG_SIZE = 48                  # Input image size
EYE_PAD = 15                   # Buffer around eyes
MOUTH_PAD = 25                 # Buffer around mouth
BUFFER_SIZE = 10               # Number of frames to smooth predictions over

# === Initialize Smoothing Buffers ===
eye_buffer = deque(maxlen=BUFFER_SIZE)    # Holds last N eye predictions
yawn_buffer = deque(maxlen=BUFFER_SIZE)   # Holds last N yawn predictions

# === MEDIAPIPE SETUP ===
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# === FACIAL LANDMARKS ===
LEFT_EYE_IDX = [33, 160, 158, 133]
RIGHT_EYE_IDX = [362, 385, 387, 263]
MOUTH_IDX = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324]

def get_box(landmarks, indices, shape, pad=10):
    """
    Calculate a padded bounding box around specific landmark indices.
    """
    h, w, _ = shape
    x_vals = [int(landmarks[i].x * w) for i in indices]
    y_vals = [int(landmarks[i].y * h) for i in indices]
    return (
        max(min(x_vals) - pad, 0),
        max(min(y_vals) - pad, 0),
        min(max(x_vals) + pad, w),
        min(max(y_vals) + pad, h)
    )

def classify_eye(roi):
    """
    Predict eye state (Open/Closed) from a cropped eye ROI.
    """
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE)).flatten().reshape(1, -1)
    pred = eye_model.predict(resized)[0]
    return "Open" if pred == 1 else "Closed"

def classify_yawn(roi, pca):
    """
    Predict yawn state from a cropped mouth ROI, using PCA-reduced input.
    """
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE)).flatten().reshape(1, -1)
    reduced = pca.transform(resized)
    pred = yawn_model.predict(reduced)[0]
    return "Yawning" if pred == 1 else "Not Yawning"

# Timers to track how long the mouth is open or eyes are closed
mouth_open_start = None
eyes_closed_start = None

# Counters for total warnings issued
yawn_warnings = 0
eye_warnings = 0

# Track last time a warning was shown and what it was
last_warning_time = 0
last_warning_text = ""
total_strikes = 0

# === START VIDEO CAPTURE ===
cap = cv2.VideoCapture(0)
print("🚀 Launching live detection... Press 'q' to quit.")

while True:
    success, frame = cap.read()
    if not success:
        break

    h, w, _ = frame.shape

    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # Get region boxes with custom padding
        lx1, ly1, lx2, ly2 = get_box(landmarks, LEFT_EYE_IDX, frame.shape, pad=EYE_PAD)
        rx1, ry1, rx2, ry2 = get_box(landmarks, RIGHT_EYE_IDX, frame.shape, pad=EYE_PAD)
        mx1, my1, mx2, my2 = get_box(landmarks, MOUTH_IDX, frame.shape, pad=MOUTH_PAD)

        # Extract eye and mouth ROIs
        left_eye_roi = frame[ly1:ly2, lx1:lx2]
        right_eye_roi = frame[ry1:ry2, rx1:rx2]
        mouth_roi = frame[my1:my2, mx1:mx2]

        # Classify only if ROI is valid size
        if left_eye_roi.size and right_eye_roi.size:
            # Optionally average both eyes
            eye_status = classify_eye(left_eye_roi)
            eye_buffer.append(eye_status)

        if mouth_roi.size:
            yawn_status = classify_yawn(mouth_roi, pca_yawn)
            yawn_buffer.append(yawn_status)

        current_time = time.time()

        # === YAWN LOGIC ===
        if yawn_status == "Yawning":
            if mouth_open_start is None:
                mouth_open_start = current_time
            elif current_time - mouth_open_start >= 2.5:
                # Yawn detected for over 2.5 seconds
                if current_time - last_warning_time > 3:
                    last_warning_text = "Stop Yawning"
                    last_warning_time = current_time
                    yawn_warnings += 1
                    total_strikes += 1
        else:
            mouth_open_start = None

        # === EYE CLOSURE LOGIC ===
        if eye_status == "Closed":
            if eyes_closed_start is None:
                eyes_closed_start = current_time
            elif current_time - eyes_closed_start >= 2:
                if current_time - last_warning_time > 3:
                    last_warning_text = "Open Your Eyes!"
                    last_warning_time = current_time
                    eye_warnings += 1
                    total_strikes += 1
        else:
            eyes_closed_start = None

        # === DISPLAY WARNINGS ON SCREEN ===
        if total_strikes >= 3:
            # Flash full screen red warning
            if current_time - last_warning_time < 5:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), -1)
                alpha = 0.6
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                cv2.putText(frame, "WAKE UP", (int(w * 0.25), int(h / 2)), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 5)
            else:
                total_strikes = 0
        else:
            if current_time - last_warning_time < 3 and last_warning_text:
                cv2.putText(frame, last_warning_text, (int(w * 0.1), int(h * 0.9)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)


        # Use most common value in buffer (smoothing)
        stable_eye = mode(eye_buffer) if eye_buffer else "Unknown"
        stable_yawn = mode(yawn_buffer) if yawn_buffer else "Unknown"

        # Draw rectangles
        cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), (0, 255, 0), 2)  # Left eye
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)  # Right eye
        cv2.rectangle(frame, (mx1, my1), (mx2, my2), (0, 0, 255), 2)  # Mouth

        # Overlay predictions
        cv2.putText(frame, f"Eyes: {stable_eye}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(frame, f"Yawn: {stable_yawn}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

    cv2.imshow("AI Drowsiness Monitor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === CLEANUP ===
cap.release()
cv2.destroyAllWindows()
