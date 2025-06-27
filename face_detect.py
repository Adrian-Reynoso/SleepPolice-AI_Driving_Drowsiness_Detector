import cv2
import joblib
import numpy as np
import mediapipe as mp

# Load the trained models from disk
eye_model = joblib.load('eye_state_model.pkl')
yawn_model = joblib.load('yawn_model.pkl')

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Landmark index ranges for eyes and mouth
LEFT_EYE_IDX = [33, 133]
RIGHT_EYE_IDX = [362, 263]
MOUTH_IDX = [78, 308]

IMG_SIZE = 48  # Must match the size used during training

# Function to predict eye state from a given image
def classify_eye(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE)).flatten().reshape(1, -1)
    pred = eye_model.predict(resized)[0]  # Predict using loaded model
    return "Open" if pred == 1 else "Closed"

# Function to predict yawn state
def classify_yawn(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE)).flatten().reshape(1, -1)
    pred = yawn_model.predict(resized)[0]
    return "Yawning" if pred == 1 else "Not Yawning"

# Start video stream
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    h, w, _ = frame.shape
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    eye_status = "Unknown"
    yawn_status = "Unknown"

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Helper function to convert landmark to pixel coordinates
            def pt(idx):
                lm = face_landmarks.landmark[idx]
                return int(lm.x * w), int(lm.y * h)

            # Extract eye and mouth bounding boxes
            lx1, ly1 = pt(LEFT_EYE_IDX[0])
            lx2, ly2 = pt(LEFT_EYE_IDX[1])
            rx1, ry1 = pt(RIGHT_EYE_IDX[0])
            rx2, ry2 = pt(RIGHT_EYE_IDX[1])
            mx1, my1 = pt(MOUTH_IDX[0])
            mx2, my2 = pt(MOUTH_IDX[1])

            # Crop regions of interest
            eye_roi = frame[min(ly1, ly2)-10:max(ly1, ly2)+10, min(lx1, lx2)-10:max(lx1, lx2)+10]
            mouth_roi = frame[min(my1, my2)-15:max(my1, my2)+15, min(mx1, mx2)-15:max(mx1, mx2)+15]

            # Predict using trained models
            if eye_roi.size:
                eye_status = classify_eye(eye_roi)
            if mouth_roi.size:
                yawn_status = classify_yawn(mouth_roi)

            break  # We only need the first face

    # Show prediction on screen
    cv2.putText(frame, f"Eyes: {eye_status}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(frame, f"Yawn: {yawn_status}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("AI Monitor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
