# SleepPolice App
### AI-Powered Drowsiness Detection with MediaPipe, OpenCV, and Scikit-learn

This project uses a webcam to detect drowsiness by classifying eye closure and yawning behavior, with future integration for Arduino-based servo feedback.

---

## Features

- Drowsiness detection through eye and mouth classification
- PCA + Logistic Regression for efficient mouth state prediction
- Flicker suppression via rolling window state smoothing
- Dataset augmentation and facial landmark cropping
- Real-time detection with webcam
- Offline AI
- venv + run.sh launcher

---

## Project Structure
  
├── face_detect.py            # Main real-time detection script. 
├── train_eye_model.py        # Trains open/closed eye classifier. 
├── train_mouth_model.py      # Trains yawn/no-yawn classifier w/ PCA. 
├── augment_utils.py          # Image augmentation for training. 
├── eye_state_model.pkl       # Saved SVM model for eye state. 
├── yawn_model.pkl            # Saved Logistic Regression model. 
├── yawn_pca.pkl              # Saved PCA model for mouth feature reduction. 
├── run.sh                    # Activates venv and runs detection. 
├── requirements.txt          # Python dependencies. 
├── venv/                     # Virtual environment. 
└── dataset/                  # Training images (Open/Closed, Yawn/No Yawn)   
  
---

## Setup Instructions (macOS/Linux)

1. Clone or download the project

    cd face_tracker

2. Create a Python virtual environment

    python3 -m venv venv

3. Activate Virtual Environment (ex. "(venv) txt-102@TXT-60 ...")

    source venv/bin/activate

4. Install dependencies

    pip install -r requirements.txt

---

## Run the App

Option A: Using the launcher script

    ./run.sh

Option B: Manual run

    source venv/bin/activate
    python face_detect.py

Press `q` while the webcam window is open to quit.

When done, remember to deactivate the VM with `Deactivate` in terminal.

---

## Requirements

- Python 3.8 or newer
- Python 3.11.9 LATEST (for mediapipe compatibility)
- macOS, Linux, or Windows (adjust serial port names on Windows)
- Webcam

---

## Troubleshooting

Problem: `ModuleNotFoundError: cv2`  
Fix: Run `pip install -r requirements.txt` again

Problem: Camera not detected  
Fix: Try `cv2.VideoCapture(1)` in face_detect.py

Problem: Permission denied to camera  
Fix: On macOS, go to System Settings → Privacy & Security → Camera → Allow Terminal

Problem: "externally-managed-environment" error  
Fix: Use virtual environment as described above

Problem: X has 2304 features, but LogisticRegression expects 100
Fix: PCA must be loaded and used before yawn prediction

Problem: mediapipe cannot be installed
Fix: Use Python 3.11.9 — newer versions may break compatibility

---

## Installing on a New Machine

If you move the project to a new computer:

    cd face_tracker
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ./run.sh

---

## Coming Next

- Integrate alert logic (soft warnings + loud “WAKE UP”)
- Add sound playback support (beep or mp3)
- Raspberry Pi deployment in vehicle
- Servo-animated police figure reacting to drowsiness state

---

## Credits

- MediaPipe: https://mediapipe.dev/
- OpenCV: https://opencv.org/
