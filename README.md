# Real-Time Face Tracking with Python (MediaPipe + OpenCV)

This project uses a webcam to detect faces in real-time using MediaPipe and OpenCV, and is designed to integrate with servo-powered animatronic eyes (via Arduino in future stages).

---

## Features

- Real-time face detection using your computer's webcam
- Lightweight AI with no internet required after setup
- Designed to integrate with Arduino + servo control
- Fully isolated Python environment via venv
- Easy one-command launch using run.sh

---

## Project Structure

face_tracker/
├── face_detect.py        # Main Python script
├── run.sh                # Script to activate venv and run detection
├── requirements.txt      # Python dependencies
└── venv/                 # Virtual environment (created locally)

---

## Setup Instructions (macOS/Linux)

1. Clone or download the project

    cd face_tracker

2. Create a Python virtual environment

    python3 -m venv venv
    source venv/bin/activate

3. Install dependencies

    pip install -r requirements.txt

---

## Run the App

Option A: Using the launcher script

    ./run.sh

Option B: Manual run

    source venv/bin/activate
    python face_detect.py

Press `q` while the webcam window is open to quit.

---

## Requirements

- Python 3.8 or newer
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

- Servo tracking via Arduino
- Smoothing movement for animatronics
- Full Raspberry Pi integration
- Face-following robot platform

---

## Credits

- MediaPipe: https://mediapipe.dev/
- OpenCV: https://opencv.org/
# face_tracker
