#!/bin/bash

# Activate the virtual environment
source venv311/bin/activate

echo "✅ Virtual environment activated."

# Install dependencies if needed
echo "📦 Installing required packages..."
pip install --upgrade pip
pip install -r requirements.txt

# Prompt user to (re)train models
echo "💡 Do you want to re-train the models? (y/n): "
read retrain

if [ "$retrain" = "y" ]; then
    echo "🎯 Training eye model..."
    ./venv311/bin/python train_eye_model.py
    echo "🎯 Training mouth model..."
    ./venv311/bin/python train_mouth_model.py
    echo "✅ Models trained and saved."
fi

echo "🚀 Launching live AI detection..."
./venv311/bin/python face_detect.py

