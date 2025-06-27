#!/bin/bash

# Activate the virtual environment
source venv311/bin/activate

echo "✅ Virtual environment activated."

# Install dependencies if needed
echo "📦 Installing required packages..."
pip install --upgrade pip
pip install -r requirements.txt

# Prompt user to (re)train models
echo ""
read -p "Do you want to re-train the models? (y/n): " choice
if [[ "$choice" == "y" ]]; then
  echo "🎯 Training eye model..."
  python train_eye_model.py
  echo "🎯 Training mouth model..."
  python train_mouth_model.py
  echo "✅ Models trained and saved."
else
  echo "⏩ Skipping training. Using existing models."
fi

# Launch real-time detection
echo "🚀 Launching live AI detection..."
python face_detect.py
