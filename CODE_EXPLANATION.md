# Codebase Explanation - Rock Paper Scissors Hand Gesture Recognition

## Project Overview
This is a desktop application that combines face detection, hand gesture recognition, and a Rock-Paper-Scissors game. The app uses computer vision to detect hand gestures in real-time through a webcam and plays RPS against the user.

## Main Components

### 1. **app.py** (Main Application - 908 lines)
- **Purpose**: Desktop GUI application built with Tkinter
- **Key Features**:
  - Real-time face detection using OpenCV Haar Cascades
  - Hand gesture recognition (rock, paper, scissors)
  - Rock-Paper-Scissors game mode
  - Optional ML model support (PyTorch MobileNetV2)
  - OpenAI integration for AI analysis (optional)

- **How it works**:
  - Uses OpenCV for webcam capture and image processing
  - Face detection: OpenCV's `haarcascade_frontalface_default.xml`
  - Hand detection: Two methods available:
    1. **OpenCV-based** (default): Uses skin color detection, contour analysis, convexity defects, and finger counting
    2. **ML model** (if trained): Uses PyTorch MobileNetV2 for more accurate gesture recognition
  - Game logic: Detects stable hand gestures (must be consistent for 5 frames), then plays RPS
  - GUI: Tkinter-based interface showing video feed, detection results, and game scores

- **Key Functions**:
  - `detect_hand_simple()`: OpenCV-based hand detection using skin color and contour analysis
  - `classify_gesture()`: Analyzes hand contours to determine rock/paper/scissors
  - `load_ml_model()`: Loads trained PyTorch model if available
  - `update_video()`: Main video processing loop
  - `determine_winner()`: RPS game logic

### 2. **train_model.py** (Model Training Script)
- **Purpose**: Trains a MobileNetV2 model for gesture recognition
- **Process**:
  - Loads images from `data/rock/`, `data/paper/`, `data/scissors/` folders
  - Uses transfer learning with ImageNet pretrained weights
  - Applies data augmentation (flips, color jitter, normalization)
  - Trains for 10-20 epochs (adjusts based on dataset size)
  - Saves model to `rps_model.pth` and class names to `class_names.json`

### 3. **download_training_images.py** (Web Image Downloader)
- **Purpose**: Automatically downloads training images from the web
- **Features**:
  - Uses DuckDuckGo image search (no API key needed)
  - Downloads 200 images per class (rock, paper, scissors)
  - Specific search queries for hand gestures:
    - Rock: "rock hand gesture fingers closed fist", "rock hand gesture closed fist"
    - Paper: "paper hand gesture palm open flat", "paper hand gesture open palm"
    - Scissors: "scissors hand gesture two fingers", "scissors hand gesture peace sign"
  - Image validation (filters small images, extreme aspect ratios)
  - Handles rate limiting automatically

### 4. **collect_data.py** (Manual Data Collection)
- **Purpose**: Manual webcam-based data collection
- **Usage**: User manually captures images by pressing SPACE key
- **Output**: Saves images to `data/{class_name}/` folders

### 5. **collect_data_auto.py** (Automatic Data Collection)
- **Purpose**: Automatic continuous image capture
- **Usage**: Press SPACE to start/stop auto-capture (captures every 0.3 seconds)

## File Structure
```
fcdtec/
├── app.py                      # Main desktop application
├── train_model.py              # Model training script
├── download_training_images.py # Web image downloader
├── collect_data.py             # Manual data collection
├── collect_data_auto.py        # Automatic data collection
├── data/                       # Training images
│   ├── rock/
│   ├── paper/
│   └── scissors/
├── rps_model.pth              # Trained model (generated)
├── class_names.json           # Class names (generated)
├── requirements.txt           # Python dependencies
└── README_TRAINING.md         # Training instructions
```

## Technologies Used
- **OpenCV**: Computer vision, face/hand detection, image processing
- **PyTorch**: Deep learning framework for gesture recognition
- **Tkinter**: Desktop GUI
- **PIL/Pillow**: Image processing
- **NumPy**: Numerical operations
- **DuckDuckGo Search**: Web image search
- **OpenAI API**: Optional AI analysis features

## Workflow

### Training Workflow:
1. Collect/download training images (200+ per class)
2. Run `train_model.py` to train MobileNetV2 model
3. Model saved as `rps_model.pth`
4. App automatically loads model if available

### Runtime Workflow:
1. User starts webcam
2. App detects faces and hands in real-time
3. Hand gestures classified as rock/paper/scissors
4. In game mode: stable gestures trigger RPS rounds
5. Scores tracked and displayed

## Key Algorithms

### Hand Gesture Classification (OpenCV method):
1. **Skin color detection**: HSV color space filtering
2. **Contour analysis**: Find hand contours
3. **Convexity defects**: Detect spaces between fingers
4. **Finger counting**: Count fingertips using local maxima
5. **Shape analysis**: Solidity, aspect ratio, area
6. **Decision logic**:
   - Rock: High solidity (>0.85) + few fingers (≤1)
   - Paper: Low solidity (<0.75) + many fingers (≥4)
   - Scissors: Medium solidity + 2-3 fingers

### ML Model (if trained):
- Uses MobileNetV2 (lightweight CNN)
- Transfer learning from ImageNet
- 3-class classification (rock, paper, scissors)
- Input: 224x224 RGB images
- Output: Class probabilities

## Game Logic
- Gesture must be stable for 5 consecutive frames
- Cooldown period between rounds
- AI randomly chooses rock/paper/scissors
- Winner determined by standard RPS rules
- Scores tracked (player vs AI)

## Dependencies
- opencv-python
- numpy
- torch, torchvision
- Pillow
- duckduckgo-search
- requests
- openai (optional)
- tkinter (usually included with Python)


