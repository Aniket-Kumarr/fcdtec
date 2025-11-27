#!/usr/bin/env python3
"""
Face Detection AI with Hand Detection and Rock-Paper-Scissors
A standalone desktop app for face detection, hand detection, and RPS game
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import os
import random
import json
from openai import OpenAI

# Try to import PyTorch for ML model
try:
    import torch
    from torchvision import transforms, models
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Using OpenCV-based detection.")

class FaceHandDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face & Hand Detection AI - Rock Paper Scissors")
        self.root.geometry("1200x800")
        
        # Initialize variables
        self.cap = None
        self.is_webcam_active = False
        self.current_frame = None
        self.detected_faces = []
        self.face_cascade = None
        self.openai_client = None
        
        # Rock-Paper-Scissors game state
        self.game_mode = False
        self.player_choice = None
        self.ai_choice = None
        self.player_score = 0
        self.ai_score = 0
        self.round_count = 0
        
        # Hand detection variables
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        self.hand_roi = None
        
        # ML Model for gesture recognition
        self.ml_model = None
        self.ml_classes = ["rock", "paper", "scissors"]
        self.use_ml_model = False
        self.ml_preprocess = None
        
        # Load face detection model
        self.load_face_cascade()
        
        # Try to load ML model
        self.load_ml_model()
        
        # Setup OpenAI client if API key is available
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            self.openai_client = OpenAI(api_key=api_key)
        
        # Create GUI
        self.create_widgets()
        
    def load_face_cascade(self):
        """Load OpenCV's face detection cascade"""
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if self.face_cascade.empty():
                raise Exception("Failed to load cascade")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load face detection model: {e}")
            self.root.quit()
    
    def load_ml_model(self):
        """Load trained PyTorch model for gesture recognition"""
        if not TORCH_AVAILABLE:
            return
        
        model_path = 'rps_model.pth'
        class_names_path = 'class_names.json'
        
        if os.path.exists(model_path) and os.path.exists(class_names_path):
            try:
                # Load class names
                with open(class_names_path, 'r') as f:
                    self.ml_classes = json.load(f)
                
                # Create model
                self.ml_model = models.mobilenet_v2(weights=None)
                self.ml_model.classifier[1] = torch.nn.Linear(self.ml_model.last_channel, len(self.ml_classes))
                self.ml_model.load_state_dict(torch.load(model_path, map_location='cpu'))
                self.ml_model.eval()
                
                # Preprocessing
                self.ml_preprocess = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                self.use_ml_model = True
                print(f"ML Model loaded successfully! Classes: {self.ml_classes}")
                # Update UI to show ML model is active
                self.root.after(0, lambda: self.model_label.config(
                    text="[ML Model Active]", foreground="green"))
            except Exception as e:
                print(f"Failed to load ML model: {e}")
                self.use_ml_model = False
        else:
            print("ML model not found. Using OpenCV-based detection.")
            print("To use ML model: run collect_data.py and train_model.py first")
    
    def detect_gesture_ml(self, frame, hand_roi):
        """Detect gesture using trained ML model"""
        if not self.use_ml_model or self.ml_model is None:
            return None, None
        
        try:
            # Extract hand region
            h, w = frame.shape[:2]
            x1, y1 = int(w * 0.3), int(h * 0.4)
            x2, y2 = w, h
            hand_crop = frame[y1:y2, x1:x2]
            
            if hand_crop.size == 0:
                return None, None
            
            # Convert to PIL Image
            hand_rgb = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB)
            hand_pil = Image.fromarray(hand_rgb)
            
            # Preprocess
            img_tensor = self.ml_preprocess(hand_pil).unsqueeze(0)
            
            # Predict
            with torch.no_grad():
                outputs = self.ml_model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence, predicted = torch.max(probabilities, 0)
                predicted_class = self.ml_classes[predicted.item()]
                confidence_score = confidence.item()
            
            # Only return if confidence is high enough
            if confidence_score > 0.5:
                debug_info = {
                    'finger_count': -1,  # Not applicable for ML
                    'defect_count': -1,
                    'solidity': -1,
                    'confidence': round(confidence_score, 2),
                    'method': 'ML'
                }
                return predicted_class, debug_info
            
            return None, None
        except Exception as e:
            print(f"ML detection error: {e}")
            return None, None
    
    def detect_gesture(self, frame, hand_mask):
        """Detect rock, paper, or scissors gesture from hand contour"""
        # Find contours
        contours, _ = cv2.findContours(hand_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Get the largest contour (hand)
        hand_contour = max(contours, key=cv2.contourArea)
        
        # Filter by area (hand should be reasonably sized)
        area = cv2.contourArea(hand_contour)
        if area < 5000:  # Too small to be a hand
            return None
        
        # Get bounding rectangle and aspect ratio
        x, y, w, h = cv2.boundingRect(hand_contour)
        aspect_ratio = float(w) / h if h > 0 else 1.0
        
        # Calculate solidity (area / convex hull area) - helps distinguish rock vs paper
        hull = cv2.convexHull(hand_contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0
        
        # Method 1: Count fingers using extreme points (topmost points)
        # Find the topmost point (wrist or base)
        topmost = tuple(hand_contour[hand_contour[:, :, 1].argmin()][0])
        bottommost = tuple(hand_contour[hand_contour[:, :, 1].argmax()][0])
        leftmost = tuple(hand_contour[hand_contour[:, :, 0].argmin()][0])
        rightmost = tuple(hand_contour[hand_contour[:, :, 0].argmax()][0])
        
        # Use the bottommost point as reference (wrist)
        wrist_y = bottommost[1]
        wrist_x = (leftmost[0] + rightmost[0]) / 2
        
        # Find all local maxima (peaks) in the upper half of the contour
        # These represent fingertips
        finger_peaks = []
        contour_points = hand_contour.reshape(-1, 2)
        
        # Find points significantly above the wrist
        threshold_y = wrist_y - (wrist_y - topmost[1]) * 0.3  # Top 30% of hand
        
        for i in range(len(contour_points)):
            px, py = contour_points[i]
            if py < threshold_y:  # Above threshold
                # Check if this is a local maximum (peak)
                is_peak = True
                for j in range(max(0, i-10), min(len(contour_points), i+10)):
                    if j != i and contour_points[j][1] < py:
                        is_peak = False
                        break
                if is_peak:
                    finger_peaks.append((px, py))
        
        # Group nearby peaks (same finger)
        if finger_peaks:
            finger_peaks = sorted(finger_peaks, key=lambda p: p[0])  # Sort by x
            distinct_peaks = []
            for peak in finger_peaks:
                if not distinct_peaks or abs(peak[0] - distinct_peaks[-1][0]) > w * 0.15:
                    distinct_peaks.append(peak)
            finger_count = len(distinct_peaks)
        else:
            finger_count = 0
        
        # Method 2: Use convexity defects
        hull_indices = cv2.convexHull(hand_contour, returnPoints=False)
        if len(hull_indices) < 3:
            # Very few hull points means likely a fist
            return "rock"
        
        defects = cv2.convexityDefects(hand_contour, hull_indices)
        defect_count = 0
        
        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(hand_contour[s][0])
                end = tuple(hand_contour[e][0])
                far = tuple(hand_contour[f][0])
                
                # Only count defects in the upper part (between fingers)
                if far[1] < threshold_y:
                    # Calculate angle
                    a = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                    b = np.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                    c = np.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                    
                    if b > 0 and c > 0:
                        angle = np.arccos(np.clip((b**2 + c**2 - a**2) / (2 * b * c), -1.0, 1.0))
                        # Valid defect: angle < 90 degrees and significant depth
                        if angle < np.pi / 2 and d > 20000:
                            defect_count += 1
        
        # Method 3: Use solidity and aspect ratio
        # Rock (fist): high solidity (>0.85), more circular
        # Paper (open): lower solidity (<0.75), wider
        # Scissors: medium solidity, elongated
        
        # Combine all methods with weights
        # Primary: finger count from peaks
        # Secondary: defect count
        # Tertiary: solidity
        
        # Store debug info for display
        debug_info = {
            'finger_count': finger_count,
            'defect_count': defect_count,
            'solidity': round(solidity, 2),
            'aspect_ratio': round(aspect_ratio, 2)
        }
        
        # Determine gesture based on combined analysis
        gesture = None
        if solidity > 0.85 and finger_count <= 1:
            # High solidity + few fingers = rock
            gesture = "rock"
        elif solidity < 0.75 and finger_count >= 4:
            # Low solidity + many fingers = paper
            gesture = "paper"
        elif 2 <= finger_count <= 3 and 0.75 <= solidity <= 0.85:
            # Medium fingers + medium solidity = scissors
            gesture = "scissors"
        elif finger_count <= 1:
            # Few fingers = rock
            gesture = "rock"
        elif finger_count >= 4:
            # Many fingers = paper
            gesture = "paper"
        elif finger_count == 2:
            # Two fingers = scissors
            gesture = "scissors"
        else:
            # Fallback based on solidity
            if solidity > 0.8:
                gesture = "rock"
            elif solidity < 0.7:
                gesture = "paper"
            else:
                gesture = "scissors"
        
        # Return gesture and debug info
        return gesture, debug_info
    
    def detect_hand_simple(self, frame, faces):
        """Simple hand detection using skin color and contour analysis, excluding face regions"""
        h, w = frame.shape[:2]
        
        # Create a mask that excludes face regions
        exclusion_mask = np.ones((h, w), dtype=np.uint8) * 255
        
        # Exclude face regions from hand detection
        for (fx, fy, fw, fh) in faces:
            # Expand the exclusion area around the face
            margin = 50
            x1 = max(0, fx - margin)
            y1 = max(0, fy - margin)
            x2 = min(w, fx + fw + margin)
            y2 = min(h, fy + fh + margin * 2)  # More margin below face
            exclusion_mask[y1:y2, x1:x2] = 0
        
        # Focus on lower half and right side of frame for hand detection
        # This is where hands typically appear
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        # Lower 60% of frame
        roi_mask[int(h * 0.4):, :] = 255
        # Right 70% of frame (to avoid face area on left)
        roi_mask[:, int(w * 0.3):] = 255
        
        # Combine exclusion and ROI masks
        combined_mask = cv2.bitwise_and(exclusion_mask, roi_mask)
        
        # Convert to HSV for better skin color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define skin color range (adjusted for better detection)
        lower_skin = np.array([0, 30, 60], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Create mask for skin color
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Apply the combined mask to exclude faces and focus on hand area
        skin_mask = cv2.bitwise_and(skin_mask, combined_mask)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)
        
        return skin_mask
    
    def determine_winner(self, player, ai):
        """Determine winner of rock-paper-scissors"""
        if player == ai:
            return "tie"
        elif (player == "rock" and ai == "scissors") or \
             (player == "paper" and ai == "rock") or \
             (player == "scissors" and ai == "paper"):
            return "player"
        else:
            return "ai"
    
    def create_widgets(self):
        """Create the GUI widgets"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Face & Hand Detection AI - Rock Paper Scissors", 
                               font=("Arial", 18, "bold"))
        title_label.grid(row=0, column=0, pady=10)
        
        # Control buttons frame
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=1, column=0, pady=10, sticky=(tk.W, tk.E))
        
        # Mode selection
        mode_frame = ttk.LabelFrame(control_frame, text="Mode", padding="10")
        mode_frame.pack(side=tk.LEFT, padx=5)
        
        self.mode_var = tk.StringVar(value="webcam")
        ttk.Radiobutton(mode_frame, text="Webcam", variable=self.mode_var, 
                       value="webcam", command=self.on_mode_change).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(mode_frame, text="Image File", variable=self.mode_var, 
                       value="image", command=self.on_mode_change).pack(side=tk.LEFT, padx=5)
        
        # Action buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(side=tk.LEFT, padx=10)
        
        self.start_btn = ttk.Button(button_frame, text="Start Webcam", 
                                    command=self.toggle_webcam)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.load_btn = ttk.Button(button_frame, text="Load Image", 
                                   command=self.load_image, state=tk.DISABLED)
        self.load_btn.pack(side=tk.LEFT, padx=5)
        
        self.detect_btn = ttk.Button(button_frame, text="Detect Faces", 
                                     command=self.detect_faces, state=tk.DISABLED)
        self.detect_btn.pack(side=tk.LEFT, padx=5)
        
        self.game_btn = ttk.Button(button_frame, text="Play Rock-Paper-Scissors", 
                                   command=self.toggle_game_mode, state=tk.DISABLED)
        self.game_btn.pack(side=tk.LEFT, padx=5)
        
        # Model indicator
        self.model_label = ttk.Label(button_frame, text="", font=("Arial", 8), foreground="gray")
        self.model_label.pack(side=tk.LEFT, padx=5)
        
        self.analyze_btn = ttk.Button(button_frame, text="Analyze with AI", 
                                      command=self.analyze_with_llm, state=tk.DISABLED)
        self.analyze_btn.pack(side=tk.LEFT, padx=5)
        
        # Main content area
        content_frame = ttk.Frame(main_frame)
        content_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        content_frame.columnconfigure(0, weight=2)
        content_frame.columnconfigure(1, weight=1)
        content_frame.rowconfigure(0, weight=1)
        
        # Video/Image display
        display_frame = ttk.LabelFrame(content_frame, text="Display", padding="10")
        display_frame.grid(row=0, column=0, padx=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        display_frame.columnconfigure(0, weight=1)
        display_frame.rowconfigure(0, weight=1)
        
        self.video_label = ttk.Label(display_frame, text="No video/image loaded", 
                                     anchor=tk.CENTER)
        self.video_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Right panel for game and info
        right_panel = ttk.Frame(content_frame)
        right_panel.grid(row=0, column=1, padx=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        right_panel.columnconfigure(0, weight=1)
        
        # Game panel
        game_frame = ttk.LabelFrame(right_panel, text="Rock-Paper-Scissors Game", padding="10")
        game_frame.grid(row=0, column=0, pady=5, sticky=(tk.W, tk.E))
        game_frame.columnconfigure(0, weight=1)
        
        self.game_status_label = ttk.Label(game_frame, text="Game not started", 
                                          font=("Arial", 12, "bold"))
        self.game_status_label.grid(row=0, column=0, pady=5)
        
        self.player_choice_label = ttk.Label(game_frame, text="Your choice: -", 
                                            font=("Arial", 10))
        self.player_choice_label.grid(row=1, column=0, pady=2)
        
        self.ai_choice_label = ttk.Label(game_frame, text="AI choice: -", 
                                        font=("Arial", 10))
        self.ai_choice_label.grid(row=2, column=0, pady=2)
        
        self.result_label = ttk.Label(game_frame, text="", 
                                     font=("Arial", 11, "bold"))
        self.result_label.grid(row=3, column=0, pady=5)
        
        self.score_label = ttk.Label(game_frame, text="Score: You 0 - 0 AI", 
                                    font=("Arial", 10))
        self.score_label.grid(row=4, column=0, pady=5)
        
        self.round_label = ttk.Label(game_frame, text="Round: 0", 
                                    font=("Arial", 10))
        self.round_label.grid(row=5, column=0, pady=2)
        
        # Instructions
        instructions = ttk.Label(game_frame, 
                                text="Show your hand in the YELLOW box:\n• Fist = Rock\n• Open hand = Paper\n• Two fingers = Scissors\n\nPlace hand in lower-right area",
                                font=("Arial", 9), justify=tk.CENTER)
        instructions.grid(row=6, column=0, pady=5)
        
        # Status and info frame
        info_frame = ttk.LabelFrame(right_panel, text="Information", padding="10")
        info_frame.grid(row=1, column=0, pady=5, sticky=(tk.W, tk.E))
        info_frame.columnconfigure(0, weight=1)
        
        self.status_label = ttk.Label(info_frame, text="Ready", 
                                      font=("Arial", 10))
        self.status_label.grid(row=0, column=0, sticky=tk.W)
        
        self.face_count_label = ttk.Label(info_frame, text="Faces detected: 0", 
                                          font=("Arial", 10))
        self.face_count_label.grid(row=1, column=0, sticky=tk.W)
        
        self.hand_status_label = ttk.Label(info_frame, text="Hand: Not detected", 
                                          font=("Arial", 10))
        self.hand_status_label.grid(row=2, column=0, sticky=tk.W)
        
        self.debug_label = ttk.Label(info_frame, text="Debug: -", 
                                     font=("Arial", 9), foreground="gray")
        self.debug_label.grid(row=3, column=0, sticky=tk.W)
        
        # AI Analysis output
        analysis_frame = ttk.LabelFrame(right_panel, text="AI Analysis", padding="10")
        analysis_frame.grid(row=2, column=0, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        analysis_frame.columnconfigure(0, weight=1)
        analysis_frame.rowconfigure(0, weight=1)
        right_panel.rowconfigure(2, weight=1)
        
        self.analysis_text = tk.Text(analysis_frame, height=8, wrap=tk.WORD, 
                                     state=tk.DISABLED)
        self.analysis_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        scrollbar = ttk.Scrollbar(analysis_frame, orient=tk.VERTICAL, 
                                  command=self.analysis_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.analysis_text.configure(yscrollcommand=scrollbar.set)
        
    def on_mode_change(self):
        """Handle mode change"""
        mode = self.mode_var.get()
        if mode == "webcam":
            self.load_btn.config(state=tk.DISABLED)
            self.start_btn.config(state=tk.NORMAL)
        else:
            if self.is_webcam_active:
                self.toggle_webcam()
            self.load_btn.config(state=tk.NORMAL)
            self.start_btn.config(state=tk.DISABLED)
            self.game_btn.config(state=tk.DISABLED)
    
    def toggle_webcam(self):
        """Start or stop webcam"""
        if not self.is_webcam_active:
            self.start_webcam()
        else:
            self.stop_webcam()
    
    def toggle_game_mode(self):
        """Toggle rock-paper-scissors game mode"""
        self.game_mode = not self.game_mode
        if self.game_mode:
            self.game_btn.config(text="Stop Game")
            self.player_score = 0
            self.ai_score = 0
            self.round_count = 0
            self.update_game_display()
            self.game_status_label.config(text="Game active - Show your hand!")
        else:
            self.game_btn.config(text="Play Rock-Paper-Scissors")
            self.game_status_label.config(text="Game not started")
            self.player_choice_label.config(text="Your choice: -")
            self.ai_choice_label.config(text="AI choice: -")
            self.result_label.config(text="")
    
    def update_game_display(self):
        """Update game display"""
        self.score_label.config(text=f"Score: You {self.player_score} - {self.ai_score} AI")
        self.round_label.config(text=f"Round: {self.round_count}")
    
    def start_webcam(self):
        """Start webcam capture"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open webcam")
                return
            
            self.is_webcam_active = True
            self.start_btn.config(text="Stop Webcam")
            self.detect_btn.config(state=tk.NORMAL)
            self.game_btn.config(state=tk.NORMAL)
            self.status_label.config(text="Webcam active")
            
            # Start video loop in separate thread
            self.video_thread = threading.Thread(target=self.update_video, daemon=True)
            self.video_thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start webcam: {e}")
    
    def stop_webcam(self):
        """Stop webcam capture"""
        self.is_webcam_active = False
        self.game_mode = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.start_btn.config(text="Start Webcam")
        self.detect_btn.config(state=tk.DISABLED)
        self.game_btn.config(state=tk.DISABLED)
        self.game_btn.config(text="Play Rock-Paper-Scissors")
        self.video_label.config(image='', text="Webcam stopped")
        self.status_label.config(text="Webcam stopped")
        self.face_count_label.config(text="Faces detected: 0")
        self.hand_status_label.config(text="Hand: Not detected")
        self.debug_label.config(text="Debug: -")
        self.detected_faces = []
    
    def update_video(self):
        """Update video frame in a loop"""
        import time
        last_gesture_time = 0
        gesture_cooldown = 2.0  # 2 seconds between game rounds
        gesture_stable_count = 0
        last_gesture = None
        
        while self.is_webcam_active and self.cap:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame.copy()
                
                # Detect faces
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )
                self.detected_faces = faces
                
                # Draw rectangles around faces
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, 'Face', (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                # Detect hands and gestures
                hand_mask = self.detect_hand_simple(frame, faces)
                
                # Try ML model first, fallback to OpenCV method
                if self.use_ml_model:
                    current_gesture, debug_data = self.detect_gesture_ml(frame, None)
                    if current_gesture is None:
                        # Fallback to OpenCV method
                        gesture_result = self.detect_gesture(frame, hand_mask)
                        if gesture_result is None:
                            current_gesture = None
                            debug_data = None
                        elif isinstance(gesture_result, tuple):
                            current_gesture, debug_data = gesture_result
                        else:
                            current_gesture = gesture_result
                            debug_data = None
                else:
                    # Use OpenCV method
                    gesture_result = self.detect_gesture(frame, hand_mask)
                    if gesture_result is None:
                        current_gesture = None
                        debug_data = None
                    elif isinstance(gesture_result, tuple):
                        current_gesture, debug_data = gesture_result
                    else:
                        current_gesture = gesture_result
                        debug_data = None
                
                # Draw hand detection area and gesture
                h, w = frame.shape[:2]
                # Draw hand detection region (lower right area)
                cv2.rectangle(frame, (int(w * 0.3), int(h * 0.4)), (w, h), (0, 255, 255), 2)
                cv2.putText(frame, "Hand Detection Area", (int(w * 0.3) + 10, int(h * 0.4) + 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Draw hand contour and gesture if detected
                if current_gesture:
                    contours, _ = cv2.findContours(hand_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        hand_contour = max(contours, key=cv2.contourArea)
                        x, y, w_hand, h_hand = cv2.boundingRect(hand_contour)
                        # Only draw if it's in the hand detection area
                        if y > h * 0.4 and x > w * 0.3:
                            cv2.rectangle(frame, (x, y), (x+w_hand, y+h_hand), (255, 0, 0), 3)
                            cv2.putText(frame, current_gesture.upper(), (x, y-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
                
                # Update UI
                self.root.after(0, lambda: self.face_count_label.config(
                    text=f"Faces detected: {len(faces)}"))
                
                hand_text = f"Hand: {current_gesture.upper()}" if current_gesture else "Hand: Not detected"
                self.root.after(0, lambda h=hand_text: self.hand_status_label.config(text=h))
                
                # Update debug info
                if debug_data:
                    if debug_data.get('method') == 'ML':
                        debug_text = f"Debug: ML Model | Confidence={debug_data['confidence']}"
                    else:
                        debug_text = f"Debug: Fingers={debug_data['finger_count']}, Solidity={debug_data['solidity']}, Defects={debug_data['defect_count']}"
                    self.root.after(0, lambda d=debug_text: self.debug_label.config(text=d))
                else:
                    self.root.after(0, lambda: self.debug_label.config(text="Debug: -"))
                
                # Game logic - require stable gesture for 5 frames
                if self.game_mode and current_gesture:
                    if current_gesture == last_gesture:
                        gesture_stable_count += 1
                    else:
                        gesture_stable_count = 0
                        last_gesture = current_gesture
                    
                    current_time = time.time()
                    if gesture_stable_count >= 5 and current_time - last_gesture_time > gesture_cooldown:
                        last_gesture_time = current_time
                        gesture_stable_count = 0
                        self.player_choice = current_gesture
                        self.ai_choice = random.choice(["rock", "paper", "scissors"])
                        self.round_count += 1
                        
                        winner = self.determine_winner(self.player_choice, self.ai_choice)
                        if winner == "player":
                            self.player_score += 1
                            result_text = "You Win! 🎉"
                        elif winner == "ai":
                            self.ai_score += 1
                            result_text = "AI Wins! 😔"
                        else:
                            result_text = "It's a Tie! 🤝"
                        
                        # Update game display
                        self.root.after(0, lambda: self.player_choice_label.config(
                            text=f"Your choice: {self.player_choice.upper()}"))
                        self.root.after(0, lambda: self.ai_choice_label.config(
                            text=f"AI choice: {self.ai_choice.upper()}"))
                        self.root.after(0, lambda: self.result_label.config(text=result_text))
                        self.root.after(0, self.update_game_display)
                elif not current_gesture:
                    gesture_stable_count = 0
                    last_gesture = None
                
                # Convert to PhotoImage and display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                frame_pil = frame_pil.resize((800, 600), Image.Resampling.LANCZOS)
                frame_tk = ImageTk.PhotoImage(image=frame_pil)
                
                def update_display(img):
                    self.video_label.config(image=img)
                    self.video_label.image = img
                
                self.root.after(0, lambda img=frame_tk: update_display(img))
    
    def load_image(self):
        """Load image from file"""
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        
        if file_path:
            try:
                img = cv2.imread(file_path)
                if img is None:
                    messagebox.showerror("Error", "Could not load image")
                    return
                
                self.current_frame = img
                self.is_webcam_active = False
                if self.cap:
                    self.cap.release()
                    self.cap = None
                
                # Display image
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                img_pil = img_pil.resize((800, 600), Image.Resampling.LANCZOS)
                img_tk = ImageTk.PhotoImage(image=img_pil)
                
                self.video_label.config(image=img_tk)
                self.video_label.image = img_tk
                self.detect_btn.config(state=tk.NORMAL)
                self.analyze_btn.config(state=tk.DISABLED)
                self.status_label.config(text=f"Image loaded: {os.path.basename(file_path)}")
                self.face_count_label.config(text="Faces detected: 0")
                self.detected_faces = []
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {e}")
    
    def detect_faces(self):
        """Detect faces in current frame"""
        if self.current_frame is None:
            messagebox.showwarning("Warning", "No image or video frame available")
            return
        
        try:
            gray = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            
            self.detected_faces = faces
            
            # Draw on frame
            frame = self.current_frame.copy()
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, 'Face', (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Update display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_pil = frame_pil.resize((800, 600), Image.Resampling.LANCZOS)
            frame_tk = ImageTk.PhotoImage(image=frame_pil)
            
            self.video_label.config(image=frame_tk)
            self.video_label.image = frame_tk
            self.face_count_label.config(text=f"Faces detected: {len(faces)}")
            self.analyze_btn.config(state=tk.NORMAL if len(faces) > 0 else tk.DISABLED)
            self.status_label.config(text=f"Detection complete: {len(faces)} face(s) found")
            
        except Exception as e:
            messagebox.showerror("Error", f"Face detection failed: {e}")
    
    def analyze_with_llm(self):
        """Analyze detected faces with LLM"""
        if len(self.detected_faces) == 0:
            messagebox.showwarning("Warning", "No faces detected. Please detect faces first.")
            return
        
        if not self.openai_client:
            message = (
                f"Detected {len(self.detected_faces)} face(s) in the image.\n\n"
                "To enable AI analysis, please set your OPENAI_API_KEY environment variable:\n"
                "export OPENAI_API_KEY=your_key_here"
            )
            self.update_analysis_text(message)
            return
        
        try:
            self.status_label.config(text="Analyzing with AI...")
            self.analyze_btn.config(state=tk.DISABLED)
            
            # Prepare prompt
            face_count = len(self.detected_faces)
            prompt = (
                f"You detected {face_count} face(s) in an image. "
                "Provide a brief, friendly analysis of what you observe about the faces. "
                "Keep it concise (2-3 sentences)."
            )
            
            # Call OpenAI API in a separate thread to avoid blocking UI
            def call_llm():
                try:
                    response = self.openai_client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a helpful AI assistant that analyzes faces detected in images. Provide friendly, concise observations."
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        max_tokens=150
                    )
                    
                    message = response.choices[0].message.content
                    self.root.after(0, lambda: self.update_analysis_text(message))
                    self.root.after(0, lambda: self.status_label.config(text="Analysis complete"))
                    self.root.after(0, lambda: self.analyze_btn.config(state=tk.NORMAL))
                    
                except Exception as e:
                    error_msg = f"LLM analysis failed: {e}"
                    self.root.after(0, lambda: self.update_analysis_text(error_msg))
                    self.root.after(0, lambda: self.status_label.config(text="Analysis failed"))
                    self.root.after(0, lambda: self.analyze_btn.config(state=tk.NORMAL))
            
            thread = threading.Thread(target=call_llm, daemon=True)
            thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to analyze: {e}")
            self.analyze_btn.config(state=tk.NORMAL)
    
    def update_analysis_text(self, text):
        """Update the analysis text widget"""
        self.analysis_text.config(state=tk.NORMAL)
        self.analysis_text.delete(1.0, tk.END)
        self.analysis_text.insert(1.0, text)
        self.analysis_text.config(state=tk.DISABLED)
    
    def on_closing(self):
        """Handle window closing"""
        if self.is_webcam_active:
            self.stop_webcam()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = FaceHandDetectionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
