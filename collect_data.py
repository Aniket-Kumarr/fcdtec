#!/usr/bin/env python3
"""
Data Collection Script for Rock-Paper-Scissors Gesture Recognition
Captures images from webcam and saves them to labeled folders
"""

import cv2
import os

classes = ["rock", "paper", "scissors"]

# Create folders
for cls in classes:
    os.makedirs(f"data/{cls}", exist_ok=True)
    print(f"Created folder: data/{cls}")

def collect(class_name):
    """Collect images for a specific class"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    count = len([f for f in os.listdir(f"data/{class_name}") if f.endswith('.jpg')])  # Start from existing count
    print(f"\n=== Collecting {class_name.upper()} images ===")
    print("Press SPACE to capture, ESC to stop.")
    print(f"Current count: {count}")
    
    save_feedback_time = 0
    save_feedback_duration = 10  # Show feedback for 10 frames
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Warning: Failed to read frame from camera")
            break
            
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        # Add instructions on frame
        cv2.putText(frame, f"Class: {class_name.upper()}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Count: {count}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "SPACE: Capture | ESC: Stop", (10, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw hand detection area
        cv2.rectangle(frame, (int(w * 0.3), int(h * 0.4)), (w, h), (0, 255, 255), 2)
        cv2.putText(frame, "Place hand here", (int(w * 0.3) + 10, int(h * 0.4) + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Show save feedback if recently saved
        if save_feedback_time > 0:
            cv2.putText(frame, "SAVED!", (w//2 - 100, h//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            save_feedback_time -= 1
        
        cv2.imshow("Capture Images", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        if key == 32:  # SPACE
            filename = f"data/{class_name}/{count}.jpg"
            # Save the current frame immediately (non-blocking)
            success = cv2.imwrite(filename, frame)
            if success:
                print(f"Saved {filename} ({count})")
                count += 1
                save_feedback_time = save_feedback_duration  # Show feedback for next few frames
            else:
                print(f"Failed to save {filename}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nFinished collecting {class_name}. Total images: {count}")

if __name__ == "__main__":
    print("=" * 50)
    print("Rock-Paper-Scissors Data Collection")
    print("=" * 50)
    print("\nAvailable classes: rock, paper, scissors")
    print("Type 'quit' to exit\n")
    
    while True:
        cls = input("Enter class to collect (rock/paper/scissors/quit): ").lower().strip()
        
        if cls == "quit":
            print("Exiting...")
            break
        elif cls in classes:
            collect(cls)
            print(f"\nNext class? (rock/paper/scissors/quit)")
        else:
            print("Invalid class. Please enter: rock, paper, or scissors")

