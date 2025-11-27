#!/usr/bin/env python3
"""
Automatic Data Collection Script - Collects images continuously
Press SPACE to start/stop collecting for current class
"""

import cv2
import os
import time

classes = ["rock", "paper", "scissors"]

# Create folders
for cls in classes:
    os.makedirs(f"data/{cls}", exist_ok=True)
    print(f"Created folder: data/{cls}")

def collect_auto(class_name, target_count=200):
    """Automatically collect images at intervals"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    count = len([f for f in os.listdir(f"data/{class_name}") if f.endswith('.jpg')])
    print(f"\n=== Auto-collecting {class_name.upper()} images ===")
    print(f"Target: {target_count} images")
    print(f"Current: {count} images")
    print("Press SPACE to start auto-capture, SPACE again to pause, ESC to finish")
    
    auto_capturing = False
    last_capture_time = 0
    capture_interval = 0.3  # Capture every 0.3 seconds when auto mode is on
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        # Add instructions on frame
        cv2.putText(frame, f"Class: {class_name.upper()}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Count: {count} / {target_count}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if auto_capturing:
            cv2.putText(frame, "AUTO-CAPTURING...", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, "SPACE: Pause | ESC: Finish", (10, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "SPACE: Start Auto | ESC: Finish", (10, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw hand detection area
        cv2.rectangle(frame, (int(w * 0.3), int(h * 0.4)), (w, h), (0, 255, 255), 2)
        cv2.putText(frame, "Place hand here", (int(w * 0.3) + 10, int(h * 0.4) + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Auto-capture logic
        current_time = time.time()
        if auto_capturing and (current_time - last_capture_time) >= capture_interval:
            if count < target_count:
                filename = f"data/{class_name}/{count}.jpg"
                success = cv2.imwrite(filename, frame)
                if success:
                    count += 1
                    last_capture_time = current_time
                    if count % 10 == 0:
                        print(f"Captured {count}/{target_count} images...")
        
        cv2.imshow("Auto Capture", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        if key == 32:  # SPACE - toggle auto-capture
            auto_capturing = not auto_capturing
            if auto_capturing:
                print("Auto-capture started. Move your hand around!")
            else:
                print("Auto-capture paused.")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nFinished collecting {class_name}. Total images: {count}")

if __name__ == "__main__":
    print("=" * 50)
    print("Automatic Rock-Paper-Scissors Data Collection")
    print("=" * 50)
    print("\nThis script auto-captures images every 0.3 seconds")
    print("Move your hand around to get variety!\n")
    
    for cls in classes:
        current_count = len([f for f in os.listdir(f"data/{cls}") if f.endswith('.jpg')])
        print(f"{cls}: {current_count} images")
    
    print("\nAvailable classes: rock, paper, scissors")
    print("Type 'quit' to exit\n")
    
    while True:
        cls = input("Enter class to collect (rock/paper/scissors/quit): ").lower().strip()
        
        if cls == "quit":
            print("Exiting...")
            break
        elif cls in classes:
            collect_auto(cls, target_count=200)
            print(f"\nNext class? (rock/paper/scissors/quit)")
        else:
            print("Invalid class. Please enter: rock, paper, or scissors")


