"""
=============================================================================
Project: Hand Gesture Controlled Virtual Mouse
Module: generate_screenshot.py
Author: Computer Vision Academic Project Team
Description:
    Generates a high-fidelity visual mock render of the running Tkinter
    dashboard and saves it to screenshots/ui_dashboard.png for presentation.
=============================================================================
"""

import os
import cv2
import numpy as np

def generate_ui_preview(output_path: str):
    w, h = 1180, 760
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (26, 17, 15)  # bg_dark #0F111A (BGR)

    # Top Header
    cv2.rectangle(img, (0, 0), (w, 65), (15, 17, 26), cv2.FILLED)
    cv2.putText(img, "HAND GESTURE CONTROLLED VIRTUAL MOUSE", (25, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "Real-Time Computer Vision HCI System | MediaPipe Hands + OpenCV + PyAutoGUI", (25, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (192, 171, 158), 1, cv2.LINE_AA)

    # Badge
    cv2.rectangle(img, (w - 230, 18), (w - 25, 50), (56, 38, 34), cv2.FILLED)
    cv2.rectangle(img, (w - 230, 18), (w - 25, 50), (255, 229, 0), 1)
    cv2.putText(img, "ACADEMIC PROJECT v1.0", (w - 215, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 229, 0), 1, cv2.LINE_AA)

    # Split: Left Viewport (740px width), Right Panel (380px width)
    # Left Video Card
    cv2.rectangle(img, (20, 80), (740, 730), (43, 29, 26), cv2.FILLED) # bg_card
    # Simulated camera viewport
    cv2.rectangle(img, (30, 90), (730, 680), (10, 10, 10), cv2.FILLED)
    
    # In-feed Active Interaction Zone
    cv2.rectangle(img, (130, 170), (630, 600), (0, 120, 255), 2)
    cv2.putText(img, "Active Interaction Zone", (140, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 120, 255), 1, cv2.LINE_AA)

    # Simulated Hand Landmarks (Index tip pointing up)
    index_tip = (380, 290)
    thumb_tip = (310, 390)
    wrist = (380, 560)
    
    # Draw hand skeleton connections
    cv2.line(img, wrist, (340, 480), (50, 100, 255), 2)
    cv2.line(img, (340, 480), (320, 440), (50, 100, 255), 2)
    cv2.line(img, (320, 440), thumb_tip, (50, 100, 255), 2)

    cv2.line(img, wrist, (380, 460), (50, 100, 255), 2)
    cv2.line(img, (380, 460), (380, 380), (50, 100, 255), 2)
    cv2.line(img, (380, 380), index_tip, (50, 100, 255), 2)

    # Draw landmarks
    cv2.circle(img, index_tip, 12, (255, 229, 0), 2)
    cv2.circle(img, index_tip, 5, (0, 229, 255), cv2.FILLED)
    cv2.circle(img, thumb_tip, 5, (0, 229, 255), cv2.FILLED)
    cv2.circle(img, wrist, 5, (0, 229, 255), cv2.FILLED)

    # Top HUD Banner on Video
    cv2.rectangle(img, (30, 90), (730, 125), (32, 22, 20), cv2.FILLED)
    cv2.putText(img, "FPS: 30", (45, 114), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 229, 0), 2, cv2.LINE_AA)
    cv2.putText(img, "Gesture: Moving Cursor", (160, 114), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (128, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(img, "MOUSE: ACTIVE", (580, 114), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

    # Bottom HUD status bar
    cv2.rectangle(img, (30, 690), (730, 720), (56, 38, 34), cv2.FILLED)
    cv2.putText(img, "FPS: 30", (45, 710), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 229, 0), 1, cv2.LINE_AA)
    cv2.putText(img, "Resolution: 640x480", (160, 710), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (192, 171, 158), 1, cv2.LINE_AA)
    cv2.putText(img, "MOUSE CONTROL ACTIVE", (540, 710), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (118, 230, 0), 1, cv2.LINE_AA)

    # Right Panel Cards
    # Card 1: Controls
    cv2.rectangle(img, (760, 80), (1160, 235), (43, 29, 26), cv2.FILLED)
    cv2.putText(img, "SYSTEM CONTROLS", (780, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 229, 0), 2, cv2.LINE_AA)
    # Buttons
    cv2.rectangle(img, (780, 122), (950, 160), (83, 200, 0), cv2.FILLED)
    cv2.putText(img, "▶ Start Camera", (800, 147), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(img, (970, 122), (1140, 160), (0, 0, 213), cv2.FILLED)
    cv2.putText(img, "⏹ Stop Camera", (990, 147), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    
    cv2.putText(img, "[X] Enable Virtual Mouse Control", (780, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    
    cv2.rectangle(img, (780, 198), (1140, 226), (102, 51, 255), cv2.FILLED)
    cv2.putText(img, "! EMERGENCY STOP (Disable Mouse)", (830, 216), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # Card 2: Telemetry
    cv2.rectangle(img, (760, 245), (1160, 385), (43, 29, 26), cv2.FILLED)
    cv2.putText(img, "LIVE RECOGNITION TELEMETRY", (780, 272), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 229, 0), 2, cv2.LINE_AA)
    cv2.rectangle(img, (780, 285), (1140, 325), (56, 38, 34), cv2.FILLED)
    cv2.putText(img, "MOVING CURSOR", (880, 312), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (118, 230, 0), 2, cv2.LINE_AA)
    cv2.putText(img, "Screen Target: X: 960 | Y: 540", (780, 348), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 229, 0), 1, cv2.LINE_AA)
    cv2.putText(img, "Fingers State [T,I,M,R,P]: [0, 1, 0, 0, 0]", (780, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # Card 3: Tuning
    cv2.rectangle(img, (760, 395), (1160, 520), (43, 29, 26), cv2.FILLED)
    cv2.putText(img, "TUNING & CALIBRATION", (780, 422), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 229, 0), 2, cv2.LINE_AA)
    cv2.putText(img, "Sensitivity: 1.2x", (780, 448), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(img, (780, 458), (1140, 464), (100, 100, 100), cv2.FILLED)
    cv2.rectangle(img, (780, 458), (920, 464), (255, 229, 0), cv2.FILLED)

    cv2.putText(img, "Smoothing Factor: 5", (780, 485), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(img, (780, 495), (1140, 501), (100, 100, 100), cv2.FILLED)
    cv2.rectangle(img, (780, 495), (960, 501), (255, 229, 0), cv2.FILLED)

    # Card 4: Gesture Guide
    cv2.rectangle(img, (760, 530), (1160, 730), (43, 29, 26), cv2.FILLED)
    cv2.putText(img, "GESTURE QUICK REFERENCE", (780, 555), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 229, 0), 2, cv2.LINE_AA)
    cv2.putText(img, "1. Index Up: Move Cursor", (780, 582), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 229, 0), 1, cv2.LINE_AA)
    cv2.putText(img, "2. Thumb + Index Pinch: Left Click", (780, 608), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (118, 230, 0), 1, cv2.LINE_AA)
    cv2.putText(img, "3. Thumb + Middle Pinch: Right Click", (780, 634), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 64, 129), 1, cv2.LINE_AA)
    cv2.putText(img, "4. Two Fingers Up: Scroll Up / Down", (780, 660), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 179, 255), 1, cv2.LINE_AA)
    cv2.putText(img, "5. Open Hand: Neutral / Idle", (780, 686), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(img, "6. Screen Corner: Emergency Failsafe", (780, 712), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (102, 51, 255), 1, cv2.LINE_AA)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)
    print(f"Generated UI preview: {output_path}")

if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "screenshots")
    generate_ui_preview(os.path.join(out_dir, "ui_dashboard.png"))

