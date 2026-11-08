"""
=============================================================================
Project: Hand Gesture Controlled Virtual Mouse
Module: generate_assets.py
Author: Computer Vision Academic Project Team
Description:
    Generates educational diagrams and reference charts illustrating:
    1. MediaPipe Hands 21-landmark topology diagram.
    2. Gesture mapping quick reference card.
    Saves outputs to the assets/ directory.
=============================================================================
"""

import os
import cv2
import numpy as np

def create_gesture_guide_asset(output_path: str):
    # Create dark-themed high-res canvas (900x500)
    w, h = 900, 520
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:] = (26, 29, 43)  # Card dark background (BGR)

    # Header
    cv2.rectangle(canvas, (0, 0), (w, 60), (15, 17, 26), cv2.FILLED)
    cv2.putText(canvas, "HAND GESTURE CONTROLLED VIRTUAL MOUSE", (25, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 229, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "GESTURE MAPPING & CONTROL REFERENCE", (w - 380, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 175, 200), 1, cv2.LINE_AA)

    cards = [
        ("1. CURSOR MOVEMENT", "Index Finger Extended Up", "Index Tip (Landmark 8) controls screen coordinates.", (0, 229, 255)),
        ("2. LEFT CLICK", "Thumb Tip + Index Tip Pinch", "Euclidean distance < 35px triggers debounced click.", (0, 230, 118)),
        ("3. RIGHT CLICK", "Thumb Tip + Middle Tip Pinch", "Pinch Thumb & Middle fingers (< 38px) for context menu.", (255, 64, 129)),
        ("4. DUAL SCROLL", "Index + Middle Fingers Up", "Move hand vertically up or down to scroll page.", (255, 179, 0)),
        ("5. NEUTRAL IDLE", "Open Palm / Closed Fist", "Zero action dispatched. Safe hand rest mode.", (180, 180, 180)),
        ("6. SAFETY FAILSAFE", "Cursor to Any Screen Corner", "PyAutoGUI hardware failsafe triggers emergency stop.", (255, 51, 102))
    ]

    for idx, (title, gesture, desc, color) in enumerate(cards):
        col = idx % 2
        row = idx // 2
        x = 30 + col * 430
        y = 80 + row * 135

        # Card container
        cv2.rectangle(canvas, (x, y), (x + 410, y + 115), (34, 38, 56), cv2.FILLED)
        cv2.rectangle(canvas, (x, y), (x + 410, y + 115), (55, 62, 90), 1)
        # Accent left stripe
        cv2.rectangle(canvas, (x, y), (x + 6, y + 115), color, cv2.FILLED)

        cv2.putText(canvas, title, (x + 20, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        cv2.putText(canvas, f"Pose: {gesture}", (x + 20, y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, desc, (x + 20, y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 175, 200), 1, cv2.LINE_AA)

    # Footer
    cv2.putText(canvas, "Built with OpenCV, MediaPipe Hands & PyAutoGUI | Academic Computer Vision Project", (25, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 135, 160), 1, cv2.LINE_AA)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, canvas)
    print(f"Generated asset: {output_path}")

if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "assets")
    create_gesture_guide_asset(os.path.join(out_dir, "gesture_reference_guide.png"))

