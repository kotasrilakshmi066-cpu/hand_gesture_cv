"""
=============================================================================
Project: Hand Gesture Controlled Virtual Mouse
Module: hand_tracking.py
Author: Member 1 - Hand Detection and Landmark Tracking Specialist
Description:
    Real-time hand detection and 21-landmark tracking wrapper leveraging Google
    MediaPipe Hands and OpenCV. Provides landmark extraction, geometric distance
    computation, finger extension status analysis, and custom visualization.
=============================================================================
"""

import cv2
import mediapipe as mp
import math
from typing import List, Tuple, Optional, Dict, Any


class HandDetector:
    """
    Robust real-time hand detector and landmark tracker using MediaPipe Hands.
    Extracts 21 3D landmarks per detected hand and provides geometric calculations.
    """

    # Tip Landmark IDs for [Thumb, Index, Middle, Ring, Pinky]
    TIP_IDS = [4, 8, 12, 16, 20]
    PIP_IDS = [3, 6, 10, 14, 18]
    MCP_IDS = [2, 5, 9, 13, 17]

    def __init__(
        self,
        mode: bool = False,
        max_hands: int = 1,
        detection_confidence: float = 0.7,
        tracking_confidence: float = 0.6
    ):
        """
        Initialize the MediaPipe Hands model with custom detection and tracking thresholds.

        :param mode: False for video streaming (temporal tracking), True for static images
        :param max_hands: Maximum number of hands to detect simultaneously
        :param detection_confidence: Minimum confidence threshold [0.0 - 1.0] for initial detection
        :param tracking_confidence: Minimum confidence threshold [0.0 - 1.0] for landmark tracking
        """
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Custom drawing specifications for a clean modern aesthetic
        self.landmark_style = self.mp_draw.DrawingSpec(
            color=(0, 229, 255),  # Cyan dots (BGR)
            thickness=2,
            circle_radius=3
        )
        self.connection_style = self.mp_draw.DrawingSpec(
            color=(255, 100, 50), # Sleek blue/purple connectors (BGR)
            thickness=2
        )

        self.results = None
        self.lm_list: List[List[int]] = []

    def find_hands(self, frame: cv2.Mat, draw: bool = True) -> cv2.Mat:
        """
        Process BGR frame with MediaPipe and optionally render hand skeleton connections.

        :param frame: Raw input frame from OpenCV webcam stream (BGR)
        :param draw: Whether to render landmark points and connectors onto frame
        :return: Processed frame with visual overlays
        """
        if frame is None:
            return frame

        # MediaPipe requires RGB format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(frame_rgb)

        if self.results.multi_hand_landmarks and draw:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=self.landmark_style,
                    connection_drawing_spec=self.connection_style
                )

        return frame

    def find_positions(
        self,
        frame: cv2.Mat,
        hand_index: int = 0,
        draw: bool = False
    ) -> Tuple[List[List[int]], Tuple[int, int, int, int]]:
        """
        Extract absolute pixel coordinates for all 21 hand landmarks and bounding box.

        :param frame: Current video frame (used to get width and height)
        :param hand_index: Target hand index in multi_hand_landmarks
        :param draw: Highlight fingertip landmarks with rings
        :return: (lm_list, bbox) where:
                 lm_list = [[id, cx, cy, cz_scaled], ...]
                 bbox = (x_min, y_min, x_max, y_max)
        """
        self.lm_list = []
        bbox = (0, 0, 0, 0)

        if frame is None:
            return self.lm_list, bbox

        h, w, _ = frame.shape
        x_list = []
        y_list = []

        if self.results and self.results.multi_hand_landmarks:
            if hand_index < len(self.results.multi_hand_landmarks):
                my_hand = self.results.multi_hand_landmarks[hand_index]

                for lm_id, lm in enumerate(my_hand.landmark):
                    # Convert normalized coordinates [0.0, 1.0] to pixel space
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cz = int(lm.z * w) # Approximate relative depth scaled by width
                    x_list.append(cx)
                    y_list.append(cy)
                    self.lm_list.append([lm_id, cx, cy, cz])

                    if draw and lm_id in self.TIP_IDS:
                        cv2.circle(frame, (cx, cy), 6, (0, 255, 0), cv2.FILLED)

                if x_list and y_list:
                    x_min, x_max = min(x_list), max(x_list)
                    y_min, y_max = min(y_list), max(y_list)
                    bbox = (x_min, y_min, x_max, y_max)

        return self.lm_list, bbox

    def fingers_up(self) -> List[int]:
        """
        Determines the open/folded state of all 5 fingers.
        
        :return: Binary list of length 5 [Thumb, Index, Middle, Ring, Pinky]
                 where 1 indicates extended/up, and 0 indicates folded/down.
        """
        fingers = [0, 0, 0, 0, 0]
        if not self.lm_list or len(self.lm_list) < 21:
            return fingers

        # ---------------------------------------------------------------------
        # Thumb Detection (Landmark 4)
        # Check orientation against Wrist (0) and Index MCP (5)
        # In a standard mirrored palm-facing webcam view:
        # If thumb tip is further outward from pinky MCP than the IP joint, it's extended.
        # Alternatively, evaluate distance between Thumb Tip (4) and Pinky MCP (17) vs Thumb IP (3) and Pinky MCP (17).
        # ---------------------------------------------------------------------
        d_tip_pinky = math.hypot(
            self.lm_list[4][1] - self.lm_list[17][1],
            self.lm_list[4][2] - self.lm_list[17][2]
        )
        d_ip_pinky = math.hypot(
            self.lm_list[3][1] - self.lm_list[17][1],
            self.lm_list[3][2] - self.lm_list[17][2]
        )
        if d_tip_pinky > d_ip_pinky * 1.05:
            fingers[0] = 1
        else:
            fingers[0] = 0

        # ---------------------------------------------------------------------
        # Four Fingers: Index (8), Middle (12), Ring (16), Pinky (20)
        # In standard webcam view, a finger is UP if TIP.y is higher than PIP.y
        # Note: In computer vision coordinates, top of frame has y = 0.
        # ---------------------------------------------------------------------
        for idx in range(1, 5):
            tip_id = self.TIP_IDS[idx]
            pip_id = self.PIP_IDS[idx]
            if self.lm_list[tip_id][2] < self.lm_list[pip_id][2]:
                fingers[idx] = 1
            else:
                fingers[idx] = 0

        return fingers

    def find_distance(
        self,
        p1_id: int,
        p2_id: int,
        frame: Optional[cv2.Mat] = None,
        draw: bool = True,
        color: Tuple[int, int, int] = (0, 255, 255)
    ) -> Tuple[float, Optional[cv2.Mat], Tuple[int, int]]:
        """
        Calculate Euclidean distance between two landmark points and optionally draw indicator.

        :param p1_id: Landmark index for first point
        :param p2_id: Landmark index for second point
        :param frame: Target frame to render visualization onto
        :param draw: Render line and midpoint indicator
        :param color: BGR color tuple for line
        :return: (distance, frame, (mid_x, mid_y))
        """
        if len(self.lm_list) < max(p1_id, p2_id) + 1:
            return 0.0, frame, (0, 0)

        x1, y1 = self.lm_list[p1_id][1], self.lm_list[p1_id][2]
        x2, y2 = self.lm_list[p2_id][1], self.lm_list[p2_id][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        distance = math.hypot(x2 - x1, y2 - y1)

        if draw and frame is not None:
            cv2.line(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (x1, y1), 5, (0, 0, 255), cv2.FILLED)
            cv2.circle(frame, (x2, y2), 5, (0, 0, 255), cv2.FILLED)
            cv2.circle(frame, (cx, cy), 6, (255, 255, 255), cv2.FILLED)

        return distance, frame, (cx, cy)

    def is_hand_present(self) -> bool:
        """Check whether any hand is currently detected in the active frame."""
        return bool(self.results and self.results.multi_hand_landmarks)

    def release(self):
        """Clean up MediaPipe Hands pipeline resources."""
        if hasattr(self, 'hands') and self.hands:
            self.hands.close()

