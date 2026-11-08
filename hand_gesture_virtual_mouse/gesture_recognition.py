"""
=============================================================================
Project: Hand Gesture Controlled Virtual Mouse
Module: gesture_recognition.py
Author: Member 2 - Gesture Recognition and Classification Specialist
Description:
    Rule-based geometric gesture classifier that maps hand landmarks and finger
    states into discrete mouse commands (Movement, Left Click, Right Click,
    Scrolling, and Idle/Neutral states).
=============================================================================
"""

import math
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import config


@dataclass
class GestureResult:
    """
    Standardized payload returned by the GestureRecognizer on every frame.
    """
    gesture_name: str                      # Identifier: "MOVE", "LEFT_CLICK", "RIGHT_CLICK", "SCROLL", "IDLE", "NO_HAND"
    display_text: str                      # User-friendly label displayed in the GUI HUD
    cursor_point: Optional[Tuple[int, int]] = None  # (x, y) of index fingertip for cursor tracking
    pinch_distance: float = 0.0            # Euclidean distance for active pinch
    pinch_point: Optional[Tuple[int, int]] = None   # Midpoint between pinched fingers for visual indicator
    scroll_delta_y: int = 0                # Directional vertical displacement for scrolling (+1: Up, -1: Down)
    fingers_state: List[int] = field(default_factory=lambda: [0, 0, 0, 0, 0])
    confidence: float = 1.0


class GestureRecognizer:
    """
    High-accuracy geometric hand gesture classifier based on Euclidean distance
    metrics, finger extension states, and temporal tracking.
    """

    def __init__(
        self,
        click_distance: float = config.LEFT_CLICK_DISTANCE,
        right_click_distance: float = config.RIGHT_CLICK_DISTANCE,
        scroll_proximity: float = config.SCROLL_FINGER_PROXIMITY
    ):
        """
        Initialize the gesture classifier with geometric threshold limits.

        :param click_distance: Pixel distance threshold for thumb-to-index pinch (Left Click)
        :param right_click_distance: Pixel distance threshold for thumb-to-middle pinch (Right Click)
        :param scroll_proximity: Max distance between index & middle tips for synchronized scroll
        """
        self.click_distance = click_distance
        self.right_click_distance = right_click_distance
        self.scroll_proximity = scroll_proximity

        # Temporal state for scroll tracking
        self.prev_scroll_y: Optional[int] = None
        self.last_gesture_time: float = time.time()
        self.current_gesture: str = "NO_HAND"

    def update_thresholds(
        self,
        click_dist: Optional[float] = None,
        right_click_dist: Optional[float] = None
    ):
        """Allow dynamic threshold calibration from configuration or UI."""
        if click_dist is not None:
            self.click_distance = click_dist
        if right_click_dist is not None:
            self.right_click_distance = right_click_dist

    def classify(
        self,
        lm_list: List[List[int]],
        fingers: List[int]
    ) -> GestureResult:
        """
        Classify the current hand pose into an actionable gesture.

        Fingers list index: [0: Thumb, 1: Index, 2: Middle, 3: Ring, 4: Pinky]

        Rules:
        1. NO_HAND: lm_list is empty.
        2. LEFT_CLICK: Thumb Tip (4) and Index Tip (8) distance < click_distance.
        3. RIGHT_CLICK: Thumb Tip (4) and Middle Tip (12) distance < right_click_distance
           AND Thumb Tip (4) is not pinched to Index Tip.
        4. SCROLL: Both Index (1) and Middle (2) fingers extended, Ring (3) & Pinky (4) down,
           and distance between (8) and (12) < scroll_proximity.
        5. MOVE: Index (1) is extended, Middle (2) is folded (or separated from thumb),
           Thumb-Index distance > click_distance.
        6. IDLE: Open hand (all extended), fist (all folded), or other non-action pose.

        :param lm_list: 21 landmarks [[id, cx, cy, cz], ...]
        :param fingers: 5-element binary list [Thumb, Index, Middle, Ring, Pinky]
        :return: GestureResult object
        """
        if not lm_list or len(lm_list) < 21:
            self.prev_scroll_y = None
            self.current_gesture = "NO_HAND"
            return GestureResult(
                gesture_name="NO_HAND",
                display_text="No Hand Detected",
                fingers_state=[0, 0, 0, 0, 0]
            )

        # Extract primary keypoints (pixel coordinates)
        # 4: Thumb Tip, 8: Index Tip, 12: Middle Tip, 16: Ring Tip, 20: Pinky Tip
        x_thumb, y_thumb = lm_list[4][1], lm_list[4][2]
        x_index, y_index = lm_list[8][1], lm_list[8][2]
        x_middle, y_middle = lm_list[12][1], lm_list[12][2]

        # Calculate key Euclidean distances
        dist_thumb_index = math.hypot(x_index - x_thumb, y_index - y_thumb)
        dist_thumb_middle = math.hypot(x_middle - x_thumb, y_middle - y_thumb)
        dist_index_middle = math.hypot(x_middle - x_index, y_middle - y_index)

        cursor_coords = (x_index, y_index)

        # ---------------------------------------------------------------------
        # Priority 1: Left Click (Pinch between Thumb and Index)
        # ---------------------------------------------------------------------
        if dist_thumb_index < self.click_distance:
            mid_x = (x_thumb + x_index) // 2
            mid_y = (y_thumb + y_index) // 2
            self.prev_scroll_y = None
            self.current_gesture = "LEFT_CLICK"
            return GestureResult(
                gesture_name="LEFT_CLICK",
                display_text="Left Click (Pinch)",
                cursor_point=cursor_coords,
                pinch_distance=dist_thumb_index,
                pinch_point=(mid_x, mid_y),
                fingers_state=fingers
            )

        # ---------------------------------------------------------------------
        # Priority 2: Right Click (Pinch between Thumb and Middle)
        # Ensure index finger is not simultaneously pinching thumb
        # ---------------------------------------------------------------------
        if dist_thumb_middle < self.right_click_distance and dist_thumb_index >= self.click_distance:
            mid_x = (x_thumb + x_middle) // 2
            mid_y = (y_thumb + y_middle) // 2
            self.prev_scroll_y = None
            self.current_gesture = "RIGHT_CLICK"
            return GestureResult(
                gesture_name="RIGHT_CLICK",
                display_text="Right Click (Pinch)",
                cursor_point=cursor_coords,
                pinch_distance=dist_thumb_middle,
                pinch_point=(mid_x, mid_y),
                fingers_state=fingers
            )

        # ---------------------------------------------------------------------
        # Priority 3: Scrolling Gesture (Index & Middle fingers extended up)
        # Ring and Pinky should be folded down to distinguish from open hand
        # ---------------------------------------------------------------------
        is_two_fingers_up = (fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0)
        if is_two_fingers_up and dist_index_middle < self.scroll_proximity:
            avg_y = (y_index + y_middle) // 2
            scroll_delta = 0

            if self.prev_scroll_y is not None:
                diff = self.prev_scroll_y - avg_y
                if diff > config.SCROLL_DEADZONE:
                    scroll_delta = 1   # Moving fingers upwards -> scroll UP
                elif diff < -config.SCROLL_DEADZONE:
                    scroll_delta = -1  # Moving fingers downwards -> scroll DOWN
                
                # Smooth the reference position
                if abs(diff) > config.SCROLL_DEADZONE:
                    self.prev_scroll_y = avg_y
            else:
                self.prev_scroll_y = avg_y

            self.current_gesture = "SCROLL"
            direction_label = "Scrolling Up" if scroll_delta > 0 else ("Scrolling Down" if scroll_delta < 0 else "Scroll Ready")
            return GestureResult(
                gesture_name="SCROLL",
                display_text=direction_label,
                cursor_point=cursor_coords,
                scroll_delta_y=scroll_delta,
                fingers_state=fingers
            )

        # Reset scroll tracking when leaving scroll mode
        self.prev_scroll_y = None

        # ---------------------------------------------------------------------
        # Priority 4: Cursor Movement (Index finger extended alone)
        # ---------------------------------------------------------------------
        is_index_moving = (fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0)
        # Also allow index moving even if thumb is loosely relaxed as long as middle/ring/pinky are down
        if is_index_moving or (fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0):
            self.current_gesture = "MOVE"
            return GestureResult(
                gesture_name="MOVE",
                display_text="Moving Cursor",
                cursor_point=cursor_coords,
                fingers_state=fingers
            )

        # ---------------------------------------------------------------------
        # Priority 5: Neutral / Idle State (Open hand, fist, or unmapped pose)
        # ---------------------------------------------------------------------
        self.current_gesture = "IDLE"
        num_extended = sum(fingers)
        state_label = "Neutral (Open Hand)" if num_extended >= 4 else ("Neutral (Fist)" if num_extended == 0 else "Neutral (Idle)")
        return GestureResult(
            gesture_name="IDLE",
            display_text=state_label,
            cursor_point=cursor_coords,
            fingers_state=fingers
        )

