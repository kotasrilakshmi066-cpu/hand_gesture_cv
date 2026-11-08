"""
=============================================================================
Project: Hand Gesture Controlled Virtual Mouse
Module: mouse_controller.py
Author: Member 3 - Mouse Movement, Clicking, and Scrolling Specialist
Description:
    Hardware interaction module wrapping PyAutoGUI. Handles coordinate mapping
    from webcam space to screen resolution, multi-stage smoothing, jitter
    suppression, debounced clicks, directional scrolling, and safety failsafes.
=============================================================================
"""

import time
import math
import numpy as np
import pyautogui
from typing import Tuple, Optional
import config


class MouseController:
    """
    Translates computer vision coordinate signals and classified gestures
    into low-latency, jitter-free operating system mouse events.
    """

    def __init__(
        self,
        frame_width: int = config.FRAME_WIDTH,
        frame_height: int = config.FRAME_HEIGHT,
        frame_reduction_x: int = config.FRAME_REDUCTION_X,
        frame_reduction_y: int = config.FRAME_REDUCTION_Y,
        smoothing: float = config.DEFAULT_SMOOTHING,
        sensitivity: float = config.DEFAULT_SENSITIVITY
    ):
        """
        Initialize mouse controller with screen dimensions and calibration constants.

        :param frame_width: Webcam frame width in pixels
        :param frame_height: Webcam frame height in pixels
        :param frame_reduction_x: Left/Right margin for active tracking zone
        :param frame_reduction_y: Top/Bottom margin for active tracking zone
        :param smoothing: Initial smoothing divisor (higher = smoother)
        :param sensitivity: Dynamic sensitivity multiplier
        """
        # Configure PyAutoGUI for real-time gaming/interactive responsiveness
        pyautogui.FAILSAFE = config.PYAUTOGUI_FAILSAFE
        pyautogui.PAUSE = config.PAUSE_BETWEEN_COMMANDS

        # Retrieve host operating system display dimensions
        self.screen_width, self.screen_height = pyautogui.size()

        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frame_reduction_x = frame_reduction_x
        self.frame_reduction_y = frame_reduction_y

        self.smoothing = smoothing
        self.sensitivity = sensitivity

        # Tracking state
        self.prev_x: float = self.screen_width / 2.0
        self.prev_y: float = self.screen_height / 2.0
        self.curr_x: float = self.prev_x
        self.curr_y: float = self.prev_y

        # Debounce and rate-limiting timers
        self.last_left_click_time: float = 0.0
        self.last_right_click_time: float = 0.0
        self.last_scroll_time: float = 0.0

        # Safety control flags
        self.is_enabled: bool = False
        self.failsafe_triggered: bool = False

    def set_enabled(self, enabled: bool):
        """Enable or disable OS mouse dispatching."""
        self.is_enabled = enabled
        if not enabled:
            self.failsafe_triggered = False

    def update_settings(self, smoothing: Optional[float] = None, sensitivity: Optional[float] = None):
        """Dynamically update smoothing and sensitivity from UI sliders."""
        if smoothing is not None:
            self.smoothing = max(config.MIN_SMOOTHING, min(config.MAX_SMOOTHING, smoothing))
        if sensitivity is not None:
            self.sensitivity = max(config.MIN_SENSITIVITY, min(config.MAX_SENSITIVITY, sensitivity))

    def map_coordinates(self, cam_x: int, cam_y: int) -> Tuple[float, float]:
        """
        Map 2D webcam coordinates inside active bounding box to screen space.
        Applies sensitivity scaling centered at screen midpoint.

        :param cam_x: Raw x-coordinate in webcam frame
        :param cam_y: Raw y-coordinate in webcam frame
        :return: (target_screen_x, target_screen_y) clamped to monitor boundaries
        """
        x_min = self.frame_reduction_x
        x_max = self.frame_width - self.frame_reduction_x
        y_min = self.frame_reduction_y
        y_max = self.frame_height - self.frame_reduction_y

        # Linear interpolation from active webcam region to full screen resolution
        raw_screen_x = np.interp(cam_x, (x_min, x_max), (0, self.screen_width))
        raw_screen_y = np.interp(cam_y, (y_min, y_max), (0, self.screen_height))

        # Apply sensitivity scaling relative to screen center
        center_x = self.screen_width / 2.0
        center_y = self.screen_height / 2.0

        scaled_x = center_x + (raw_screen_x - center_x) * self.sensitivity
        scaled_y = center_y + (raw_screen_y - center_y) * self.sensitivity

        # Clamp strictly within screen resolution
        target_x = float(np.clip(scaled_x, 0, self.screen_width - 1))
        target_y = float(np.clip(scaled_y, 0, self.screen_height - 1))

        return target_x, target_y

    def move_cursor(self, cam_x: int, cam_y: int) -> Tuple[int, int]:
        """
        Smoothly interpolate cursor position and dispatch move event to OS.

        :param cam_x: Index fingertip x-coordinate
        :param cam_y: Index fingertip y-coordinate
        :return: (actual_x, actual_y) dispatched position
        """
        target_x, target_y = self.map_coordinates(cam_x, cam_y)

        # Apply exponential moving average (smoothing)
        self.curr_x = self.prev_x + (target_x - self.prev_x) / self.smoothing
        self.curr_y = self.prev_y + (target_y - self.prev_y) / self.smoothing

        # Jitter suppression (deadband filter): ignore micro-tremors
        delta_dist = math.hypot(self.curr_x - self.prev_x, self.curr_y - self.prev_y)
        if delta_dist < config.JITTER_THRESHOLD:
            # Revert to previous position to maintain rock-solid stillness
            self.curr_x = self.prev_x
            self.curr_y = self.prev_y

        out_x = int(self.curr_x)
        out_y = int(self.curr_y)

        if self.is_enabled:
            try:
                pyautogui.moveTo(out_x, out_y)
            except pyautogui.FailSafeException:
                self.failsafe_triggered = True
                self.is_enabled = False
            except Exception as e:
                # Catch any unexpected OS display driver interruptions gracefully
                pass

        self.prev_x = self.curr_x
        self.prev_y = self.curr_y

        return out_x, out_y

    def left_click(self) -> bool:
        """
        Execute debounced left mouse click.

        :return: True if click was fired, False if blocked by cooldown debounce
        """
        now = time.time()
        if now - self.last_left_click_time >= config.CLICK_COOLDOWN:
            self.last_left_click_time = now
            if self.is_enabled:
                try:
                    pyautogui.click(button='left')
                    return True
                except pyautogui.FailSafeException:
                    self.failsafe_triggered = True
                    self.is_enabled = False
                except Exception:
                    pass
        return False

    def right_click(self) -> bool:
        """
        Execute debounced right mouse click.

        :return: True if right click was fired, False if blocked by cooldown debounce
        """
        now = time.time()
        if now - self.last_right_click_time >= config.RIGHT_CLICK_COOLDOWN:
            self.last_right_click_time = now
            if self.is_enabled:
                try:
                    pyautogui.click(button='right')
                    return True
                except pyautogui.FailSafeException:
                    self.failsafe_triggered = True
                    self.is_enabled = False
                except Exception:
                    pass
        return False

    def scroll(self, direction: int):
        """
        Execute directional scrolling.

        :param direction: +1 for Up, -1 for Down
        """
        if direction == 0 or not self.is_enabled:
            return

        now = time.time()
        # Limit scroll frequency to 30 Hz for fluid feel
        if now - self.last_scroll_time >= 0.03:
            self.last_scroll_time = now
            scroll_amount = direction * config.SCROLL_SPEED
            try:
                pyautogui.scroll(scroll_amount)
            except pyautogui.FailSafeException:
                self.failsafe_triggered = True
                self.is_enabled = False
            except Exception:
                pass

    def emergency_stop(self):
        """Instant safety abort: disables control and resets cursor state."""
        self.is_enabled = False
        self.failsafe_triggered = True

    def get_status(self) -> dict:
        """Return diagnostic dictionary of current controller state."""
        return {
            "enabled": self.is_enabled,
            "failsafe_triggered": self.failsafe_triggered,
            "screen_width": self.screen_width,
            "screen_height": self.screen_height,
            "cursor_pos": (int(self.curr_x), int(self.curr_y)),
            "smoothing": self.smoothing,
            "sensitivity": self.sensitivity,
        }

