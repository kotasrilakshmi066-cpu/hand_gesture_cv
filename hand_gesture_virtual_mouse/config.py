"""
=============================================================================
Project: Hand Gesture Controlled Virtual Mouse
Module: config.py
Author: Computer Vision Academic Project Team
Description:
    Centralized configuration file defining camera parameters, MediaPipe
    hyperparameters, gesture distance thresholds, coordinate mapping regions,
    smoothing factors, safety limits, and UI visual theme definitions.
=============================================================================
"""

import os

# =============================================================================
# 1. CAMERA CONFIGURATION
# =============================================================================
CAMERA_INDEX = 0             # Default primary webcam index
FRAME_WIDTH = 640            # Target capture frame width (pixels)
FRAME_HEIGHT = 480           # Target capture frame height (pixels)
TARGET_FPS = 30              # Target processing frame rate

# =============================================================================
# 2. MEDIAPIPE HAND TRACKING HYPERPARAMETERS
# =============================================================================
STATIC_IMAGE_MODE = False        # Video streaming mode (temporal tracking)
MAX_NUM_HANDS = 1                # Track single primary hand to prevent multi-cursor conflicts
MIN_DETECTION_CONFIDENCE = 0.70  # Hand detection model confidence threshold
MIN_TRACKING_CONFIDENCE = 0.60   # Landmark tracking model confidence threshold

# =============================================================================
# 3. COORDINATE MAPPING & ACTIVE BOUNDARY (ROI)
# =============================================================================
# Margin reduction (in pixels) applied to the webcam frame.
# This creates an active interaction box so the user doesn't need to stretch
# their hand to the physical limits of the webcam view to reach screen corners.
FRAME_REDUCTION_X = 100          # Horizontal margin on left and right
FRAME_REDUCTION_Y = 80           # Vertical margin on top and bottom

# =============================================================================
# 4. MOUSE MOVEMENT & SMOOTHING DYNAMICS
# =============================================================================
DEFAULT_SMOOTHING = 5.0          # Smoothing factor (higher = smoother, lower = snappier)
MIN_SMOOTHING = 1.0              # Direct raw tracking
MAX_SMOOTHING = 15.0             # Heavily damped tracking

DEFAULT_SENSITIVITY = 1.2        # Speed multiplier applied to cursor delta
MIN_SENSITIVITY = 0.5            # Precision fine-grained mode
MAX_SENSITIVITY = 2.5            # Fast sweep mode

JITTER_THRESHOLD = 3             # Minimal pixel change required to issue mouse move (noise gate)

# =============================================================================
# 5. GESTURE RECOGNITION THRESHOLDS & TIMERS
# =============================================================================
# Euclidean distance thresholds (in frame pixel space) for finger pinches
LEFT_CLICK_DISTANCE = 35         # Thumb tip (4) to Index tip (8) distance threshold
RIGHT_CLICK_DISTANCE = 38        # Thumb tip (4) to Middle tip (12) distance threshold
DOUBLE_CLICK_DISTANCE = 32       # Close pinch for double-click confirmation
SCROLL_FINGER_PROXIMITY = 45     # Maximum distance between Index (8) and Middle (12) for scroll mode

# Debounce cooldowns (seconds) to prevent accidental repeated clicks
CLICK_COOLDOWN = 0.35            # Minimum interval between consecutive left clicks
RIGHT_CLICK_COOLDOWN = 0.50      # Minimum interval between consecutive right clicks
DOUBLE_CLICK_INTERVAL = 0.40     # Maximum time window between two pinches to count as double click

# Scrolling sensitivity
SCROLL_SPEED = 20                # PyAutoGUI scroll units per movement tick
SCROLL_DEADZONE = 8              # Minimum vertical movement threshold before scrolling fires

# =============================================================================
# 6. SAFETY & FAILSAFE CONFIGURATION
# =============================================================================
PYAUTOGUI_FAILSAFE = True        # Moving mouse to any corner of screen aborts PyAutoGUI
EMERGENCY_STOP_ENABLED = True    # Global safety flag toggleable from UI
PAUSE_BETWEEN_COMMANDS = 0.001   # Minimal PyAutoGUI pause (removes default 0.1s delay for real-time responsiveness)

# =============================================================================
# 7. UI THEME & DESIGN TOKENS (DARK PROFESSIONAL THEME)
# =============================================================================
THEME = {
    "bg_dark": "#0F111A",        # Deep obsidian background
    "bg_card": "#1A1D2B",        # Surface card container
    "bg_card_alt": "#222638",    # Elevated card / widget container
    "text_primary": "#FFFFFF",   # High-contrast primary header and text
    "text_secondary": "#9EABC0", # Subtext, descriptors, and units
    "accent_cyan": "#00E5FF",    # Cyber cyan for active indicators & primary highlights
    "accent_purple": "#7C4DFF",  # Deep purple for secondary accents
    "status_green": "#00E676",   # Active / Online / Connected indicator
    "status_red": "#FF3366",     # Inactive / Emergency Stop / Error
    "status_amber": "#FFB300",   # Standby / Calibrating / Tracking lost
    "font_family": "Segoe UI",   # Clean modern Windows system font
    "font_mono": "Consolas",     # Monospace font for telemetry readouts
}

# Landmark IDs for quick reference (MediaPipe Hands 21 landmarks)
LANDMARKS = {
    "WRIST": 0,
    "THUMB_CMC": 1,
    "THUMB_MCP": 2,
    "THUMB_IP": 3,
    "THUMB_TIP": 4,
    "INDEX_FINGER_MCP": 5,
    "INDEX_FINGER_PIP": 6,
    "INDEX_FINGER_DIP": 7,
    "INDEX_FINGER_TIP": 8,
    "MIDDLE_FINGER_MCP": 9,
    "MIDDLE_FINGER_PIP": 10,
    "MIDDLE_FINGER_DIP": 11,
    "MIDDLE_FINGER_TIP": 12,
    "RING_FINGER_MCP": 13,
    "RING_FINGER_PIP": 14,
    "RING_FINGER_DIP": 15,
    "RING_FINGER_TIP": 16,
    "PINKY_MCP": 17,
    "PINKY_PIP": 18,
    "PINKY_DIP": 19,
    "PINKY_TIP": 20,
}

