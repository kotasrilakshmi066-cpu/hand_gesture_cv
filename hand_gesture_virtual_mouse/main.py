"""
=============================================================================
Project: Hand Gesture Controlled Virtual Mouse
Module: main.py
Author: Member 4 - Application Interface, Testing, Integration & Architecture
Description:
    Primary desktop application entry point. Features a professional dark-themed
    Tkinter dashboard with embedded real-time OpenCV webcam preview, HUD telemetry,
    dynamic sensitivity/smoothing sliders, safety failsafes, and gesture guides.
=============================================================================
"""

import cv2
import time
import math
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from typing import Optional

# Local project modules
import config
from hand_tracking import HandDetector
from gesture_recognition import GestureRecognizer, GestureResult
from mouse_controller import MouseController


class VirtualMouseApp:
    """
    Main application controller managing GUI layout, webcam capture loop,
    telemetry diagnostics, and integration between Computer Vision modules.
    """

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Hand Gesture Controlled Virtual Mouse - CV Project")
        self.root.geometry("1180x760")
        self.root.minsize(1050, 700)
        self.root.configure(bg=config.THEME["bg_dark"])

        # Configure window icon if available
        try:
            self.root.iconbitmap(default="")
        except Exception:
            pass

        # Core CV and Hardware Controllers
        self.detector = HandDetector(
            mode=config.STATIC_IMAGE_MODE,
            max_hands=config.MAX_NUM_HANDS,
            detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            tracking_confidence=config.MIN_TRACKING_CONFIDENCE
        )
        self.recognizer = GestureRecognizer()
        self.controller = MouseController()

        # Camera state
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_camera_running: bool = False
        self.camera_device_index: int = config.CAMERA_INDEX

        # Telemetry & Performance Metrics
        self.prev_frame_time: float = 0.0
        self.fps: float = 0.0
        self.current_gesture_display: str = "Standby"
        self.last_action_feedback: str = "Awaiting hand detection"
        self.action_feedback_color: str = config.THEME["text_secondary"]

        # UI state variables
        self.var_mouse_enabled = tk.BooleanVar(value=False)
        self.var_draw_landmarks = tk.BooleanVar(value=True)
        self.var_draw_roi = tk.BooleanVar(value=True)
        self.var_sensitivity = tk.DoubleVar(value=config.DEFAULT_SENSITIVITY)
        self.var_smoothing = tk.DoubleVar(value=config.DEFAULT_SMOOTHING)

        # Build interface components
        self._setup_styles()
        self._build_ui()

        # Automatically start the camera stream upon launch
        self.root.after(400, self.start_camera)

        # Handle window closure gracefully
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _setup_styles(self):
        """Configure custom TTK styles matching the dark professional palette."""
        self.style = ttk.Style()
        self.style.theme_use("clam")

        # Configure general styles
        self.style.configure(".", background=config.THEME["bg_dark"], foreground=config.THEME["text_primary"])
        self.style.configure(
            "TFrame",
            background=config.THEME["bg_dark"]
        )
        self.style.configure(
            "Card.TFrame",
            background=config.THEME["bg_card"],
            relief="flat"
        )
        self.style.configure(
            "Header.TLabel",
            background=config.THEME["bg_dark"],
            foreground=config.THEME["text_primary"],
            font=(config.THEME["font_family"], 18, "bold")
        )
        self.style.configure(
            "Subheader.TLabel",
            background=config.THEME["bg_dark"],
            foreground=config.THEME["text_secondary"],
            font=(config.THEME["font_family"], 10)
        )
        self.style.configure(
            "CardTitle.TLabel",
            background=config.THEME["bg_card"],
            foreground=config.THEME["accent_cyan"],
            font=(config.THEME["font_family"], 11, "bold")
        )
        self.style.configure(
            "CardText.TLabel",
            background=config.THEME["bg_card"],
            foreground=config.THEME["text_primary"],
            font=(config.THEME["font_family"], 9)
        )
        self.style.configure(
            "CardSub.TLabel",
            background=config.THEME["bg_card"],
            foreground=config.THEME["text_secondary"],
            font=(config.THEME["font_family"], 8)
        )
        self.style.configure(
            "Status.TLabel",
            background=config.THEME["bg_card_alt"],
            foreground=config.THEME["text_primary"],
            font=(config.THEME["font_mono"], 10, "bold"),
            padding=6
        )

    def _build_ui(self):
        """Construct the responsive grid layout dashboard."""
        # Top Header Bar
        header_frame = ttk.Frame(self.root, padding="15 12 15 8")
        header_frame.pack(side=tk.TOP, fill=tk.X)

        title_box = ttk.Frame(header_frame)
        title_box.pack(side=tk.LEFT)
        lbl_title = ttk.Label(
            title_box,
            text="HAND GESTURE CONTROLLED VIRTUAL MOUSE",
            style="Header.TLabel"
        )
        lbl_title.pack(anchor=tk.W)

        lbl_subtitle = ttk.Label(
            title_box,
            text="Real-Time Computer Vision HCI System | MediaPipe Hands + OpenCV + PyAutoGUI",
            style="Subheader.TLabel"
        )
        lbl_subtitle.pack(anchor=tk.W)

        # Team / Academic Badge on Top Right
        badge_frame = ttk.Frame(header_frame)
        badge_frame.pack(side=tk.RIGHT, pady=4)
        lbl_badge = tk.Label(
            badge_frame,
            text=" ACADEMIC PROJECT v1.0 ",
            bg=config.THEME["bg_card_alt"],
            fg=config.THEME["accent_cyan"],
            font=(config.THEME["font_mono"], 9, "bold"),
            relief="groove",
            bd=1,
            padx=8,
            pady=4
        )
        lbl_badge.pack()

        # Separator Line
        sep = tk.Frame(self.root, height=1, bg="#2C324B")
        sep.pack(fill=tk.X, padx=15, pady=4)

        # Main Content Split (Left: Webcam Viewport | Right: Controls & Diagnostics)
        content_frame = ttk.Frame(self.root, padding="15 10 15 15")
        content_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # =====================================================================
        # LEFT PANEL: WEBCAM FEED & VIDEO HUD
        # =====================================================================
        left_panel = ttk.Frame(content_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Viewport Card Container
        video_card = ttk.Frame(left_panel, style="Card.TFrame", padding="8")
        video_card.pack(fill=tk.BOTH, expand=True)

        # Video Canvas / Label
        self.video_label = tk.Label(
            video_card,
            bg="#000000",
            relief="sunken",
            bd=1,
            text="Camera feed offline.\n\nClick 'Start Camera' below to launch the video stream.",
            fg=config.THEME["text_secondary"],
            font=(config.THEME["font_family"], 11)
        )
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # Video HUD Status Bar directly below video
        video_hud_bar = tk.Frame(video_card, bg=config.THEME["bg_card_alt"], height=34, padx=10, pady=4)
        video_hud_bar.pack(fill=tk.X, pady=(6, 0))

        self.lbl_hud_fps = tk.Label(
            video_hud_bar,
            text="FPS: --",
            bg=config.THEME["bg_card_alt"],
            fg=config.THEME["accent_cyan"],
            font=(config.THEME["font_mono"], 9, "bold")
        )
        self.lbl_hud_fps.pack(side=tk.LEFT, padx=5)

        self.lbl_hud_res = tk.Label(
            video_hud_bar,
            text=f"Resolution: {config.FRAME_WIDTH}x{config.FRAME_HEIGHT}",
            bg=config.THEME["bg_card_alt"],
            fg=config.THEME["text_secondary"],
            font=(config.THEME["font_mono"], 9)
        )
        self.lbl_hud_res.pack(side=tk.LEFT, padx=15)

        self.lbl_hud_status = tk.Label(
            video_hud_bar,
            text="CAMERA OFFLINE",
            bg=config.THEME["bg_card_alt"],
            fg=config.THEME["status_red"],
            font=(config.THEME["font_mono"], 9, "bold")
        )
        self.lbl_hud_status.pack(side=tk.RIGHT, padx=5)

        # =====================================================================
        # RIGHT PANEL: CONTROL DASHBOARD & TELEMETRY
        # =====================================================================
        right_panel = ttk.Frame(content_frame, width=380)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False)
        right_panel.pack_propagate(False)

        # Card 1: Camera & Mouse Primary Controls
        control_card = ttk.Frame(right_panel, style="Card.TFrame", padding="12")
        control_card.pack(fill=tk.X, pady=(0, 10))

        lbl_c1_title = ttk.Label(control_card, text="SYSTEM CONTROLS", style="CardTitle.TLabel")
        lbl_c1_title.pack(anchor=tk.W, pady=(0, 8))

        btn_grid = tk.Frame(control_card, bg=config.THEME["bg_card"])
        btn_grid.pack(fill=tk.X, pady=2)

        self.btn_start = tk.Button(
            btn_grid,
            text="▶ Start Camera",
            bg="#00C853",
            fg="#FFFFFF",
            activebackground="#00E676",
            font=(config.THEME["font_family"], 10, "bold"),
            relief="flat",
            cursor="hand2",
            padx=12,
            pady=6,
            command=self.start_camera
        )
        self.btn_start.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        self.btn_stop = tk.Button(
            btn_grid,
            text="⏹ Stop Camera",
            bg="#D50000",
            fg="#FFFFFF",
            activebackground="#FF1744",
            font=(config.THEME["font_family"], 10, "bold"),
            relief="flat",
            cursor="hand2",
            state=tk.DISABLED,
            padx=12,
            pady=6,
            command=self.stop_camera
        )
        self.btn_stop.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(5, 0))

        # Mouse Control Enable Toggle
        toggle_frame = tk.Frame(control_card, bg=config.THEME["bg_card"], pady=8)
        toggle_frame.pack(fill=tk.X)

        self.chk_mouse = tk.Checkbutton(
            toggle_frame,
            text="Enable Virtual Mouse Control",
            variable=self.var_mouse_enabled,
            onvalue=True,
            offvalue=False,
            bg=config.THEME["bg_card"],
            fg=config.THEME["text_primary"],
            selectcolor=config.THEME["bg_dark"],
            activebackground=config.THEME["bg_card"],
            activeforeground=config.THEME["accent_cyan"],
            font=(config.THEME["font_family"], 10, "bold"),
            cursor="hand2",
            command=self._on_mouse_toggle
        )
        self.chk_mouse.pack(anchor=tk.W)

        # Emergency Stop Button
        self.btn_emergency = tk.Button(
            control_card,
            text="⚠ EMERGENCY STOP (Disable Mouse)",
            bg="#FF3366",
            fg="#FFFFFF",
            activebackground="#FF5252",
            font=(config.THEME["font_family"], 9, "bold"),
            relief="flat",
            cursor="hand2",
            pady=6,
            command=self.emergency_stop
        )
        self.btn_emergency.pack(fill=tk.X, pady=(6, 0))

        # Card 2: Real-Time Telemetry & Gesture Detection
        telemetry_card = ttk.Frame(right_panel, style="Card.TFrame", padding="12")
        telemetry_card.pack(fill=tk.X, pady=(0, 10))

        lbl_c2_title = ttk.Label(telemetry_card, text="LIVE RECOGNITION TELEMETRY", style="CardTitle.TLabel")
        lbl_c2_title.pack(anchor=tk.W, pady=(0, 6))

        # Current Gesture Display Badge
        self.lbl_gesture_badge = tk.Label(
            telemetry_card,
            text="STANDBY",
            bg=config.THEME["bg_card_alt"],
            fg=config.THEME["text_secondary"],
            font=(config.THEME["font_family"], 12, "bold"),
            relief="ridge",
            bd=1,
            pady=8
        )
        self.lbl_gesture_badge.pack(fill=tk.X, pady=(0, 6))

        # Telemetry detail readouts
        tele_grid = tk.Frame(telemetry_card, bg=config.THEME["bg_card"])
        tele_grid.pack(fill=tk.X)

        lbl_pos_title = tk.Label(tele_grid, text="Screen Cursor Target:", bg=config.THEME["bg_card"], fg=config.THEME["text_secondary"], font=(config.THEME["font_family"], 8))
        lbl_pos_title.grid(row=0, column=0, sticky=tk.W, pady=1)
        self.lbl_pos_val = tk.Label(tele_grid, text="X: 0 | Y: 0", bg=config.THEME["bg_card"], fg=config.THEME["accent_cyan"], font=(config.THEME["font_mono"], 8, "bold"))
        self.lbl_pos_val.grid(row=0, column=1, sticky=tk.E, pady=1)

        lbl_fingers_title = tk.Label(tele_grid, text="Fingers State [T,I,M,R,P]:", bg=config.THEME["bg_card"], fg=config.THEME["text_secondary"], font=(config.THEME["font_family"], 8))
        lbl_fingers_title.grid(row=1, column=0, sticky=tk.W, pady=1)
        self.lbl_fingers_val = tk.Label(tele_grid, text="[0, 0, 0, 0, 0]", bg=config.THEME["bg_card"], fg=config.THEME["text_primary"], font=(config.THEME["font_mono"], 8))
        self.lbl_fingers_val.grid(row=1, column=1, sticky=tk.E, pady=1)

        # Card 3: Calibration Tuning (Sensitivity & Smoothing)
        tuning_card = ttk.Frame(right_panel, style="Card.TFrame", padding="12")
        tuning_card.pack(fill=tk.X, pady=(0, 10))

        lbl_c3_title = ttk.Label(tuning_card, text="TUNING & CALIBRATION", style="CardTitle.TLabel")
        lbl_c3_title.pack(anchor=tk.W, pady=(0, 6))

        # Sensitivity Slider
        sens_row = tk.Frame(tuning_card, bg=config.THEME["bg_card"])
        sens_row.pack(fill=tk.X)
        tk.Label(sens_row, text="Sensitivity Multiplier:", bg=config.THEME["bg_card"], fg=config.THEME["text_primary"], font=(config.THEME["font_family"], 8, "bold")).pack(side=tk.LEFT)
        self.lbl_sens_val = tk.Label(sens_row, text=f"{config.DEFAULT_SENSITIVITY:.1f}x", bg=config.THEME["bg_card"], fg=config.THEME["accent_cyan"], font=(config.THEME["font_mono"], 8, "bold"))
        self.lbl_sens_val.pack(side=tk.RIGHT)

        slider_sens = ttk.Scale(
            tuning_card,
            from_=config.MIN_SENSITIVITY,
            to=config.MAX_SENSITIVITY,
            variable=self.var_sensitivity,
            orient=tk.HORIZONTAL,
            command=self._on_tuning_changed
        )
        slider_sens.pack(fill=tk.X, pady=(2, 6))

        # Smoothing Slider
        smooth_row = tk.Frame(tuning_card, bg=config.THEME["bg_card"])
        smooth_row.pack(fill=tk.X)
        tk.Label(smooth_row, text="Cursor Smoothing Factor:", bg=config.THEME["bg_card"], fg=config.THEME["text_primary"], font=(config.THEME["font_family"], 8, "bold")).pack(side=tk.LEFT)
        self.lbl_smooth_val = tk.Label(smooth_row, text=f"{config.DEFAULT_SMOOTHING:.0f}", bg=config.THEME["bg_card"], fg=config.THEME["accent_cyan"], font=(config.THEME["font_mono"], 8, "bold"))
        self.lbl_smooth_val.pack(side=tk.RIGHT)

        slider_smooth = ttk.Scale(
            tuning_card,
            from_=config.MIN_SMOOTHING,
            to=config.MAX_SMOOTHING,
            variable=self.var_smoothing,
            orient=tk.HORIZONTAL,
            command=self._on_tuning_changed
        )
        slider_smooth.pack(fill=tk.X, pady=(2, 6))

        # Visual overlay checkboxes
        overlay_box = tk.Frame(tuning_card, bg=config.THEME["bg_card"])
        overlay_box.pack(fill=tk.X, pady=(2, 0))
        tk.Checkbutton(
            overlay_box,
            text="Skeleton Landmarks",
            variable=self.var_draw_landmarks,
            bg=config.THEME["bg_card"],
            fg=config.THEME["text_secondary"],
            selectcolor=config.THEME["bg_dark"],
            font=(config.THEME["font_family"], 8)
        ).pack(side=tk.LEFT)
        tk.Checkbutton(
            overlay_box,
            text="Active Zone Box",
            variable=self.var_draw_roi,
            bg=config.THEME["bg_card"],
            fg=config.THEME["text_secondary"],
            selectcolor=config.THEME["bg_dark"],
            font=(config.THEME["font_family"], 8)
        ).pack(side=tk.RIGHT)

        # Card 4: Gesture Guide / Quick Reference
        guide_card = ttk.Frame(right_panel, style="Card.TFrame", padding="12")
        guide_card.pack(fill=tk.BOTH, expand=True)

        lbl_c4_title = ttk.Label(guide_card, text="GESTURE QUICK REFERENCE", style="CardTitle.TLabel")
        lbl_c4_title.pack(anchor=tk.W, pady=(0, 6))

        gestures_guide = [
            ("☝ Index Up", "Move cursor smoothly across screen"),
            ("🤏 Thumb + Index Pinch", "Trigger Left Click (Debounced)"),
            ("👌 Thumb + Middle Pinch", "Trigger Right Click (Context Menu)"),
            ("✌ Two Fingers Up & Move", "Scroll Up / Down by hand motion"),
            ("✋ Open Hand / Fist", "Neutral Idle (No action triggered)"),
            ("🛑 Screen Corner / Esc", "PyAutoGUI Safety Failsafe Trip")
        ]

        for icon_name, desc in gestures_guide:
            row = tk.Frame(guide_card, bg=config.THEME["bg_card"], pady=2)
            row.pack(fill=tk.X)
            tk.Label(row, text=icon_name, bg=config.THEME["bg_card"], fg=config.THEME["accent_cyan"], font=(config.THEME["font_family"], 8, "bold"), width=22, anchor=tk.W).pack(side=tk.LEFT)
            tk.Label(row, text=desc, bg=config.THEME["bg_card"], fg=config.THEME["text_secondary"], font=(config.THEME["font_family"], 8), anchor=tk.W).pack(side=tk.LEFT, fill=tk.X, expand=True)

    def _on_mouse_toggle(self):
        """Callback when Virtual Mouse control checkbox is toggled."""
        enabled = self.var_mouse_enabled.get()
        self.controller.set_enabled(enabled)
        if enabled:
            self.lbl_hud_status.config(text="MOUSE CONTROL ACTIVE", fg=config.THEME["status_green"])
        else:
            self.lbl_hud_status.config(text="CAMERA ONLY (MOUSE IDLE)", fg=config.THEME["status_amber"])

    def _on_tuning_changed(self, _event=None):
        """Update controller smoothing and sensitivity dynamically."""
        sens = round(self.var_sensitivity.get(), 1)
        smooth = round(self.var_smoothing.get(), 1)
        self.lbl_sens_val.config(text=f"{sens:.1f}x")
        self.lbl_smooth_val.config(text=f"{smooth:.0f}")
        self.controller.update_settings(smoothing=smooth, sensitivity=sens)

    def emergency_stop(self):
        """Immediately disable virtual mouse control and trigger visual alert."""
        self.var_mouse_enabled.set(False)
        self.controller.emergency_stop()
        self.lbl_hud_status.config(text="EMERGENCY STOPPED", fg=config.THEME["status_red"])
        self.lbl_gesture_badge.config(
            text="EMERGENCY STOP",
            bg=config.THEME["status_red"],
            fg="#FFFFFF"
        )

    def start_camera(self):
        """Initialize OpenCV webcam device and start the frame loop."""
        if self.is_camera_running:
            return

        try:
            # Multi-index and multi-backend webcam detector
            opened = False
            for idx in [self.camera_device_index, 0, 1, 2]:
                for backend in [cv2.CAP_DSHOW, None]:
                    try:
                        if backend is not None:
                            self.cap = cv2.VideoCapture(idx, backend)
                        else:
                            self.cap = cv2.VideoCapture(idx)
                        if self.cap and self.cap.isOpened():
                            ret, test_frame = self.cap.read()
                            if ret and test_frame is not None:
                                self.camera_device_index = idx
                                opened = True
                                break
                            else:
                                self.cap.release()
                    except Exception:
                        pass
                if opened:
                    break

            if not opened or not self.cap or not self.cap.isOpened():
                messagebox.showerror(
                    "Camera Error",
                    "Unable to access webcam.\n\n"
                    "Please verify:\n"
                    "1. Another app (Teams, Zoom, browser) is not currently locking the webcam.\n"
                    "2. Windows Camera Privacy Settings allow desktop apps to access the camera.\n"
                    "3. Your webcam is plugged in."
                )
                self.cap = None
                return

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, config.TARGET_FPS)

            self.is_camera_running = True
            self.btn_start.config(state=tk.DISABLED, bg="#333333")
            self.btn_stop.config(state=tk.NORMAL, bg="#D50000")
            self.lbl_hud_status.config(text="CAMERA ONLINE", fg=config.THEME["status_green"])

            # Begin video polling loop
            self.update_frame()

        except Exception as e:
            messagebox.showerror("Initialization Error", f"Failed to initialize camera:\n{str(e)}")
            self.is_camera_running = False

    def stop_camera(self):
        """Safely release webcam stream and reset UI viewport."""
        self.is_camera_running = False
        self.controller.set_enabled(False)
        self.var_mouse_enabled.set(False)

        if self.cap:
            self.cap.release()
            self.cap = None

        self.btn_start.config(state=tk.NORMAL, bg="#00C853")
        self.btn_stop.config(state=tk.DISABLED, bg="#333333")
        self.lbl_hud_status.config(text="CAMERA OFFLINE", fg=config.THEME["status_red"])
        self.lbl_hud_fps.config(text="FPS: --")
        self.lbl_gesture_badge.config(
            text="STANDBY",
            bg=config.THEME["bg_card_alt"],
            fg=config.THEME["text_secondary"]
        )
        self.video_label.config(
            image="",
            text="Camera stream stopped.\nClick 'Start Camera' to restart."
        )

    def update_frame(self):
        """
        Primary Computer Vision Processing Pipeline:
        1. Capture frame from OpenCV.
        2. Horizontal flip for mirror view.
        3. MediaPipe hand landmark detection.
        4. Geometric gesture classification.
        5. Coordinate mapping & mouse action dispatching.
        6. In-frame visual HUD rendering.
        7. Presentation onto Tkinter canvas.
        """
        if not self.is_camera_running or self.cap is None:
            return

        success, frame = self.cap.read()
        if not success or frame is None:
            # Temporary frame drop; retry shortly
            self.root.after(15, self.update_frame)
            return

        # Step 1: Flip frame horizontally for intuitive mirror orientation
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Step 2: Calculate instantaneous FPS
        current_time = time.time()
        if self.prev_frame_time > 0:
            instant_fps = 1.0 / max(1e-5, current_time - self.prev_frame_time)
            self.fps = 0.9 * self.fps + 0.1 * instant_fps  # Smoothed FPS
        self.prev_frame_time = current_time

        # Step 3: Landmark detection
        draw_landmarks = self.var_draw_landmarks.get()
        frame = self.detector.find_hands(frame, draw=draw_landmarks)
        lm_list, bbox = self.detector.find_positions(frame, draw=False)
        fingers = self.detector.fingers_up()

        # Step 4: Gesture Classification
        gesture_res: GestureResult = self.recognizer.classify(lm_list, fingers)

        # Step 5: Mouse Action Execution
        cursor_x_screen, cursor_y_screen = 0, 0
        if gesture_res.gesture_name == "MOVE" and gesture_res.cursor_point:
            ix, iy = gesture_res.cursor_point
            cursor_x_screen, cursor_y_screen = self.controller.move_cursor(ix, iy)
            # Highlight index tip with cyan radar ring
            cv2.circle(frame, (ix, iy), 10, (255, 229, 0), 2)
            cv2.circle(frame, (ix, iy), 4, (0, 229, 255), cv2.FILLED)

        elif gesture_res.gesture_name == "LEFT_CLICK":
            if gesture_res.cursor_point:
                ix, iy = gesture_res.cursor_point
                cursor_x_screen, cursor_y_screen = self.controller.move_cursor(ix, iy)
            fired = self.controller.left_click()
            if gesture_res.pinch_point:
                px, py = gesture_res.pinch_point
                # Ripple animation circle for click feedback
                cv2.circle(frame, (px, py), 16, (0, 255, 0), 3)
                cv2.putText(frame, "CLICK", (px + 12, py - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        elif gesture_res.gesture_name == "RIGHT_CLICK":
            if gesture_res.cursor_point:
                ix, iy = gesture_res.cursor_point
                cursor_x_screen, cursor_y_screen = self.controller.move_cursor(ix, iy)
            fired = self.controller.right_click()
            if gesture_res.pinch_point:
                px, py = gesture_res.pinch_point
                cv2.circle(frame, (px, py), 16, (255, 0, 255), 3)
                cv2.putText(frame, "RIGHT CLICK", (px + 12, py - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        elif gesture_res.gesture_name == "SCROLL":
            if gesture_res.scroll_delta_y != 0:
                self.controller.scroll(gesture_res.scroll_delta_y)
            if gesture_res.cursor_point:
                ix, iy = gesture_res.cursor_point
                dir_symbol = "^" if gesture_res.scroll_delta_y > 0 else "v"
                cv2.putText(frame, f"SCROLL {dir_symbol}", (ix - 30, iy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 229, 255), 2)

        # Check for PyAutoGUI Failsafe trip
        if self.controller.failsafe_triggered:
            self.emergency_stop()
            messagebox.showwarning(
                "Failsafe Triggered",
                "PyAutoGUI Safety Failsafe was triggered because the cursor reached a screen corner.\n"
                "Mouse control has been safely disabled."
            )

        # Step 6: Render In-Frame Active Tracking Zone (ROI Box)
        if self.var_draw_roi.get():
            rx1 = config.FRAME_REDUCTION_X
            ry1 = config.FRAME_REDUCTION_Y
            rx2 = w - config.FRAME_REDUCTION_X
            ry2 = h - config.FRAME_REDUCTION_Y
            # Draw semi-transparent or subtle neon border
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 120, 0), 2)
            cv2.putText(
                frame,
                "Active Interaction Zone",
                (rx1 + 8, ry1 + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 120, 0),
                1,
                cv2.LINE_AA
            )

        # Step 7: In-Frame HUD Banner
        hud_bg = np.zeros((40, w, 3), dtype=np.uint8)
        # Overlay top banner
        cv2.rectangle(frame, (0, 0), (w, 36), (20, 22, 32), cv2.FILLED)
        cv2.putText(
            frame,
            f"FPS: {int(self.fps)}",
            (15, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 229, 255),
            2,
            cv2.LINE_AA
        )
        gesture_color = (0, 255, 128) if gesture_res.gesture_name in ("MOVE", "LEFT_CLICK", "RIGHT_CLICK", "SCROLL") else (180, 180, 180)
        cv2.putText(
            frame,
            f"Gesture: {gesture_res.display_text}",
            (130, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            gesture_color,
            2,
            cv2.LINE_AA
        )

        mouse_badge = "MOUSE: ACTIVE" if self.controller.is_enabled else "MOUSE: OFF"
        mouse_color = (0, 255, 0) if self.controller.is_enabled else (0, 0, 255)
        cv2.putText(
            frame,
            mouse_badge,
            (w - 180, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            mouse_color,
            2,
            cv2.LINE_AA
        )

        # Step 8: Update Tkinter GUI Diagnostics
        self.lbl_hud_fps.config(text=f"FPS: {int(self.fps)}")
        self.lbl_gesture_badge.config(
            text=gesture_res.display_text.upper(),
            bg=config.THEME["bg_card_alt"],
            fg=config.THEME["status_green"] if gesture_res.gesture_name != "NO_HAND" else config.THEME["status_amber"]
        )
        self.lbl_pos_val.config(text=f"X: {cursor_x_screen} | Y: {cursor_y_screen}")
        self.lbl_fingers_val.config(text=str(fingers))

        # Step 9: Render onto Tkinter Viewport
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)

        # Responsive scaling to fit video viewport smoothly
        view_w = self.video_label.winfo_width()
        view_h = self.video_label.winfo_height()

        if view_w > 100 and view_h > 100:
            # Preserve aspect ratio
            aspect = config.FRAME_WIDTH / config.FRAME_HEIGHT
            target_w = view_w
            target_h = int(target_w / aspect)
            if target_h > view_h:
                target_h = view_h
                target_w = int(target_h * aspect)
            img_pil = img_pil.resize((target_w, target_h), Image.Resampling.BILINEAR)

        img_tk = ImageTk.PhotoImage(image=img_pil)
        self.video_label.img_tk = img_tk  # Keep reference to prevent GC
        self.video_label.config(image=img_tk, text="")

        # Schedule next frame at ~30 FPS (approx 15ms-30ms)
        self.root.after(15, self.update_frame)

    def on_closing(self):
        """Resource cleanup on application exit."""
        self.stop_camera()
        self.detector.release()
        self.root.destroy()


def main():
    """Application bootstrap function."""
    root = tk.Tk()
    app_instance = VirtualMouseApp(root)
    root.mainloop()


# =============================================================================
# Vercel Serverless & Cloud Web Hosting Entrypoint
# Allows repository to deploy cleanly on Vercel as an interactive web showcase
# while preserving full native Tkinter desktop functionality when run locally.
# =============================================================================
import os

def app(environ, start_response):
    """WSGI application entrypoint for Vercel and cloud platforms."""
    status = '200 OK'
    headers = [('Content-type', 'text/html; charset=utf-8')]
    start_response(status, headers)
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    if not os.path.exists(html_path):
        html_path = os.path.join(os.path.dirname(__file__), "..", "index.html")
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception:
        content = "<html><body><h1>Hand Gesture Virtual Mouse</h1><p>Please run locally via python main.py or run.bat for webcam mouse control.</p></body></html>"
    return [content.encode('utf-8')]

application = app
handler = app


if __name__ == "__main__":
    main()

