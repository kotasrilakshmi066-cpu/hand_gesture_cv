"""
=============================================================================
Project: Hand Gesture Controlled Virtual Mouse
Module: test_suite.py
Author: Computer Vision Academic Project Team
Description:
    Comprehensive verification and unit test suite covering:
    1. MediaPipe hand tracking pipeline & landmark extraction.
    2. Geometric distance calculation and finger state detection.
    3. Rule-based gesture recognition classification accuracy.
    4. Screen coordinate mapping, smoothing, sensitivity, and debouncing.
    5. Operating system mouse events and safety failsafe tripping.
    6. Camera availability and frame acquisition.
=============================================================================
"""

import sys
import os
import time
import math
import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from hand_tracking import HandDetector
from gesture_recognition import GestureRecognizer, GestureResult
from mouse_controller import MouseController


def test_hand_detector_initialization():
    print("[TEST 1/8] Testing HandDetector initialization and cleanup...")
    detector = HandDetector(mode=False, max_hands=1)
    assert detector.hands is not None, "MediaPipe Hands was not initialized!"
    
    # Process a synthetic blank frame
    dummy_frame = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)
    processed = detector.find_hands(dummy_frame)
    assert processed is not None, "find_hands returned None!"
    
    lm_list, bbox = detector.find_positions(dummy_frame)
    assert isinstance(lm_list, list), "lm_list should be a list!"
    assert len(lm_list) == 0, "Empty frame should detect 0 landmarks!"
    assert detector.is_hand_present() is False, "Empty frame should report no hand present!"
    
    detector.release()
    print("  -> PASSED: HandDetector initializes, processes blank frame, and releases resources cleanly.")


def test_geometric_distance():
    print("[TEST 2/8] Testing geometric Euclidean distance calculation...")
    detector = HandDetector()
    # Mock landmark list with known coordinates: p1 at (100, 100), p2 at (130, 140)
    # Expected Euclidean distance: sqrt(30^2 + 40^2) = 50.0
    detector.lm_list = [[i, 0, 0, 0] for i in range(21)]
    detector.lm_list[4] = [4, 100, 100, 0]
    detector.lm_list[8] = [8, 130, 140, 0]
    
    dist, _, midpoint = detector.find_distance(4, 8, frame=None, draw=False)
    assert math.isclose(dist, 50.0, rel_tol=1e-3), f"Expected distance 50.0, got {dist}"
    assert midpoint == (115, 120), f"Expected midpoint (115, 120), got {midpoint}"
    print(f"  -> PASSED: Euclidean distance calculation accurate: {dist} px, midpoint {midpoint}.")


def test_gesture_classification():
    print("[TEST 3/8] Testing GestureRecognizer classification logic...")
    recognizer = GestureRecognizer()

    # Case A: No hand
    res_no_hand = recognizer.classify([], [0, 0, 0, 0, 0])
    assert res_no_hand.gesture_name == "NO_HAND", f"Failed: expected NO_HAND, got {res_no_hand.gesture_name}"

    # Setup base landmarks (all coordinates default far apart)
    mock_lms = [[i, i * 20 + 50, i * 15 + 50, 0] for i in range(21)]

    # Case B: Cursor Move Gesture (Only Index finger is UP)
    fingers_move = [0, 1, 0, 0, 0]
    mock_lms[8] = [8, 300, 200, 0]   # Index tip
    mock_lms[4] = [4, 150, 300, 0]   # Thumb tip (far from index)
    mock_lms[12] = [12, 350, 400, 0] # Middle tip (folded)
    res_move = recognizer.classify(mock_lms, fingers_move)
    assert res_move.gesture_name == "MOVE", f"Failed: expected MOVE, got {res_move.gesture_name}"
    assert res_move.cursor_point == (300, 200), f"Expected cursor point (300, 200), got {res_move.cursor_point}"

    # Case C: Left Click (Thumb Tip 4 and Index Tip 8 pinched close)
    mock_lms[4] = [4, 305, 205, 0]   # ~7 pixels distance from Index tip (300, 200)
    mock_lms[12] = [12, 450, 400, 0] # Middle tip far away
    res_left = recognizer.classify(mock_lms, [1, 1, 0, 0, 0])
    assert res_left.gesture_name == "LEFT_CLICK", f"Failed: expected LEFT_CLICK, got {res_left.gesture_name}"
    assert res_left.pinch_distance < config.LEFT_CLICK_DISTANCE

    # Case D: Right Click (Thumb Tip 4 touches Middle Tip 12 while Index is far)
    mock_lms[4] = [4, 452, 403, 0]   # ~4 pixels from Middle tip (450, 400)
    mock_lms[8] = [8, 200, 100, 0]   # Index tip far away (>100px)
    mock_lms[12] = [12, 450, 400, 0]
    res_right = recognizer.classify(mock_lms, [1, 0, 1, 0, 0])
    assert res_right.gesture_name == "RIGHT_CLICK", f"Failed: expected RIGHT_CLICK, got {res_right.gesture_name}"

    # Case E: Scroll Gesture (Index & Middle fingers both UP and adjacent)
    mock_lms[8] = [8, 300, 200, 0]
    mock_lms[12] = [12, 320, 205, 0] # ~20 px from index tip (< scroll proximity)
    mock_lms[4] = [4, 100, 300, 0]   # Thumb away
    res_scroll1 = recognizer.classify(mock_lms, [0, 1, 1, 0, 0])
    assert res_scroll1.gesture_name == "SCROLL", f"Failed: expected SCROLL, got {res_scroll1.gesture_name}"
    
    # Simulate upward motion (y decreases)
    mock_lms[8] = [8, 300, 170, 0]
    mock_lms[12] = [12, 320, 175, 0]
    res_scroll_up = recognizer.classify(mock_lms, [0, 1, 1, 0, 0])
    assert res_scroll_up.scroll_delta_y == 1, f"Expected scroll_delta_y = 1 (Up), got {res_scroll_up.scroll_delta_y}"

    # Case F: Neutral / Open Hand
    res_idle = recognizer.classify(mock_lms, [1, 1, 1, 1, 1])
    assert res_idle.gesture_name == "IDLE", f"Failed: expected IDLE, got {res_idle.gesture_name}"

    print("  -> PASSED: All gesture classifications (MOVE, LEFT_CLICK, RIGHT_CLICK, SCROLL, IDLE, NO_HAND) verified.")


def test_mouse_controller_mapping_and_smoothing():
    print("[TEST 4/8] Testing MouseController coordinate interpolation and smoothing...")
    controller = MouseController()
    
    # Verify screen size detection
    sw, sh = controller.screen_width, controller.screen_height
    assert sw > 0 and sh > 0, f"Invalid screen size: {sw}x{sh}"
    
    # Test coordinate mapping: center of frame active zone -> center of screen
    cam_center_x = config.FRAME_WIDTH // 2
    cam_center_y = config.FRAME_HEIGHT // 2
    mapped_x, mapped_y = controller.map_coordinates(cam_center_x, cam_center_y)
    
    assert math.isclose(mapped_x, sw / 2, abs_tol=10.0), f"Center mapping X mismatch: {mapped_x} vs {sw/2}"
    assert math.isclose(mapped_y, sh / 2, abs_tol=10.0), f"Center mapping Y mismatch: {mapped_y} vs {sh/2}"
    
    # Test smoothing filter without enabling OS dispatch
    controller.set_enabled(False)
    controller.prev_x = 0.0
    controller.prev_y = 0.0
    controller.smoothing = 5.0
    
    # Step move
    out_x, out_y = controller.move_cursor(cam_center_x, cam_center_y)
    expected_step_x = 0.0 + (mapped_x - 0.0) / 5.0
    assert math.isclose(out_x, expected_step_x, abs_tol=2.0), f"Smoothing mismatch: {out_x} vs {expected_step_x}"
    
    print(f"  -> PASSED: Screen mapping ({sw}x{sh}) and smoothing exponential interpolation verified.")


def test_click_debouncing():
    print("[TEST 5/8] Testing click debounce protection...")
    controller = MouseController()
    controller.set_enabled(False) # Test timer logic without firing physical OS clicks
    
    # First click attempt
    controller.last_left_click_time = 0.0
    # Left click should succeed
    res1 = controller.left_click()
    assert controller.last_left_click_time > 0, "Timestamp was not recorded!"
    
    # Rapid successive attempt within 50ms (well below 350ms cooldown)
    res2 = controller.left_click()
    assert res2 is False, "Debounce failed: Second rapid click was not blocked!"
    
    # Right click debounce test
    controller.last_right_click_time = 0.0
    controller.right_click()
    res_right_debounce = controller.right_click()
    assert res_right_debounce is False, "Right click debounce failed to block rapid click!"
    
    print("  -> PASSED: Left and Right click debouncing correctly prevents accidental duplicate clicks.")


def test_safety_and_emergency_stop():
    print("[TEST 6/8] Testing safety failsafe and emergency stop...")
    controller = MouseController()
    controller.set_enabled(True)
    assert controller.is_enabled is True
    
    # Trigger emergency stop
    controller.emergency_stop()
    assert controller.is_enabled is False, "Emergency stop failed to disable mouse control!"
    assert controller.failsafe_triggered is True, "Emergency stop should flag failsafe!"
    
    # Verify no movements occur when disabled
    st = controller.get_status()
    assert st["enabled"] is False
    print("  -> PASSED: Emergency stop immediately cuts mouse interaction.")


def test_camera_detection():
    print("[TEST 7/8] Testing physical / virtual webcam access...")
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if cap.isOpened():
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None:
            print(f"  -> PASSED: Physical webcam detected! Frame shape: {frame.shape}")
        else:
            print("  -> WARNING: Camera opened but returned no frame (busy or virtual device). Handled gracefully.")
    else:
        print(f"  -> NOTICE: No physical webcam at index {config.CAMERA_INDEX}. Application UI gracefully alerts the user.")


def test_full_pipeline_synthetic_simulation():
    print("[TEST 8/8] Testing end-to-end processing pipeline on synthetic image...")
    detector = HandDetector(mode=True)
    recognizer = GestureRecognizer()
    controller = MouseController()
    
    # Create synthetic test canvas with white background
    test_img = np.full((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), 240, dtype=np.uint8)
    
    # Process through pipeline
    processed = detector.find_hands(test_img, draw=True)
    lms, bbox = detector.find_positions(processed)
    fingers = detector.fingers_up()
    result = recognizer.classify(lms, fingers)
    
    assert result.gesture_name == "NO_HAND"
    assert result.display_text == "No Hand Detected"
    
    detector.release()
    print("  -> PASSED: Full end-to-end pipeline executes without runtime exceptions.")


def run_all_tests():
    print("=" * 72)
    print("  HAND GESTURE CONTROLLED VIRTUAL MOUSE - COMPREHENSIVE TEST SUITE")
    print("=" * 72)
    start_time = time.time()
    
    test_hand_detector_initialization()
    test_geometric_distance()
    test_gesture_classification()
    test_mouse_controller_mapping_and_smoothing()
    test_click_debouncing()
    test_safety_and_emergency_stop()
    test_camera_detection()
    test_full_pipeline_synthetic_simulation()
    
    elapsed = time.time() - start_time
    print("=" * 72)
    print(f"  ALL TESTS PASSED SUCCESSFULLY in {elapsed:.3f} seconds!")
    print("=" * 72)


if __name__ == "__main__":
    run_all_tests()

