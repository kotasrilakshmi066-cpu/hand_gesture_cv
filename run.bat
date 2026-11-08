@echo off
title Hand Gesture Controlled Virtual Mouse
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" hand_gesture_virtual_mouse\main.py
) else (
    python hand_gesture_virtual_mouse\main.py
)
pause

