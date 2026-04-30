import cv2
import subprocess

# 1. Check camera devices using system command
print("--- Checking connected cameras (v4l2-ctl) ---")
try:
    result = subprocess.run(['v4l2-ctl', '--list-devices'], capture_output=True, text=True)
    print(result.stdout if result.stdout else "No devices found. Is v4l2-ctl installed?")
except FileNotFoundError:
    print("v4l2-ctl not found. Please install with: sudo apt install v4l-utils")

print("\n--- Testing OpenCV VideoCapture indices (0 to 5) ---")
for i in range(6):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"OpenCV can open camera index {i}")
        cap.release()
    else:
        print(f"OpenCV cannot open camera index {i}")
