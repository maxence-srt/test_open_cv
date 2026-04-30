from ultralytics import YOLO
import cv2

# 1. Use nano model (much faster)
model = YOLO('yolov8n.pt')

# 2. Open camera with optimized settings
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)   # Linux V4L2 backend
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)    # Lower resolution
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # MJPEG compression
cap.set(cv2.CAP_PROP_FPS, 30)

# Get actual dimensions after settings
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
screen_center_x = frame_width // 2
screen_center_y = frame_height // 2

# Simple status tracking (per frame, no persistent ID list)
frame_count = 0
PROCESS_EVERY_N_FRAMES = 1   # set to 2 to skip every other frame

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % PROCESS_EVERY_N_FRAMES != 0:
        # Skip processing for this frame (just show the raw frame if you want)
        cv2.imshow('Center Point Check', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Draw center point
    cv2.circle(frame, (screen_center_x, screen_center_y), 6, (0, 0, 255), -1)

    # 3. Use predict() instead of track() – faster
    results = model(frame, conf=0.4, classes=[0], verbose=False)  # class 0 = person
    boxes = results[0].boxes

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Inner rectangle (smaller region for "center" check)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            inner_width = (x2 - x1) // 4
            inner_height = (y2 - y1) // 2
            inner_x1 = center_x - inner_width // 2
            inner_y1 = center_y - inner_height // 2
            inner_x2 = center_x + inner_width // 2
            inner_y2 = center_y + inner_height // 2

            # Check if screen center is inside inner rectangle
            center_in_inner = (screen_center_x >= inner_x1 and screen_center_x <= inner_x2 and
                               screen_center_y >= inner_y1 and screen_center_y <= inner_y2)

            # Determine left/right relative to inner rectangle
            if center_in_inner:
                left_right = "inside"
            elif inner_x1 > screen_center_x:
                left_right = "left"
            else:
                left_right = "right"

            # Colors
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (inner_x1, inner_y1), (inner_x2, inner_y2), (0, 255, 200), 2)

            # Label (no tracking ID to save CPU)
            label = f"Person: {left_right}"
            cv2.putText(frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2)

    # Show the frame
    cv2.imshow('Center Point Check (RPi Optimized)', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
