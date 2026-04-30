from ultralytics import YOLO
import cv2

# Load the smallest YOLO model for speed
# Download the model if needed: pip install ultralytics
model = YOLO('yolov8n.pt')

# Start the camera
cap = cv2.VideoCapture(0)

while True:
    # Grab a single frame
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection with a low confidence threshold
    # Lowering 'conf' helps find people, even with lower image quality
    results = model(frame, conf=0.4, classes=[0], verbose=False)

    # Process the results
    for r in results:
        boxes = r.boxes
        if boxes is not None:
            for box in boxes:
                # Get the bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Draw a green rectangle around the person
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Show the annotated frame
    cv2.imshow('Human Detection', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
