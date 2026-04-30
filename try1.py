from ultralytics import YOLO
import cv2

model = YOLO('yolov8s.pt')
cap = cv2.VideoCapture(0)
Status_screen_past=[]
Status_screen=[]
# Get screen dimensions
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
screen_center_x = frame_width // 2
screen_center_y = frame_height // 2

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Draw center point
    cv2.circle(frame, (screen_center_x, screen_center_y), 6, (0, 0, 255), -1)
    
    results = model.track(frame, persist=True, conf=0.6, classes=[0], verbose=False)[0]
    
    if results.boxes is not None:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Inner rectangle (4x smaller)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            inner_width = (x2 - x1) // 4
            inner_height = (y2 - y1) // 2
            inner_x1 = center_x - inner_width // 2
            inner_y1 = center_y - inner_height // 2
            inner_x2 = center_x + inner_width // 2
            inner_y2 = center_y + inner_height // 2
            
            # Check if center point is inside inner rectangle
            center_in_inner = (
                screen_center_x >= inner_x1 and 
                screen_center_x <= inner_x2 and 
                screen_center_y >= inner_y1 and 
                screen_center_y <= inner_y2
            )
            
            # Colors based on the check
            
            # Draw rectangles
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (inner_x1, inner_y1), (inner_x2, inner_y2), (0, 255, 200), 2)
            
            # Get ID
            track_id = int(box.id) if box.id is not None else "?"
            

            
            # Display info
            status = "True" if center_in_inner else "False"
            
            if status == "True":
                left_right="inside"
            elif inner_x1> screen_center_x:
                left_right="left"
            else:
                left_right="right"
            
            
            
            label = f"ID {track_id}: Center {status} and is {left_right}"
            if box.id != None:
                if int(box.id)> len(Status_screen):
                    Status_screen.append(eval(status))
                else:
                    Status_screen[int(box.id)-1]=eval(status)
                print(Status_screen)
                if Status_screen != Status_screen_past:
                    print(Status_screen)
                    Status_screen_past=Status_screen
            
            cv2.putText(frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2)
    
    cv2.imshow('Center Point Check', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
