import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

# ---------- Settings ----------
MODEL_PATH = "detect.tflite"
CONFIDENCE_THRESHOLD = 0.5
CAMERA_INDEX = 0
PERSON_CLASS_ID = 0
# ------------------------------

def load_model(model_path):
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def detect_persons(interpreter, image, threshold):
    """Run inference and return list of (ymin, xmin, ymax, xmax, score) for persons."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_height, input_width = input_details[0]['shape'][1:3]

    img_resized = cv2.resize(image, (input_width, input_height))
    input_data = np.expand_dims(img_resized, axis=0)

    if input_details[0]['dtype'] == np.uint8:
        input_data = input_data.astype(np.uint8)
    else:
        input_data = (input_data.astype(np.float32) / 127.5) - 1.0

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    persons = []
    for i in range(len(scores)):
        if scores[i] >= threshold and int(classes[i]) == PERSON_CLASS_ID:
            persons.append((boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3], scores[i]))
    return persons

def shrink_box(ymin, xmin, ymax, xmax, factor=0.5):
    """Return a new box half the size, centred at the same point."""
    center_x = (xmin + xmax) / 2.0
    center_y = (ymin + ymax) / 2.0
    new_w = (xmax - xmin) * factor
    new_h = (ymax - ymin) * factor
    return (center_y - new_h/2, center_x - new_w/2,
            center_y + new_h/2, center_x + new_w/2)

def main():
    print("Loading model...")
    interpreter = load_model(MODEL_PATH)
    print("Model loaded.")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        persons = detect_persons(interpreter, frame, CONFIDENCE_THRESHOLD)

        # ----------  NEW: keep only the person with highest confidence ----------
        if persons:
            best_person = max(persons, key=lambda p: p[4])   # p[4] is the score
            persons = [best_person]
        # ------------------------------------------------------------------------

        img_h, img_w = frame.shape[:2]
        for (ymin, xmin, ymax, xmax, score) in persons:
            # ---------- NEW: shrink the box to half size ----------
            new_ymin, new_xmin, new_ymax, new_xmax = shrink_box(ymin, xmin, ymax, xmax, factor=0.5)

            # Convert to pixel coordinates
            x1 = int(new_xmin * img_w)
            x2 = int(new_xmax * img_w)
            y1 = int(new_ymin * img_h)
            y2 = int(new_ymax * img_h)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"person {score:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # --------------------------------------------------------

        cv2.imshow("Person Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
