sudo apt update
sudo apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev -y
curl https://pyenv.run | bash

echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc

exec $SHELL

pyenv install 3.11.9

pyenv shell 3.11.9        # temporarily use Python 3.11 in this terminal
python -m venv person_detector_env
source person_detector_env/bin/activate

python --version  

pip install --upgrade pip
pip install tflite-runtime numpy opencv-python 

wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
unzip coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
python person_detector.py



source person_detector_env/bin/activate



import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

# ---------- Settings ----------
MODEL_PATH = "detect.tflite"
CONFIDENCE_THRESHOLD = 0.5
CAMERA_INDEX = 0                # 0 = first webcam, change if needed
PERSON_CLASS_ID = 0             # COCO class ID for 'person'
# ------------------------------

def load_model(model_path):
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def detect_persons(interpreter, image, threshold):
    """Run inference and return list of (ymin, xmin, ymax, xmax, score) for persons."""
    # Get model input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_height, input_width = input_details[0]['shape'][1:3]

    # Preprocess image: resize and add batch dimension
    img_resized = cv2.resize(image, (input_width, input_height))
    input_data = np.expand_dims(img_resized, axis=0)

    # Handle model input type (quantized = uint8, float = float32)
    if input_details[0]['dtype'] == np.uint8:
        input_data = input_data.astype(np.uint8)
    else:
        input_data = (input_data.astype(np.float32) / 127.5) - 1.0

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[0]['index'])[0]   # [N, 4]
    classes = interpreter.get_tensor(output_details[1]['index'])[0] # [N]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # [N]

    persons = []
    for i in range(len(scores)):
        if scores[i] >= threshold and int(classes[i]) == PERSON_CLASS_ID:
            persons.append((boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3], scores[i]))
    return persons

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

        # Detect persons in the frame
        persons = detect_persons(interpreter, frame, CONFIDENCE_THRESHOLD)

        # Draw bounding boxes
        img_h, img_w = frame.shape[:2]
        for (ymin, xmin, ymax, xmax, score) in persons:
            xmin = int(xmin * img_w)
            xmax = int(xmax * img_w)
            ymin = int(ymin * img_h)
            ymax = int(ymax * img_h)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, f"person {score:.2f}", (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Person Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
