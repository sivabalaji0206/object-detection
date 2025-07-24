import os
import cv2
import numpy as np
import requests

# ========== 1. Download yolov3.weights from Google Drive if not exists ==========
FILE_ID = "115b1K2j7DGb7UXEVUrPlAoZn9fsmIA3H"
FILE_NAME = "yolov3.weights"

def download_from_google_drive(file_id, destination):
    print("Downloading yolov3.weights from Google Drive...")
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={'id': file_id}, stream=True)
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

    print(f"{destination} downloaded successfully.")

if not os.path.exists(FILE_NAME):
    download_from_google_drive(FILE_ID, FILE_NAME)

# ========== 2. Paths ==========
CONFIG_FILE = "yolov3.cfg"
WEIGHTS_FILE = "yolov3.weights"
LABELS_FILE = "coco.names"

# ========== 3. Load Model ==========
net = cv2.dnn.readNetFromDarknet(CONFIG_FILE, WEIGHTS_FILE)
layer_names = net.getLayerNames()
output_layers_indices = net.getUnconnectedOutLayers()
output_layers = [layer_names[i - 1] for i in output_layers_indices]

with open(LABELS_FILE, "r") as f:
    class_labels = f.read().strip().split("\n")

# ========== 4. Open Webcam ==========
webcam = cv2.VideoCapture(0)

while True:
    ret, frame = webcam.read()
    if not ret:
        print("Failed to grab frame")
        break

    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outputs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(class_labels[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
