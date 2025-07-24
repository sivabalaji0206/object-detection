import os
import cv2
import numpy as np
import requests

# File info
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
else:
    print(f"{FILE_NAME} already exists.")

# Paths to model files (must be in same directory)
CONFIG_FILE = "yolov3.cfg"
LABELS_FILE = "coco.names"

# Load YOLO
net = cv2.dnn.readNet("yolov3.onnx")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

with open(LABELS_FILE, "r") as f:
    class_labels = f.read().strip().split("\n")

# Web server for deployment
from flask import Flask, Response

app = Flask(__name__)

@app.route('/')
def index():
    return "YOLOv3 Flask App is running!"

@app.route('/detect')
def detect():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return "Camera failed", 500

    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x, center_y, w, h = (detection[0:4] * [width, height, width, height]).astype("int")
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = f"{class_labels[class_ids[i]]} {confidences[i]:.2f}"
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    _, img_encoded = cv2.imencode('.jpg', frame)
    return Response(img_encoded.tobytes(), mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
