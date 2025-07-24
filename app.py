from flask import Flask, request, jsonify
from ultralytics import YOLO
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load model
model = YOLO("yolov8n.pt")  # Lightweight model for fast inference

@app.route("/", methods=["GET"])
def home():
    return "✅ YOLOv8 Flask API is live!", 200

@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}, 200

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded under 'file' field"}), 400

    file = request.files['file']
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Read and convert image
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # Inference
    results = model.predict(img, conf=0.25, imgsz=640, verbose=False)

    # Parse detections
    detections = []
    for r in results:
        for b in r.boxes:
            detections.append({
                "class_id": int(b.cls[0]),
                "class_name": r.names[int(b.cls[0])],
                "confidence": float(b.conf[0]),
                "bbox_xyxy": b.xyxy[0].tolist()
            })

    return jsonify({
        "num_detections": len(detections),
        "detections": detections
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
