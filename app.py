from flask import Flask, request, jsonify
from ultralytics import YOLO
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load a small CPU-friendly model. This will auto-download on first run.
model = YOLO("yolov8n.pt")  # you can switch to yolov5s.pt as well

@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}, 200

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file field named 'file' in form-data"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Read image
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # Run inference
    results = model.predict(img, conf=0.25, imgsz=640, verbose=False)

    # Parse results
    detections = []
    for r in results:
        boxes = r.boxes
        for b in boxes:
            xyxy = b.xyxy[0].tolist()  # [x1, y1, x2, y2]
            conf = float(b.conf[0])
            cls_id = int(b.cls[0])
            cls_name = r.names[cls_id]
            detections.append({
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": conf,
                "bbox_xyxy": xyxy
            })

    return jsonify({
        "num_detections": len(detections),
        "detections": detections
    })

@app.route("/", methods=["GET"])
def home():
    return "✅ YOLOv8 Flask API is live!", 200

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files['image']
    results = model(image)  # Run YOLOv8 inference

    detections = []
    for box in results[0].boxes:
        detections.append({
            "class": int(box.cls[0]),
            "confidence": float(box.conf[0]),
            "bbox": box.xyxy[0].tolist()
        })

    return jsonify({"detections": detections})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
