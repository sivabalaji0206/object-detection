from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import io

app = Flask(__name__)

# Load model (YOLOv8 nano version)
model = YOLO("yolov8n.pt")

@app.route("/", methods=["GET"])
def home():
    return "✅ YOLOv8 Flask API is live!", 200

@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}, 200

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files and 'image' not in request.files:
        return jsonify({"error": "No image uploaded (use field name 'file' or 'image')"}), 400

    image_file = request.files.get('file') or request.files.get('image')
    
    if image_file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        img_bytes = image_file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Failed to read image: {str(e)}"}), 400

    # Run inference
    results = model.predict(img, conf=0.25, imgsz=640, verbose=False)

    # Parse results
    detections = []
    for r in results:
        for b in r.boxes:
            xyxy = b.xyxy[0].tolist()
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
