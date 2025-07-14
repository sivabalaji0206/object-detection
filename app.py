import cv2
import numpy as np

# Define paths to the model configuration and weights
CONFIG_FILE = r"D:\College\Projects\Object Detection\yolov3.cfg"
WEIGHTS_FILE = r"D:\College\Projects\Object Detection\yolov3.weights"
LABELS_FILE = r"D:\College\Projects\Object Detection\coco.names"

# Load the YOLO model
net = cv2.dnn.readNetFromDarknet(CONFIG_FILE, WEIGHTS_FILE)

# Get the names of all layers
layer_names = net.getLayerNames()

# Get the indices of the output layers
output_layers_indices = net.getUnconnectedOutLayers()

# OpenCV returns 1-based indexing, so we subtract 1 for 0-based indexing
output_layers = [layer_names[i - 1] for i in output_layers_indices]

# Load class labels from coco.names file
with open(LABELS_FILE, "r") as f:
    class_labels = f.read().strip().split("\n")

# Set up webcam (or use another camera)
webcam = cv2.VideoCapture(0)

while True:
    # Read frame from webcam
    ret, frame = webcam.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Get the shape of the frame
    height, width, channels = frame.shape

    # Convert the frame to a blob (this is what YOLO expects)
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    # Set the blob as input to the network
    net.setInput(blob)

    # Run forward pass to get output of the layers
    outputs = net.forward(output_layers)

    # Initialize lists to hold detected objects
    class_ids = []
    confidences = []
    boxes = []

    # Process each output layer
    for out in outputs:
        for detection in out:
            scores = detection[5:]  # the first 5 are for the object box
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # If the confidence is above a threshold, we proceed
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Get the top-left corner of the bounding box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Append the results
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # Apply non-maximum suppression to remove overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw the bounding boxes and labels
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(class_labels[class_ids[i]])
            confidence = str(round(confidences[i], 2))

            # Draw rectangle around the object
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Object Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
webcam.release()
cv2.destroyAllWindows()
