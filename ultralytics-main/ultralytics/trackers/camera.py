from ultralytics import YOLO

# Load the YOLOv8 model (choose one)
model = YOLO("yolov8n.pt")  # Nano (fastest, least accurate)
# model = YOLO("yolov8s.pt")  # Small
# model = YOLO("yolov8m.pt")  # Medium
# model = YOLO("yolov8l.pt")  # Large
# model = YOLO("yolov8x.pt")  # XLarge (slowest, most accurate)

# Run inference on the webcam (source=0)
results = model(source=0, show=True, conf=0.5, stream=True)  # 'stream' for real-time

# Press 'q' to quit
for r in results:
    pass  # No extra processing needed if just displaying