from ultralytics import YOLO

# Load the Triton Server model
model = YOLO("http://localhost:8000/yolo", task="detect")

# Run inference on the server
results = model(["images/bus.jpg", "images/img.jpg"], save = True)