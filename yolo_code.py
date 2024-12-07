from ultralytics import YOLO

# Load the model
model = YOLO("yolo11n.pt")

# Train the model using CPU
train_results = model.train(
    data="C:/yolo11/data.yaml",  # Use a raw string or double backslashes
    epochs=2,
    imgsz=640,
    device='cpu'  # Specify 'cpu' for CPU training
)