from ultralytics import YOLO

# Function to load the YOLO model
def load_model(model_path):
    model = YOLO(model_path)
    return model

# Function to train the YOLO model
def train_model(model, data_path, epochs=10, img_size=640, device="cpu"):
    results = model.train(
        data=data_path,
        epochs=epochs,
        imgsz=img_size,
        device=device
    )
    return results
