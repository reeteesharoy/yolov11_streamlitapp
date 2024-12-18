import streamlit as st
from yolo_code import load_model, train_model  # Import functions from yolo_code.py

# Streamlit app interface for user input
st.title("YOLOv11 Model Training")

# Set the default values as specified
model_path = "C:/yolo11/yolo11n.pt"
data_path = "C:/yolo11/data.yaml"
epochs = 2
img_size = 640
device = "cpu"

# Display and let users see these as default inputs
model_path = st.text_input("Enter the model path (e.g., yolo11n.pt):", model_path)
data_path = st.text_input("Enter the data.yaml path:", data_path)
epochs = st.number_input("Enter the number of epochs:", min_value=1, max_value=100, value=epochs)
img_size = st.number_input("Enter the image size:", min_value=320, max_value=1280, value=img_size)
device = st.selectbox("Choose the device:", ["cpu", "cuda"], index=["cpu", "cuda"].index(device))

# Display the chosen settings for confirmation
st.write("Training will proceed with the following settings:")
st.write(f"Model Path: {model_path}")
st.write(f"Data Path: {data_path}")
st.write(f"Epochs: {epochs}")
st.write(f"Image Size: {img_size}")
st.write(f"Device: {device}")

# Button to start training
if st.button("Start Training"):
    # Load the YOLO model
    st.write("Loading the YOLO model...")
    model = load_model(model_path)
    
    # Train the model
    st.write("Starting model training...")
    train_results = train_model(model, data_path, epochs, img_size, device)
    
    # Display training results
    st.write("Training completed successfully!")
    st.write("Training Results:", train_results)
