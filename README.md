# Triton YOLO

This repository contains code and assets to deploy and test a YOLO (You Only Look Once) object detection model using NVIDIA Triton Inference Server. The setup includes client and server scripts, ONNX model files, and example Python scripts to facilitate object detection with YOLO models.

## Repository Structure

- **tmp/triton_repo/yolo/**: Directory containing the YOLO model repository for Triton.
  - **1/**
    - **model.onnx**: The YOLO model file in ONNX format, optimized for deployment with Triton.
    - **config.pbtxt**: Configuration file for the YOLO model, specifying input and output parameters for Triton.
- **bus.jpg**: Example image file used for testing object detection.
- **main.py**: Main script to set up and run the YOLO model.
- **print_in4_onnx.py**: Script to print information about the ONNX model, useful for debugging and model inspection.
- **test_client_triton.py**: Client script to connect to Triton Server and perform inference using the YOLO model.
- **triton_server.py**: Script to configure and start Triton Server for serving the YOLO model.
- **use_triton_without_ultralytics.py**: Script for using Triton with YOLO models without relying on the Ultralytics library.
- **yolov8n.pt**: YOLO model file (in PyTorch format), which can be converted to ONNX if needed.

## Requirements

- NVIDIA Triton Inference Server
- Python 3.x
- Required Python libraries (listed in `requirements.txt` if available)
- See tutorials to Docker can see GPU: https://saturncloud.io/blog/how-to-use-gpu-from-a-docker-container-a-guide-for-data-scientists-and-software-engineers/

## Getting Started

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/triton_yolo.git
   cd triton_yolo
   ```

2. **Run Client Script**:

   - Use `triton_server.py` to start server Triton.
   - Use `test_client_triton.py` to send requests to the Triton Server and get predictions.

3. **Developing Scripts**:

   - Run `print_in4_onnx.py` to check the details of the ONNX model.
   - Run `use_triton_without_ultralytics.py` if you prefer not to use the Ultralytics library.
