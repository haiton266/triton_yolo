import numpy as np
import tritonclient.http as httpclient
from PIL import Image

# Load and preprocess the image
image_path = "/home/hai/code/triton_yolo/images/bus.jpg"
image = Image.open(image_path).resize((640, 640))
image = np.array(image).astype(np.float32) / 255.0
image = np.transpose(image, (2, 0, 1))  # Convert to CHW format
image = np.expand_dims(image, axis=0)  # Add batch dimension

# Initialize Triton client
model_name = "yolo"
client = httpclient.InferenceServerClient(url="localhost:8000")

# Prepare the input and output
inputs = [httpclient.InferInput("images", image.shape, "FP32")]
inputs[0].set_data_from_numpy(image)

outputs = [httpclient.InferRequestedOutput("output0")]

# Perform inference
results = client.infer(model_name, inputs, outputs=outputs)

# Process the output
output_data = results.as_numpy("output0")
print("Output shape:", output_data.shape)
print("Output data:", output_data)