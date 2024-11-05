import numpy as np
from PIL import Image
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput

# Define your Triton server URL
TRITON_SERVER_URL = "localhost:8000"

# Initialize the Triton client
client = InferenceServerClient(url=TRITON_SERVER_URL, verbose=False)

# Model and input settings
model_name = "yolo"
model_version = "1"  # Use the specific model version if required
input_name = "images"  # As specified in the config
output_name = "output0"  # As specified in the config

# Function to preprocess an image
def preprocess_image(image_path, target_size=(640, 640)):
    # Open image, resize, and normalize (example preprocessing, adjust as needed)
    image = Image.open(image_path).convert("RGB")
    image = image.resize(target_size)
    image = np.array(image).astype(np.float32)
    
    # Normalize the image (0-1 range) if required by the model
    image = image / 255.0

    # Change shape from (H, W, C) to (C, H, W) for models expecting NCHW format
    image = np.transpose(image, (2, 0, 1))
    
    # Add a batch dimension to the image (1, C, H, W)
    image = np.expand_dims(image, axis=0)
    
    return image

# Load and preprocess images
image1 = preprocess_image("bus.jpg")
image2 = preprocess_image("img.jpg")

# Concatenate images along the batch dimension
batch_data = np.concatenate([image1, image2])  # Shape will be (2, 3, 640, 640)

# Create the InferInput object with the batched input data
infer_input = InferInput(input_name, batch_data.shape, "FP32")
infer_input.set_data_from_numpy(batch_data)

# Specify the output you want to receive
output = InferRequestedOutput(output_name)

# Send the request to the Triton server
response = client.infer(
    model_name=model_name,
    model_version=model_version,
    inputs=[infer_input],
    outputs=[output]
)

# Retrieve the results
result = response.as_numpy(output_name)
print("Inference results for the batch:")
print(result)
