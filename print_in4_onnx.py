import onnx

# Load the ONNX model
model = onnx.load("/home/hai/code/yolo_project/tmp/triton_repo/yolo/1/model.onnx")

# Retrieve input information
input_info = [
    {
        "name": input.name,
        "shape": [dim.dim_value if (dim.dim_value != 0) else 'dynamic' for dim in input.type.tensor_type.shape.dim],
        "data_type": input.type.tensor_type.elem_type
    }
    for input in model.graph.input
]

# Retrieve output information
output_info = [
    {
        "name": output.name,
        "shape": [dim.dim_value if (dim.dim_value != 0) else 'dynamic' for dim in output.type.tensor_type.shape.dim],
        "data_type": output.type.tensor_type.elem_type
    }
    for output in model.graph.output
]

# Display input and output information
print("Inputs:")
for info in input_info:
    print(f"Name: {info['name']}, Shape: {info['shape']}, Data Type: {info['data_type']}")

print("\nOutputs:")
for info in output_info:
    print(f"Name: {info['name']}, Shape: {info['shape']}, Data Type: {info['data_type']}")
