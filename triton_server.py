import contextlib
import subprocess
import time
from pathlib import Path
from tritonclient.http import InferenceServerClient

# Define image https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver
tag = "nvcr.io/nvidia/tritonserver:23.09-py3"  # 6.4 GB

# Pull the image, only run once
# subprocess.call(f"docker pull {tag}", shell=True)

# Use absolute path for triton_repo_path
triton_repo_path = Path("tmp") / "triton_repo"
absolute_triton_repo_path = triton_repo_path.resolve()  # Convert to absolute path
model_name = "yolo"
triton_model_path = absolute_triton_repo_path / model_name

# Run the Triton server and capture the container ID
container_id = (
    subprocess.check_output(
        f"docker run --rm --gpus all -v {absolute_triton_repo_path}:/models "
        f"-p 8000:8000 -p 8001:8001 {tag} "
        f"tritonserver --model-repository=/models --http-port=8000 --grpc-port=8001",
        shell=True,
    )
    .decode("utf-8")
    .strip()
)



# Wait for the Triton server to start
triton_client = InferenceServerClient(url="localhost:8000", verbose=False, ssl=False)

# Wait until model is ready
for _ in range(10):
    with contextlib.suppress(Exception):
        assert triton_client.is_model_ready(model_name)
        break
    time.sleep(1)

# To turn off
#  docker ps
# docker stop <container_id>
