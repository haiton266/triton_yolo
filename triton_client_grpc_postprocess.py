import numpy as np
import tritonclient.grpc as grpcclient
from PIL import Image
import cv2
import matplotlib.pyplot as plt

# Load and preprocess the image
image_path = "/home/hai/code/triton_yolo/images/bus.jpg"
image = Image.open(image_path).resize((640, 640))
image_np = np.array(image).astype(np.float32) / 255.0
image = np.transpose(image_np, (2, 0, 1))  # Convert to CHW format
image = np.expand_dims(image, axis=0)  # Add batch dimension

# Initialize Triton client
model_name = "yolo"
client = grpcclient.InferenceServerClient(url="localhost:8001")

# Prepare the input and output
inputs = [grpcclient.InferInput("images", image.shape, "FP32")]
inputs[0].set_data_from_numpy(image)

outputs = [grpcclient.InferRequestedOutput("output0")]

# Perform inference
results = client.infer(model_name, inputs, outputs=outputs)

# Process the output
output_data = results.as_numpy("output0")
print("Output shape:", output_data.shape)
print("Output data:", output_data)

# Hàm NMS (Non-Maximum Suppression)
def non_max_suppression(boxes, scores, threshold=0.4):
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.5, nms_threshold=threshold)
    return indices.flatten() if len(indices) > 0 else []

# Hàm giải mã output của YOLO
def post_process(output_data, conf_threshold=0.5, iou_threshold=0.4):
    boxes = []
    confidences = []
    class_ids = []
    
    # Duyệt qua các kết quả dự đoán
    for detection in output_data[0]:
        # Kiểm tra các box có confidence lớn hơn threshold
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        
        if confidence > conf_threshold:
            # Tính toán tọa độ box
            center_x, center_y, width, height = detection[0:4]
            left = int((center_x - width / 2) * 640)
            top = int((center_y - height / 2) * 640)
            right = int((center_x + width / 2) * 640)
            bottom = int((center_y + height / 2) * 640)
            
            # Lưu các thông tin của box
            boxes.append([left, top, right, bottom])
            confidences.append(float(confidence))
            class_ids.append(class_id)
    
    # Áp dụng Non-Maximum Suppression (NMS) để loại bỏ các bounding boxes chồng chéo
    indices = non_max_suppression(boxes, confidences, threshold=iou_threshold)
    
    return [(boxes[i], confidences[i], class_ids[i]) for i in indices]

# Sử dụng hàm post-process
post_processed_results = post_process(output_data)

# Vẽ bounding boxes lên ảnh
image_with_boxes = (image_np.copy() * 255).astype(np.uint8)  # Convert back to [0, 255] range for display

for box, confidence, class_id in post_processed_results:
    left, top, right, bottom = box
    # Vẽ bounding box và text
    cv2.rectangle(image_with_boxes, (left, top), (right, bottom), (0, 255, 0), 2)  # Vẽ hình chữ nhật
    label = f"Class {class_id}, Conf {confidence:.2f}"
    cv2.putText(image_with_boxes, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# Hiển thị kết quả
plt.imshow(image_with_boxes)
plt.axis('off')  # Tắt trục
plt.show()
