import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO('yolov8n.pt')

# Open the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read frame.")
        break
    
    # Predict using the YOLO model
    results = model.predict(frame)
    
    # Visualize the results on the frame
    annotated_frame = results[0].plot()
    annotated_frame = cv2.resize(annotated_frame, (1080, 720))
    # Display the frame
    cv2.imshow("Webcam Feed", annotated_frame)
    
    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
