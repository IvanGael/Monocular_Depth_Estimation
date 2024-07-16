import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# Initialize the Depth Estimation model
model_name = "Intel/dpt-hybrid-midas"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForDepthEstimation.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Initialize video capture
cap = cv2.VideoCapture("video.mp4")  # Use 0 for default camera

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    inputs = processor(images=frame, return_tensors="pt").to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # Normalize depth map for visualization
    depth_map = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=frame.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze().cpu().numpy()
    
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    depth_map = (depth_map * 255).astype(np.uint8)

    # Display the frame and depth map
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title("RGB Frame")

    plt.subplot(1, 2, 2)
    plt.imshow(depth_map, cmap='plasma')
    plt.title("Depth Map")

    plt.show()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()