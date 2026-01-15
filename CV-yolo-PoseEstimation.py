from ultralytics import YOLO
import cv2
from pathlib import Path

# Load a model
model = YOLO("yolo26x-pose.pt")  # load an official model

# Predict with the model
results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image

# # Access the results
# for result in results:
#     xy = result.keypoints.xy  # x and y coordinates
#     xyn = result.keypoints.xyn  # normalized
#     kpts = result.keypoints.data  # x, y, visibility (if available)

# Render + save (handles multiple images; here it's just one)
out_dir = Path(".")
out_dir.mkdir(parents=True, exist_ok=True)

for i, r in enumerate(results):
    rendered = r.plot()  # numpy array (BGR) with boxes/labels drawn

    out_path = out_dir / f"IMG_rendered_{i}_PE.jpg"
    cv2.imwrite(str(out_path), rendered)
    print("Saved:", out_path)