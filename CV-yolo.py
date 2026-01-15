from ultralytics import YOLO
import cv2
from pathlib import Path

# Load a model
model = YOLO("yolo26x.pt")

# Predict
#results = model("./test/IMG_9788.jpeg")
results = model("./bus.jpg")

# Render + save (handles multiple images; here it's just one)
out_dir = Path(".")
out_dir.mkdir(parents=True, exist_ok=True)

for i, r in enumerate(results):
    rendered = r.plot()  # numpy array (BGR) with boxes/labels drawn

    out_path = out_dir / f"IMG_rendered_{i}.jpg"
    cv2.imwrite(str(out_path), rendered)
    print("Saved:", out_path)