from ultralytics import YOLO

#model = YOLO("yolo26s.pt")
model = YOLO("yolo26s-pose.pt")
#model = YOLO("yolo26x.pt")
#model = YOLO("yolo26x-pose.pt")


# 0 = default Mac camera
results = model(source=0, show=True)
