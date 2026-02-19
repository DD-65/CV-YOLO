# CV-YOLO
Basic computer vision / pose estimation using YOLO models.

## Installation

1. Create and activate a virtual environment (optional but recommended):
```bash
conda create -n cv-yolo python=3.11 -y 
conda activate cv-yolo
```
2. Clone repo:
```bash
git clone https://github.com/DD-65/CV-YOLO.git
cd CV-YOLO
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Model Files

The scripts reference local model weights such as `yolo26s-pose.pt`, `yolo26x.pt`, and `yolo26x-pose.pt` in the project root.  
If a model file is missing, add it to the repo root or update the model path inside the script you are running.

## Run

```bash
python live-cam.py
python pose-checker.py
...
```
