from ultralytics import YOLO

# 1. Load the Segmentation model (the -seg version)
model = YOLO('yolo11n-seg.pt')

# 2. Start Training
model.train(
    data='data.yaml',
    epochs=100,
    imgsz=320,
    batch=8,
    patience=10,
    device='cpu'
)