import cv2
import os
from ultralytics import YOLO

if __name__ == '__main__':
    model = ("yolov8n.pt")   # n/s/m/l/x — pick a size

    # 2) Train
    results = model.train(
        data="parameters.yml",
        imgsz=640,
        epochs=100,
        batch=16,
        workers=8,
        device="mps",     # "0" for CUDA GPU; "mps" for Apple Silicon; omit for CPU
        project="soccer_py",
        name="exp1",
        cache=True,       # speed up after first epoch
        patience=30,      # early stop
        cos_lr=True,
    )

    # 3) Validate (optional; also happens at end of training)
    metrics = model.val(data="parameters.yml", batch=16)

    # 4) Inference test
    model.predict(source="test_images/", conf=0.25, save=True)

    # 5) Export (for deployment)
    model.export(format="onnx") 
