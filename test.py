import cv2, os, torch
import numpy as np
import yaml
from ultralytics import YOLO

if __name__ == '__main__':

    with open('parameters.yml','r') as file:
        parameters = yaml.safe_load(file)['model']
    model = YOLO("yolo11l.pt")  # pretrained YOLO11n model

    # Run batched inference on a list of images
    results = model(["figures/frame7.png", "figures/frame8.png"],stream=True)  # return a list of Results objects

    # Process results list
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        result.show()  # display to screen
        result.save(filename="results/result.png")  # save to disk
