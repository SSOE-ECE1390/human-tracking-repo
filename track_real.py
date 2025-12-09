import cv2
import yaml
from ultralytics import YOLO


if __name__ == '__main__':
    with open('settings.yml','r') as file:
        parameters = yaml.safe_load(file)
        device = parameters['device']
        imgsz = parameters['imgsz']
    # Load the YOLO11 model
    # model = YOLO("/Users/stonyxiong/github-classroom/SSOE-ECE1390/project/playeriden/exp1/weights/last.pt")
    model = YOLO("Team_iden_kmean/exp_zero2/weights/last.pt") # kmean
    # model = YOLO("Team_iden/exp_zero/weights/last.pt")      # brightness
    # model = YOLO("yolo11n.pt")                              # vanilla

    # Open the video file
    video_path = "soccer_noaudio.mp4"
    results = model.track(video_path,imgsz = imgsz, conf=0.35,device=device, show=True)

    cv2.destroyAllWindows()