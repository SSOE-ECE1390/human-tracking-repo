from ultralytics import YOLO
import cv2, os, re, yaml
import numpy as np
from tqdm import tqdm

def chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]
def go_next(lst):
    for l in lst:
        yield l
if __name__ == "__main__":
    with open('parameters.yml','r') as file:
        parameters = yaml.safe_load(file)['generate_training']
        device = parameters['device']
        imgsz = parameters['imgsz']

    img_folder = 'data/images'
    figs = os.listdir(img_folder)
    figs = sorted(figs,key=lambda x : float(re.findall(r'\d+\.\d+|\d+',x)[0]))
    figsdir = [os.path.join(img_folder, f) for f in figs]
    model = YOLO("yolo11l.pt")  # pretrained YOLO11n model
    figsbatchlist = [figsdir[i:i+20] for i in range(0,len(figsdir),20)]
    label_dir = 'data/labels'
    figgen = go_next(figs)
    # 5 逐帧输出标签

    for batch in tqdm(chunks(figsdir, 20), total=len(figsdir)//20, desc="Processing batches"):
        results = model.predict(batch, stream=True, device=device, imgsz=imgsz, batch=1, verbose=False)

        for i, result in enumerate(results):
            h, w = result.orig_shape
            label_path = os.path.join(label_dir, os.path.splitext(next(figgen))[0] + ".txt")

            with open(label_path, "w") as f:
                for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
                    x1, y1, x2, y2 = box.cpu().numpy()
                    # 转 YOLO 格式 (x_center, y_center, width, height) 并归一化
                    x_center = ((x1 + x2) / 2) / w
                    y_center = ((y1 + y2) / 2) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    f.write(f"{int(cls)} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n")
