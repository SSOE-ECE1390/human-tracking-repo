import os, cv2, yaml
import numpy as np
from tqdm import tqdm
def safe_range(x1,y1,x2,y2,w,h):
    x1, y1, x2, y2 = np.clip([x1, y1, x2, y2], 0, [w - 1, h - 1, w -1, h - 1])
    return x1, y1, x2, y2
def draw_box(im_path, label_path, color=(255,0,0),thickness=2, class_names=None):
    im = cv2.imread(im_path)
    if im is None:
        raise IOError(f"Cannot read image: {im_path}")
    h, w = im.shape[:2]
    if not os.path.exists(label_path):
        return im
    with open(label_path,'r') as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            conf = float(parts[5]) if len(parts) >= 6 else None
            xc, yc, bw, bh = map(float, parts[1:5])
            x1 = int((xc - bw/2) * w)
            y1 = int((yc - bh/2) * h)
            x2 = int((xc + bw/2) * w)
            y2 = int((yc + bh/2) * h)
            x1, y1, x2, y2 = safe_range(x1, y1, x2, y2, w, h)
            if class_names:
                label_text = class_names[cls] if cls < len(class_names) else str(cls)
            else:
                label_text = str(cls)
            if conf is not None:
                label_text += f" {conf:.2f}"

            # 画框 + 文本
            cv2.rectangle(im, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(im, label_text, (x1, max(0, y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return im

if __name__ == '__main__':

    img_folder = 'data/images'
    lbl_folder = 'data/labels'
    out_folder = 'data/plkmean'
    os.makedirs(out_folder, exist_ok=True)

    imgs = sorted([f for f in os.listdir(img_folder) if f.lower().endswith(('.png'))])
    for img_name in tqdm(imgs,total=len(imgs),desc='processing'):
        stem = os.path.splitext(img_name)[0]
        img_path = os.path.join(img_folder, img_name)
        lbl_path = os.path.join(lbl_folder, stem + '.txt')   # same stem

        vis = draw_box(img_path, lbl_path)
        cv2.imwrite(os.path.join(out_folder, img_name), vis)  # use positional args