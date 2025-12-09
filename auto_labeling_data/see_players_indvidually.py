import os, cv2
import numpy as np
import matplotlib.pyplot as plt
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
def box(im_path, label_path, class_names=None):
    im = cv2.imread(im_path)
    sqares = []
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
            xc, yc, bw, bh = map(float, parts[1:5])
            x1 = int((xc - bw/2) * w)
            y1 = int((yc - bh/2) * h)
            x2 = int((xc + bw/2) * w)
            y2 = int((yc + bh/2) * h)
            x1, y1, x2, y2 = safe_range(x1, y1, x2, y2, w, h)
            sqares.append(im[y1:y2,x1:x2])
    return sqares

if __name__ == "__main__":

    img = "data/images/frame32.590295147573784.png"
    label = "data/labels/frame32.590295147573784.txt"
    vis = box(img, label)
    print(vis)
    chunk = [vis[i:i+6] for i in range(0,len(vis),6)]
    cols = 6
    rows = len(chunk)       # 每个 ch 是一行
    plt.figure(dpi=200)

    for i, ch in enumerate(chunk):       # i: 第几行
        for j, im in enumerate(ch):      # j: 这一行第几个
            im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
            if im.size == 0:
                print("skip empty image at group", i, "index", j, "shape:", im.shape)
                continue
            idx = i * cols + j + 1       # 全局第几个子图
            plt.subplot(rows, cols, idx)
            plt.imshow(im)
            plt.axis("off")
    plt.tight_layout()
    plt.show()
    # cv2.imwrite(os.path.join(out_folder, img_name), vis)  # use positional args