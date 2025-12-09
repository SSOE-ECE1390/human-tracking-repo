from ultralytics import YOLO
import os, re, yaml
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import argparse
import joblib

def chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]
def go_next(lst):
    for l in lst:
        yield l

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, choices=["generate","kmeans"])
    args = parser.parse_args()
    if args.mode == "generate":
        with open('settings.yml','r') as file:
            parameters = yaml.safe_load(file)
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
        kmeans = joblib.load("kmeans_model.pkl")
        # 5 逐帧输出标签
        color = []
        for batch in tqdm(chunks(figsdir, 20), total=len(figsdir)//20, desc="Processing batches"):
            results = model.predict(batch, stream=True, device=device, imgsz=imgsz, batch=1, verbose=False, conf=0.15,)

            for i, result in enumerate(results):
                h, w = result.orig_shape
                label_path = os.path.join(label_dir, os.path.splitext(next(figgen))[0] + ".txt")
                img = result.orig_img
                
                with open(label_path, "w") as f:
                    for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
                        if int(cls) == 0:
                            x1, y1, x2, y2 = np.uint32(box.cpu().numpy())
                            X = np.arange(x1,x2).reshape(1,-1)
                            Y = np.arange(y1,y2).reshape(-1,1)
                            square = img[Y,X]
                            channelmean = square.mean(axis=(0,1)).reshape(1,-1)
                            write_cls = kmeans.predict(channelmean)[0]
                            # brightness = square.mean()
                            if not(write_cls==2 and x2 - x1>70):
                                color.append(channelmean)
                                # if brightness > 66.5:
                                    # write_cls = 0 if brightness < 99.6 else 1
                                    # 转 YOLO 格式 (x_center, y_center, width, height) 并归一化
                                x_center = ((x1 + x2) / 2) / w
                                y_center = ((y1 + y2) / 2) / h
                                bw = (x2 - x1) / w
                                bh = (y2 - y1) / h
                                f.write(f"{write_cls} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n")
        vec = np.array(color)
        np.save('vec.npy', vec)
    elif args.mode == "kmeans":
        vec = np.load('vec.npy')
        n = 3
        kmeans = KMeans(n_clusters=n, random_state=0).fit(vec)
        joblib.dump(kmeans, "kmeans_model.pkl")
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        cut = vec[:150]
        cut_label = labels[:150]
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        
        for i in range(n):
            pts = cut[cut_label == i]
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], label=f'Cluster {i}', s=10)

        # 绘制聚类中心
        ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
                c='black', marker='X', s=100, label='Centers')
        ax.set_xlabel('B')
        ax.set_ylabel('G')
        ax.set_zlabel('R')
        plt.show()
    else:
        raise ValueError("mode generate or kmean")
    # 画频率直方图
    # plt.hist(color, bins=100, color='steelblue', edgecolor='black')
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Data')
    # plt.show()