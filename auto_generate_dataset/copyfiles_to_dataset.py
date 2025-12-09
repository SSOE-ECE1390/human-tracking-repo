import shutil
import os
from sklearn.model_selection import train_test_split

src_img_dir = 'data/images'
src_label_dir = 'data/labels'
dst_img_dir = 'dataset/images'
dst_label_dir = 'dataset/labels'
name_paths = os.listdir(src_img_dir)
name_paths = [os.path.splitext(f)[0] for f in name_paths]
img_train_dir = os.path.join(dst_img_dir,'train')
img_val_dir = os.path.join(dst_img_dir,'val')
label_train_dir = os.path.join(dst_label_dir,'train')
label_val_dir = os.path.join(dst_label_dir,'val')

label_paths = []
image_paths = []
for path in name_paths:
    image_paths.append(path)
    label_paths.append(path)

img_paths_train, img_paths_val, lb_paths_train, lb_paths_val = \
    train_test_split(image_paths,label_paths, test_size=0.25, random_state=42)

for label_train, img_train in zip(lb_paths_train, img_paths_train):
    # img_tr_real = img_train + '.png'
    label_tr_real = label_train + '.txt'
    # src_img_train = os.path.join(src_img_dir,img_tr_real)
    src_label_train = os.path.join(src_label_dir,label_tr_real)
    # shutil.copy(src_img_train,img_train_dir)
    shutil.copy(src_label_train,label_train_dir)

for label_val, img_val in zip(lb_paths_val, img_paths_val):
    # img_val_real = img_val + '.png'
    label_val_real = label_val + '.txt'
    # src_img_val = os.path.join(src_img_dir,img_val_real)
    src_label_val = os.path.join(src_label_dir,label_val_real)
    # shutil.copy(src_img_val,img_val_dir)
    shutil.copy(src_label_val,label_val_dir)