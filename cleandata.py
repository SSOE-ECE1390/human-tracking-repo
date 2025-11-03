import os,re
from tqdm import tqdm

def onlykeep(src_path, dirs, keyword='0'):
    for path in tqdm(dirs,total=len(dirs),desc=f'Processing {os.path.basename(src_path)}'):
        target_path = os.path.join(src_path,path)

        with open(target_path, "r") as f:
             lines = f.readlines()

        # Keep only lines that do NOT contain the keyword
        lines = [ln for ln in lines if ln.strip() and ln[0] == keyword]

        with open(target_path, "w") as f:
            f.writelines(lines)

if __name__ == '__main__':
    src_path = 'dataset/labels'
    src_path_train = os.path.join(src_path,'train')
    src_path_val = os.path.join(src_path,'val')
    train_dirs = os.listdir(src_path_train)
    val_dirs = os.listdir(src_path_val)

    onlykeep(src_path_train,train_dirs)
    onlykeep(src_path_val,val_dirs)
    
    
