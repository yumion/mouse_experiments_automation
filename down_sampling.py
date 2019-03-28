from glob import glob
import os
import shutil
import numpy as np

cls_0 = glob('dataset/0/*')
cls_1 = glob('dataset/1/*')

os.makedirs('downsampling', exist_ok=True)
train_dir = os.path.join('downsampling', '0')
os.makedirs(train_dir, exist_ok=True)
test_dir = os.path.join('downsampling', '1')
os.makedirs(test_dir, exist_ok=True)

for f in cls_1:
    to_dir = os.path.join('downsampling/1', os.path.basename(f))
    shutil.copyfile(f, to_dir)

np.random.shuffle(cls_0)
for f in cls_0[:len(cls_1)]:
    to_dir = os.path.join('downsampling/0', os.path.basename(f))
    shutil.copyfile(f, to_dir)
