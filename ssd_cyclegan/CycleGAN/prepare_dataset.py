
import os
import shutil

root = 'G:/datasets/VOCdevkit/'
voc_2007 = 'VOC2007/'
voc_2012 = 'VOC2012/'
clipart = 'clipart/'
txt = 'ImageSets/Main/trainval.txt'
dataset = 'CycleGAN/'
with open(os.path.join(root, voc_2007, txt), 'r') as f:
    for data in f.readlines():
        img_name = data.strip('\n')+'.jpg'
        file_path = os.path.join(root, voc_2007, 'JPEGImages', img_name)
        shutil.copy(file_path, os.path.join(root, dataset, 'VOC2007', 'trainA/'))
with open(os.path.join(root, voc_2012, txt), 'r') as f:
    for data in f.readlines():
        img_name = data.strip('\n')+'.jpg'
        file_path = os.path.join(root, voc_2012, 'JPEGImages', img_name)
        shutil.copy(file_path, os.path.join(root, dataset, 'VOC2012', 'trainA/'))
for file in os.listdir(os.path.join(root, clipart, 'JPEGImages')):
    shutil.copy(os.path.join(root, clipart, 'JPEGImages', file), os.path.join(root, dataset, 'VOC2007', 'trainB/'))
    shutil.copy(os.path.join(root, clipart, 'JPEGImages', file), os.path.join(root, dataset, 'VOC2012', 'trainB/'))




