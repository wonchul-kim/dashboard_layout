import cv2
import os
from tqdm import tqdm
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from skimage.measure import label, regionprops

from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
import glob
import torch 
from PIL import Image
import copy
import torchvision.transforms as T

def convert2yolo(size, bbox):
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

    def sorting(l1, l2):
        if l1 > l2:
            lmax, lmin = l1, l2
            return lmax, lmin
        else:
            lmax, lmin = l2, l1
            return lmax, lmin

    xmax, xmin = sorting(x1, x2)
    ymax, ymin = sorting(y1, y2)
    
    dw = 1./size[1]
    dh = 1./size[0]
    
    x = (xmin + xmax)/2.0
    y = (ymin + ymax)/2.0
    w = xmax - xmin
    h = ymax - ymin
    
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    
    return [x, y, w, h]

def convert_from_yolo(size, x, y, w, h):

    dh, dw = size[0], size[1]
    l = ((x - w / 2) * dw)
    r = ((x + w / 2) * dw)
    t = ((y - h / 2) * dh)
    b = ((y + h / 2) * dh)
    
    if l < 0:
        l = 0
    if r > dw - 1:
        r = dw - 1
    if t < 0:
        t = 0
    if b > dh - 1:
        b = dh - 1   

    return [l, t, r, b]

thresholds = [64, 128]
root_path = './res1/bad2/tensors'
save_path = os.path.join(root_path, 'labels')
if not os.path.exists(save_path):
    os.mkdir(save_path)

img_files = list(sorted(glob.glob(os.path.join(root_path, '*.png'))))

for img_file in tqdm(img_files):
    img_name = img_file.split('/')[-1].split('.')[0]
    txt = open(os.path.join(save_path, img_name + '.txt'), 'w')

    mask = cv2.imread(os.path.join(root_path, img_name + '.png'))
    # mask = cv2.imread(img_file)
    # mask = np.load(img_file)[0]
    # print(mask.shape)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # print(np.unique(mask))
    mask_ = copy.deepcopy(mask)

    for idx, thresh in enumerate(thresholds):
        if idx == 0:
            bw = closing((mask == 64), square(3))
        elif idx == 1:
            bw = closing((mask == 128), square(3))
 
        cleared = clear_border(bw)

        label_image = label(cleared)

        props = regionprops(label_image)
        for prop in props:
            # if abs(prop.bbox[1] - prop.bbox[3])*abs(prop.bbox[0] - prop.bbox[2]) > 5:
            bbox = convert2yolo(mask.shape[:2], (prop.bbox[1], prop.bbox[0], prop.bbox[3], prop.bbox[2]))
            txt.write('{} {:.3f} {:.3f} {:.3f} {:.3f}'.format(idx, bbox[0], bbox[1], bbox[2], bbox[3]))
            txt.write('\n')
            # print(idx, img_name, prop.bbox)

            cv2.rectangle(mask_, (prop.bbox[1], prop.bbox[0]), (prop.bbox[3], prop.bbox[2]), (255, 0, 0), 2)
        # print("------------------------------------------------------------------------------------")


    # plt.imshow(mask_)
    # plt.show()

    txt.close()

