import datetime
import os
import time

import torch
import torch.utils.data
from torch import nn
import torchvision

from coco_utils import get_coco
import presets
import utils

from torchvision import transforms as transforms
import seaborn as sns 
import matplotlib.pyplot as plt 
import matplotlib 
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
import argparse
import numpy as np
from PIL import Image
import glob 
import cv2 
import math 
from pytictoc import TicToc
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import pandas as pd 
from tqdm import tqdm

IDS = [0, 32, 64, 128, 192, 256]
VALUES = [0., 1., 2., 3., 4., 5.]
t2l = { val : id_ for val, id_ in zip(VALUES, IDS) }

def get_transform(train):
    base_size = 800
    crop_size = 640

    return presets.SegmentationPresetTrain(base_size, crop_size) if train else presets.SegmentationPresetEval(base_size)

def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks

class ConvertCocoPolysToMask(object):
    def __call__(self, image, anno):
        w, h = image.size
        segmentations = [obj["segmentation"] for obj in anno]
        cats = [obj["category_id"] for obj in anno]
        if segmentations:
            masks = convert_coco_poly_to_mask(segmentations, h, w)
            cats = torch.as_tensor(cats, dtype=masks.dtype)
            # merge all instance masks into a single segmentation map
            # with its corresponding categories
            target, _ = (masks * cats[:, None, None]).max(dim=0)
            # discard overlapping instances
            target[masks.sum(0) > 1] = 255
        else:
            target = torch.zeros((h, w), dtype=torch.uint8)
        target = Image.fromarray(target.numpy())
        return image, target

# def get_transform(train):
#     base_size = 800
#     crop_size = 640

#     return presets.SegmentationPresetTrain(base_size, crop_size) if train else presets.SegmentationPresetEval(base_size)

def get_circle(img, fname):
    np_img = np.array(transforms.ToPILImage()(img[0]))
    cv_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    blr = cv2.GaussianBlur(gray, (0, 0), 1)
    circles = cv2.HoughCircles(blr, cv2.HOUGH_GRADIENT, 1, 50,
                                param1=150, param2=40, minRadius=450,maxRadius=480)

    if len(circles) != 1:
        print("ERROR for the number of circles: ", fname)
    else:
        c = np.array(circles[0][0], np.int32)
        # img_ = cv2.circle(np_img, (c[0], c[1]), c[2], (255, 0, 0), 4)
        # plt.scatter(c[0], c[1], c='r')
        # plt.imshow(img_)
        # plt.show()

        return cv_img, c[0], c[1], c[2]

def exceptions(tensor_pred, cx, cy, r, offset1, offset2):
    tensor_pred = np.array(transforms.ToPILImage()(tensor_pred[0].byte()))
    idxes = np.where(tensor_pred == 64)
    # print(tensor_pred.shape)
    for x, y in zip(idxes[0], idxes[1]):
        if math.sqrt((cx - x)**2 + (cy - y)**2) > r - offset1//2 and \
            math.sqrt((cx - x)**2 + (cy - y)**2) < r + offset1:
            tensor_pred[idxes[0], idxes[1]] = 0
        
        if math.sqrt((cx - x)**2 + (cy - y)**2) > r + offset2:
            tensor_pred[idxes[0], idxes[1]] = 0

    
    idxes_ = np.where(tensor_pred != 0)

    tensor_pred = cv2.circle(tensor_pred, (cx, cy), r - offset1//2, (255, 0, 0), 1)
    tensor_pred = cv2.circle(tensor_pred, (cx, cy), r + offset1, (255, 0, 0), 1)
    # plt.imshow(tensor_pred)
    # plt.show()

    return tensor_pred, idxes_

def save_as_images(img, tensor_pred, folder, image_name):
    # img = transforms.ToPILImage()(img[0])
    # np.save(os.path.join(folder, 'tensors', image_name + '.npy'), tensor_pred.cpu().numpy())
    # tensor_pred = transforms.ToPILImage()(tensor_pred[0].byte())

    filename = os.path.join(folder, image_name + '.png')

    fig = plt.figure()
    # plt.imshow(img, alpha=0.8)
    # plt.imshow(tensor_pred, alpha=0.5)
    # plt.savefig(filename)
    # plt.close()
    # fig = plt.gca()
    # fig.axes.get_xaxis().set_visible(False)
    # fig.axes.get_yaxis().set_visible(False)
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(tensor_pred)#, alpha=0.7)
    # plt.subplot(133)
    # plt.imshow(tensor_pred)
    # plt.show()
    plt.savefig(filename)
    plt.close()

    # tensor_pred.save(os.path.join(folder, image_name + '.png'))

def main(args):
    t = TicToc()
    offset1 = 10
    offset2 = 50
    device = torch.device(args.device)

    transform = transforms.Compose([
                    get_transform(train=False)
                    ])

    # transform = transforms.ToTensor()
    model = torch.load(args.resume)
    model.to(device)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(os.path.join(args.output_dir, 'over')):
        os.makedirs(os.path.join(args.output_dir, 'over'))
    lens_types = os.listdir(args.data_path)

    df = pd.DataFrame(columns=['fname', 'defect'])
    model.eval()
    
    nb = {'Rhapsody': 0, 
                         'SoulBrown': 0,
                         'ANW': 0, 'FLAMINGO': 0, "PIA": 0}

    with torch.no_grad():
        over_cnt = 0
        cnt = 1
        img_files = glob.glob(os.path.join(args.data_path, '*.jpg'))
        
        for img_file in tqdm(img_files):
            t.tic()
            fname = os.path.split(os.path.splitext(img_file)[0])[-1]
            image = cv2.imread(img_file)
            # image = image[50:1150, 200:1400, :]
            image = transforms.ToTensor()(image)
            image = torch.unsqueeze(image, 0)
            # print(image.size())
            
            output = model(image)
            output = output['out']
            preds = torch.nn.functional.softmax(output, dim=1)
            preds_labels = torch.argmax(preds, dim=1)
            preds_labels = preds_labels.float()

            
            preds_labels = preds_labels.to('cpu')
            _, x, y = preds_labels.size()
            preds_labels.apply_(lambda x: t2l[x])
            preds_labels = transforms.Resize((1100, 1200), interpolation=Image.NEAREST)(preds_labels)
            # plt.imshow(preds_labels[0])
            # plt.show()
            # print(preds_labels.size(), np.unique(preds_labels.cpu()))
            
            image = transforms.Resize((1100, 1200))(image)
            # target = transforms.Resize((1100, 1200))(target)

            cnt += 1
            image, cx, cy, r = get_circle(image, fname)
            preds_labels, idxes = exceptions(preds_labels, cx, cy, r, offset1, offset2)

            if len(idxes[0]) != 0:
                print("\nOVER >>>>>>>>>>>>>>>>>>>>>>>>>>> {}".format(fname))
                print(nb)
                over_cnt += 1
                # for i in range(len(idxes[0])):
                #     info = [fname, preds_labels[idxes[0][i], idxes[1][i]]]
                #     df.loc[len(df)] = info
                nb[fname.split('-')[0]] += 1
            #     save_as_images(image, preds_labels, os.path.join(args.output_dir, 'over'), fname)    
            # else:
            #     save_as_images(image, preds_labels, args.output_dir, fname)    

            # if cnt > 13:
            #     break
    df.to_csv(os.path.join(args.output_dir, 'info.csv'), index=False)
    print(nb)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--data-path', default='/home/wonchul/projects/interojo_good/good_crop_')
    parser.add_argument('--output-dir', default='./res1/good')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=4, type=int)
    parser.add_argument('--nb-workers', default=8, type=int, metavar='N')
    parser.add_argument('--resume', default='./models/deeplabv3_resnet101/model_190.pt')

    args = parser.parse_args()

    main(args)
