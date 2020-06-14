from data import COCODetection, get_label_map, MEANS, COLORS
from yolact import Yolact
from utils.augmentations import BaseTransform, FastBaseTransform, Resize
from utils.functions import MovingAverage, ProgressBar
from layers.box_utils import jaccard, center_size, mask_iou
from utils import timer
from utils.functions import SavePath
from layers.output_utils import postprocess, undo_image_transformation
import pycocotools

from data import cfg, set_cfg, set_dataset

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import argparse
import time
import random
import cProfile
import pickle
import json
import os
from collections import defaultdict
from pathlib import Path
from collections import OrderedDict
from PIL import Image

import matplotlib.pyplot as plt
import cv2

iou_thresholds = [x / 100 for x in range(50, 100, 5)]
coco_cats = {}  # Call prep_coco_cats to fill this
coco_cats_inv = {}
color_cache = defaultdict(lambda: {})



def prep_display_mod(dets_out, img, h, w,depth_map, undo_transform=True, mask_alpha=1.0 ):  # was mask_alpha=0.45
    """
    Note: If undo_transform=False then im_h and im_w are allowed to be None.
    """
    score_threshold = 0.15
    top_k = 15

    if undo_transform:
        img_numpy = undo_image_transformation(img, w, h)
        img_gpu = torch.Tensor(img_numpy).cuda()
    else:
        img_gpu = img / 255.0
        h, w, _ = img.shape

    with timer.env('Postprocess'):
        save = cfg.rescore_bbox
        cfg.rescore_bbox = True
        t = postprocess(dets_out, w, h, score_threshold = score_threshold)
        cfg.rescore_bbox = save

    with timer.env('Copy'):
        idx = t[1].argsort(0, descending=True)[:top_k]  # top_k = 15

        if cfg.eval_mask_branch:
            # Masks are drawn on the GPU, so don't copy
            masks = t[3][idx]
        classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]
    num_dets_to_consider = min(top_k, classes.shape[0])
    for j in range(num_dets_to_consider):
        if scores[j] < score_threshold:
            num_dets_to_consider = j
            break
    classes = classes[:num_dets_to_consider]  # added

    # Quick and dirty lambda for selecting the color for a particular index
    # Also keeps track of a per-gpu color cache for maximum speed
    def get_color(j, on_gpu=None):
        global color_cache
        # color_idx = (classes[j] * 5 if class_color else j * 5) % len(COLORS)          #original
        color_idx = j  # black
        if on_gpu is not None and color_idx in color_cache[on_gpu]:
            return color_cache[on_gpu][color_idx]
        else:
            color = COLORS[color_idx]
            if not undo_transform:
                # The image might come in as RGB or BRG, depending
                color = (color[2], color[1], color[0])
            if on_gpu is not None:
                color = torch.Tensor(color).to(on_gpu).float() / 255.
                color_cache[on_gpu][color_idx] = color
            return color

    # First, draw the masks on the GPU where we can do it really fast
    # Beware: very fast but possibly unintelligible mask-drawing code ahead
    # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice
    if num_dets_to_consider > 0:  # was ...>0
        # After this, mask is of size [num_dets, h, w, 1]
        masks = masks[:num_dets_to_consider, :, :, None]
        # print("masks_og.shape", masks.shape)

        # begin added       // filter out the person masks and class indices
        temp_masks = []
        classes_to_mask = []
        x = []  # save the center points of the boxes in the same order as the masks
        y = []
        for i, j in enumerate(classes):
            if j == 0:  # j = 0 for person class
                temp_masks.append(i)
                classes_to_mask.append(j)
                x1, y1, x2, y2 = boxes[i, :]
                x.append(int((x1 + x2) / 2))
                y.append(int((y1 + y2) / 2))
        num_dets_to_consider = len(classes_to_mask)
        print("x: ", x)
        print("y: ", y)

        x = np.array(y)
        y = np.array(x)

        for i in range(x.size):
            print("depth at object i: ", x[i], y[i], " : ", depth_map[x[i], y[i], 0])

        if num_dets_to_consider == 0:
            return ((img_gpu * 0).byte().cpu().numpy())  # make it black before returning
        # print("num_dets_to_consider: ", num_dets_to_consider)
        # print("filtered classes : ", classes_to_mask)
        # temp_masks = np.array(temp_masks).T
        # print("temp_masks ", temp_masks)
        # print(temp_masks.shape)
        np.array(temp_masks).T.tolist()
        # print("temp masks", temp_masks)
        masks = masks[temp_masks]
        # masks = masks[:,:,:,0]
        # print("masks : ", masks)
        # print("masks_filtered.shape", masks.shape)
        # print("masks.shape[0]", masks.shape[0])
        # end added

        # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])
        # colors = torch.cat([get_color(j, on_gpu=img_gpu.device.index).view(1, 1, 1, 3) for j in range(num_dets_to_consider)], dim=0)   #original
        # colors = torch.cat([get_color(j, on_gpu=img_gpu.device.index).view(1, 1, 1, 3) for j in classes_to_mask],     dim=0)  # added
        colors = torch.cat([get_color(0, on_gpu=img_gpu.device.index).view(1, 1, 1, 3)], dim=0)  # added
        # masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha  # original
        # This is 1 everywhere except for 1-mask_alpha where the mask is
        # inv_alph_masks = masks * (-mask_alpha) + 1      #original

        # begin added        // make an union of the stacked masks
        num_dets_to_consider = 1
        tmp = masks[0]
        if num_dets_to_consider > 1:
            for msk in masks[1:]:
                tmp = tmp + msk
        # print("masks.shape: ", masks.shape)
        # print("tmp.shape: ", (tmp.unsqueeze(0)).shape)
        masks = tmp.unsqueeze(0)
        masks[masks != 0.0] = 1.0

        inv_alph_masks = masks * (-mask_alpha) + 1
        masks_color = (inv_alph_masks.repeat(1, 1, 1, 3)) * colors * mask_alpha
        inv_alph_masks = masks.repeat(1, 1, 1, 3)

        # inv_alph_masks = masks
        # inv_alph_masks = masks
        # print("masks : ", masks)
        # masks = (masks-1.)*-1.
        # print("masks : ", masks)
        # inv_alph_masks = masks * (-mask_alpha)+1
        # masks_color = masks_color*0.5
        # end added

        # I did the math for this on pen and paper. This whole block should be equivalent to:
        #    for j in range(num_dets_to_consider):
        #        img_gpu = img_gpu * inv_alph_masks[j] + masks_color[j]
        # masks_color_summand = masks_color[0]
        # if num_dets_to_consider > 1:
        #     inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider - 1)].cumprod(dim=0)
        #     masks_color_cumul = masks_color[1:] * inv_alph_cumul
        #     masks_color_summand += masks_color_cumul.sum(dim=0)

        # img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand  # original
        # print("inv_alph_masks.shape: ", (torch.squeeze(inv_alph_masks,0)).shape)
        # print("masks_color.shape: ", (torch.squeeze(masks_color,0)).shape)
        img_gpu = img_gpu * torch.squeeze(inv_alph_masks, 0) + torch.squeeze(masks_color, 0)  # added
        # img_gpu = img_gpu

    img_numpy = (img_gpu * 255.0).byte().cpu().numpy()

    return img_numpy






def evalimage_mod(net: Yolact, img, depth_map):
    # frame = torch.from_numpy(cv2.imread(path)).cuda().float()
    frame = img
    batch = FastBaseTransform()(frame.unsqueeze(0))
    preds = net(batch)

    img_numpy = prep_display_mod(preds, frame, None, None, depth_map, undo_transform=False)

    return img_numpy
    # if save_path is None:
    #     img_numpy = img_numpy[:, :, (2, 1, 0)]
    #
    # if save_path is None:
    #     plt.imshow(img_numpy)
    #     plt.title(path)
    #     plt.show()
    # else:
    #     cv2.imwrite(save_path, img_numpy)




def evaluate_mod(net: Yolact, img, depth_map):
    img_out = evalimage_mod(net, img, depth_map)
    return img_out




# if __name__ == '__main__':
#
#     set_cfg('yolact_plus_resnet50_config')
#     cfg.mask_proto_debug = False
#
#     with torch.no_grad():
#         if not os.path.exists('results'):
#             os.makedirs('results')
#
#         if torch.cuda.is_available():
#             cudnn.fastest = True
#             torch.set_default_tensor_type('torch.cuda.FloatTensor')
#         else:
#             torch.set_default_tensor_type('torch.FloatTensor')
#
#         print('Loading model...', end='')
#         net = Yolact()
#         net.load_weights('/content/yolact/weights/yolact_plus_resnet50_54_800000.pth')
#         if torch.cuda.is_available():
#             net = net.cuda()
#         print(' Done.')
#         net.eval()
#
#         img = torch.from_numpy(cv2.imread('/content/yolact/photo.jpg')).cuda().float()
#
#         img_out_numpy = evaluate_mod(net, img)
#
#         # save the output image
#
#         # img_numpy = img_out_numpy[:, :, (2, 1, 0)]
#         cv2.imwrite('/content/yolact/photo_out.jpg', img_out_numpy)
