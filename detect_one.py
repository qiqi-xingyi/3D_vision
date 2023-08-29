import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, check_requirements, non_max_suppression, apply_classifier, scale_coords, \
    xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
import os
import shutil


################################################################################################
def yolo_detect(im0s , model ,imgsz=640):

    set_logging()
    device = select_device('0')
    half = device.type != 'cpu'

    # Load model
    #model = attempt_load('weights/yolov5x.pt', map_location=device)

    imgsz = check_img_size(640, s=model.stride.max())
    if half:
        model.half()

    # classify = False
    # if classify:
    #     modelc = load_classifier(name='resnet101', n=2)  # initialize
    #     modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])

    model.to(device).eval()
    names = model.module.names if hasattr(model, 'module') else model.names

    #colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    # img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    # print(img)
    # _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

# #################################################################################################

    # dataset = LoadImages(path , img_size=imgsz)
    # for path, img, im0s, vid_cap in dataset:

    img = letterbox(im0s, new_shape=640)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  #半精度
    img /= 255.0                               #图像归一化
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img, augment=False)[0]
    pred = pred.float()
    pred = non_max_suppression(pred, 0.4, 0.5, classes=None, agnostic=False)

    pred_boxes = []

    # for i, det in enumerate(pred):  # detections per image
    for det in pred:
        #print("2")
        # gn = torch.tensor(im0s.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        #print(det)
        # box_points = []
        # all_points = []  # del

        # pred_boxes = []

        if det is not None and len(det):

            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()


            # for *xyxy, conf, cls_id in reversed(det):
                # x,y,w,h = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                #
                # label = f'{names[int(cls_id)]} {conf:.2f}'
                #
                # all_points.append([left, top, right, bottom, names[int(cls)]])
                # box_points.append([(left + right) / 2, bottom])

            # for *x, conf, cls_id in reversed(det):
            for *x, conf, cls_id in det:
                lbl = names[int(cls_id)]
                label = f'{names[int(cls_id)]} {conf:.2f}'
                x1, y1 = int(x[0]), int(x[1])
                x2, y2 = int(x[2]), int(x[3])
                pred_boxes.append((x1, y1, x2, y2, lbl, conf))
                # print(conf)
                #x1, y1, x2, y2, cls_id, conf

                # plot_one_box(x , im0s, label=label,  line_thickness=3)

            # >>>>>>>>>接入deepsort<<<<<<<<<<

    return im0s , pred_boxes
