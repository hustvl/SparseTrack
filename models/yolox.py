#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import torch 
import torch.nn as nn
from detectron2.structures import Instances, Boxes
from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN
from utils.boxes import postprocess

class YOLOX(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, head=None, confthre=0.1, nmsthre=0.6):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
            head = YOLOXHead(80)

        self.backbone = backbone
        self.head = head
        self.device = None
        self.confthre = confthre
        self.nmsthre = nmsthre

    def forward(self, batch_data):
        targets = None
        if self.training:
            x = batch_data[0].to(self.device) # b c h w
            targets = batch_data[1].to(self.device) # b num_gt 6
            fpn_outs = self.backbone(x)
            assert targets is not None
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
                fpn_outs, targets, x
            )
            outputs = {
                "total_loss": loss,
            }
            return outputs
        else:
            ori_image_sizes, input_image_sizes, input_images = self._batch_data_prerocess(batch_data)
            fpn_outs = self.backbone(torch.stack(input_images))
            batch_outputs = self.head(fpn_outs)
            batch_outputs = postprocess(batch_outputs, self.head.num_classes, self.confthre, self.nmsthre)
            
            results = []
            # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
            for batch_id, (output_per_image, input_img_size, ori_img_size) in enumerate(zip(
                batch_outputs, input_image_sizes, ori_image_sizes
            )):
                result = Instances(ori_img_size)
                pred_bboxes = output_per_image[:, :4]
                scale = min(input_img_size[0] / float(ori_img_size[0]), input_img_size[1] / float(ori_img_size[1]))
                pred_bboxes /= scale
                result.pred_boxes = Boxes(pred_bboxes)
                result.scores = output_per_image[:, 4] * output_per_image[:, 5]
                result.pred_classes = output_per_image[:, 6]
                results.append({"instances": result})
                # detections preprocess
            return results

    def _batch_data_prerocess(self, batch_data):
        ori_image_sizes, input_image_sizes, input_images= [], [], []
        for data_per_img in batch_data:
            ori_image_sizes.append([data_per_img['height'], data_per_img['width']])
            input_image_sizes.append([data_per_img['image'].shape[1], data_per_img['image'].shape[2]])
            input_images.append(data_per_img['image'])
        return ori_image_sizes, input_image_sizes, input_images