import os
import os.path as osp
import cv2, torch
import numpy as np
import copy
import logging

__all__ = ["MOTtestMapper", ]

class MOTtestMapper:
    def __init__(
        self,
        test_size,
        preproc=None,
    ):
        self.img_size =  test_size
        self.preproc = preproc
        

    def __call__(self, data_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
            {   "file_name":, "height":,  "width":,  "image_id":, 
                "annotations":[{"iscrowd":, "bbox":, "category_id":, "bbox_mode":,}, {} ...]  }
        Returns:
            dict: a format that builtin models in detectron2 accept
        """
      
        data_dict = copy.deepcopy(data_dict)  # it will be modified by code below
        img0 = cv2.imread(data_dict["file_name"])
        
        objs = []
        for obj in data_dict["annotations"]:
            x1 = obj["bbox"][0]
            y1 = obj["bbox"][1]
            x2 = x1 + obj["bbox"][2]
            y2 = y1 + obj["bbox"][3]
            if obj["bbox"][2] * obj["bbox"][3] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)
        
        num_objs = len(objs)
        target = np.zeros((num_objs, 6))
        for ix, obj in enumerate(objs):
            cls_id = obj["category_id"]
            target[ix, 0:4] = obj["clean_bbox"]
            target[ix, 4] = cls_id
            target[ix, 5] = -1
      
        if self.preproc is not None:
            img, target = self.preproc(img0, target, self.img_size)
    
        data_dict["image"] = torch.as_tensor(np.ascontiguousarray(img))
        data_dict["ori_img"] = img0
        data_dict.pop("annotations", None)
      
        return data_dict
    #{"file_name":, "height":,  "width":,  "image_id":, "image":  "ori_img":}