import torch
import torch.nn as nn


def get_model(model_type = 'yolox', depth = 1.33, width = 1.25, num_classes = 1, confthre = 0.001, nmsthre = 0.7):
    from models import YOLOPAFPN, YOLOX, YOLOXHead
    
    def init_yolo(M):
        for m in M.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03

    if model_type == "yolox":
        in_channels = [256, 512, 1024] 
        backbone = YOLOPAFPN(depth, width, in_channels=in_channels)
        head = YOLOXHead(num_classes, width, in_channels=in_channels)
        model = YOLOX(backbone, head, confthre, nmsthre)
    elif model_type == "yolov7x_fpn":
        from models import Yolov7_fpn
        assert width == 1
        in_channels =  [320, 640, 1280]
        backbone = Yolov7_fpn()
        head = YOLOXHead(num_classes, width, in_channels=in_channels)
        model = YOLOX(backbone, head, confthre, nmsthre)

    model.apply(init_yolo)
    model.head.initialize_biases(1e-2)
    
    return model