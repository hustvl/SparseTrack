from detectron2.config import LazyCall as L
from omegaconf import OmegaConf
from .datasets.builder import build_test_loader
from .models.yolo import DetectionModel

# build dataloader
dataloader = OmegaConf.create()
dataloader.test = L(build_test_loader)(
    test_size = (800, 1440),
    infer_batch = 1,
    test_num_workers  = 4,
    max_dets = 1000
)
 
# build model
model = L(DetectionModel)(
    cfg='yolov8l.yaml',  
    ch=3,  
    nc=1,  
    cls_idx = [0,], 
    conf=0.01, 
    iou=0.75, 
    agnostic=False, 
    multi_label=False, 
    num_max_dets=1000, 
    verbose=True
)

# build train cfg 
train = dict(
    output_dir="./yolov8_mix17",
    init_checkpoint="/data/zelinliu/yolov8/yolov8_mix17/model_0146239.pth",
    # model ema
    model_ema = dict(
        enabled=True,
        use_ema_weights_for_eval_only = True,
        decay = 0.9998,
        device = "cuda",
        after_backward = False
    ),
    device="cuda",
    seed = 0
)

# build tracker
track = dict(
    experiment_name = "yolov8_mix17_det",
    track_thresh = 0.5,
    track_buffer = 60,
    match_thresh = 0.8,
    min_box_area = 100,
    down_scale = 4,
    depth_levels = 1,
    depth_levels_low = 3,
    confirm_thresh = 0.8,
    mot20 = False,
    byte = False,
    deep = True,
    fp16 = True,
    fuse = True,
    val_ann = "train.json",  
)
 
 
 
 





