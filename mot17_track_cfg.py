from detectron2.config import LazyCall as L
from omegaconf import OmegaConf
from .datasets.builder import build_test_loader
from .models.model_utils import get_model

# build dataloader
dataloader = OmegaConf.create()
dataloader.test = L(build_test_loader)(
    test_size = (800, 1440),
    infer_batch = 1 # for tracking process frame by frame 
)
 
# build model
model = L(get_model)(
    model_type = 'yolox',
    depth = 1.33,
    width = 1.25,
    num_classes = 1,
    confthre = 0.01,
    nmsthre = 0.7
)

# build train cfg 
train = dict(
    output_dir="./yolox_mix17",
    init_checkpoint="/data/zelinliu/sparsetrack/pretrain/bytetrack_x_mot17.pth.tar",
    # model ema
    model_ema = dict(
        enabled=False,
        use_ema_weights_for_eval_only = False,
        decay = 0.9998,
        device = "cuda",
        after_backward = False
    ),
    device="cuda",
)

# build tracker
track = dict(
    experiment_name = "yolox_mix17_det",
    track_thresh = 0.6,
    track_buffer = 30,
    match_thresh = 0.75,
    min_box_area = 100,
    down_scale = 4,
    depth_levels = 1,
    depth_levels_low = 3,
    confirm_thresh = 0.8,
    # is fuse scores
    mot20 = False,
    # trackers
    byte = False,
    deep = True,
    bot = False,
    sort = False,
    ocsort = False,
    # detector model settings
    fp16 = True,
    fuse = True,
    val_ann = "train.json",
    # is public dets using 
    is_public = False
)
 
 
 
 





