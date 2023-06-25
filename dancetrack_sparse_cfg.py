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
    output_dir="./yolox_dance_sparse",
    init_checkpoint="/data/zelinliu/sparsetrack/pretrain/bytetrack_dance.pth.tar",
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
    experiment_name = "yolox_dance_sparse_det",
    track_thresh = 0.7,
    track_buffer = 60,
    match_thresh = 0.85,
    min_box_area = 100,
    down_scale = 4,
    depth_levels = 1,
    depth_levels_low = 12,
    confirm_thresh = 0.7,
    mot20 = False,
    byte = False,
    deep = True,
    fp16 = True,
    fuse = True,
    val_ann = "test.json"
)

# For dancetrack--unenabled GMC: 368 - 373 in sparse_tracker.py 
# Change the thresh 0.3 to 0.35 during low-score matching 
 
 
 
 





