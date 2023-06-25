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
    output_dir="./yolox_bot_dance",
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
    experiment_name = "yolox_bot_dance_det",
    track_buffer = 30,
    track_high_thresh = 0.6,
    track_thresh = 0.6,
    track_low_thresh = 0.1,
    match_thresh = 0.7,
    min_box_area = 10,
    new_track_thresh = 0.7,
    mot20 = True,
    byte = False,
    deep = False,
    bot = True,
    fp16 = True,
    fuse = True,
    val_ann = "test.json"
)
 
 
                
 
 





