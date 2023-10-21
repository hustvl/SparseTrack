import logging, cv2
import os
import torch
import torch.backends.cudnn as cudnn
from detectron2.utils import comm
from detectron2.config import LazyConfig, instantiate
from detectron2.engine.defaults import create_ddp_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import (
    launch,
    default_argument_parser,
    default_setup
)
from datasets.data.data_augment import preproc
from tracker.byte_tracker import BYTETracker
from tracker.bot_sort import BoTSORT
from tracker.sparse_tracker import SparseTracker
from tracker.eval.timer import Timer
from utils import ema
from utils.model_utils import fuse_model
from utils.visualization import plot_tracking
logger = logging.getLogger("detectron2")

def do_demo(cfg, model):
    logger = logging.getLogger("detectron2")
    if cfg.train.model_ema.enabled and cfg.train.model_ema.use_ema_weights_for_eval_only:
        logger.info("Run evaluation with EMA.")
    else:
        logger.info("Run evaluation without EMA.")
    cudnn.benchmark = True
        
    # set environment variables for distributed inference    
    file_name = os.path.join(cfg.train.output_dir, cfg.track.experiment_name)
    if comm.is_main_process():
        os.makedirs(file_name, exist_ok=True)
    results_folder = os.path.join(file_name, "track_infer")    
    os.makedirs(results_folder, exist_ok=True)    

    # build evaluator
    model.eval()
    if cfg.track.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)#

    # start evaluate
    # TODO half to amp_test
    timer = Timer()
    model = model.eval()
    if cfg.track.fp16:
        model = model.half()
    
    if cfg.track.byte:
        tracker = BYTETracker(cfg.track)
    elif cfg.track.deep:
        tracker = SparseTracker(cfg.track)
    elif cfg.track.bot:
        tracker = BoTSORT(cfg.track)

    video_path = cfg.video.path
    video_imgs_name = sorted(os.listdir(video_path))
    dataloader = [ os.path.join(video_path, n) for n in video_imgs_name ]
    
    if cfg.video.save_video_result:
        height, width = cv2.imread(dataloader[0]).shape[:2]
        vid_writer = cv2.VideoWriter(
            os.path.join(results_folder, cfg.video.name), cv2.VideoWriter_fourcc(*"mp4v"), cfg.video.fps, (int(width), int(height))
        )
    
    for cur_iter, (img_data, img_name) in enumerate(zip(dataloader, video_imgs_name)):
        with torch.no_grad():
            img_info = {}
            frame_id = int(img_name[:-4])
            img = cv2.imread(img_data)
            ori_img = img.copy()
            img_info['height'] = img.shape[0]
            img_info['width'] = img.shape[1]
            img, _ = preproc(img, cfg.video.test_size, cfg.video.rgb_means, cfg.video.std)
            img = torch.from_numpy(img)
            img_info["image"] = img
            if cfg.track.fp16:
                img_info["image"] = img.half()  # to FP16
            timer.tic()
            outputs = model([img_info])  
                
        # run tracking
        if outputs[0]["instances"] is not None:
            # import pdb;pdb.set_trace()
            online_targets = tracker.update(
                outputs[0]["instances"], ori_img
            )
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > cfg.track.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
            timer.toc()
            online_im = plot_tracking(
                ori_img, online_tlwhs, online_ids, frame_id=frame_id, fps=1. / timer.average_time
            )
         
        if cfg.video.save_img_result:
            cv2.imwrite(os.path.join(results_folder, img_name), online_im)
        if cfg.video.save_video_result:
            vid_writer.write(online_im)
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
    logger.info("Complete inference !")


def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)
    
    model = instantiate(cfg.model)
    model.to(cfg.train.device)
    model.device = torch.device(cfg.train.device)
    model = create_ddp_model(model)
    
    # using ema for evaluation
    ema.may_build_model_ema(cfg, model)
    DetectionCheckpointer(model, **ema.may_get_ema_checkpointer(cfg, model)).load(cfg.train.init_checkpoint)
    # Apply ema state for evaluation
    if cfg.train.model_ema.enabled and cfg.train.model_ema.use_ema_weights_for_eval_only:
        ema.apply_model_ema(model)
    do_demo(cfg, model)

if __name__ == "__main__":
    args = default_argument_parser(epilog = "SparseTrack Eval").parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
'''
CUDA_VISIBLE_DEVICES=0 python track.py  --num-gpus 1  --config-file demo_track_cfg.py  
'''
