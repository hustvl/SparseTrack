import logging
import os
import glob
import motmetrics as mm
from pathlib import Path
from collections import OrderedDict
import torch
import torch.backends.cudnn as cudnn

from detectron2.utils import comm
from detectron2.config import CfgNode
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger
from detectron2.config import LazyConfig, instantiate
from detectron2.engine.defaults import create_ddp_model, _try_get_key, _highlight
from detectron2.utils.collect_env import collect_env_info
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import (
    launch,
    default_argument_parser,
)

from utils import ema 
from utils.model_utils import fuse_model
from tracker.eval.evaluators import MOTEvaluator
from register_data import *

logger = logging.getLogger("detectron2")

def compare_dataframes(gts, ts):
    accs = []
    names = []
    for k, tsacc in ts.items():
        if k in gts:            
            logger.info('Comparing {}...'.format(k))
            accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.5))
            names.append(k)
        else:
            logger.warning('No ground truth for {}, skipping.'.format(k))

    return accs, names


def default_track_setup(cfg, args):
    """
    Perform some basic common setups at the beginning of a job, including:

    1. Set up the detectron2 logger
    2. Log basic information about environment, cmdline arguments, and config
    3. Backup the config to the output directory

    Args:
        cfg (CfgNode or omegaconf.DictConfig): the full config to be used
        args (argparse.NameSpace): the command line arguments to be logged
    """
    output_dir = _try_get_key(cfg, "OUTPUT_DIR", "output_dir", "train.output_dir")
    if comm.is_main_process() and output_dir:
        PathManager.mkdirs(output_dir)

    rank = comm.get_rank()
    setup_logger(output_dir, distributed_rank=rank, name="fvcore")
    logger = setup_logger(output_dir, distributed_rank=rank)

    logger.info("Rank of current process: {}. World size: {}".format(rank, comm.get_world_size()))
    logger.info("Environment info:\n" + collect_env_info())

    logger.info("Command line arguments: " + str(args))
    if hasattr(args, "config_file") and args.config_file != "":
        logger.info(
            "Contents of args.config_file={}:\n{}".format(
                args.config_file,
                _highlight(PathManager.open(args.config_file, "r").read(), args.config_file),
            )
        )

    if comm.is_main_process() and output_dir:
        # Note: some of our scripts may expect the existence of
        # config.yaml in output directory
        path = os.path.join(output_dir, "config.yaml")
        if isinstance(cfg, CfgNode):
            logger.info("Running with full config:\n{}".format(_highlight(cfg.dump(), ".yaml")))
            with PathManager.open(path, "w") as f:
                f.write(cfg.dump())
        else:
            LazyConfig.save(cfg, path)
        logger.info("Full config saved to {}".format(path))
    
    
def do_track(cfg, model):
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
    results_folder = os.path.join(file_name, "track_results")    
    os.makedirs(results_folder, exist_ok=True)    

    # build evaluator
    evaluator = MOTEvaluator(
        args=cfg,
        dataloader=instantiate(cfg.dataloader.test),
    )

    model.eval()
    if cfg.track.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)#

    # start evaluate
    evaluator.evaluate(
        model,  cfg.track.fp16, results_folder
    )

    # evaluate MOTA
    mm.lap.default_solver = 'lap'

    if cfg.track.val_ann == 'val_half.json':
        gt_type = '_val_half'
    else:
        gt_type = ''
    print('gt_type', gt_type)
    if cfg.track.mot20:
        gtfiles = glob.glob(os.path.join('/data/zelinliu/MOT20/train', '*/gt/gt{}.txt'.format(gt_type)))
    else:
        gtfiles = glob.glob(os.path.join('/data/zelinliu/MOT17/train', '*/gt/gt{}.txt'.format(gt_type)))
    print('gt_files', gtfiles)
    tsfiles = [f for f in glob.glob(os.path.join(results_folder, '*.txt')) if not os.path.basename(f).startswith('eval')]

    logger.info('Found {} groundtruths and {} test files.'.format(len(gtfiles), len(tsfiles)))
    logger.info('Available LAP solvers {}'.format(mm.lap.available_solvers))
    logger.info('Default LAP solver \'{}\''.format(mm.lap.default_solver))
    logger.info('Loading files.')
    
    gt = OrderedDict([(Path(f).parts[-3], mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=1)) for f in gtfiles])
    ts = OrderedDict([(os.path.splitext(Path(f).parts[-1])[0], mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=-1)) for f in tsfiles])    
    
    mh = mm.metrics.create()    
    accs, names = compare_dataframes(gt, ts)
    
    logger.info('Running metrics')
    metrics = ['recall', 'precision', 'num_unique_objects', 'mostly_tracked',
               'partially_tracked', 'mostly_lost', 'num_false_positives', 'num_misses',
               'num_switches', 'num_fragmentations', 'mota', 'motp', 'num_objects']
    summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)

    div_dict = {
        'num_objects': ['num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations'],
        'num_unique_objects': ['mostly_tracked', 'partially_tracked', 'mostly_lost']}
    for divisor in div_dict:
        for divided in div_dict[divisor]:
            summary[divided] = (summary[divided] / summary[divisor])
    fmt = mh.formatters
    change_fmt_list = ['num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations', 'mostly_tracked',
                       'partially_tracked', 'mostly_lost']
    for k in change_fmt_list:
        fmt[k] = fmt['mota']
    print(mm.io.render_summary(summary, formatters=fmt, namemap=mm.io.motchallenge_metric_names))

    metrics = mm.metrics.motchallenge_metrics + ['num_objects']
    summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
    print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))
    logger.info('Completed')


def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_track_setup(cfg, args)
    
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
    do_track(cfg, model)

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
CUDA_VISIBLE_DEVICES=0 python3 track.py  --num-gpus 1  --config-file mot17_track_cfg.py 

CUDA_VISIBLE_DEVICES=0 python3 track.py  --num-gpus 1  --config-file mot20_track_cfg.py 

CUDA_VISIBLE_DEVICES=0 python3 track.py  --num-gpus 1  --config-file mot17_ab_track_cfg.py 

CUDA_VISIBLE_DEVICES=0 python3 track.py  --num-gpus 1  --config-file mot20_ab_track_cfg.py  

CUDA_VISIBLE_DEVICES=0 python3 track.py  --num-gpus 1  --config-file dancetrack_bot_cfg.py  

CUDA_VISIBLE_DEVICES=0 python3 track.py  --num-gpus 1  --config-file dancetrack_sparse_cfg.py  
'''