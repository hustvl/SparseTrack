from detectron2.data import (
    build_detection_test_loader,
)
from detectron2.evaluation import (
    COCOEvaluator,
)
from detectron2.config import get_cfg
import torch.distributed as dist
import os, torch


# test data cfg
opt = get_cfg()
opt.DATASETS.TEST = ("my_val",)
opt.OUTPUT_DIR = "./yolox_mix17"
opt.DATALOADER.NUM_WORKERS = 4
opt.TEST.DETECTIONS_PER_IMAGE = 1200

    
def build_train_loader(
    data_dir, json_file, input_size, degrees, translate, scale, shear, perspective, enable_mixup, num_workers, batch_size, is_distributed, no_aug=False
):
    from datasets.data import (
        MOTDataset,
        TrainTransform,
        YoloBatchSampler,
        DataLoader,
        InfiniteSampler,
        MosaicDetection,
    )

    dataset = MOTDataset(
        data_dir= data_dir,
        json_file=json_file,
        name='train',
        img_size= input_size,
        preproc=TrainTransform(
            rgb_means=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_labels=600,
        ),
    )

    dataset = MosaicDetection(
        dataset,
        mosaic=not no_aug,
        img_size= input_size,
        preproc=TrainTransform(
            rgb_means=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_labels=1200,
        ),
        degrees= degrees,
        translate= translate,
        scale= scale,
        shear= shear,
        perspective= perspective,
        enable_mixup= enable_mixup,
    )

    if is_distributed:
        batch_size = batch_size // dist.get_world_size()

    sampler = InfiniteSampler(
        len(dataset), seed=0
    )

    batch_sampler = YoloBatchSampler(
        sampler=sampler,
        batch_size=batch_size,
        drop_last=False,
        input_dimension= input_size,
        mosaic=not no_aug,
    )

    dataloader_kwargs = {"num_workers": num_workers , "pin_memory": True}
    dataloader_kwargs["batch_sampler"] = batch_sampler
    train_loader = DataLoader(dataset, **dataloader_kwargs)

    return train_loader

def build_test_loader(test_size, infer_batch = 1):
    """
    Returns:
        iterable

    It now calls :func:`detectron2.data.build_detection_test_loader`.
    Overwrite it if you'd like a different data loader.
    """
    from datasets.mot_mapper import MOTtestMapper
    from datasets.data import ValTransform
   
    print(" building val loader ...")
    mapper = MOTtestMapper(
        test_size = test_size,
        preproc = ValTransform(
            rgb_means=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    )
    return build_detection_test_loader(opt, dataset_name = opt.DATASETS.TEST[0], mapper=mapper, batch_size = infer_batch)

def build_evaluator(output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    print(" building val evaluator ...")
    if output_folder is None:
        output_folder = os.path.join(opt.OUTPUT_DIR, "inference")
    return COCOEvaluator(opt.DATASETS.TEST[0], output_dir=output_folder)
