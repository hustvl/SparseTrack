# SparseTrack
####  SparseTrack integrated with the YOLOv8 detector. 

**Here includes the implementation of the [Detectron2](https://github.com/facebookresearch/detectron2) version of YOLOv8 detector and its usage in SparseTrack for tracking.**

## Usage
```shell
git clone -b v8 https://github.com/hustvl/SparseTrack.git
# rename 'SparseTrack' to 'yolov8'
cd yolov8
pip install -r requirements.txt
pip install Cython  
pip install cython_bbox
```

## Data preparation
Download [MOT17](https://motchallenge.net/), [MOT20](https://motchallenge.net/), [CrowdHuman](https://www.crowdhuman.org/), [Cityperson](https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/DATASET_ZOO.md), [ETHZ](https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/DATASET_ZOO.md) and put them under ROOT/ in the following structure:
```
ROOT
   |
   |——————yolov8(repo)
   |           └—————mix
   |                  └——————mix_17/annotations
   |                  └——————mix_20/annotations
   |                  └——————ablation_17/annotations
   |                  └——————ablation_20/annotations
   |——————MOT17
   |        └——————train
   |        └——————test
   └——————crowdhuman
   |         └——————Crowdhuman_train
   |         └——————Crowdhuman_val
   |         └——————annotation_train.odgt
   |         └——————annotation_val.odgt
   └——————MOT20
   |        └——————train
   |        └——————test
   └——————Citypersons
   |        └——————images
   |        └——————labels_with_ids
   └——————ETHZ
   |        └——————eth01
   |        └——————...
   |        └——————eth07
   └——————dancetrack
               └——————train
               └——————train_seqmap.txt
               └——————test
               └——————test_seqmap.txt
               └——————val
               └——————val_seqmap.txt

   
```
Then, you need to turn the datasets to COCO format and mix different training data:
```
cd <ROOT>/yolov8
python3 tools/convert_mot17_to_coco.py
python3 tools/convert_mot20_to_coco.py
python3 tools/convert_crowdhuman_to_coco.py
python3 tools/convert_cityperson_to_coco.py
python3 tools/convert_ethz_to_coco.py
python3 tools/convert_dance_to_coco.py
```
Creating different training mix_data:
```
cd <ROOT>/yolov8

# training on CrowdHuman and MOT17 half train, evaluate on MOT17 half val.
python3 tools/mix_data_ablation.py

# training on CrowdHuman and MOT20 half train, evaluate on MOT20 half val.
python3 tools/mix_data_ablation_20.py

# training on MOT17, CrowdHuman, ETHZ, Citypersons, evaluate on MOT17 train.
python3 tools/mix_data_test_mot17.py

# training on MOT20 and CrowdHuman, evaluate on MOT20 train.
python3 tools/mix_data_test_mot20.py
``` 


## Training
All training is conducted on a unified script. You need to change the **VAL_JSON** and **VAL_PATH** in [register_data.py](https://github.com/hustvl/SparseTrack/blob/main/register_data.py), and then run as follows：
```
# training on MOT17, CrowdHuman, ETHZ, Citypersons, evaluate on MOT17 train set.
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --num-gpus 4  --config-file mot17_train_config.py 


# training on MOT20, CrowdHuman, evaluate on MOT20 train set.
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --num-gpus 4  --config-file mot20_train_config.py 
```
**Notes**: 
For MOT20, you need to clip the bounding boxes inside the image.

Add clip operation in line 138-139 in [data_augment.py](https://github.com/hustvl/SparseTrack/blob/v8/datasets/data/data_augment.py), line 118-121 in [mosaicdetection.py](https://github.com/hustvl/SparseTrack/blob/v8/datasets/data/datasets/mosaicdetection.py), line 213-221 in mosaicdetection.py, line 115-118 in [boxes.py](https://github.com/hustvl/SparseTrack/blob/v8/utils/boxes.py).

## Tracking
All tracking experimental scripts are run in the following manner. You first place the model weights in the **<ROOT/SparseTrack/pretrain/>**, and change the **VAL_JSON** and **VAL_PATH** in [register_data.py](https://github.com/hustvl/SparseTrack/blob/v8/register_data.py).
```
# tracking on mot17 train set or test set
CUDA_VISIBLE_DEVICES=0 python3 track.py  --num-gpus 1  --config-file mot17_track_cfg.py 


# tracking on mot20 train set or test set
CUDA_VISIBLE_DEVICES=0 python3 track.py  --num-gpus 1  --config-file mot20_track_cfg.py 
```

## Citation -->
If you find SparseTrack is useful in your research or applications, please consider giving us a star 🌟 and citing it by the following BibTeX entry.
```bibtex
@inproceedings{SparseTrack,
  title={SparseTrack: Multi-Object Tracking by Performing Scene Decomposition based on Pseudo-Depth},
  author={Liu, Zelin and Wang, Xinggang and Wang, Cheng and Liu, Wenyu and Bai, Xiang},
  journal={arXiv preprint arXiv:2306.05238},
  year={2023}
}
```

