

# SparseTrack
####  SparseTrack is a simply and strong multi-object tracker. 

<!-- <p align="center"><img src="assets/DCM.png" width="500"/></p>  -->

> [**SparseTrack: Multi-Object Tracking by Performing Scene Decomposition based on Pseudo-Depth**](https://arxiv.org/abs/2306.05238)
> 
> Zelin Liu, Xinggang Wang, Cheng Wang, Wenyu Liu, Xiang Bai
> 
> *[arXiv 2306.05238](https://arxiv.org/abs/2306.05238)*

## News
The code will be released in a week

## Abstract
Exploring robust and efficient association methods has always been an important issue in multiple-object tracking (MOT).
Although existing tracking methods have achieved impressive performance, congestion and frequent occlusions still pose challenging problems in multi-object tracking. We reveal that performing sparse decomposition on dense scenes is a crucial step to enhance the performance of associating occluded targets. To this end, we propose a pseudo-depth estimation method for obtaining the relative depth of targets from 2D images. Secondly, we design a depth cascading matching (DCM) algorithm, which can use the obtained depth information to convert a dense target set into multiple sparse target subsets and perform data association on these sparse target subsets in order from near to far. By integrating the pseudo-depth method and the DCM strategy into the data association process, we propose a new tracker, called SparseTrack. SparseTrack provides a new perspective for solving the challenging crowded scene MOT problem and achieves comparable performance with state-of-the-art (SOTA) on the MOT17 and MOT20 test set.

<p align="center"><img src="DCM.png" width="500"/></p> 
 
## Tracking performance
### Results on MOT challenge test set
| Dataset    | HOTA | MOTA | IDF1 | MT | ML | FP | FN | IDs |
|------------|-------|-------|------|------|-------|-------|------|------|
|MOT17       | 65.1 | 81.0 | 80.1 | 54.6% | 14.3% | 23904 | 81927 | 1170 |
|MOT20       | 63.4 | 78.2 | 77.3 | 69.9% | 9.2%  | 25108 | 86720 | 1116 |

 ### Comparison on DanceTrack test set
|  Method  | HOTA | DetA | AssA | MOTA | IDF1 |
|------------|-------|-------|------|------|-------|
| SparseTrack | 55.5 (**+7.8**) | 78.9 (**+7.9**) | 39.1 (**+7.0**) | 91.3 (**+1.7**) | 58.3 (**+4.4**) |
| ByteTrack  |  47.7 | 71.0 | 32.1 | 89.6 | 53.9 | 
    
**Notes**: 
- All the inference experiments are performed on 1 NVIDIA GeForce RTX 3090 GPUs. 
- Each experiment uses the same detector and model weights as [ByteTrack](https://github.com/ifzhang/ByteTrack) . 
- SparseTrack relies on IoU distance association only and do not use any appearance embedding, learnable motion, and attention components.
 
## Installation
#### Dependence
This project is an implementation version of [Detectron2](https://github.com/facebookresearch/detectron2) and requires the compilation of [OpenCV](https://opencv.org/), [Boost](https://www.boost.org), and [pbcvt](https://github.com/Algomorph/pyboostcvconverter).
#### Install
```shell
git clone https://github.com/hustvl/SparseTrack.git
cd SparseTrack
pip install -r requirements.txt
pip install Cython  
pip install cython_bbox
```

## Data preparation
Download [MOT17](https://motchallenge.net/), [MOT20](https://motchallenge.net/), [CrowdHuman](https://www.crowdhuman.org/), [Cityperson](https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/DATASET_ZOO.md), [ETHZ](https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/DATASET_ZOO.md) and put them under ROOT/ in the following structure:
```
ROOT
   |
   |——————SparseTrack(repo)
   |
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
            └——————eth01
            └——————...
            └——————eth07
```
Then, you need to turn the datasets to COCO format and mix different training data:
```
cd <ROOT>/SparseTrack
python3 tools/convert_mot17_to_coco.py
python3 tools/convert_mot20_to_coco.py
python3 tools/convert_crowdhuman_to_coco.py
python3 tools/convert_cityperson_to_coco.py
python3 tools/convert_ethz_to_coco.py
```
Creating different training mix_data:
```
cd <ROOT>/SparseTrack

# training on CrowdHuman and MOT17 half train, evaluate on MOT17 half val.
python3 tools/mix_data_ablation.py

# training on CrowdHuman and MOT20 half train, evaluate on MOT20 half val.
python3 tools/mix_data_ablation_20.py

# training on MOT17, CrowdHuman, ETHZ, Citypersons, evaluate on MOT17 train.
python3 tools/mix_data_test_mot17.py

# training on MOT20 and CrowdHuman, evaluate on MOT20 train.
python3 tools/mix_data_test_mot20.py
```

## Model zoo
See [ByteTrack.model_zoo](https://github.com/ifzhang/ByteTrack#model-zoo). All tracking results are obtained via the corresponding YOLOX_X model weights for inference.

Additionally, we conducted joint training on MOT20 train half and Crowdhuman, and evaluated on MOT20 val half. The model as follows:[yolox_x_mot20_ablation]

The model trained on DanceTrack can be available at [yolox_x_dancetrack](https://drive.google.com/drive/folders/1-uxcNTi7dhuDNGC5MmzXyllLzmVbzXay?usp=sharing).  

**Notes**: 
- All models have been published. We do not train any additional models.


## Training

## Tracking
 
## Demo
 
## Acknowledgements
A large part of the code is borrowed from [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX), [FairMOT](https://github.com/ifzhang/FairMOT), [ByteTrack](https://github.com/ifzhang/ByteTrack), [Detectron2](https://github.com/facebookresearch/detectron2). 
 Many thanks for their wonderful works.

## Citation -->
If you find SparseTrack is useful in your research or applications, please consider giving us a star 🌟 and citing it by the following BibTeX entry.
```bibtex
@inproceedings{SparseTrack,
  title={SparseTrack: Multi-Object Tracking by Performing Scene Decomposition based on Pseudo-Depth},
  author={Liu, Zelin and Wang, Xinggang and Wang, Cheng and Liu, Wenyu and Bai, Xiang},
  journal={arXiv preprint arXiv:2006.13164}
  year={2023}
}
```

