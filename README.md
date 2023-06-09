
# SparseTrack
####  SparseTrack is a simply and strong multi-object tracker. 

 **SparseTrack: Multi-Object Tracking by Performing Scene Decomposition based on Pseudo-Depth**
 



## News
The code will be released in a week


## Abstract
Exploring robust and efficient association methods has always been an important issue in multiple-object tracking (MOT).
Although existing tracking methods have achieved impressive performance, congestion and frequent occlusions still pose challenging problems in multi-object tracking. 
We reveal that performing sparse decomposition on dense scenes is a crucial step to enhance the performance of associating occluded targets. 
To this end, we propose a pseudo-depth estimation method for obtaining the relative depth of targets from 2D images.  
Secondly, we design a depth cascading matching (DCM) algorithm, which can use the obtained depth information to convert a dense target set into multiple sparse target subsets and perform data association on these sparse target subsets in order from near to far. 
By integrating the pseudo-depth method and the DCM strategy into the data association process, we propose a new tracker, called SparseTrack. 
SparseTrack provides a new perspective for solving the challenging crowded scene MOT problem and achieves comparable performance with state-of-the-art (SOTA) on the MOT17 and MOT20 test set.
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
<!-- - [Installation](docs/install.md)
- [Prepare Dataset](docs/prepare_dataset.md)
- [Train and Eval](docs/train_eval.md)
- [Visualization](docs/visualization.md) -->


## Data preparation
<!-- 
- [ ] centerline detection & topology support
- [x] multi-modal checkpoints
- [x] multi-modal code
- [ ] lidar modality code
- [x] argoverse2 dataset 
- [x] Nuscenes dataset 
- [x] MapTR checkpoints
- [x] MapTR code
- [x] Initialization -->

## Model zoo
 
## Training

## Tracking
 
## Demo
 
## Acknowledgements

A large part of the code is borrowed from [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX), [FairMOT](https://github.com/ifzhang/FairMOT), [ByteTrack](https://github.com/ifzhang/ByteTrack), [Detectron2](https://github.com/facebookresearch/detectron2). 
 Many thanks for their wonderful works.

<!-- ## Citation -->
<!-- If you find MapTR is useful in your research or applications, please consider giving us a star 🌟 and citing it by the following BibTeX entry.
```bibtex
@inproceedings{MapTR,
  title={MapTR: Structured Modeling and Learning for Online Vectorized HD Map Construction},
  author={Liao, Bencheng and Chen, Shaoyu and Wang, Xinggang and Cheng, Tianheng, and Zhang, Qian and Liu, Wenyu and Huang, Chang},
  booktitle={International Conference on Learning Representations},
  year={2023}
} -->
```
