import json
import os
# ROOT = '/data/zelinliu' --- /path/to/your/root/
mot_json = json.load(open('/data/zelinliu/MOT17/annotations/train_half.json','r'))

img_list = list()
for img in mot_json['images']:
    img['file_name'] = '/data/zelinliu/MOT17/train/' + img['file_name']
    img_list.append(img)

ann_list = list()
for ann in mot_json['annotations']:
    ann_list.append(ann)

video_list = mot_json['videos']
category_list = mot_json['categories']

print('mot17')

max_img = 100000
max_ann = 200000000
max_video = 30

crowdhuman_json = json.load(open('/data/zelinliu/crowdhuman/annotations/train.json','r'))
img_id_count = 0
for img in crowdhuman_json['images']:
    img_id_count += 1
    img['file_name'] = '/data/zelinliu/crowdhuman/Crowdhuman_train/' + img['file_name']
    img['frame_id'] = img_id_count
    img['prev_image_id'] = img['id'] + max_img
    img['next_image_id'] = img['id'] + max_img
    img['id'] = img['id'] + max_img
    img['video_id'] = max_video
    img_list.append(img)
    
for ann in crowdhuman_json['annotations']:
    ann['id'] = ann['id'] + max_ann
    ann['image_id'] = ann['image_id'] + max_img
    ann_list.append(ann)
    
video_list.append({
    'id': max_video,
    'file_name': 'crowdhuman_train'
})

print('crowdhuman_train')

max_img = 10000000
max_ann = 200000000000

crowdhuman_val_json = json.load(open('/data/zelinliu/crowdhuman/annotations/val.json','r'))
img_id_count = 0
for img in crowdhuman_val_json['images']:
    img_id_count += 1
    img['file_name'] = '/data/zelinliu/crowdhuman/Crowdhuman_val/' + img['file_name']
    img['frame_id'] = img_id_count
    img['prev_image_id'] = img['id'] + max_img
    img['next_image_id'] = img['id'] + max_img
    img['id'] = img['id'] + max_img
    img['video_id'] = max_video
    img_list.append(img)
    
for ann in crowdhuman_val_json['annotations']:
    ann['id'] = ann['id'] + max_ann
    ann['image_id'] = ann['image_id'] + max_img
    ann_list.append(ann)
    
video_list.append({
    'id': max_video,
    'file_name': 'crowdhuman_val'
})

print('crowdhuman_val')

mix_json = dict()
mix_json['images'] = img_list
mix_json['annotations'] = ann_list
mix_json['videos'] = video_list
mix_json['categories'] = category_list
json.dump(mix_json, open('/data/zelinliu/sparsetrack/mix/ablation_17/annotations/train.json','w'))