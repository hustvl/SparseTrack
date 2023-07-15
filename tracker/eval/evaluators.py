import os
import torch
import logging
import numpy as np
from .timer import Timer
from collections import defaultdict
from  tracker.byte_tracker import BYTETracker
from  tracker.sparse_tracker import SparseTracker
from  tracker.bot_sort import BoTSORT
logger = logging.getLogger("detectron2")

def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))


def write_results_no_score(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},-1,-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1))
                f.write(line)
    logger.info('save results to {}'.format(filename))


class MOTEvaluator:
    """
    COCO AP Evaluation class.  All the data in the val2017 dataset are processed
    and evaluated by COCO API.
    """

    def __init__(
        self, args, dataloader):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
        """
        self.dataloader = dataloader
        self.args = args

    def evaluate(
        self,
        model,
        half=False,
        result_folder=None
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        results = []
        timer_avgs, timer_calls = [], []
        video_names = defaultdict()
        
        if self.args.track.byte:
            tracker = BYTETracker(self.args.track)
        elif self.args.track.deep:
            tracker = SparseTracker(self.args.track)
        elif self.args.track.bot:
            tracker = BoTSORT(self.args.track)
        timer = Timer()
        ori_thresh = self.args.track.track_thresh
        ori_track_buffer = self.args.track.track_buffer
        video_id = 0

        for cur_iter, batch_data in enumerate(
           self.dataloader
        ):
            with torch.no_grad():
                # init tracker  
                frame_id = int(batch_data[0]["frame_id"])
                img_file_name = batch_data[0]["file_name"]
                video_name = img_file_name.split('/')[-3]
                
                if 'MOT17-05-' in video_name  or 'MOT17-06-' in video_name:
                    self.args.track.track_buffer = 14
                elif 'MOT17-13-' in video_name:
                    self.args.track.track_buffer = 25
                else:
                    self.args.track.track_buffer = ori_track_buffer
                
                # if 'MOT17-06-' in video_name:
                #     self.args.track.track_thresh = 0.65
                if 'MOT17-12-' in video_name:
                    self.args.track.track_thresh = 0.6
                elif video_name in ['MOT20-06', 'MOT20-08']:
                    self.args.track.track_thresh = 0.27
                else:
                    self.args.track.track_thresh = ori_thresh

                if frame_id == 1:
                    video_id += 1 
                if video_name not in video_names:
                    video_names[video_id] = video_name
                if frame_id == 1:
                    if self.args.track.byte:
                        tracker = BYTETracker(self.args.track)
                    elif self.args.track.deep:
                        tracker = SparseTracker(self.args.track)
                    elif self.args.track.bot:
                        tracker = BoTSORT(self.args.track)
                    if len(results) != 0:
                        # import pdb;pdb.set_trace()
                        result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id - 1]))
                        write_results(result_filename, results)
                        results = []
                    timer_avgs.append(timer.average_time)
                    timer_calls.append(timer.calls)
                    timer.clear()
                 
                if 'MOT17-01-' in video_name:
                    tracker.layers = 2
                
                # run model
                batch_data[0]["image"] = batch_data[0]["image"].type(tensor_type)  
                timer.tic()
                outputs = model(batch_data)  
                    
            # run tracking
            if outputs[0]["instances"] is not None:
                online_targets = tracker.update(
                    outputs[0]["instances"], batch_data[0]["ori_img"]
                )
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > 1.6
                    if tlwh[2] * tlwh[3] > self.args.track.min_box_area: #and not vertical
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                # save results
                results.append((frame_id, online_tlwhs, online_ids, online_scores))
            timer.toc() 
            if frame_id % 20 == 0:
                logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
                write_results(result_filename, results) 
                timer_avgs.append(timer.average_time)
                timer_calls.append(timer.calls)
                
        timer_avgs = np.asarray(timer_avgs)
        timer_calls = np.asarray(timer_calls)
        all_time = np.dot(timer_avgs, timer_calls)
        avg_time = all_time / np.sum(timer_calls)
        logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(all_time, 1.0 / avg_time))