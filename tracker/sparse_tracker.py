import numpy as np
from collections import deque
 
import os.path as osp
import copy
import torch
import torch.nn.functional as F
from tracker import pbcvt 
from .kalman_filter import KalmanFilter
from .matching import *
from .basetrack import BaseTrack, TrackState

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.deep_vector = self._get_deep_vec()

        self.score = score
        self.tracklet_len = 0
        
    def _get_deep_vec(self):
        cx = self._tlwh[0] + 0.5 * self._tlwh[2]
        y2 = self._tlwh[1] +  self._tlwh[3]
        lendth = 2000 - y2
        return np.asarray([cx, y2, lendth], dtype=np.float)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0

        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][6] = 0
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()

        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xywh(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):

        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_track.tlwh))
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh

        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_tlwh))

        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            R = H[:2, :2]
            R8x8 = np.kron(np.eye(4, dtype=float), R)# np.kron 
            t = H[:2, 2]

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = R8x8.dot(mean)
                mean[:2] += t
                cov = R8x8.dot(cov).dot(R8x8.transpose())

                stracks[i].mean = mean
                stracks[i].covariance = cov
    
    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret
    @property
    def xywh(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2.0
        return ret
    @property
    # @jit(nopython=True)
    def deep_vec(self):
        """Convert bounding box to format `((top left, bottom right)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        cx = ret[0] + 0.5 * ret[2]
        y2 = ret[1] +  ret[3]
        lendth = 2000 - y2
        return np.asarray([cx, y2, lendth], dtype=np.float)

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlwh_to_xywh(tlwh):
        """Convert bounding box to format `(center x, center y, width,
        height)`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret
    
    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)
    
    def to_xywh(self):
        return self.tlwh_to_xywh(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class SparseTracker(object):
    def __init__(self, args, frame_rate = 30):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.args = args
            
        self.pre_img = None
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()
        self.down_scale = args.down_scale
        self.layers = args.depth_levels 
           
    def get_deep_range(self, obj, step):
        col = []
        for t in obj:
            lend = (t.deep_vec)[2]
            col.append(lend)
        max_len, mix_len = max(col), min(col)
        if max_len != mix_len:
            deep_range =np.arange(mix_len, max_len, (max_len - mix_len + 1) / step)
            if deep_range[-1] < max_len:
                deep_range = np.concatenate([deep_range, np.array([max_len],)])
                deep_range[0] = np.floor(deep_range[0])
                deep_range[-1] = np.ceil(deep_range[-1])
        else:    
            deep_range = [mix_len,] 
        mask = self.get_sub_mask(deep_range, col)      
        return mask
    
    def get_sub_mask(self, deep_range, col):
        mix_len=deep_range[0]
        max_len=deep_range[-1]
        if max_len == mix_len:
            lc = mix_len   
        mask = []
        for d in deep_range:
            if d > deep_range[0] and d < deep_range[-1]:
                mask.append((col >= lc) & (col < d)) 
                lc = d
            elif d == deep_range[-1]:
                mask.append((col >= lc) & (col <= d)) 
                lc = d 
            else:
                lc = d
                continue
        return mask
    
    def DCM(self, detections, tracks, activated_starcks, refind_stracks, levels, thresh, is_fuse):
        if len(detections) > 0:
            det_mask = self.get_deep_range(detections, levels) 
        else:
            det_mask = []

        if len(tracks)!=0:
            track_mask = self.get_deep_range(tracks, levels)
        else:
            track_mask = []

        u_detection, u_tracks, res_det, res_track = [], [], [], []
        if len(track_mask) != 0:
            if  len(track_mask) < len(det_mask):
                for i in range(len(det_mask) - len(track_mask)):
                    idx = np.argwhere(det_mask[len(track_mask) + i] == True)
                    for idd in idx:
                        res_det.append(detections[idd[0]])
            elif len(track_mask) > len(det_mask):
                for i in range(len(track_mask) - len(det_mask)):
                    idx = np.argwhere(track_mask[len(det_mask) + i] == True)
                    for idd in idx:
                        res_track.append(tracks[idd[0]])
        
            for dm, tm in zip(det_mask, track_mask):
                det_idx = np.argwhere(dm == True)
                trk_idx = np.argwhere(tm == True)
                
                # search det 
                det_ = []
                for idd in det_idx:
                    det_.append(detections[idd[0]])
                det_ = det_ + u_detection
                # search trk
                track_ = []
                for idt in trk_idx:
                    track_.append(tracks[idt[0]])
                # update trk
                track_ = track_ + u_tracks
                
                dists = iou_distance(track_, det_)
                if (not self.args.mot20) and is_fuse:
                    dists = fuse_score(dists, det_)
                matches, u_track_, u_det_ = linear_assignment(dists, thresh)
                for itracked, idet in matches:
                    track = track_[itracked]
                    det = det_[idet]
                    if track.state == TrackState.Tracked:
                        track.update(det_[idet], self.frame_id)
                        activated_starcks.append(track)
                    else:
                        track.re_activate(det, self.frame_id, new_id=False)
                        refind_stracks.append(track)
                u_tracks = [track_[t] for t in u_track_]
                u_detection = [det_[t] for t in u_det_]
                
            u_tracks = u_tracks + res_track
            u_detection = u_detection + res_det

        else:
            u_detection = detections
            
        return activated_starcks, refind_stracks, u_tracks, u_detection
        
        
    def update(self, output_results, curr_img = None):
        self.frame_id += 1
        if self.frame_id == 1:
            self.pre_img = None
            
        # init stracks
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        # current detections
        bboxes = output_results.pred_boxes.tensor.cpu().numpy()# x1y1x2y2 
        scores = output_results.scores.cpu().numpy()

        # divide high-score dets and low-scores dets
        remain_inds = scores > self.args.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.args.track_thresh
        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]
        
        # tracks preprocess
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
        
        
        # init high-score dets
        if len(dets) > 0:
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets, scores_keep)]   
        else:
            detections = []
        # get strack_pool   
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        
        # predict the current location with KF
        STrack.multi_predict(strack_pool)
        
        # use GMC: for dancetrack--unenabled GMC: 368 - 373
        if self.pre_img is not None:
            warp = pbcvt.GMC(curr_img, self.pre_img, self.down_scale)
        else:
            warp = np.eye(3,3)
        STrack.multi_gmc(strack_pool, warp[:2, :])
        STrack.multi_gmc(unconfirmed, warp[:2, :])
        
        # DCM
        activated_starcks, refind_stracks, u_track, u_detection_high = self.DCM(
                                                                                detections, 
                                                                                strack_pool, 
                                                                                activated_starcks,
                                                                                refind_stracks, 
                                                                                self.layers, 
                                                                                self.args.match_thresh, 
                                                                                is_fuse=True)  
            
            
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []
        r_tracked_stracks = [t for t in u_track if t.state == TrackState.Tracked]   
        
        # DCM
        activated_starcks, refind_stracks, u_strack, u_detection_sec = self.DCM(
                                                                                detections_second, 
                                                                                r_tracked_stracks, 
                                                                                activated_starcks, 
                                                                                refind_stracks, 
                                                                                self.args.depth_levels_low, 
                                                                                0.35, 
                                                                                is_fuse=False) 
        for track in u_strack:
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)  

        
        # Deal with unconfirmed tracks, usually tracks with only one beginning frame 
        detections = [d for d in u_detection_high ]
        dists = iou_distance(unconfirmed, detections)
        if not self.args.mot20:
            dists = fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = linear_assignment(dists, thresh = self.args.confirm_thresh) 
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        # self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        self.pre_img = curr_img
        return output_stracks

def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


# def remove_duplicate_stracks(stracksa, stracksb):
#     pdist = iou_distance(stracksa, stracksb)
#     pairs = np.where(pdist < 0.15)
#     dupa, dupb = list(), list()
#     for p, q in zip(*pairs):
#         timep = stracksa[p].frame_id - stracksa[p].start_frame
#         timeq = stracksb[q].frame_id - stracksb[q].start_frame
#         if timep > timeq:
#             dupb.append(q)
#         else:
#             dupa.append(p)
#     resa = [t for i, t in enumerate(stracksa) if not i in dupa]
#     resb = [t for i, t in enumerate(stracksb) if not i in dupb]
#     return resa, resb