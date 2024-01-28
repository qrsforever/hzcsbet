#!/usr/bin/env python3

from aux.executor import ExecutorBase
from aux.message import SharedResult
from aux.constant import ByteTrackerArgs

import numpy as np


class TrackExecutor(ExecutorBase):
    _name = 'Tracking'

    def run(self, frame: np.ndarray, msg: SharedResult, cache: dict) -> SharedResult:
        xyxy_list, conf_list = msg.boxes_xyxy, msg.boxes_conf
        state = np.array([[*xyxy, conf] for xyxy, conf in zip(xyxy_list, conf_list)], dtype=float)
        tracks = cache['tracker'].update(output_results=state, img_info=frame.shape, img_size=frame.shape)
        track_list = [track.tlbr for track in tracks]

        tracklet_ids = [0] * len(xyxy_list)
        ious = cache['box_iou_batch'](track_list, xyxy_list)
        for tracker_index, detection_index in enumerate(np.argmax(ious, axis=1)):
            if ious[tracker_index, detection_index] != 0:
                tracklet_ids[detection_index] = tracks[tracker_index].track_id
        msg.tracklet_ids = tracklet_ids
        return msg

    def pre_loop(self, cache):
        from yolox.tracker.byte_tracker import BYTETracker
        from cython_bbox import bbox_overlaps as bbox_ious
        import numpy as np
        np.float = float # TODO work around

        def box_iou_batch(atlbrs, btlbrs):
            ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=float)
            if ious.size == 0:
                return ious
            atlbrs, btlbrs = np.array(atlbrs), np.array(btlbrs)
            ious = bbox_ious(
                np.ascontiguousarray(atlbrs, dtype=float),
                np.ascontiguousarray(btlbrs, dtype=float))
            return ious
        cache['tracker'] = BYTETracker(ByteTrackerArgs)
        cache['box_iou_batch'] = box_iou_batch
