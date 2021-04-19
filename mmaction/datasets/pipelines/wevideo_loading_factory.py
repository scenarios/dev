import io
import os
import os.path as osp
import shutil
import warnings

import mmcv
import numpy as np
import torch
from mmcv.fileio import FileClient
from torch.nn.modules.utils import _pair

from ...utils import get_random_string, get_shm_dir, get_thread_id
from ..registry import PIPELINES

from .loading import SampleFrames

@PIPELINES.register_module()
class SampleWeVideoFrames(SampleFrames):

    def __init__(self, clip_len, frame_interval=2, test_mode=False):

        super().__init__(clip_len, frame_interval, test_mode=test_mode)

    def _get_clips(self, center_index, skip_offsets, shot_info):
        start = center_index - (self.clip_len // 2) * self.frame_interval
        end = center_index + ((self.clip_len + 1) // 2) * self.frame_interval
        frame_inds = list(range(start, end, self.frame_interval))
        if not self.test_mode:
            frame_inds = frame_inds + skip_offsets
        frame_inds = np.clip(frame_inds, shot_info[0], shot_info[1] - 1)
        return frame_inds

    def __call__(self, results):
        fps = results['fps']
        timestamp = results['timestamp']
        timestamp_start = results['timestamp_start']
        timestamp_end = results['timestamp_end']
        shot_info = results['shot_info']

        num_frames = fps * (timestamp_end - timestamp_start)
        if num_frames < self.frame_interval * self.clip_len + 1: # +1 for safety
            return None

        if timestamp:
            timestamp = int(timestamp)
            center_index = fps * (timestamp - timestamp_start) + 1
        else:
            center_index = np.random.randint(
                low=self.clip_len//2*self.frame_interval,
                high=max(num_frames-self.clip_len//2*self.frame_interval, self.clip_len//2*self.frame_interval+1)
            )
        skip_offsets = np.random.randint(
            -self.frame_interval // 2, (self.frame_interval + 1) // 2,
            size=self.clip_len)
        frame_inds = self._get_clips(center_index, skip_offsets, shot_info)

        results['frame_inds'] = np.array(frame_inds, dtype=np.int)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = self.frame_interval
        results['num_clips'] = 1
        results['crop_quadruple'] = np.array([0, 0, 1, 1], dtype=np.float32)
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'clip_len={self.clip_len}, '
                    f'frame_interval={self.frame_interval}, '
                    f'test_mode={self.test_mode})')
        return repr_str

