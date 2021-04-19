import csv
import copy
import os
import warnings
import os.path as osp
from collections import defaultdict
from datetime import datetime

from abc import ABCMeta, abstractmethod

import mmcv
import numpy as np
from mmcv.utils import print_log

from ..core.evaluation.ava_utils import ava_eval, read_labelmap, results2csv
from ..utils import get_root_logger
from .registry import DATASETS

from .pipelines import Compose

import torch
from torch.utils.data import Dataset


class WeVideoBaseDataset(Dataset, metaclass=ABCMeta):
    """Base class for datasets.

    All datasets to process video should subclass it.
    All subclasses should overwrite:

    - Methods:`load_annotations`, supporting to load information from an
    annotation file.
    - Methods:`prepare_train_frames`, providing train data.
    - Methods:`prepare_test_frames`, providing test data.

    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        data_prefix (str | None): Path to a directory where videos are held.
            Default: None.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
        multi_class (bool): Determines whether the dataset is a multi-class
            dataset. Default: False.
        num_video_classes (int | None): Number of video classes of the dataset, used in
            multi-class datasets. Default: None.
        num_action_classes (int | None): Number of action classes of the dataset, used in
            multi-class datasets. Default: None.
        start_index (int): Specify a start index for frames in consideration of
            different filename format. However, when taking videos as input,
            it should be set to 0, since frames loaded from videos count
            from 0. Default: 1.
        modality (str): Modality of data. Support 'RGB', 'Flow', 'Audio'.
            Default: 'RGB'.
        sample_by_class (bool): Sampling by class, should be set `True` when
            performing inter-class data balancing. Only compatible with
            `multi_class == False`. Only applies for training. Default: False.
        power (float | None): We support sampling data with the probability
            proportional to the power of its label frequency (freq ^ power)
            when sampling data. `power == 1` indicates uniformly sampling all
            data; `power == 0` indicates uniformly sampling all classes.
            Default: None.
    """

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_prefix=None,
                 test_mode=False,
                 multi_class=False,
                 num_video_classes=None,
                 num_action_classes=None,
                 start_index=1,
                 modality='RGB',
                 sample_by_class=False,
                 power=None):
        super().__init__()

        self.ann_file = ann_file
        self.data_prefix = osp.realpath(
            data_prefix) if data_prefix is not None and osp.isdir(
                data_prefix) else data_prefix
        self.test_mode = test_mode
        self.multi_class = multi_class
        self.num_video_classes = num_video_classes
        self.num_action_classes = num_action_classes
        self.start_index = start_index
        self.modality = modality
        self.sample_by_class = sample_by_class
        self.power = power
        assert not (self.multi_class and self.sample_by_class)

        self._check_label_format()
        self.pipeline = Compose(pipeline)
        self.video_infos = self.load_annotations()
        if self.sample_by_class:
            self.video_infos_by_class = self.parse_by_class()

    def _check_label_format(self):
        self.video_label_format = 'ONE_INDEXED'
        self.action_label_format = 'ONE_INDEXED'
        with open(self.ann_file, 'r') as fin:
            reader = csv.DictReader(fin)
            for line in reader:
                if int(line['video_class']) == 0:
                    self.video_label_format = 'ZERO_INDEXED'
                    break
            for line in reader:
                if line['action_class'] and int(line['action_class']) == 0:
                    self.video_label_format = 'ZERO_INDEXED'
                    break

    @abstractmethod
    def load_annotations(self):
        """Load the annotation according to ann_file into video_infos."""

    # json annotations already looks like video_infos, so for each dataset,
    # this func should be the same
    def load_json_annotations(self):
        """Load json annotation file to get video information."""
        video_infos = mmcv.load(self.ann_file)
        num_videos = len(video_infos)
        path_key = 'frame_dir' if 'frame_dir' in video_infos[0] else 'filename'
        for i in range(num_videos):
            path_value = video_infos[i][path_key]
            if self.data_prefix is not None:
                path_value = osp.join(self.data_prefix, path_value)
            video_infos[i][path_key] = path_value
            if self.multi_class:
                assert self.num_action_classes is not None and self.num_video_classes is not None
            else:
                assert len(video_infos[i]['label']) == 1
                video_infos[i]['label'] = video_infos[i]['label'][0]
        return video_infos

    def parse_by_class(self):
        video_infos_by_class = defaultdict(list)
        for item in self.video_infos:
            label = item['label']
            video_infos_by_class[label].append(item)
        return video_infos_by_class

    @staticmethod
    def label2array(num, label):
        arr = np.zeros(num, dtype=np.float32)
        arr[label] = 1.
        return arr

    @staticmethod
    def evaluate(self,
                 results,
                 metrics='top_k_accuracy',
                 metric_options=dict(top_k_accuracy=dict(topk=(1, 5))),
                 logger=None,
                 **deprecated_kwargs):
        """Perform evaluation for common datasets.

        Args:
            results (list): Output results.
            metrics (str | sequence[str]): Metrics to be performed.
                Defaults: 'top_k_accuracy'.
            metric_options (dict): Dict for metric options. Options are
                ``topk`` for ``top_k_accuracy``.
                Default: ``dict(top_k_accuracy=dict(topk=(1, 5)))``.
            logger (logging.Logger | None): Logger for recording.
                Default: None.
            deprecated_kwargs (dict): Used for containing deprecated arguments.
                See 'https://github.com/open-mmlab/mmaction2/pull/286'.

        Returns:
            dict: Evaluation results dict.
        """

    @staticmethod
    def dump_results(results, out):
        """Dump data to json/yaml/pickle strings or files."""
        return mmcv.dump(results, out)

    @staticmethod
    def prepare_train_frames(self, idx):
        """Prepare the frames for training given the index."""

    @staticmethod
    def prepare_test_frames(self, idx):
        """Prepare the frames for testing given the index."""


    def __len__(self):
        """Get the size of the dataset."""
        return len(self.video_infos)

    def __getitem__(self, idx):
        """Get the sample for either training or testing given index."""
        if self.test_mode:
            return self.prepare_test_frames(idx)

        return self.prepare_train_frames(idx)


@DATASETS.register_module()
class WeVideoDataset(WeVideoBaseDataset):
    """WeVideo dataset for spatial temporal detection.

    Based on WeVideo video annotation file format, the dataset loads raw frames,
    bounding boxes, proposals and applies specified transformations to return
    a dict containing the frame tensors and other information.

    This datasets can load information from the following files:

    .. code-block:: txt

        ann_file -> wevideo_{train, val}.csv
        exclude_file -> wevideo_{train, val}_excluded_timestamps.csv
        label_file -> wevideo_action_list.pbtxt /
        proposal_file -> wevideo_dense_proposals_{train, val}.pkl

    Particularly, the proposal_file is a pickle file which contains
    ``img_key`` (in format of ``{video_id},{timestamp}``). Example of a pickle
    file:

    .. code-block:: JSON

        {
            ...
            '0f39OWEqJ24,0902':
                array([[0.011   , 0.157   , 0.655   , 0.983   , 0.998163]]),
            '0f39OWEqJ24,0912':
                array([[0.054   , 0.088   , 0.91    , 0.998   , 0.068273],
                       [0.016   , 0.161   , 0.519   , 0.974   , 0.984025],
                       [0.493   , 0.283   , 0.981   , 0.984   , 0.983621]]),
            ...
        }

    Args:
        ann_file (str): Path to the annotation file like
            ``ava_{train, val}_{v2.1, v2.2}.csv``.
        exclude_file (str): Path to the excluded timestamp file like
            ``ava_{train, val}_excluded_timestamps_{v2.1, v2.2}.csv``.
        pipeline (list[dict | callable]): A sequence of data transforms.
        label_file (str): Path to the label file like
            ``ava_action_list_{v2.1, v2.2}.pbtxt`` or
            ``ava_action_list_{v2.1, v2.2}_for_activitynet_2019.pbtxt``.
            Default: None.
        filename_tmpl (str): Template for each filename.
            Default: 'img_{:05}.jpg'.
        proposal_file (str): Path to the proposal file like
            ``ava_dense_proposals_{train, val}.FAIR.recall_93.9.pkl``.
            Default: None.
        person_det_score_thr (float): The threshold of person detection scores,
            bboxes with scores above the threshold will be used. Default: 0.9.
            Note that 0 <= person_det_score_thr <= 1. If no proposal has
            detection score larger than the threshold, the one with the largest
            detection score will be used.
        num_video_classes (int | None): Number of video classes of the dataset, used in
            multi-class datasets. Default: None. (another 1-dim is added for potential
            usage)
        num_action_classes (int | None): Number of action classes of the dataset, used in
            multi-class datasets. Default: None. (another 1-dim is added for potential
            usage)
        custom_video_classes (list[int]): A subset of video class ids from origin dataset.
            Please note that 0 should NOT be selected, and ``num_video_classes``
            should be equal to ``len(custom_video_classes) + 1``
        custom_action_classes (list[int]): A subset of action class ids from origin dataset.
            Please note that 0 should NOT be selected, and ``num_action_classes``
            should be equal to ``len(custom_action_classes) + 1``
        data_prefix (str): Path to a directory where videos are held.
            Default: None.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
        modality (str): Modality of data. Support 'RGB', 'Flow'.
                        Default: 'RGB'.
        num_max_proposals (int): Max proposals number to store. Default: 1000.
        timestamp_start (int): The start point of included timestamps. The
            default value is referred from the official website. Default: 902.
        timestamp_end (int): The end point of included timestamps. The
            default value is referred from the official website. Default: 1798.
    """

    def __init__(self,
                 ann_file,
                 exclude_file,
                 pipeline,
                 video_label_file = None,
                 action_label_file =None,
                 filename_tmpl='img_{:05}.jpg',
                 proposal_file=None,
                 person_det_score_thr=0.9,
                 num_video_classes=81,
                 num_action_classes=81,
                 custom_video_labels=None,
                 custom_action_labels=None,
                 data_prefix=None,
                 test_mode=False,
                 modality='RGB',
                 num_max_proposals=1000):
        # since it inherits from `BaseDataset`, some arguments
        # should be assigned before performing `load_annotations()`

        self.custom_video_labels = custom_video_labels
        self.custom_action_labels = custom_action_labels
        self.video_label_file = video_label_file
        self.action_label_file = action_label_file

        self.exclude_file = exclude_file
        self.proposal_file = proposal_file
        assert 0 <= person_det_score_thr <= 1, (
            'The value of '
            'person_det_score_thr should in [0, 1]. ')
        self.person_det_score_thr = person_det_score_thr
        self.filename_tmpl = filename_tmpl
        self.num_max_proposals = num_max_proposals
        self.logger = get_root_logger()

        self._init_custom_labels()

        super().__init__(
            ann_file,
            pipeline,
            data_prefix,
            test_mode,
            modality=modality,
            num_video_classes=num_video_classes,
            num_action_classes = num_action_classes)

        if self.proposal_file is not None:
            self.proposals = mmcv.load(self.proposal_file)
        else:
            self.proposals = None

        if not test_mode:
            valid_indexes = self.filter_exclude_file()
            self.logger.info(
                f'{len(valid_indexes)} out of {len(self.video_infos)} '
                f'frames are valid.')
            self.video_infos = [self.video_infos[i] for i in valid_indexes]

    def _init_custom_labels(self):
        if self.custom_video_labels is not None:
            assert self.num_video_classes == len(self.custom_video_labels) + 1
            assert 0 not in self.custom_video_labels
            _, class_whitelist = read_labelmap(open(self.video_label_file))
            assert set(self.custom_video_labels).issubset(class_whitelist)

            self.custom_video_labels = tuple([0] + self.custom_video_labels)
        if self.custom_action_labels is not None:
            assert self.num_action_classes == len(self.custom_action_labels) + 1
            assert 0 not in self.custom_action_labels
            _, class_whitelist = read_labelmap(open(self.action_label_file))
            assert set(self.custom_action_labels).issubset(class_whitelist)

            self.custom_action_labels = tuple([0] + self.custom_action_labels)

    def parse_img_record(self, img_records):
        bboxes, action_labels, entity_ids, video_labels = [], [], [], []

        if img_records[0]['video_label'] is not None:
            video_label_idx = np.array([
                record['video_label']
                for record in img_records
            ])
            video_label = np.zeros(self.num_video_classes, dtype=np.float32)
            video_label[video_label_idx] = 1.
            video_labels.append(video_label)

        if img_records[0]['entity_box'] is not None:
            while len(img_records) > 0:
                img_record = img_records[0]
                num_img_records = len(img_records)
                selected_records = list(
                    filter(
                        lambda x: np.array_equal(x['entity_box'], img_record[
                            'entity_box']), img_records))
                num_selected_records = len(selected_records)
                img_records = list(
                    filter(
                        lambda x: not np.array_equal(x['entity_box'], img_record[
                            'entity_box']), img_records))
                assert len(img_records) + num_selected_records == num_img_records

                bboxes.append(img_record['entity_box'])
                valid_action_labels = np.array([
                    selected_record['action_label']
                    for selected_record in selected_records
                ])

                # The format can be directly used by BCELossWithLogits
                action_label = np.zeros(self.num_action_classes, dtype=np.float32)
                action_label[valid_action_labels] = 1.

                action_labels.append(action_label)
                entity_ids.append(img_record['entity_id'])
        elif img_records[0]['action_label'] is not None:
            action_label_idx = np.array([
                record['action_label']
                for record in img_records
            ])
            action_label = np.zeros(self.num_action_classes, dtype=np.float32)
            action_label[action_label_idx] = 1.
            action_labels.append(action_label)
        else:
            pass

        action_labels = np.stack(action_labels) if action_labels else None
        video_labels = np.stack(video_labels) if video_labels else None
        bboxes = np.stack(bboxes) if bboxes else None
        entity_ids = np.stack(entity_ids) if entity_ids else None

        return bboxes, action_labels, entity_ids, video_labels

    def filter_exclude_file(self):
        valid_indexes = []
        if self.exclude_file is None:
            valid_indexes = list(range(len(self.video_infos)))
        else:
            exclude_video_infos = [
                x.strip().split(',') for x in open(self.exclude_file)
            ]
            for i, video_info in enumerate(self.video_infos):
                valid_indexes.append(i)
                for video_id, timestamp in exclude_video_infos:
                    if (video_info['video_id'] == video_id
                            and video_info['timestamp'] == int(timestamp)):
                        valid_indexes.pop()
                        break
        return valid_indexes

    def load_annotations(self):
        video_infos = []
        records_dict_by_img = defaultdict(list)
        with open(self.ann_file, 'r') as fin:
            reader = csv.DictReader(fin)
            for line in reader:
                video_label = int(line['video_class']) if line['video_class'] else None
                video_label = video_label + 1 if \
                    line['video_class'] and self.video_label_format == 'ZERO_INDEXED' else video_label
                if self.custom_video_labels is not None:
                    if video_label not in self.custom_video_labels:
                        continue
                    video_label = self.custom_video_labels.index(video_label)

                action_label = int(line['action_class']) if line['action_class'] else None
                action_label = action_label + 1 if \
                    line['action_class'] and self.action_label_format == 'ZERO_INDEXED' else action_label
                if self.custom_action_labels is not None:
                    if action_label not in self.custom_action_labels:
                        continue
                    action_label = self.custom_action_labels.index(action_label)

                if action_label is None and video_label is None:
                    continue

                video_id = line['video_id']
                timestamp = line['time_stamp']
                img_key = f'{video_id},{timestamp}'

                entity_box = np.array(list(map(float, [line['top'], line['left'], line['bottom'], line['right']]))) if \
                    line['top'] and line['left'] and line['bottom'] and line['right'] else None

                entity_id = int(line['person_id']) if line['person_id'] else None

                self.timestamp_start = int(line['clip_time_start'])
                self.timestamp_end = int(line['clip_time_end'])
                self._FPS = int(line['fps'])

                shot_info = (0, (self.timestamp_end - self.timestamp_start) *
                             self._FPS)

                video_info = dict(
                    video_id=video_id,
                    timestamp=timestamp,
                    entity_box=entity_box,
                    video_label=video_label,
                    action_label=action_label,
                    entity_id=entity_id,
                    shot_info=shot_info,
                    storage_path=line['relative_path'])
                records_dict_by_img[img_key].append(video_info)

        for img_key in records_dict_by_img:
            video_id, timestamp = img_key.split(',')
            bboxes, action_labels, entity_ids, video_labels = self.parse_img_record(
                records_dict_by_img[img_key])
            ann = dict(
                gt_bboxes=bboxes, gt_action_labels=action_labels, entity_ids=entity_ids, gt_video_labels=video_labels)
            frame_dir = records_dict_by_img[img_key][0]['storage_path']
            if self.data_prefix is not None:
                frame_dir = osp.join(self.data_prefix, frame_dir)
            video_info = dict(
                frame_dir=frame_dir,
                video_id=video_id,
                timestamp=timestamp,
                img_key=img_key,
                shot_info=shot_info,
                fps=self._FPS,
                ann=ann)
            video_infos.append(video_info)

        return video_infos

    def prepare_train_frames(self, idx):
        """Prepare the frames for training given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        img_key = results['img_key']

        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = self.modality
        results['start_index'] = self.start_index
        results['timestamp_start'] = self.timestamp_start
        results['timestamp_end'] = self.timestamp_end

        if self.proposals is not None:
            if img_key not in self.proposals:
                results['proposals'] = np.array([[0, 0, 1, 1]])
                results['scores'] = np.array([1])
            else:
                proposals = self.proposals[img_key]
                assert proposals.shape[-1] in [4, 5]
                if proposals.shape[-1] == 5:
                    thr = min(self.person_det_score_thr, max(proposals[:, 4]))
                    positive_inds = (proposals[:, 4] >= thr)
                    proposals = proposals[positive_inds]
                    proposals = proposals[:self.num_max_proposals]
                    results['proposals'] = proposals[:, :4]
                    results['scores'] = proposals[:, 4]
                else:
                    proposals = proposals[:self.num_max_proposals]
                    results['proposals'] = proposals

        ann = results.pop('ann')
        results['gt_bboxes'] = ann['gt_bboxes']
        results['gt_video_labels'] = ann['gt_video_labels']
        results['gt_action_labels'] = ann['gt_action_labels']
        results['entity_ids'] = ann['entity_ids']

        return self.pipeline(results)

    def prepare_test_frames(self, idx):
        """Prepare the frames for testing given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        img_key = results['img_key']

        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = self.modality
        results['start_index'] = self.start_index
        results['timestamp_start'] = self.timestamp_start
        results['timestamp_end'] = self.timestamp_end

        if self.proposals is not None:
            if img_key not in self.proposals:
                results['proposals'] = np.array([[0, 0, 1, 1]])
                results['scores'] = np.array([1])
            else:
                proposals = self.proposals[img_key]
                assert proposals.shape[-1] in [4, 5]
                if proposals.shape[-1] == 5:
                    thr = min(self.person_det_score_thr, max(proposals[:, 4]))
                    positive_inds = (proposals[:, 4] >= thr)
                    proposals = proposals[positive_inds]
                    proposals = proposals[:self.num_max_proposals]
                    results['proposals'] = proposals[:, :4]
                    results['scores'] = proposals[:, 4]
                else:
                    proposals = proposals[:self.num_max_proposals]
                    results['proposals'] = proposals

        ann = results.pop('ann')
        # Follow the mmdet variable naming style.
        results['gt_bboxes'] = ann['gt_bboxes']
        results['gt_video_labels'] = ann['gt_video_labels']
        results['gt_action_labels'] = ann['gt_action_labels']
        results['entity_ids'] = ann['entity_ids']

        return self.pipeline(results)

    def dump_results(self, results, out):
        assert out.endswith('csv')
        results2csv(self, results, out, self.custom_classes)

    def evaluate(self,
                 results,
                 metrics=('mAP', ),
                 metric_options=None,
                 logger=None):
        # need to create a temp result file
        assert len(metrics) == 1 and metrics[0] == 'mAP', (
            'For evaluation on AVADataset, you need to use metrics "mAP" '
            'See https://github.com/open-mmlab/mmaction2/pull/567 '
            'for more info.')
        time_now = datetime.now().strftime('%Y%m%d_%H%M%S')
        temp_file = f'AVA_{time_now}_result.csv'
        results2csv(self, results, temp_file, self.custom_classes)

        ret = {}
        for metric in metrics:
            msg = f'Evaluating {metric} ...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            eval_result = ava_eval(
                temp_file,
                metric,
                self.label_file,
                self.ann_file,
                self.exclude_file,
                custom_classes=self.custom_classes)
            log_msg = []
            for k, v in eval_result.items():
                log_msg.append(f'\n{k}\t{v: .4f}')
            log_msg = ''.join(log_msg)
            print_log(log_msg, logger=logger)
            ret.update(eval_result)

        os.remove(temp_file)

        return ret

    def unit_test(self):
        return self.video_infos
