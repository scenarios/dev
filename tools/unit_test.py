from mmaction.datasets import WeVideoDataset


def UnitTest():
    dataset = WeVideoDataset(
        ann_file='/Users/yizhouzhou/Documents/workspace/projects/spatiotemporal_event_detection/data_acquisition/validate_reorged.csv',
        exclude_file=None,
        pipeline=[dict(type='SampleWeVideoFrames', clip_len=8, frame_interval=3)],
        video_label_file = None,
        action_label_file =None,
        filename_tmpl='img_{:05}.jpg',
        proposal_file=None,
        person_det_score_thr=0.9,
        num_video_classes=701,
        num_action_classes=81,
        custom_video_labels=None,
        custom_action_labels=None,
        data_prefix=None,
        test_mode=False,
        modality='RGB',
        num_max_proposals=1000)

    video_infos = dataset.unit_test()
    print('Done')


if __name__ == '__main__':
    UnitTest()