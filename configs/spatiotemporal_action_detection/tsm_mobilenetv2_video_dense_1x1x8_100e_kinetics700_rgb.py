_base_ = [
    '../_base_/schedules/sgd_tsm_mobilenet_v2_100e.py',
    '../_base_/default_runtime.py'
]

model = dict(
    type='MixRecognizer2D',
    backbone=dict(
        type='MobileNetV2TSM',
        shift_div=8,
        num_segments=8,
        is_shift=True,
        pretrained=None),
    cls_head=dict(
        type='TSMHead',
        num_segments=8,
        num_classes=700,
        in_channels=1280,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.5,
        init_std=0.001,
        is_shift=True),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))

# dataset settings
dataset_type = 'WeVideoDataset'
data_root = '/mnt/wfs/mmcommwfssz/project_mm-base-vision/harryizzhou/projects/video_understanding/data/kinetics-700-2020'
data_root_val = '/mnt/wfs/mmcommwfssz/project_mm-base-vision/harryizzhou/projects/video_understanding/data/kinetics-700-2020'
ann_file_train = '/mnt/wfs/mmcommwfssz/project_mm-base-vision/harryizzhou/projects/video_understanding/data/kinetics700_list/train_reorged.csv'
ann_file_val = '/mnt/wfs/mmcommwfssz/project_mm-base-vision/harryizzhou/projects/video_understanding/data/kinetics700_list/validate_reorged.csv'
ann_file_test = '/mnt/wfs/mmcommwfssz/project_mm-base-vision/harryizzhou/projects/video_understanding/data/kinetics700_list/validate_reorged.csv'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='DecordInit'),
    dict(type='DenseSampleFrames', clip_len=1, frame_interval=1, num_clips=8),
    dict(type='DecordDecode'),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1,
        num_fixed_crops=13),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='DenseSampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='DenseSampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        exclude_file=None,
        data_prefix=data_root,
        pipeline=train_pipeline,
        num_video_classes=701),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        exclude_file=None,
        data_prefix=data_root_val,
        pipeline=val_pipeline,
        num_video_classes=701),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        exclude_file=None,
        data_prefix=data_root_val,
        pipeline=test_pipeline,
        num_video_classes=701))

evaluation = dict(
    interval=5, metrics=['top_k_accuracy', 'mean_class_accuracy'])

# optimizer
optimizer = dict(
    lr=0.01,  # this lr is used for 8 gpus
)

# runtime settings
checkpoint_config = dict(interval=5)
work_dir = './work_dirs/tsm_mobilenetv2_dense_video_1x1x8_100e_kinetics700_rgb/'  # noqa
