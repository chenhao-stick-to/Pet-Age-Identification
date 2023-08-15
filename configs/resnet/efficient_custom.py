_base_ = [
    '../_base_/models/efficientnet_v2/efficientnetv2_s.py',
    '../_base_/default_runtime.py',
]
#
model = dict(
    backbone=dict(
            init_cfg=dict(
            type='Pretrained',
            checkpoint='/home/chenh/work/pet_age/mmpretrain/data/efficientnetv2-s_3rdparty_in1k_20221220-f0eaff9d.pth',
            prefix='backbone',
        )),
    neck=dict(type='GlobalAveragePooling'),
   head=dict(
        type='LinearClsHead',
        num_classes=192,
        in_channels=1280,
        loss=dict(type='GaussSmoothLoss',std=2,loss_weight=1.0,num_samples=11,reduction='mean'),
        topk=(10,),
    ))


# dataset settings
dataset_type = 'CustomDataset'
data_preprocessor = dict(
    num_classes=192,
    # RGB format normalization parameters
    mean=[127.5, 127.5, 127.5],
    std=[127.5, 127.5, 127.5],
    # convert image from BGR to RGB
    to_rgb=True,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='EfficientNetRandomCrop', scale=300, crop_padding=0),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='EfficientNetCenterCrop', crop_size=384, crop_padding=0),
    dict(type='PackInputs'),
]

classid=[str(i) for i in range(192)]
train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root='data/petimage',  # `ann_flie` 和 `data_prefix` 共同的文件路径前缀
        ann_file='meta/train1.txt',      # 相对于 `data_root` 的标注文件路径
        data_prefix='train',            # `ann_file` 中文件路径的前缀，相对于 `data_root`
        classes=classid,  # 每个类别的名称
        pipeline=train_pipeline,
        with_label=True),
    sampler=dict(type='DefaultSampler', shuffle=True),
)
val_dataloader = dict(
    batch_size=16,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root='data/petimage',  # `ann_flie` 和 `data_prefix` 共同的文件路径前缀
        ann_file='meta/val1.txt',      # 相对于 `data_root` 的标注文件路径
        data_prefix='val',            # `ann_file` 中文件路径的前缀，相对于 `data_root`
        classes=classid,  # 每个类别的名称
        pipeline=test_pipeline,
        with_label=True),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = dict(type='self_Accuracy', topk=(10,))

test_dataloader = dict(
    batch_size=32,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        data_root='data/petimage/test1',  # `ann_flie` 和 `data_prefix` 共同的文件路径前缀
        pipeline=test_pipeline,
       ),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
test_evaluator = dict(type='self_Accuracy', topk=(10,))



#训练策略重载
# optim_wrapper = dict(
#     optimizer=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001))
optim_wrapper = dict(
    optimizer=dict(
        type='Adam',
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0))

param_scheduler = dict(
    type='MultiStepLR', by_epoch=True, milestones=[100, 150, 200], gamma=0.1)
# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=1000, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=256)
