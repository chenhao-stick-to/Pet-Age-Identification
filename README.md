# 环境
系统:Linux x86 ubuntu20.04
 mmcv:2.0.1
 mmpretrain:1.0.1
 python:3.8.16
 GPU:RTX 3080ti
# 1.mmpretrain的简单训练学习

## 1.配置文件结构学习

配置文件都放置在configs/下面,我们实际训练使用时的例子：

CUDA_VISIBLE_DEVICES=0,1,2,3  bash ./tools/dist_train.sh /home/chenh/work/pet_age/mmpretrain/configs/resnet/resnet101_8xb32_custom.py 4

其中resnet101_8xb32_custom.py在configs文件下

```
_base_ = [
    '../_base_/models/resnet101.py', '../_base_/datasets/imagenet_bs32.py',
    '../_base_/schedules/imagenet_bs256.py', '../_base_/default_runtime.py'
]
```

而该配置文件一般由model，data相关设置，训练策略（优化器选择，步长 策略等），运行时的设置（保存权重策略，日志打印策略）

## 2.model设置

```python
# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))
#原始的resnet101.py的设置，但是如果我们需要使用预训练网络，需要像这样
model = dict(
    # type='ImageClassifier',
    backbone=dict(
        frozen_stages=3,
        # type='Pretrained',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/home/chenh/work/pet_age/mmpretrain/data/resnet101_8xb32_in1k_20210831-539c63f8.pth',
            prefix='backbone',
        )),
    # backbone=dict(
    #     type='ResNet',
    #     depth=101,
    #     num_stages=4,
    #     out_indices=(3, ),
    #     style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=192,
        in_channels=2048,
        loss=dict(type='GaussSmoothLoss',std=2,loss_weight=1.0,num_samples=11,reduction='mean'),
        topk=(10,),
    ))
```

### （1）通常type字段可以设置不同任务类型

- 对于图像分类任务，通常为 `ImageClassifier`，更多细节请参考 [API 文档](https://mmpretrain.readthedocs.io/zh_CN/latest/api/models.html#module-mmpretrain.models.classifiers)。
- 对于自监督任务，有多种类型的算法，例如 `MoCoV2`, `BEiT`, `MAE` 等。更多细节请参考 [API 文档](https://mmpretrain.readthedocs.io/zh_CN/latest/api/models.html#module-mmpretrain.models.selfsup)。
- 对于图像检索任务，通常为 `ImageToImageRetriever`，更多细节请参考 [API 文档](https://mmpretrain.readthedocs.io/zh_CN/latest/api/models.html#module-mmpretrain.models.retrievers).

### （2）选定type=”ImageClassifier“后，对backbone，neck，head进行设置，可使用诸多主干网络，具体使用

设置查看api文档。特别neck涉及loss的设置，当没有相关loss时需要具体设计，例如设计了一个GaussSmoothLoss，那么需要在mmpretrain目录下创建一个新的loss的py文件，类比其他loss文件输出保持一致，输入的形式也要确定，编写完成后，需要将这个loss类注册到loss目录下的__init__.py文件中。

```
from .asymmetric_loss import AsymmetricLoss, asymmetric_loss
from .cae_loss import CAELoss
from .cosine_similarity_loss import CosineSimilarityLoss
from .cross_correlation_loss import CrossCorrelationLoss
from .cross_entropy_loss import (CrossEntropyLoss, binary_cross_entropy,
                                 cross_entropy)
from .focal_loss import FocalLoss, sigmoid_focal_loss
from .label_smooth_loss import LabelSmoothLoss
from .reconstruction_loss import PixelReconstructionLoss
from .seesaw_loss import SeesawLoss
from .swav_loss import SwAVLoss
from .crossentropy_partentropy_loss import GaussSmoothLoss
from .utils import (convert_to_one_hot, reduce_loss, weight_reduce_loss,
                    weighted_loss)


__all__ = [
    'asymmetric_loss',
    'AsymmetricLoss',
    'cross_entropy',
    'binary_cross_entropy',
    'CrossEntropyLoss',
    'reduce_loss',
    'weight_reduce_loss',
    'LabelSmoothLoss',
    'weighted_loss',
    'FocalLoss',
    'sigmoid_focal_loss',
    'convert_to_one_hot',
    'SeesawLoss',
    'CAELoss',
    'CosineSimilarityLoss',
    'CrossCorrelationLoss',
    'PixelReconstructionLoss',
    'SwAVLoss',
    'GaussSmoothLoss',
]

```

后面就可以通过配置文件直接使用。

### （3）其他配置。

```python
model = dict(
    backbone=...,
    neck=...,
    head=...,
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.8),
        dict(type='CutMix', alpha=1.0),
    ], probs=[0.3, 0.7])
)
##在model阶段使用train_cfg可以对其进行一个混合的增强，可以设置它的增强操作以及增强的概率。
model = dict(
    backbone = ...,
    neck = ...,
    head = ...,
    data_preprocessor = dict(
                         mean=[123.675, 116.28, 103.53],
                         std=[58.395, 57.12, 57.375],
                         to_rgb=True)
    train_cfg=...,
)
##model阶段使用data_preprocessor对数据集进行处理，效果与后面data.py的data_preprocessor的设置是一样，但优先级更高。

```

## 3.data的配置

data_preprocessor/训练，测试，验证集的dataloader的构建；评估器evalutor的构建

## （1）可使用其已经包含的数据集或者自定义使用数据集使用dataset_type = 'CustomDataset'来指定自定义的数据集。一般来说我们训练是使用自定义数据集，前者照教程走就行，后者来说。

首先，自定义数据集有两种组织方式：

```
#子文件夹方式
data_prefix/
├── class_x
│   ├── xxx.png
│   ├── xxy.png
│   └── ...
│       └── xxz.png
└── class_y
    ├── 123.png
    ├── nsdf3.png
    ├── ...
    └── asd932_.png
    
#文件方式
data_root
├── meta
│   ├── test.txt     # 测试数据集的标注文件
│   ├── train.txt    # 训练数据集的标注文件
│   └── val.txt      # 验证数据集的标注文件

├── train
│   ├── 123.png
│   ├── folder_1
│   │   ├── xxx.png
│   │   └── xxy.png
│   └── nsdf3.png
├── test
└── val
##如图设置
# 训练数据设置
train_dataloader = dict(
    dataset=dict(
        type='CustomDataset',
        data_root='path/to/data_root',  # `ann_flie` 和 `data_prefix` 共同的文件路径前缀
        ann_file='meta/train.txt',      # 相对于 `data_root` 的标注文件路径
        data_prefix='train',            # `ann_file` 中文件路径的前缀，相对于 `data_root`
        classes=['A', 'B', 'C', 'D', ...],  # 每个类别的名称
        pipeline=...,    # 处理数据集样本的一系列变换操作
    )
    ...
)

#classes是以字符列表的形式传入，自己的设置
classid=[str(i) for i in range(192)]
train_dataloader = dict(
    batch_size=32,
    num_workers=8,
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
##在train—_dataloader中，
batch_size: 每个 GPU 的 batch size
num_workers: 每个 GPU 的线程数
sampler是采样器的设置
##tain_pipeline设置
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=224),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]
#里面存在许多的预处理选项，需要使用时查阅api文档获取
```

## （2）评估器的使用

```python
val_evaluator = dict(type='self_Accuracy', topk=(10,))
test_evaluator = dict(type='self_Accuracy', topk=(10,))
#设置评估器，官方内置多种评估器，具体使用可查阅官方api文档
#当需要自定义评估指标时，需要自定义实现

```

自定义的self_Accuracy类方法的calculate函数实现

```python
def calculate(
        pred: Union[torch.Tensor, np.ndarray, Sequence],
        target: Union[torch.Tensor, np.ndarray, Sequence],
        topk: Sequence[int] = (1, ),
        thrs: Sequence[Union[float, None]] = (0., ),
    ) -> Union[torch.Tensor, List[List[torch.Tensor]]]:
        """Calculate the accuracy.

        Args:
            pred (torch.Tensor | np.ndarray | Sequence): The prediction
                results. It can be labels (N, ), or scores of every
                class (N, C).
            target (torch.Tensor | np.ndarray | Sequence): The target of
                each prediction with shape (N, ).
            thrs (Sequence[float | None]): Predictions with scores under
                the thresholds are considered negative. It's only used
                when ``pred`` is scores. None means no thresholds.
                Defaults to (0., ).
            thrs (Sequence[float]): Predictions with scores under
                the thresholds are considered negative. It's only used
                when ``pred`` is scores. Defaults to (0., ).

        Returns:
            torch.Tensor | List[List[torch.Tensor]]: Accuracy.

            - torch.Tensor: If the ``pred`` is a sequence of label instead of
              score (number of dimensions is 1). Only return a top-1 accuracy
              tensor, and ignore the argument ``topk` and ``thrs``.
            - List[List[torch.Tensor]]: If the ``pred`` is a sequence of score
              (number of dimensions is 2). Return the accuracy on each ``topk``
              and ``thrs``. And the first dim is ``topk``, the second dim is
              ``thrs``.
        """

        pred = to_tensor(pred)
        target = to_tensor(target).to(torch.int64)
        num = pred.size(0)
        assert pred.size(0) == target.size(0), \
            f"The size of pred ({pred.size(0)}) doesn't match "\
            f'the target ({target.size(0)}).'

        if pred.ndim == 1:
            # For pred label, ignore topk and acc
            pred_label = pred.int()
            correct = pred.eq(target).float().sum(0, keepdim=True)
            acc = correct.mul_(100. / num)
            return acc
        else:
            # For pred score, calculate on all topk and thresholds.
            pred = pred.float()
            maxk = max(topk)
            if maxk > pred.size(1):
                raise ValueError(
                    f'Top-{maxk} accuracy is unavailable since the number of '
                    f'categories is {pred.size(1)}.')

            pred_score, pred_label = pred.topk(maxk,dim=1)
            target_num=target.numpy()
            results = []
            max_index=5
            for k in topk:
                results.append([])
                for thr in thrs:
                    # Only prediction values larger than thr are counted
                    # as correct
                    temp=pred_label[:,:max_index]
                    temp1=pred_label[:,:max_index].numpy()
                    print(temp)
                    temp_argmax=temp.argmax(dim=1).tolist()
                    temp_argmin=temp.argmin(dim=1).tolist()
                    for i in range(pred_label.size(0)):
                        temp[i][temp_argmax[i]]=-1
                        temp[i][temp_argmin[i]]=-1
                    temp=temp.numpy()
                    sum_result=0
                    sum_result1=0
                    sum_result2=0
                    sum_result3=0
                    for i in range(pred_label.size(0)):
                        sum=0
                        for j in range(max_index):
                            if temp[i,j]!=-1:
                                sum=sum+temp[i,j] 
                        sum_result=sum_result+abs((int)(np.floor(sum/(max_index-2)))-target_num[i]) 
                    acc = sum_result/ num
                    results[-1].append(acc)
                    for i in range(pred_label.size(0)):
                        sum_result1=sum_result1+abs(temp1[i][0]-target_num[i])
                    acc1=sum_result1/num
                    results[-1].append(acc1)
                    for i in range(pred_label.size(0)):
                        sum=0
                        for j in range(2):
                           sum=sum+temp1[i][j] 
                        sum_result2=sum_result2+abs((int)(np.floor(sum/2))-target_num[i])
                    acc2=sum_result2/num
                    results[-1].append(acc2)
                    for i in range(pred_label.size(0)):
                        sum=0
                        for j in range(3):
                           sum=sum+temp1[i][j] 
                        sum_result3=sum_result3+abs((int)(np.floor(sum/3))-target_num[i])
                    acc2=sum_result3/num
                    results[-1].append(acc2)
                print(results)
            return results
      ##重点是构造出自己的评估方法，再根据具体的输入，来进行评估算法的编写，最后也需要注册到evaluation当前#文件夹下的__init__.py文件下
```

## 3.训练策略的设置

主要设置一些在训练过程中的一些策略。

```python
optim_wrapper = dict(
    # 使用 SGD 优化器来优化参数
    optimizer=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001))

# 学习率参数的调整策略
# 'MultiStepLR' 表示使用多步策略来调度学习率（LR）。
param_scheduler = dict(
    type='MultiStepLR', by_epoch=True, milestones=[30, 60, 90], gamma=0.1)

# 训练的配置，迭代 100 个 epoch，每一个训练 epoch 后都做验证集评估
# 'by_epoch=True' 默认使用 `EpochBaseLoop`,  'by_epoch=False' 默认使用 `IterBaseLoop`
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
# 使用默认的验证循环控制器
val_cfg = dict()
# 使用默认的测试循环控制器
test_cfg = dict()

# 通过默认策略自动缩放学习率，此策略适用于总批次大小 256
# 如果你使用不同的总批量大小，比如 512 并启用自动学习率缩放
# 我们将学习率扩大到 2 倍
auto_scale_lr = dict(base_batch_size=256)
```

## 4.运行设置

本部分主要包括保存权重策略、日志配置、训练参数、断点权重路径和工作目录等等。

```
# 默认所有注册器使用的域
default_scope = 'mmpretrain'

# 配置默认的 hook
default_hooks = dict(
    # 记录每次迭代的时间。
    timer=dict(type='IterTimerHook'),

    # 每 100 次迭代打印一次日志。
    logger=dict(type='LoggerHook', interval=100),

    # 启用默认参数调度 hook。
    param_scheduler=dict(type='ParamSchedulerHook'),

    # 每个 epoch 保存检查点。
    checkpoint=dict(type='CheckpointHook', interval=1),

    # 在分布式环境中设置采样器种子。
    sampler_seed=dict(type='DistSamplerSeedHook'),

    # 验证结果可视化，默认不启用，设置 True 时启用。
    visualization=dict(type='VisualizationHook', enable=False),
)

# 配置环境
env_cfg = dict(
   # 是否开启 cudnn benchmark
    cudnn_benchmark=False,

    # 设置多进程参数
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),

    # 设置分布式参数
    dist_cfg=dict(backend='nccl'),
)

# 设置可视化工具
vis_backends = [dict(type='LocalVisBackend')] # 使用磁盘(HDD)后端
visualizer = dict(
    type='UniversalVisualizer', vis_backends=vis_backends, name='visualizer')

# 设置日志级别
log_level = 'INFO'

# 从哪个检查点加载
load_from = None

# 是否从加载的检查点恢复训练
resume = False
```

## 5.文件的继承与修改

可以继原来的四个__base__文件,来修改相关的一些变量，例如

### （1）直接继承原文间并修改

```python
# 在 'configs/resnet/' 创建此文件
_base_ = './resnet50_8xb32_in1k.py'

# 模型在之前的基础上使用 CutMix 训练增强
model = dict(
    train_cfg=dict(
        augments=dict(type='CutMix', alpha=1.0)
    )
)

# 优化策略在之前基础上训练更多个 epoch
train_cfg = dict(max_epochs=300, val_interval=10)  # 训练 300 个 epoch，每 10 个 epoch 评估一次
param_scheduler = dict(step=[150, 200, 250])   # 学习率调整也有所变动

# 使用自己的数据集目录
train_dataloader = dict(
    dataset=dict(data_root='mydata/imagenet/train'),
)
val_dataloader = dict(
    batch_size=64,                  # 验证时没有反向传播，可以使用更大的 batchsize
    dataset=dict(data_root='mydata/imagenet/val'),
)
test_dataloader = dict(
    batch_size=64,                  # 测试时没有反向传播，可以使用更大的 batchsize
    dataset=dict(data_root='mydata/imagenet/val'),
)
```

## （2）使用配置文件中间量

```python
bgr_mean = [103.53, 116.28, 123.675]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=224, backend='pillow', interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='RandAugment',
        policies='timm_increasing',
        num_policies=2,
        total_level=10,
        magnitude_level=6,
        magnitude_std=0.5,
        hparams=dict(pad_val=[round(x) for x in bgr_mean], interpolation='bicubic')),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=236, edge='short', backend='pillow', interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs')
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=val_pipeline))
test_dataloader = dict(dataset=dict(pipeline=val_pipeline))
```

## （3）忽略基础配置文件里的部分内容

```python
_base_ = '../../configs/resnet/resnet50_8xb32_in1k.py'

# 学习率调整策略
param_scheduler = dict(type='CosineAnnealingLR', by_epoch=True, _delete_=True)
#需要使用_delete_=True来避免原域param_scheduler的字段step继承。
```

## （4）直接引用配置文件的变量

```python
_base_ = [
    '../_base_/models/resnest50.py', '../_base_/datasets/imagenet_bs64.py',
    '../_base_/default_runtime.py', './_randaug_policies.py',
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandAugment',
        policies={{_base_.policies}},    # 这里使用了 _base_ 里的 `policies` 参数。
        num_policies=2,
        magnitude_level=12),
    dict(type='EfficientNetRandomCrop', scale=224, backend='pillow'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='ColorJitter', brightness=0.4, contrast=0.4, saturation=0.4),
    dict(
        type='Lighting',
        eigval=EIGVAL,
        eigvec=EIGVEC,
        alphastd=0.1,
        to_rgb=False),
    dict(type='PackInputs'),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
```

# 2.简单的运行宠物年龄分类程序

## 1.数据集获取

进入大赛官网：https://tianchi.aliyun.com/competition/entrance/532092/information获取原始数据

将数据集按上述方式进行组织。

## 2.训练

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./tools/dist_train.sh /home/chenh/work/pet_age/mmpretrain/configs/resnet/resnet101_8xb32_custom.py 4
##训练模型保存为pth文件，方便下面测试
```

## 3.测试

```shell
 bash  ./tools/dist_test.sh /home/chenh/work/pet_age/mmpretrain/configs/resnet/resnet101_8xb32_custom.py /home/chenh/work/pet_age/mmpretrain/work_dirs/resnet101_8xb32_custom/epoch_7.pth 4 --work-dir /home/chenh/work/pet_age/mmpretrain/dataload --out /home/chenh/work/pet_age/mmpretrain/dataload/result7epoch.pkl
 #将数据保存为pkl文件，后面再自己写数据的到处脚本完成到txt文件的转换。
 ##在实际运行时需要自己去确认各种路径是否正确。。。。。
```

