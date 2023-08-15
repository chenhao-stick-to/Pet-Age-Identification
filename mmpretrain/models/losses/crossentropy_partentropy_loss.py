# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from mmpretrain.registry import MODELS
from .cross_entropy_loss import CrossEntropyLoss
from .utils import convert_to_one_hot
import numpy as np
from scipy.stats import norm

@MODELS.register_module()
class GaussSmoothLoss(nn.Module):
    r"""Initializer for the label smoothed cross entropy loss.

    Refers to `Rethinking the Inception Architecture for Computer Vision
    <https://arxiv.org/abs/1512.00567>`_

    This decreases gap between output scores and encourages generalization.
    Labels provided to forward can be one-hot like vectors (NxC) or class
    indices (Nx1).
    And this accepts linear combination of one-hot like labels from mixup or
    cutmix except multi-label task.

    Args:
        label_smooth_val (float): The degree of label smoothing.
        num_classes (int, optional): Number of classes. Defaults to None.
        mode (str): Refers to notes, Options are 'original', 'classy_vision',
            'multi_label'. Defaults to 'original'.
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid of
            softmax. Defaults to None, which means to use sigmoid in
            "multi_label" mode and not use in other modes.
        reduction (str): The method used to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to 'mean'.
        loss_weight (float):  Weight of the loss. Defaults to 1.0.

    Notes:
        - if the mode is **"original"**, this will use the same label smooth
          method as the original paper as:

          .. math::
              (1-\epsilon)\delta_{k, y} + \frac{\epsilon}{K}

          where :math:`\epsilon` is the ``label_smooth_val``, :math:`K` is the
          ``num_classes`` and :math:`\delta_{k, y}` is Dirac delta, which
          equals 1 for :math:`k=y` and 0 otherwise.

        - if the mode is **"classy_vision"**, this will use the same label
          smooth method as the facebookresearch/ClassyVision repo as:

          .. math::
              \frac{\delta_{k, y} + \epsilon/K}{1+\epsilon}

        - if the mode is **"multi_label"**, this will accept labels from
          multi-label task and smoothing them as:

          .. math::
              (1-2\epsilon)\delta_{k, y} + \epsilon
    """

    def __init__(self,
                 std,
                 num_samples,
                 num_classes=None,
                 use_sigmoid=None,
                 reduction='mean',
                 loss_weight=1.0,
                 class_weight=None,
                 pos_weight=None):
        super().__init__()
        self.num_classes = num_classes
        self.loss_weight = loss_weight
        self.num_samples=num_samples
        assert (isinstance(std, int)
                and 0 < std), \
            f'GaussSmoothLoss accepts a int std ' \
            f'std > 0, but gets {std}'
        self.std = std

        accept_reduction = { 'mean', 'sum'}
        assert reduction in accept_reduction, \
            f'Gaussian supports reduction {accept_reduction}, ' \
            f'but gets {reduction}.'
        self.reduction = reduction
        self.gauss_label = self.original_gauss_label
        use_sigmoid = False if use_sigmoid is None else use_sigmoid

        self.ce = CrossEntropyLoss(
            use_sigmoid=use_sigmoid,
            use_soft=not use_sigmoid,
            reduction=reduction,
            class_weight=class_weight,
            pos_weight=pos_weight)

    def generate_one_hot_like_label(self, label):
        """This function takes one-hot or index label vectors and computes one-
        hot like label vectors (float)"""
        # check if targets are inputted as class integers
        if label.dim() == 1 or (label.dim() == 2 and label.shape[1] == 1):
            label = convert_to_one_hot(label.view(-1, 1), self.num_classes)
        return label.float()

    def original_gauss_label(self, one_hot_like_label):
        assert self.num_classes > 0
        std_dev = self.std  # 标准差
        column_indices_list = []
        for row in one_hot_like_label:
            column_indices = (row == 1).nonzero(as_tuple=False).squeeze().tolist()
            column_indices_list.append(column_indices)
        # 生成随机数
        num_samples = self.num_samples  # 生成样本数量
        gauss_label=one_hot_like_label
        index=0
        # f = open("/home/chenh/work/pet_age/mmpretrain/dataload/data.txt","a+")
        # for row in one_hot_like_label.cpu().numpy():
        #     row_str = ' '.join(str(x.item()) for x in row)
        #     f.write(row_str + '\n')
        for elem in column_indices_list:
            if (elem-np.floor(num_samples/2))<0:
                samples = norm.pdf(np.linspace(elem-np.floor(num_samples/2),elem+np.floor(num_samples/2) , num_samples), loc=elem, scale=std_dev)
                normalized_samples = samples / np.sum(samples)
                first_zero_index = np.where(np.linspace(elem-np.floor(num_samples/2),elem+np.floor(num_samples/2), num_samples) == 0)[0][0]
                for i in range(first_zero_index,len(samples)):
                    gauss_label[index,i-first_zero_index]=normalized_samples[i]
            elif (elem+np.floor(num_samples/2))>191:
                samples = norm.pdf(np.linspace(elem-np.floor(num_samples/2),elem+np.floor(num_samples/2) , num_samples), loc=elem, scale=std_dev)
                normalized_samples = samples / np.sum(samples)
                first_max_index = np.where(np.linspace(elem-np.floor(num_samples/2),elem+np.floor(num_samples/2), num_samples) == 192)[0][0]
                for i in range(first_max_index):
                    gauss_label[index,elem-(int)(np.floor(num_samples/2)+i)]=normalized_samples[i]
            else:
                samples = norm.pdf(np.linspace(elem-np.floor(num_samples/2),elem+np.floor(num_samples/2) , num_samples), loc=elem, scale=std_dev)
                normalized_samples = samples / np.sum(samples)
                for i in range(len(samples)):
                    gauss_label[index,(int)(elem-np.floor(num_samples/2)+i)]=normalized_samples[i]
            index=index+1
        # for row in gauss_label.cpu().numpy():
        #     row_str = ' '.join(str(x.item()) for x in row)
        #     f.write(row_str + '\n')
        # f.close()
        return gauss_label

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        r"""Label smooth loss.

        Args:
            pred (torch.Tensor): The prediction with shape (N, \*).
            label (torch.Tensor): The ground truth label of the prediction
                with shape (N, \*).
            weight (torch.Tensor, optional): Sample-wise loss weight with shape
                (N, \*). Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The method used to reduce the
                loss into a scalar. Options are "none", "mean" and "sum".
                Defaults to None.

        Returns:
            torch.Tensor: Loss.
        """
        if self.num_classes is not None:
            assert self.num_classes == cls_score.shape[1], \
                f'num_classes should equal to cls_score.shape[1], ' \
                f'but got num_classes: {self.num_classes} and ' \
                f'cls_score.shape[1]: {cls_score.shape[1]}'
        else:
            self.num_classes = cls_score.shape[1]

        one_hot_like_label = self.generate_one_hot_like_label(label=label)
        assert one_hot_like_label.shape == cls_score.shape, \
            f'LabelSmoothLoss requires output and target ' \
            f'to be same shape, but got output.shape: {cls_score.shape} ' \
            f'and target.shape: {one_hot_like_label.shape}'

        gauss_label = self.gauss_label(one_hot_like_label)
        return self.loss_weight * self.ce.forward(
            cls_score,
            gauss_label,
            weight=weight,
            avg_factor=avg_factor,
            reduction_override=reduction_override,
            **kwargs)
