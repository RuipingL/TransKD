import numpy as np
import torch.nn as nn
from mmcv.cnn import ConvModule
import torch
from mmseg.ops import resize
# from ..builder import HEADS
# from .decode_head import BaseDecodeHead
# from IPython import embed

# @HEADS.register_module()
class FPNHead(nn.Module):
    """Panoptic Feature Pyramid Networks.
    This head is the implementation of `Semantic FPN
    <https://arxiv.org/abs/1901.02446>`_.
    Args:
        feature_strides (tuple[int]): The strides for input feature maps.
            stack_lateral. All strides suppose to be power of 2. The first
            one is of largest resolution.
    """

    def __init__(self, in_channels=[256, 256, 256, 256],
                feature_strides=[4, 8, 16, 32],
                channels=128, 
                in_index=[0, 1, 2, 3],
                dropout_ratio=0.1,
                num_classes=19, 
                norm_cfg=None,
                # norm_cfg=dict(type='SyncBN', requires_grad=True), 
                align_corners=False) :
        super(FPNHead, self).__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.norm_cfg = norm_cfg
        self.align_corners = align_corners
        self.in_index = in_index
        self.input_transform = 'multiple_select'
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        self.scale_heads = nn.ModuleList()
        for i in range(len(feature_strides)):
            head_length = max(
                1,
                int(np.log2(feature_strides[i]) - np.log2(feature_strides[0])))
            scale_head = []
            for k in range(head_length):
                scale_head.append(
                    ConvModule(
                        self.in_channels[i] if k == 0 else self.channels,
                        self.channels,
                        3,
                        padding=1,
                        conv_cfg=None,
                        norm_cfg=self.norm_cfg,
                        act_cfg=dict(type='ReLU')))
                if feature_strides[i] != feature_strides[0]:
                    scale_head.append(
                        nn.Upsample(
                            scale_factor=2,
                            mode='bilinear',
                            align_corners=self.align_corners))
            self.scale_heads.append(nn.Sequential(*scale_head))
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def forward(self, inputs):

        x = self._transform_inputs(inputs)

        output = self.scale_heads[0](x[0])
        for i in range(1, len(self.feature_strides)):
            # non inplace
            output = output + resize(
                self.scale_heads[i](x[i]),
                size=output.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)

        # output = self.cls_seg(output)
        if self.dropout is not None:
            output = self.dropout(output)
        output = self.conv_seg(output)
        # embed(header='123123')
        return output
