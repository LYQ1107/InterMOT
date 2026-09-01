"""PSTR PartAttention shim for stock mmcv-full 1.7.x.

PSTR's bundled mmcv adds ``PartAttention`` (pure PyTorch grid_sample, no CUDA
ops). This registers the same class into the stock mmcv ATTENTION registry so
the official PSTR config can be built against mmcv-full 1.7.1.
"""

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv import deprecated_api_warning
from mmcv.cnn import constant_init, xavier_init
from mmcv.cnn.bricks.registry import ATTENTION
from mmcv.runner import BaseModule


@ATTENTION.register_module()
class PartAttention(BaseModule):
    """Part attention used by PSTR (Deformable-DETR style, grid_sample)."""

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=False,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first

        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError('invalid input for _is_power_of_2')
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in PartAttention to make the "
                "dimension of each attention head a power of 2")

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.normalize_fact = float(embed_dims / num_heads) ** -0.5
        self.init_weights()

    def init_weights(self):
        constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
                         self.num_heads, 1, 1,
                         2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        self._is_init = True

    @deprecated_api_warning({'residual': 'identity'}, cls_name='PartAttention')
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets \
                / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                + sampling_offsets / self.num_points \
                * reference_points[:, :, None, :, None, 2:] \
                * 0.5
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')

        bs, _, num_heads, embed_dims = value.shape
        _, num_queries, num_heads, num_levels, num_points, _ = \
            sampling_locations.shape

        value_list = value.split([H_ * W_ for H_, W_ in spatial_shapes],
                                 dim=1)
        sampling_grids = 2 * sampling_locations - 1
        sampling_value_list = []
        for level, (H_, W_) in enumerate(spatial_shapes):
            value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(
                bs * num_heads, embed_dims, H_, W_)
            sampling_grid_l_ = sampling_grids[:, :, :,
                                              level].transpose(1, 2).flatten(0, 1)
            sampling_value_l_ = F.grid_sample(
                value_l_,
                sampling_grid_l_,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=False)
            sampling_value_list.append(sampling_value_l_)
        output = torch.stack(sampling_value_list, dim=-2).flatten(-2)

        output = output.permute(0, 2, 3, 1)
        reid_tokens = torch.mean(output, dim=2, keepdim=True)
        output = torch.cat([reid_tokens, output], dim=2)
        weights = torch.einsum("bqnc,bqcl->bqnl",
                               output * self.normalize_fact,
                               output.transpose(2, 3))
        weights = F.softmax(weights, dim=-1)
        output1 = torch.einsum("bqnl,bqlc->bqnc", weights, output)
        output1 = output1[:, :, 0, :]
        output1 = output1.permute(1, 0, 2).contiguous().view(
            num_queries, bs, num_heads * embed_dims)
        return output1
