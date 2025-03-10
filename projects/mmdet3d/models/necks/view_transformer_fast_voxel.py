import torch
import torch.nn as nn
from torch.nn import functional as F
from mmcv.runner import BaseModule, force_fp32
from mmcv.cnn import build_conv_layer, ConvModule
from ..builder import NECKS, BACKBONES
from .. import builder
import numpy as np
from termcolor import colored
from einops import rearrange
import math

from mmseg.ops import resize

import os
from torch import nn
import torch.utils.checkpoint as cp

from mmdet.models import NECKS
from mmcv.runner import auto_fp16
from mmdet3d.models.utils.self_print import print2file
from mmcv.utils import TORCH_VERSION, digit_version


class FastVoxelGen(nn.Module):
    def __init__(
        self,
        occ_size=(200, 200, 16),
        pc_range=[-51.2, -51.2, -5, 51.2, 51.2, 3],
        multi_frame_range=1,
        device="cuda",
    ):
        self.occ_size = occ_size
        self.pc_range = pc_range
        self.multi_frame_range = multi_frame_range
        self.device = device

    def get_voxel_points(self):
        x = torch.linspace(0, 1, self.occ_size[0], device=self.device)
        y = torch.linspace(0, 1, self.occ_size[1], device=self.device)
        z = torch.linspace(0, 1, self.occ_size[2], device=self.device)

        xyz = torch.stack(torch.meshgrid[x, y, z]).permute(1, 2, 3, 0)

        return xyz
    
    @force_fp32()
    def get_projection(self, rots, trans, intrins, post_rots, post_trans, bda):
        B, N, _, _ = rots.shape

        bda = bda.view(B, 1, 3, 3).repeat(1, N, 1, 1)
        inv_sensor2ego = torch.inverse(rots)
        lidar2img_R = intrins.matmul(inv_sensor2ego).matmul(torch.inverse(bda))
        lidar2img_t = -intrins.matmul(inv_sensor2ego).matmul(trans.unsqueeze(-1))
        lidar2img = torch.cat((lidar2img_R, lidar2img_t), -1)
        img_aug = torch.cat((post_rots, post_trans.unsqueeze(-1)), -1)
        return lidar2img, img_aug

    def get_cams_sampling_point(self, reference_points, pc_range, img_metas):
        lidar2img = []
        for img_meta in img_metas:
            lidar2img.append(img_meta["lidar2img"])
        lidar2img = np.asarray(lidar2img)
        lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4)

        B, NT, _, _ = lidar2img.shape

        lidar2img = lidar2img.view(
            B, NT / self.multi_frame_range, self.multi_frame_range, 4, 4
        )  # B, N, T, 4, 4

        reference_points = reference_points.clone()

        reference_points[..., 0:1] = (
            reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
        )
        reference_points[..., 1:2] = (
            reference_points[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
        )
        reference_points[..., 2:3] = (
            reference_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
        )

        reference_points = torch.cat(
            (reference_points, torch.ones_like(reference_points[..., :1])), -1
        )

        reference_points = reference_points.permute(1, 0, 2, 3)
        D, B, num_query = reference_points.size()[:3]
        num_cam = lidar2img.size(1)

        reference_points = (
            reference_points.view(D, B, 1, num_query, 4)
            .repeat(1, 1, num_cam, 1, 1)
            .unsqueeze(-1)
        )

        lidar2img = lidar2img.view(1, B, num_cam, 1, 4, 4).repeat(
            D, 1, 1, num_query, 1, 1
        )

        reference_points_cam = torch.matmul(
            lidar2img.to(torch.float32), reference_points.to(torch.float32)
        ).squeeze(-1)

        eps = 1e-5

        bev_mask = reference_points_cam[..., 2:3] > eps
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3],
            torch.ones_like(reference_points_cam[..., 2:3]) * eps,
        )

        # reference_points_cam[..., 0] /= img_metas[0]["img_shape"][0][1]
        # reference_points_cam[..., 1] /= img_metas[0]["img_shape"][0][0]

        bev_mask = (
            bev_mask
            & (reference_points_cam[..., 1:2] > 0.0)
            & (reference_points_cam[..., 1:2] < 1.0)
            & (reference_points_cam[..., 0:1] < 1.0)
            & (reference_points_cam[..., 0:1] > 0.0)
        )

        if digit_version(TORCH_VERSION) >= digit_version("1.8"):
            bev_mask = torch.nan_to_num(bev_mask)
        else:
            bev_mask = bev_mask.new_tensor(np.nan_to_num(bev_mask.cpu().numpy()))

        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)

        return reference_points_cam, bev_mask

    def fast_voxel_fill(self, img_feats, points, valid):
        n_images, n_channels, _, _ = img_feats.shape

        x = points[..., 0].round().long()  # n_cams n_points 1
        y = points[..., 1].round().long()  # n_cams n_points 1

        # x = points[..., 0]
        # y = points[..., 1]

        valid = valid.squeeze(-1) if len(valid.shape) == 3 else valid

        # method2：特征填充，只填充有效特征，重复特征直接覆盖
        volume = torch.zeros(
            (n_channels, points.shape[-2]), device=img_feats.device
        ).type_as(img_feats)

        for i in range(n_images):
            volume[:, valid[i]] = img_feats[i, :, y[i, valid[i]], x[i, valid[i]]]

        volume = volume.view(
            n_channels, self.occ_size[0], self.occ_size[1], self.occ_size[2]
        )  # C, X, Y, Z

        return volume

    def forward(self, img_feats, img_inputs=None, img_metas=None):
        
        reference_voxel_point = self.get_voxel_points()

        cams_point, valid_mask = self.get_cams_sampling_point(
            reference_voxel_point, self.pc_range, img_metas=img_metas
        )

        fast_voxel = self.fast_voxel_fill(
            img_feats=img_feats, points=cams_point, valid=valid_mask
        )

        return fast_voxel
