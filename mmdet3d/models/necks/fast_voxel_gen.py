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

try:
    from kornia.utils.grid import create_meshgrid3d
    from kornia.geometry.linalg import transform_points
except Exception as e:
    # Note: Kornia team will fix this import issue to try to allow the usage of lower torch versions.
    print(
        "Warning: kornia is not installed correctly, please ignore this warning if you do not use CaDDN. Otherwise, it is recommended to use torch version greater than 1.2 to use kornia properly."
    )




def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = torch.Tensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])
    return dx, bx, nx


@torch.no_grad()
def get_lidar_to_cams_project(rots, trans, cam_inner):
    """
    NOTE:
        cams_load_order:  ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
         'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
    """
    _device = rots.device
    _dtype = rots.dtype
    rots = rots.cpu().numpy()
    trans = trans.cpu().numpy()
    cam_inner = cam_inner.cpu().numpy()

    lidar2img_rts = []
    for INDICES in range(len(rots)):
        # inverse sensor2lidar_rotation
        lidar2cam_r = np.linalg.inv(rots[INDICES])
        lidar2cam_t = trans[INDICES] @ lidar2cam_r.T
        lidar2cam_rt = np.eye(4)
        lidar2cam_rt[:3, :3] = lidar2cam_r.T
        lidar2cam_rt[3, :3] = -lidar2cam_t
        intrinsic = cam_inner[INDICES]
        viewpad = np.eye(4)
        viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
        lidar2img_rt = viewpad @ lidar2cam_rt.T
        lidar2img_rt = torch.tensor(lidar2img_rt, dtype=_dtype)
        lidar2img_rts.append(lidar2img_rt)

    assert len(lidar2img_rts) == 6

    lidar2img_rts = torch.stack(lidar2img_rts).to(_device)

    return lidar2img_rts


class ResModule2D(nn.Module):
    def __init__(self, n_channels, norm_cfg=dict(type="BN2d"), groups=1):
        super().__init__()
        self.conv0 = ConvModule(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=3,
            padding=1,
            groups=groups,
            conv_cfg=dict(type="Conv2d"),
            norm_cfg=norm_cfg,
            # act_cfg=dict(type="ReLU", inplace=True),
            act_cfg=dict(type="LeakyReLU", inplace=True),
        )
        self.conv1 = ConvModule(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=3,
            padding=1,
            groups=groups,
            conv_cfg=dict(type="Conv2d"),
            norm_cfg=norm_cfg,
            act_cfg=None,
        )
        # self.activation = nn.ReLU(inplace=True)
        self.activation = nn.LeakyReLU(inplace=True)

    @auto_fp16()
    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): of shape (N, C, N_x, N_y, N_z).

        Returns:
            torch.Tensor: 5d feature map.
        """
        identity = x
        x = self.conv0(x)
        x = self.conv1(x)
        x = identity + x
        x = self.activation(x)
        return x


class ResModule2D_LReLU(ResModule2D):
    def __init__(self, n_channels, norm_cfg=dict(type="BN2d"), groups=1):
        super().__init__()
        self.conv0 = ConvModule(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=3,
            padding=1,
            groups=groups,
            conv_cfg=dict(type="Conv2d"),
            norm_cfg=norm_cfg,
            act_cfg=dict(type="LeakyReLU", inplace=True),
        )
        self.conv1 = ConvModule(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=3,
            padding=1,
            groups=groups,
            conv_cfg=dict(type="Conv2d"),
            norm_cfg=norm_cfg,
            act_cfg=None,
        )
        self.activation = nn.LeakyReLU(inplace=True)


@NECKS.register_module()
class M2BevNeck(nn.Module):
    """Neck for M2BEV."""

    def __init__(
        self,
        in_channels,
        out_channels,
        num_layers=2,
        norm_cfg=dict(type="BN2d"),
        stride=2,
        fuse=None,
        with_cp=False,
    ):
        super().__init__()

        self.with_cp = with_cp

        if fuse is not None:
            self.fuse = nn.Conv2d(
                fuse["in_channels"], fuse["out_channels"], kernel_size=1
            )
        else:
            self.fuse = None

        model = nn.ModuleList()
        model.append(ResModule2D(in_channels, norm_cfg))
        model.append(
            ConvModule(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                conv_cfg=dict(type="Conv2d"),
                norm_cfg=norm_cfg,
                act_cfg=dict(type="ReLU", inplace=True),
            )
        )
        for i in range(num_layers):
            model.append(ResModule2D(out_channels, norm_cfg))
            model.append(
                ConvModule(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    conv_cfg=dict(type="Conv2d"),
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type="ReLU", inplace=True),
                )
            )
        self.model = nn.Sequential(*model)

    @auto_fp16()
    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): of shape (N, C_in, N_x, N_y, N_z).

        Returns:
            list[torch.Tensor]: of shape (N, C_out, N_y, N_x).
        """

        def _inner_forward(x):
            out = self.model.forward(x)
            return out

        if self.fuse is not None:
            x = self.fuse(x)

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)

        return x


@NECKS.register_module()
class M2BevNeck_LReLU(M2BevNeck):
    """Neck for M2BEV."""

    def __init__(
        self,
        in_channels,
        out_channels,
        num_layers=2,
        norm_cfg=dict(type="BN2d"),
        stride=2,
    ):
        super().__init__()

        model = nn.ModuleList()
        model.append(ResModule2D_LReLU(in_channels, norm_cfg))
        model.append(
            ConvModule(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                conv_cfg=dict(type="Conv2d"),
                norm_cfg=norm_cfg,
                # act_cfg=dict(type="ReLU", inplace=True),
                act_cfg=dict(type="LeakyReLU", inplace=True),
            )
        )
        for i in range(num_layers):
            model.append(ResModule2D_LReLU(out_channels, norm_cfg))
            model.append(
                ConvModule(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    conv_cfg=dict(type="Conv2d"),
                    norm_cfg=norm_cfg,
                    # act_cfg=dict(type="ReLU", inplace=True),
                    act_cfg=dict(type="LeakyReLU", inplace=True),
                )
            )
        self.model = nn.Sequential(*model)


class SELikeModule(nn.Module):
    def __init__(self, in_channel=512, feat_channel=256, intrinsic_channel=33):
        super(SELikeModule, self).__init__()
        self.input_conv = nn.Conv2d(in_channel, feat_channel, kernel_size=1, padding=0)
        self.fc = nn.Sequential(
            nn.BatchNorm1d(intrinsic_channel),
            nn.Linear(intrinsic_channel, feat_channel),
            nn.Sigmoid(),
        )

    def forward(self, x, cam_params):
        x = self.input_conv(x)
        b, c, _, _ = x.shape
        y = self.fc(cam_params).view(b, c, 1, 1)
        return x * y.expand_as(x)


@NECKS.register_module()
class InterDistill_FastBEV(BaseModule):
    def __init__(
        self,
        grid_config=None,
        data_config=None,
        numC_input=512,
        pct_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        voxel_size=[0.1, 0.1, 0.2],
        numC_Trans=64,
        unproject=True,
        unproject_postion="after",
        z_axis_times=4,
        downsample=16,
        accelerate=False,
        max_drop_point_rate=0.0,
        use_bev_pool=True,
        is_ms_bev=False,
        loss_depth_weight=None,
        extra_depth_net=None,
        se_config=dict(),
        dcn_config=dict(bias=True),
        selfcalib_conv_config=None,
        bev_channel_reduce=None,
        **kwargs,
    ):
        super(InterDistill_FastBEV, self).__init__()
        if grid_config is None:
            grid_config = {
                "xbound": [-51.2, 51.2, 0.8],
                "ybound": [-51.2, 51.2, 0.8],
                "zbound": [-10.0, 10.0, 20.0],
                "dbound": [1.0, 60.0, 1.0],
            }
        self.grid_config = grid_config
        dx, bx, nx = gen_dx_bx(
            self.grid_config["xbound"],
            self.grid_config["ybound"],
            self.grid_config["zbound"],
        )
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        if data_config is None:
            data_config = {"input_size": (256, 704)}
        self.data_config = data_config
        self.downsample = downsample

        self.pct_range = pct_range
        self.voxel_size = voxel_size
        self.D = 60
        self.unproject = unproject
        self.unproject_postion = unproject_postion
        self.z_axis_times = z_axis_times
        self.is_ms_bev = is_ms_bev
        self.bev_channel_reduce = bev_channel_reduce

        # NOTE the following is the implemented of the original BEVPooling with LSS-BEVDet
        self.numC_input = numC_input
        self.numC_Trans = numC_Trans
        self.depthnet = nn.Conv2d(
            self.numC_input, self.D + self.numC_Trans, kernel_size=1, padding=0
        )
        self.geom_feats = None
        self.accelerate = accelerate
        self.max_drop_point_rate = max_drop_point_rate
        self.use_bev_pool = use_bev_pool
        self.is_ms_bev = is_ms_bev

        self.n_voxels = kwargs["n_voxels"] if "n_voxels" in kwargs.keys() else None

        self.featnet = nn.Conv2d(
            self.numC_input, self.numC_Trans, kernel_size=1, padding=0
        )

        # config the cams depth estimationm, not suitable for fastbev
        self.loss_depth_weight = loss_depth_weight
        if loss_depth_weight > 0:
            self.loss_depth_weight = loss_depth_weight
            self.extra_depthnet = builder.build_backbone(extra_depth_net)
            self.dcn = nn.Sequential(
                *[
                    build_conv_layer(
                        dict(type="DCNv2", deform_groups=1),
                        extra_depth_net["num_channels"][0],
                        extra_depth_net["num_channels"][0],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        dilation=1,
                        **dcn_config,
                    ),
                    nn.BatchNorm2d(extra_depth_net["num_channels"][0]),
                ]
            )

            self.depthnet = nn.Conv2d(
                extra_depth_net["num_channels"][0], self.D, kernel_size=1, padding=0
            )
            self.se = SELikeModule(
                self.numC_input,
                feat_channel=extra_depth_net["num_channels"][0],
                **se_config,
            )

        self.selfcalib_conv_config = selfcalib_conv_config
        if selfcalib_conv_config is None:
            # 是否需要 normal？？？？？？？？？
            self.bev_reduce = nn.Sequential(
                ConvModule(
                    self.z_axis_times * numC_Trans,
                    64,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=dict(type="BN", eps=1e-3, momentum=0.01),
                    act_cfg=dict(type="ReLU"),
                ),
                ConvModule(
                    64,
                    64,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=dict(type="BN", eps=1e-3, momentum=0.01),
                    act_cfg=dict(type="ReLU"),
                ),
            )
            self.ch_reduce = ConvModule(
                self.z_axis_times * numC_Trans,
                64,
                kernel_size=1,
                stride=1,
                norm_cfg=dict(type="BN", eps=1e-3, momentum=0.01),
                act_cfg=dict(type="ReLU"),
            )

        else:
            self.bev_reduce = BACKBONES.build(selfcalib_conv_config)
            self.ch_reduce = ConvModule(
                self.z_axis_times * numC_Trans,
                64,
                kernel_size=1,
                stride=1,
                norm_cfg=dict(type="BN", eps=1e-3, momentum=0.01),
                act_cfg=dict(type="ReLU"),
            )

        self.conv3d_modify = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.other_args = kwargs

    def wrap_lidar2img_project(
        self,
        rots,
        trans,
        cam_inner,
        img_metas=None,
        device=None,
        dtype=None,
        lidar2img=False,
    ):
        """
        Args:
            rots: b
        Return:

        """
        lidar2img_project = None

        if img_metas is None:
            lidar2img_project = get_lidar_to_cams_project(rots, trans, cam_inner)
        elif lidar2img:
            lidar2img_project = [
                torch.tensor(lidar2imgs["lidar2img"]) for lidar2imgs in img_metas
            ]
            lidar2img_project = torch.stack(lidar2img_project).to(device).to(dtype)
        else:
            pass

        return lidar2img_project

    @torch.no_grad()
    def scenes_voxel_point(self, z_times=4):
        if self.n_voxels is None:
            n_voxel_x = self.nx[0]
            n_voxel_y = self.nx[1]
            n_voxel_z = self.nx[2] * z_times

        else:
            n_voxel_x = self.n_voxels[0]
            n_voxel_y = self.n_voxels[1]
            n_voxel_z = self.n_voxels[2]

        voxel_points = torch.stack(
            torch.meshgrid(
                [
                    torch.arange(n_voxel_x),
                    torch.arange(n_voxel_y),
                    torch.arange(n_voxel_z),
                ]
            )
        )

        # print(voxel_points, voxel_points.shape)

        return voxel_points  # (3, x, y, z)

    def voxel_grid_to_lidar_points(
        self,
        bs,
        pc_range,
        voxel_size,
        voxel_points,
        trans_version="v3",
        data_type=None,
        device=None,
    ):
        """
        Calculate grid to LiDAR unprojection for each plane
        Args:
            pc_min: [x_min, y_min, z_min], Minimum of point cloud range (m)
            voxel_size: [x, y, z], Size of each voxel (m)
            trans_version [ default : v1]: Controls the method used to complete predefined voxel point transformations, Configured as v2, which means using the official code of fastbev. It is more recommended to use v1 or v3.
        Params:
            unproject: (4, 4), Voxel grid to LiDAR unprojection matrix
        Return:
            lidar_points (torch.tensor): [B, N, 3]
        """
        trans_version = self.other_args["trans_version"]
        B = bs
        if trans_version == "v1":
            assert len(voxel_points.shape) == 3  # B N 3
            x_size, y_size, z_size = voxel_size
            x_min, y_min, z_min = pc_range[:3]
            unproject = torch.tensor(
                [
                    [x_size, 0, 0, x_min],
                    [0, y_size, 0, y_min],
                    [0, 0, z_size, z_min],
                    [0, 0, 0, 1],
                ],
                dtype=data_type,
                device=device,
            )  # (4, 4)

            lidar_points = transform_points(
                trans_01=unproject.unsqueeze(0), points_1=voxel_points
            )
            # lidar_points = lidar_points.squeeze()
            return lidar_points  # B  N  3

        elif trans_version == "v2":
            voxel_size = voxel_points.new_tensor(voxel_size)
            pc_range = np.array(pc_range, dtype=np.float32)
            origin = torch.tensor(
                (pc_range[:3] + pc_range[3:]) / 2.0,
                device=voxel_points.device,
                dtype=data_type,
            )

            n_voxels = self.nx.cpu().numpy()
            n_voxels = torch.tensor(
                np.array([n_voxels[0], n_voxels[1], n_voxels[2] * self.z_axis_times]),
                device=voxel_points.device,
                dtype=data_type,
            )

            new_origin = origin - n_voxels / 2.0 * voxel_size
            lidar_points = voxel_points * voxel_size.view(1, 1, 3) + new_origin.view(
                1, 1, 3
            )

            return lidar_points

        elif trans_version == "v3":
            # self.z_axis_times = 1 if self.n_voxels is not None else self.z_axis_times
            # # Redefine the generated voxel points
            # voxel_shape = (
            #     self.nx.cpu().numpy().astype(np.int)
            #     if self.n_voxels is None
            #     else self.n_voxels
            # )
            if self.n_voxels is None:
                voxel_shape = self.nx.cpu().numpy().astype(np.int)
            else:
                self.z_axis_times = 1
                voxel_shape = self.n_voxels

            # _width = torch.linspace(
            #     0.5, voxel_shape[0] - 0.5, voxel_shape[0], device=device
            # )
            # _hight = torch.linspace(
            #     0.5, voxel_shape[1] - 0.5, voxel_shape[1], device=device
            # )

            # _depth_voxel_shape = voxel_shape[2] * self.z_axis_times
            # _depth = torch.linspace(
            #     0.5, _depth_voxel_shape - 0.5, _depth_voxel_shape, device=device
            # )

            _width = torch.linspace(0, 1, voxel_shape[0], device=device)
            _hight = torch.linspace(0, 1, voxel_shape[1], device=device)

            _depth_voxel_shape = voxel_shape[2] * self.z_axis_times
            _depth = torch.linspace(0, 1, _depth_voxel_shape, device=device)

            reference_voxel = torch.stack(
                torch.meshgrid([_width, _hight, _depth]), dim=-1
            )  # x y z 3

            reference_voxel = reference_voxel.unsqueeze(0).repeat(B, 1, 1, 1, 1)

            pc_range = np.array(pc_range)

            reference_voxel[..., 0:1] = (
                reference_voxel[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
            )  # X
            reference_voxel[..., 1:2] = (
                reference_voxel[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
            )  # Y
            reference_voxel[..., 2:3] = (
                reference_voxel[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
            )  # Z

            # import pdb
            # pdb.set_trace()

            # B n_points 3
            # to lidar point from the predefined voxel point.
            return reference_voxel  # b x y z 3

        else:
            raise NotImplementedError

    @force_fp32(apply_to=())
    def lidar_points_to_cams_points(
        self,
        lidar_points,
        rots,
        trans,
        cam_inner,
        feats_shape,
        post_rots,
        post_trans,
        unproject=True,
        img_metas=None,
        unproject_order=1,
        normal_lidar2cams_point=False,
    ):
        """lidar_points_to_cams_points

        - execute fastbev transformation
        - N for points, n for num of multi-view
        - lidar_points  : b N 3
        - rots          : b n 3 3
        - trans         : b n 1 3
        - post_rots     : b n 3 3
        - post_trans    : b n 1 3
        - cams_inner    : b n 3 3

        Args:
            lidar_points (torch.tensor): [b, N, 3] or [b x y z 3]
            unproject_order: 0 for first unproject, 1 for after unprojcet
            normal_lidar2cams_point: normalization the porjected lidar point( points in cams space )
        Return:
            cams_points (torch.tensor): [B, n, N, 2]
            mask (torch.tensor): [B, n, N, 1]
        """

        B = lidar_points.size(0)
        NV = rots.size(1)

        # post trans and rots to lidar to cams space, before trans the lidar_point to cams 2d points
        if self.unproject_postion == "before" and self.unproject:
            if len(lidar_points.shape) == 5:
                lidar_points = rearrange(lidar_points, "b x y z 3 -> b (x y z) 3")
            lidar_points = lidar_points[:, None, ...].repeat(1, NV, 1, 1)

            # b n N 3 -> b n N 3 1
            # b n 3 3 -> b n 1 3 3
            # b n 1 3 3 @ b n N 3 1 -> b n N 3 1 -> b n N 3
            lidar_points = torch.matmul(
                post_rots.unsqueeze(2), lidar_points.unsqueeze(-1)
            ).squeeze(-1)
            # b n N 3 + b n 1 3
            lidar_points = lidar_points + post_trans.unsqueeze(2)

        lidar_points = torch.cat(
            [lidar_points, torch.ones_like(lidar_points[..., :1])], -1
        )  # b N 3 -> b N 4  or b x y z 3 -> b x y z 3

        if len(lidar_points.shape) == 5:
            lidar_points = lidar_points.flatten(1, 3)

        n_points = lidar_points.size(1)
        lidar_points = (
            lidar_points.view(B, 1, n_points, 4).repeat(1, NV, 1, 1)
            if len(lidar_points.shape) == 3
            else lidar_points
        )  # b n_cams n_points 4

        lidar2img_project = self.wrap_lidar2img_project(
            rots,
            trans,
            cam_inner,
            img_metas=img_metas,
            device=lidar_points.device,
            dtype=lidar_points.dtype,
            lidar2img=False,
        )  # b n_cams 4 4

        if lidar2img_project is None:
            cams_points, ref_depth = self.l2c_w_rots_trans(
                lidar_points[..., :3],
                rots=rots,
                trans=trans,
                intrins=cam_inner,
                post_rots=post_rots,
                post_trans=post_trans,
            )
        else:
            lidar_points = lidar_points.unsqueeze(-1)  # b n_cams n_points 4 1
            lidar2img_project = lidar2img_project.view(B, NV, 1, 4, 4)  # b n 1 4 4

            # if fp16_enabled:
            #     lidar2img_project = lidar2img_project.half()
            #     lidar_points = lidar_points.half()  # b n_cams n_points 4

            cams_points = torch.matmul(lidar2img_project, lidar_points).squeeze(
                -1
            )  # b n_cams n_points 4

        eps = 1e-5
        # cams_points_ref_depth = cams_points[..., 2:3].clone()
        # mask = cams_points_ref_depth > eps  # b n_cams n_points 1
        mask = cams_points[..., 2:3] > eps

        cams_points = cams_points[..., 0:2] / torch.maximum(
            cams_points[..., 2:3], torch.ones_like(cams_points[..., 2:3]) * eps
        )

        # image enhancement inverse mapping
        if (
            self.unproject_postion == "after" and self.unproject
        ):  # Inverse mapping transformation of image enhancement
            cams_points[..., :2] = (
                cams_points[..., :2] + post_trans.unsqueeze(-2)[..., :2]
            )
            cams_points[..., :2] = torch.matmul(
                post_rots[..., :2, :2].unsqueeze(-3), cams_points[..., :2].unsqueeze(-1)
            ).squeeze(-1)
            # b n_cams n_points 2 @ b n_cams 3[:2] 3[:2]
            # cams_points[..., :2] = torch.matmul(
            #     cams_points.unsqueeze(-2), post_rots[..., :2, :2].unsqueeze(-3)
            # ).squeeze(-2)

        cams_points[..., 0] /= self.data_config["input_size"][1]
        cams_points[..., 1] /= self.data_config["input_size"][0]

        mask = (
            mask
            & (cams_points[..., 0:1] > 0.0)
            & (cams_points[..., 1:2] > 0.0)
            & (cams_points[..., 0:1] < 1.0)
            & (cams_points[..., 1:2] < 1.0)
        )

        H, W = feats_shape
        # cams_points[..., 0] *= W / self.data_config["input_size"][1]
        # cams_points[..., 1] *= H / self.data_config["input_size"][0]

        # mask = (
        #     mask
        #     & (cams_points[..., 0:1] > 0)
        #     & (cams_points[..., 1:2] > 0)
        #     & (cams_points[..., 0:1] < W)
        #     & (cams_points[..., 1:2] < H)
        # )

        masks = mask  # b n_cams n_points 1

        # cams_points = []
        # masks = []

        # for bs_indices in range(B):
        #     H, W = feats_shape  # feat size shape

        #     cams_point_bs = []
        #     mask_bs = []
        #     # for view_indices in range(NV):
        #     #     pass

        #     lidar2img_project = _inner_wrap_lidar2img_project(
        #         rots[bs_indices], trans[bs_indices], cam_inner[bs_indices], img_metas=img_metas[bs_indices])

        #     eps = 1e-5
        #     cams_points_single_view = torch.matmul(lidar2img_project.view(NV, 1, 4, 4), lidar_points[bs_indices].view(
        #         NV, n_points, 4, 1)).squeeze(-1)  # n 1 4 4 @ n N 4 1 -> n N 4 1 -> n  N  4
        #     cams_points_single_view_depth = cams_points_single_view[..., 2:3].clone(
        #     )

        #     mask = (cams_points_single_view_depth > eps)

        #     cams_points_single_view = cams_points_single_view[..., 0:2] / torch.maximum(
        #         cams_points_single_view[..., 2:3], torch.ones_like(cams_points_single_view[..., 2:3]) * eps)

        #     if unproject_order == 1:  # Inverse mapping transformation of image enhancement

        #         if unproject:
        #             cams_points_single_view[..., :2] = torch.matmul(post_rots[bs_indices][:, :2, :2].unsqueeze(
        #                 1), cams_points_single_view[..., :2].unsqueeze(-1)).squeeze(-1)
        #             cams_points_single_view[..., :2] = cams_points_single_view[...,
        #                                                                        :2] + post_trans[bs_indices][..., None, :2]
        #         else:
        #             pass

        #     # Camera mapping point scaling based on downsampling of image features
        #     cams_points_single_view[...,
        #                             0] *= (W/self.data_config['input_size'][1])
        #     cams_points_single_view[...,
        #                             1] *= (H/self.data_config['input_size'][0])

        #     if normal_lidar2cams_point:
        #         cams_points_single_view[...,
        #                                 0] /= self.data_config['input_size'][1]
        #         cams_points_single_view[...,
        #                                 1] /= self.data_config['input_size'][0]
        #         cams_points_single_view = (cams_points_single_view - 0.5) * 2
        #         # use normalized intra-interval parameter thresholds
        #         mask = (mask & (cams_points_single_view[..., 0:1] > -1)
        #                 & (cams_points_single_view[..., 1:2] > -1)
        #                 & (cams_points_single_view[..., 0:1] < 1)
        #                 & (cams_points_single_view[..., 1:2] < 1))
        #     else:
        #         # threshold limits directly against image feature size
        #         # maybe this is error, wrong
        #         if True:
        #             mask = (mask & (cams_points_single_view[..., 0:1] > 0)
        #                     & (cams_points_single_view[..., 1:2] > 0)
        #                     & (cams_points_single_view[..., 0:1] < W)
        #                     & (cams_points_single_view[..., 1:2] < H))
        #         else:
        #             W = self.data_config['input_size'][1]
        #             H = self.data_config['input_size'][0]
        #             mask = (mask & (cams_points_single_view[..., 0:1] > 0)
        #                     & (cams_points_single_view[..., 1:2] > 0)
        #                     & (cams_points_single_view[..., 0:1] < W)
        #                     & (cams_points_single_view[..., 1:2] < H))

        #     cams_points.append(cams_points_single_view)  # n N 2
        #     masks.append(mask)

        # cams_points = torch.stack(cams_points)  # B n N 2
        # masks = torch.stack(masks)  # B n N 1

        return cams_points, masks

    def fastbev_trans_single(self, img_feats, points, valid):
        """
        function: 2d feature + predefined point cloud -> 3d volume
        input:
            img features: [n_cams, C, H, W]
            points: [ n_cams, n_point, 2]
            valid: [n_cams, n_points, 1]
        output:
            volume: [C, *n_voxels]
        """
        n_images, n_channels, _, _ = img_feats.shape

        if self.n_voxels is None:
            n_x_voxels, n_y_voxels, n_z_voxels = (
                int(self.nx[0]),
                int(self.nx[1]),
                int(self.nx[2] * self.z_axis_times),
            )
        else:
            n_x_voxels, n_y_voxels, n_z_voxels = self.n_voxels

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

        volume = volume.view(n_channels, n_x_voxels, n_y_voxels, n_z_voxels)

        return volume

    def get_volumes(
        self,
        rots,
        trans,
        post_rots,
        post_trans,
        cams_inner,
        img_feats,
        unproject=True,
        img_metas=None,
    ):
        """lidar points project to cams pixel points
        Args:
            rots (torch.tensr): [B,N,3,3] the rotation from lidar points to cams points
            trans (torch.tensor): [B,N,3,1] the translate from lidar points to cams
            post_rots (torch.tensor): [B,N,2,2] the rotation for augmentation imgs
            post_trans (torch.tensor): [B,N,2,1] ...
            cams_inner (torch.tensor): [B,N,3,3] ...
            img_feats (torch.tensor): [B, n, C, H, W] ...
        Return:
            volumes (torch.tensor): [B, C_cams, X, Y, Z]
        """
        B, _, _, H, W = img_feats.shape

        if self.other_args["trans_version"] == "v3":
            predefined_voxel_points = None
        else:
            # 3,x,y,z -> 3,N -> B,3,N -> B,N,3
            predefined_voxel_points = (
                self.scenes_voxel_point(z_times=self.z_axis_times)
                .view(1, 3, -1)
                .repeat(B, 1, 1)
                .permute(0, 2, 1)
                .to(img_feats.device)
            )

        lidar_points = self.voxel_grid_to_lidar_points(
            B,
            self.pct_range,
            self.voxel_size,
            predefined_voxel_points,
            data_type=img_feats.dtype,
            device=img_feats.device,
        )  # b N 3

        feats_shape = (H, W)
        cams_points, masks = self.lidar_points_to_cams_points(
            lidar_points,
            rots,
            trans,
            cams_inner,
            feats_shape,
            post_rots,
            post_trans,
            unproject,
            img_metas=img_metas,
        )  # b n_cams N_points 2, b n_cams N_points 1

        volumes = []
        for bs_indice in range(B):
            volume = self.fastbev_trans_single(
                img_feats[bs_indice], cams_points[bs_indice], masks[bs_indice]
            )
            volumes.append(volume)  # b...  |  c x y z

        # volumes = torch.stack(volumes).permute(0, 1, 4, 3, 2) # b c x y z -> b c z y x
        volumes = torch.stack(volumes)

        return volumes

    def forward(self, input, img_metas):
        x, rots, trans, intrins, post_rots, post_trans, _ = input
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)

        if C != self.numC_Trans:
            img_feat = self.featnet(x)
        else:
            img_feat = x

        depth_digit = 0
        if self.loss_depth_weight > 0:
            depth_feat = x
            cam_params = torch.cat(
                [
                    intrins.reshape(B * N, -1),
                    post_rots.reshape(B * N, -1),
                    post_trans.reshape(B * N, -1),
                    rots.reshape(B * N, -1),
                    trans.reshape(B * N, -1),
                ],
                dim=1,
            )

            depth_feat = self.se(depth_feat, cam_params)
            depth_feat = self.extra_depthnet(depth_feat)[0]
            depth_feat = self.dcn(depth_feat)
            depth_digit = self.depthnet(depth_feat)
            # depth_prob = self.get_depth_dist(depth_digit)

        volumes = self.get_volumes(
            rots,
            trans,
            post_rots,
            post_trans,
            intrins,
            img_feat.view(B, N, -1, H, W),
            self.unproject,
            img_metas=img_metas,
        )

        # volumes = self.conv3d_modify(volumes.contiguous())

        ll_voxel_feats = volumes.clone()  # b, 64, 12, 128, 128
        # ll_voxel_feats = volumes  # b, 64, 12, 128, 128

        # print(ll_voxel_feats)

        def _inner_bev_process(volumes):
            # b c d h w -> [(b c h w)-1, (b c h w)2，...,(b c h w)-d]
            if False:
                volumes_list = torch.unbind(volumes, dim=2)
                _bev_feats = torch.cat(volumes_list, dim=1)
            else:
                # B, C, X, Y, Z = volumes.shape

                # volumes_list = torch.unbind(volumes, dim=2)

                # _bev_feats = volumes.reshape(B, X, Y, Z*C).permute(0, 3, 1, 2)
                _bev_feats = rearrange(volumes, "b c x y z -> b x y (z c)")

            if self.selfcalib_conv_config:
                _bev_feats = self.bev_reduce(_bev_feats)

            if self.bev_channel_reduce:
                _bev_feats = self.ch_reduce(_bev_feats)  # 256 -> 64 warining
            else:
                pass

            return _bev_feats  # b, 64 128, 128

        bev_feat = _inner_bev_process(volumes=volumes)

        # print(colored(bev_feat, 'yellow'), bev_feat.shape)
        # raise NotImplemented

        return bev_feat, depth_digit, ll_voxel_feats

    def original_fastbev_imple(self, features, points, projection):
        """
        function: 2d feature + predefined point cloud -> 3d volume
        input:
            img features: [6, 64, 225, 400]
            points: [3, 200, 200, 12]
            projection (Optinal): [6, 3, 4]
        output:
            volume: [64, 200, 200, 12]
        """
        n_images, n_channels, height, width = features.shape
        n_x_voxels, n_y_voxels, n_z_voxels = points.shape[-3:]
        # [3, 200, 200, 12] -> [1, 3, 480000] -> [6, 3, 480000]
        points = points.view(1, 3, -1).expand(n_images, 3, -1)
        # [6, 3, 480000] -> [6, 4, 480000]
        points = torch.cat((points, torch.ones_like(points[:, :1])), dim=1)
        # ego_to_cam
        # [6, 3, 4] * [6, 4, 480000] -> [6, 3, 480000]

        points_2d_3 = torch.bmm(projection, points)  # lidar2img
        x = (points_2d_3[:, 0] / points_2d_3[:, 2]).round().long()  # [6, 480000]
        y = (points_2d_3[:, 1] / points_2d_3[:, 2]).round().long()  # [6, 480000]
        z = points_2d_3[:, 2]  # [6, 480000]
        valid = (
            (x >= 0) & (y >= 0) & (x < width) & (y < height) & (z > 0)
        )  # [6, 480000]

        # method2：特征填充，只填充有效特征，重复特征直接覆盖
        volume = torch.zeros(
            (n_channels, points.shape[-1]), device=features.device
        ).type_as(features)
        for i in range(n_images):
            volume[:, valid[i]] = features[i, :, y[i, valid[i]], x[i, valid[i]]]

        volume = volume.view(n_channels, n_x_voxels, n_y_voxels, n_z_voxels)
        return volume


@NECKS.register_module()
class InterDistill_FastBEV_L2C_V2(InterDistill_FastBEV):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @force_fp32(apply_to=("rots", "trans", "intrins", "post_rots", "post_trans"))
    def l2c_w_rots_trans(self, points, rots, trans, intrins, post_rots, post_trans):
        """
        Args:
            points:  B n_cams n_points 3
            rots :   B n_cams 3 3
            trans:   B n_cams 3 1
            intrins: B n_cams 3 3
        Return:
            points_img: B n_cams n_points 2
        """
        assert points.size(-1) <= 3  # b n_cams n_points 3
        combine = rots.matmul(torch.inverse(intrins))
        combine_inv = torch.inverse(combine)  # b n_cams 3 3

        # points_img = (points - trans.squeeze(-1).unsqueeze(-2)).matmul(
        #     combine_inv.permute(0, 1, 3, 2)
        # )  # b n_cams n_points 3

        points_img = (
            (points - trans.squeeze(-1).unsqueeze(-2))
            .matmul(torch.inverse(rots).permute(0, 1, 3, 2))
            .matmul(intrins.permute(0, 1, 3, 2))
        )
        reference_depth = points_img[..., 2:3]

        points_img = torch.cat(
            [points_img[..., :2] / points_img[..., 2:3], points_img[..., 2:3]], -1
        )

        points_img = points_img.matmul(
            post_rots.permute(0, 1, 3, 2)
        ) + post_trans.squeeze(-1).unsqueeze(-2)

        return points_img[..., :2], reference_depth

    @force_fp32(
        apply_to=(
            "lidar_points",
            "rots",
            "trans",
            "cam_inner",
            "post_rots",
            "post_trans",
        )
    )
    def lidar_points_to_cams_points(
        self,
        lidar_points,
        rots,
        trans,
        cam_inner,
        feats_shape,
        post_rots,
        post_trans,
        unproject=True,
        img_metas=None,
        unproject_order=1,
        normal_lidar2cams_point=False,
    ):
        """lidar_points_to_cams_points

        - execute fastbev transformation
        - N for points, n for num of multi-view
        - lidar_points  : b N 3
        - rots          : b n 3 3
        - trans         : b n 1 3
        - post_rots     : b n 3 3
        - post_trans    : b n 1 3
        - cams_inner    : b n 3 3

        Args:
            lidar_points (torch.tensor): [b, N, 3] or [b x y z 3]
            unproject_order: 0 for first unproject, 1 for after unprojcet
            normal_lidar2cams_point: normalization the porjected lidar point( points in cams space ).
        Return:
            cams_points (torch.tensor): [B, n, N, 2]
            mask (torch.tensor): [B, n, N, 1]
        """
        print(colored("v2 version excuting!!", "red"))

        B = lidar_points.size(0)
        NV = rots.size(1)

        if len(lidar_points.shape) == 5:
            lidar_points = lidar_points.flatten(1, 3)
        n_points = lidar_points.size(1)
        lidar_points = lidar_points.view(B, 1, n_points, 3).repeat(1, NV, 1, 1)
        # b n_cams n_points 4

        cams_points, ref_depth = self.l2c_w_rots_trans(
            lidar_points,
            rots=rots,
            trans=trans,
            intrins=cam_inner,
            post_rots=post_rots,
            post_trans=post_trans,
        )

        cams_points = cams_points / torch.maximum(
            ref_depth, torch.ones_like(ref_depth) * 1e-5
        )

        # cams_points[..., 0:1] /= ref_depth
        # cams_points[..., 1:2] /= ref_depth

        mask = ref_depth > 0

        # cams_points[..., 0] /= self.data_config["input_size"][1]
        # cams_points[..., 1] /= self.data_config["input_size"][0]

        # print(colored(cams_points, "yellow"))
        # print(colored(ref_depth, "yellow"))
        # mask = (
        #     mask
        #     & (cams_points[..., 0:1] > 0.0)
        #     & (cams_points[..., 1:2] > 0.0)
        #     & (cams_points[..., 0:1] < 1.0)
        #     & (cams_points[..., 1:2] < 1.0)
        # )

        H, W = feats_shape
        # cams_points[..., 0] *= W / self.data_config["input_size"][1]
        # cams_points[..., 1] *= H / self.data_config["input_size"][0]

        cams_points[..., 0] /= W
        cams_points[..., 1] /= H

        # print(colored(cams_points, "green"))

        mask = (
            mask
            & (cams_points[..., 0:1] > 0.0)
            & (cams_points[..., 1:2] > 0.0)
            & (cams_points[..., 0:1] < 1.0)
            & (cams_points[..., 1:2] < 1.0)
        )

        masks = mask  # b n_cams n_points 1

        return cams_points, masks


@NECKS.register_module()
class InterDistill_FastBEV_L2C_V3(InterDistill_FastBEV):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @force_fp32(apply_to=("lidar_points", "img_metas"))
    def lidar_points_to_cams_points(
        self,
        lidar_points,
        rots,
        trans,
        cam_inner,
        feats_shape,
        post_rots,
        post_trans,
        unproject=True,
        img_metas=None,
        unproject_order=1,
        normal_lidar2cams_point=False,
    ):
        print(colored("v3 version excuting!!", "red"))

        assert "lidar2imgs" in img_metas[0].keys()

        lidar2img = []
        for data in img_metas:
            lidar2img.append(data["lidar2imgs"])

        lidar2imgs = torch.stack(lidar2img).to(lidar_points.device)  # b num_cams 4 4

        B = lidar_points.size(0)
        lidar_points = F.pad(lidar_points, (0, 1), value=1)
        num_cams = img_metas[0]["lidar2imgs"].size(0)
        lidar_points = (
            lidar_points.unsqueeze(1).repeat(1, num_cams, 1, 1, 1, 1).unsqueeze(-1)
        )  # b num_cams x y z 4 1

        # b ncams 4 4 @ b ncams x y z 4 1 -> b ncams x y z 4
        img_points = torch.matmul(
            lidar2imgs.to(torch.float32).view(B, num_cams, 1, 1, 1, 4, 4),
            lidar_points.to(torch.float32),
        ).squeeze(-1)

        eps = 1e-5
        masks = img_points[..., 2:3] > eps
        img_points = img_points[..., 0:2] / torch.maximum(
            img_points[..., 2:3],
            torch.ones_like(img_points[..., 2:3]) * eps,
        )  # b ncams x y z 2

        img_points = img_points - post_trans[..., :2].view(B, num_cams, 1, 1, 1, 2)
        img_points = (
            torch.inverse(post_rots[..., :2, :2])
            .view(B, num_cams, 1, 1, 1, 2, 2)
            .matmul(img_points.unsqueeze(-1))
        ).squeeze(-1)

        resize_img_shape = self.data_config["input_size"]

        img_points[..., 0] /= resize_img_shape[1]
        img_points[..., 1] /= resize_img_shape[0]

        # img_points[..., 0] /= feats_shape[1]
        # img_points[..., 1] /= feats_shape[0]

        masks = (
            masks
            & (img_points[..., 1:2] > 0.0)
            & (img_points[..., 1:2] < 1.0)
            & (img_points[..., 0:1] < 1.0)
            & (img_points[..., 0:1] > 0.0)
        )

        img_points = img_points.flatten(2, -2)
        masks = masks.flatten(2, -2)

        return img_points, masks


@NECKS.register_module()
class InterDistill_FastBEV_MS_BEV(InterDistill_FastBEV_L2C_V3):
    """Still testing! ! !"""

    def __init__(
        self,
        is_ms_bev=False,
        num_ms_fusion=4,
        ms_bev_fusion=None,
        neck_bev=None,
        se_like=None,
        xy_transpose=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if not is_ms_bev:
            assert "Multi-scale bev adopts this class!"

        self.is_ms_bev = is_ms_bev
        self.is_neck_bev = neck_bev
        self.num_ms_fusion = num_ms_fusion
        self.se_like = se_like

        self.img_fusion_feats_fuse_1 = nn.Conv2d(
            self.numC_Trans * 4, self.numC_Trans, 3, 1, 1
        )
        self.img_fusion_feats_fuse_2 = nn.Conv2d(
            self.numC_Trans * 3, self.numC_Trans, 3, 1, 1
        )
        self.img_fusion_feats_fuse_3 = nn.Conv2d(
            self.numC_Trans * 2, self.numC_Trans, 3, 1, 1
        )

        self.conv3d_modify = nn.Conv3d(
            self.numC_Trans * num_ms_fusion, self.numC_Trans, kernel_size=1
        )

        self.ms_bev_fusion = ConvModule(
            self.numC_Trans * num_ms_fusion * self.n_voxels[2],
            ms_bev_fusion.out_channel,
            kernel_size=1,
            act_cfg=ms_bev_fusion.act_cfg if ms_bev_fusion.act_cfg else None,
            norm_cfg=ms_bev_fusion.norm_cfg if ms_bev_fusion.norm_cfg else None,
        )

        if neck_bev:
            self.neck_bev = NECKS.build(neck_bev)

        if self.bev_channel_reduce:
            self.ms_bev_ch_reduce = ConvModule(
                self.bev_channel_reduce.in_channel,
                self.bev_channel_reduce.out_channel,
                1,
                act_cfg=self.bev_channel_reduce.act_cfg,
            )

        if se_like:
            self.att = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(
                    se_like.in_channel, se_like.in_channel, kernel_size=1, stride=1
                ),
                nn.Sigmoid(),
            )

        if self.selfcalib_conv_config:
            self.selfcalib_encoder = BACKBONES.build(self.selfcalib_conv_config)
        self.xy_transpose = xy_transpose

    def img_unify_resize_and_fusion(self, ms_img_feats, index=1):
        assert isinstance(ms_img_feats, tuple) or isinstance(ms_img_feats, list)

        ms_level_start = 0
        img_feats_0_size = ms_img_feats[0].shape[-2:]
        # print(img_feats_0_size, '============')
        fuse_feats = [ms_img_feats[0]]
        for i in range(ms_level_start + 1, len(ms_img_feats)):
            fuse_feats.append(
                resize(
                    ms_img_feats[i],
                    img_feats_0_size,
                    mode="bilinear",
                    align_corners=False,
                )
            )
        if len(fuse_feats) > 1:
            fuse_feats = torch.cat(fuse_feats, dim=1)  # bn c*4 h w
        else:
            fuse_feats = fuse_feats[0]

        fuse_feats = self.__getattr__(f"img_fusion_feats_fuse_{index}")(fuse_feats)

        return fuse_feats

    def forward(self, input, img_metas):
        x, rots, trans, intrins, post_rots, post_trans, _ = input
        B = rots.size(0)
        depth_logits = 0

        # assert isinstance(x, tuple)  # Multi-scale image features

        low_voxel_feats = None

        ms_img_feats = []
        ms_img_feats.append(x[0])
        # ms_img_feats.append(self.img_unify_resize_and_fusion(x, index=1))
        # ms_img_feats.append(self.img_unify_resize_and_fusion(x[1:], index=2))
        # ms_img_feats.append(self.img_unify_resize_and_fusion(x[2:], index=3))

        assert ms_img_feats.__len__() == self.num_ms_fusion

        ms_volumes = []
        for i, img_feat in enumerate(ms_img_feats):
            _, C, H, W = img_feat.shape
            img_feat = img_feat.view(B, -1, C, H, W)

            # n_ms b c x y z
            ms_volumes.append(
                self.get_volumes(
                    rots,
                    trans,
                    post_rots,
                    post_trans,
                    intrins,
                    img_feats=img_feat,
                    unproject=True,
                    img_metas=img_metas,
                )
            )

        # ms_volumes = torch.stack(ms_volumes)
        # b c*num_ms_fusion x y z

        if self.num_ms_fusion > 1:
            ms_volumes = torch.cat(ms_volumes, dim=1)
        else:
            ms_volumes = ms_volumes[0]

        # print(
        #     colored(
        #         f"{ms_volumes},{ torch.sum(torch.where(ms_volumes > 0, 1, 0))}", "red"
        #     )
        # )

        # if ms_volumes.size(1) == self.numC_Trans * self.num_ms_fusion:
        #     low_voxel_feats = self.conv3d_modify(ms_volumes.permute(0, 1, 4, 2, 3))
        # else:
        #     low_voxel_feats = ms_volumes

        _, C, X, Y, Z = ms_volumes.shape

        if self.xy_transpose:
            ms_volumes = (
                ms_volumes.permute(0, 2, 3, 4, 1)
                .reshape(B, X, Y, Z * C)
                .permute(0, 3, 1, 2)
            )
        else:
            ms_volumes = (
                ms_volumes.permute(0, 2, 3, 4, 1)
                .reshape(B, X, Y, Z * C)
                .permute(0, 3, 2, 1)
            )

        bev_feats = self.ms_bev_fusion(ms_volumes)

        if self.is_neck_bev:
            bev_feats = self.neck_bev(bev_feats)

        if self.se_like:
            bev_feats = bev_feats * self.att(bev_feats)

        if self.bev_channel_reduce:
            bev_feats = self.ms_bev_ch_reduce(bev_feats)

        if self.selfcalib_conv_config:
            bev_feats = self.selfcalib_encoder(bev_feats)

        if self.xy_transpose:
            bev_feats = bev_feats.transpose(-1, -2)

        # print(bev_feats.shape, '===========',bev_feats)

        return bev_feats, depth_logits, low_voxel_feats
