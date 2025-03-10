import torch
from torch.nn import functional as F
from mmcv.runner import force_fp32
from mmcv.cnn import ConvModule
from mmdet3d.models.utils.self_print import feats_to_img
from .. import builder
from ..builder import DETECTORS, build_loss
from .centerpoint import CenterPoint
import torch.nn as nn
import numpy as np
from mmdet3d.ops import Voxelization
from mmdet3d.core import bbox3d2result
import matplotlib.pyplot as plt
from termcolor import colored
from mmdet.models.backbones.resnet import ResNet
from mmdet3d.models.detectors.bevdet import BEVStereo4D
from mmdet.utils import get_root_logger
from mmdet3d.models.utils.self_print import feats_to_img
from mmdet3d.models.detectors.bevdet import BEVDepth4D,BEVStereo4D
from mmdet3d.models.backbones.swin import SwinTransformer
from mmdet3d.models.backbones.swin_bev import SwinTransformerBEVFT
import time


@DETECTORS.register_module()
class D3PD(BEVDepth4D):
    def __init__(
        self,
        teacher_pretrained=None,
        student_pretrained=None,
        vovnet_pretrained=None,
        load_with_teacher_head=None,
        freeze_img_backbone=False,
        radar_voxel_layer=None,
        radar_pillar_encoder=None,
        radar_middle_encoder=None,
        radar_backbone=None,
        radar_neck=None,
        rc_bev_fusion=None,
        pts_bbox_head_tea=None,
        distillation=None,
        **kwargs,
    ):
        super(D3PD, self).__init__(**kwargs)
        self.extra_ref_frames = 1
        self.temporal_frame = self.num_frame
        self.num_frame += self.extra_ref_frames

        self.student_pretrained = student_pretrained
        self.teacher_pretrained = teacher_pretrained
        self.vovnet_pretrained = vovnet_pretrained
        self.load_with_teacher_head = load_with_teacher_head
        self.freeze_img_backbone = freeze_img_backbone
        self.distillation = distillation

        self.rc_bev_fusion = (
            builder.build_neck(rc_bev_fusion) if rc_bev_fusion is not None else None
        )
        self._init_teacher_model(pts_bbox_head_tea, **kwargs)
        self._init_student_model()
        self._init_radar_net(
            radar_voxel_layer,
            radar_pillar_encoder,
            radar_middle_encoder,
            radar_backbone,
            radar_neck,
        )
        self._init_img_backbone(close=self.freeze_img_backbone)
        self._init_distill_module()

        self.time_recoder = {}

    def _init_teacher_model(self, pts_bbox_head_tea=None, **kwargs):
        logger = get_root_logger()
        teacher_weight = torch.load(
            self.teacher_pretrained,
            map_location="cuda:{}".format(torch.cuda.current_device()),
        )["state_dict"]
        load_pts = ["pts_middle_encoder", "pts_backbone", "pts_neck"]
        for load_key in load_pts:
            # 首先，将加载的模型的 key - value 进行处理，保证预训练参数可以让每一个模块加载
            dict_load = {
                _key.replace(load_key + ".", ""): teacher_weight[_key]
                for _key in teacher_weight
                if load_key in _key
            }
            getattr(self, load_key).load_state_dict(dict_load, strict=False)
            logger.info("Loaded pretrained {}".format(load_key))
            assert len(dict_load) > 0

        if self.pts_middle_encoder:
            for param in self.pts_middle_encoder.parameters():
                param.requires_grad = False
            self.pts_middle_encoder.eval()

        if self.pts_backbone:
            for param in self.pts_backbone.parameters():
                param.requires_grad = False
            self.pts_backbone.eval()

        if self.pts_neck:
            for param in self.pts_neck.parameters():
                param.requires_grad = False
            self.pts_neck.eval()

        if pts_bbox_head_tea:
            train_cfg = kwargs["train_cfg"]
            test_cfg = kwargs["test_cfg"]
            pts_train_cfg = train_cfg.pts if train_cfg else None
            pts_bbox_head_tea.update(train_cfg=pts_train_cfg)
            pts_test_cfg = test_cfg.pts if test_cfg else None
            pts_bbox_head_tea.update(test_cfg=pts_test_cfg)
            self.pts_bbox_head_tea = builder.build_head(pts_bbox_head_tea)

            dict_load = {
                _key.replace("pts_bbox_head.", ""): teacher_weight[_key]
                for _key in teacher_weight
                if "pts_bbox_head" in _key
            }

            self.pts_bbox_head_tea.load_state_dict(dict_load, strict=False)

            for param in self.pts_bbox_head_tea.parameters():
                param.requires_grad = False
            self.pts_bbox_head_tea.eval()

    def _init_student_model(self):

        if self.student_pretrained is not None:

            logger = get_root_logger()
            student_pretrained_load = [
                "img_neck",
                "img_view_transformer",
                "img_bev_encoder_backbone",
                "img_bev_encoder_neck",
                "pre_process",
            ]

            sutdent_model = torch.load(
                self.student_pretrained,
                map_location="cuda:{}".format(torch.cuda.current_device()),
            )["state_dict"]

            for load_key in student_pretrained_load:

                dict_load = {
                    _key.replace(load_key + ".", ""): sutdent_model[_key]
                    for _key in sutdent_model
                    if load_key in _key
                }

                if "pre_process" in load_key:
                    load_key = "pre_process_net"

                getattr(self, load_key).load_state_dict(dict_load, strict=False)

                print("Loaded pretrained {}".format(load_key))
                logger.info("Loaded pretrained {}".format(load_key))
                assert len(dict_load) > 0

            # for module in student_pretrained_load:

            #     for param in getattr(self, module).parameters():
            #         param.requires_grad = False
            #     getattr(self, module).eval()

        if self.load_with_teacher_head:
            teacher_weight = torch.load(
                self.teacher_pretrained,
                map_location="cuda:{}".format(torch.cuda.current_device()),
            )["state_dict"]

            dict_load = {
                _key.replace("pts_bbox_head.task_heads.", ""): teacher_weight[_key]
                for _key in teacher_weight
                if "pts_bbox_head.task_heads" in _key
            }

            self.pts_bbox_head.task_heads.load_state_dict(dict_load, strict=False)

    # def _init_img_transformation(self):
    #     pass

    def _init_radar_net(
        self,
        radar_voxel_layer,
        radar_pillar_encoder,
        radar_middle_encoder,
        radar_backbone,
        radar_neck,
    ):
        if radar_voxel_layer:
            self.radar_voxel_layer = Voxelization(**radar_voxel_layer)
        else:
            self.radar_voxel_layer = None

        if radar_pillar_encoder:
            self.radar_pillar_encoder = builder.build_voxel_encoder(
                radar_pillar_encoder
            )

        if radar_middle_encoder:
            self.radar_middle_cfg = radar_middle_encoder
            self.radar_middle_encoder = builder.build_middle_encoder(
                radar_middle_encoder
            )

        if radar_backbone is not None:
            self.with_radar_backbone = True
            self.radar_backbone = builder.build_backbone(radar_backbone)
        else:
            self.with_radar_backbone = False

        if radar_neck is not None:
            self.with_radar_neck = True
            self.radar_neck = builder.build_neck(radar_neck)
        else:
            self.with_radar_neck = False

    def _init_img_backbone(self, close=False):

        if self.img_backbone.__class__.__name__ == "VoVNetCP":
            img_backbone_weight = self.vovnet_pretrained

            img_backbone_load = torch.load(
                img_backbone_weight,
                map_location="cuda:{}".format(torch.cuda.current_device()),
            )["model"]

            img_backbone_dict_load = {
                _key.replace("backbone.bottom_up.", ""): img_backbone_load[_key]
                for _key in img_backbone_load
                if "backbone.bottom_up" in _key
            }

            self.img_backbone.load_state_dict(img_backbone_dict_load, strict=False)
            logger = get_root_logger()
            logger.warn("Loaded pretrained {}".format("backbone.bottom_up"))
            # print(colored("Loaded pretrained {}".format("backbone.bottom_up"), "green"))
            assert len(img_backbone_dict_load) > 0

        if False:
            for param in self.img_backbone.parameters():
                param.requires_grad = False
            self.img_backbone.eval()

    def _init_distill_module(self):

        if self.distillation.get("sparse_feats_distill"):
            # Construct sparse feature distillation, including knowledge transfer of image bev and radar bev two view features
            self.sparse_feats_distill_img_bev = (
                build_loss(self.distillation.sparse_feats_distill.img_bev_distill)
                if "img_bev_distill" in self.distillation.sparse_feats_distill.keys()
                else None
            )

            self.sparse_feats_distill_radar_bev = (
                build_loss(self.distillation.sparse_feats_distill.radar_bev_distill)
                if "radar_bev_distill" in self.distillation.sparse_feats_distill.keys()
                else None
            )
            self.radar_ms_feats_distill = (
                build_loss(self.distillation.sparse_feats_distill.radar_ms_distill)
                if "sparse_feats_distill" in self.distillation.keys()
                and "radar_ms_distill" in self.distillation.sparse_feats_distill.keys()
                else None
            )
        else:
            self.sparse_feats_distill_img_bev = None
            self.sparse_feats_distill_radar_bev = None
            self.radar_ms_feats_distill = None

        # self.sampling_pos_distill = (
        #     build_loss(self.distillation.sampling_pos_distill)
        #     if "sampling_pos_distill" in self.distillation.keys()
        #     else None
        # )
        # self.sampling_feats_distill = (
        #     build_loss(self.distillation.sampling_feats_distill)
        #     if "sampling_feats_distill" in self.distillation.keys()
        #     else None
        # )

        self.det_feats_distill = (
            build_loss(self.distillation.det_feats_distill)
            if "det_feats_distill" in self.distillation.keys()
            else None
        )

        if "det_result_distill" in self.distillation.keys():
            self.ret_sum = self.distillation.det_result_distill["ret_sum"]
            self.dcdistill_loss = build_loss(self.distillation.det_result_distill)
        else:
            self.ret_sum = None
            self.dcdistill_loss = None

        self.smfd_distill_loss = (
            build_loss(self.distillation.mask_bev_feats_distill)
            if "mask_bev_feats_distill" in self.distillation.keys()
            else None
        )

        self.heatmap_aug_distill_loss = (
            build_loss(self.distillation.heatmap_aug_distill)
            if "heatmap_aug_distill" in self.distillation.keys()
            else None
        )

    @torch.no_grad()
    @force_fp32()
    def radar_voxelization(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points, number of points
                per voxel, and coordinates.
        """
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.radar_voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode="constant", value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    def extract_pts_feat(self, pts, img_feats, img_metas):
        """Extract features of points."""
        if not self.pts_middle_encoder:
            return None
        if not self.with_pts_bbox:
            return None

        voxels, num_points, coors = self.voxelize(pts)

        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)

        if self.radar_ms_feats_distill:
            pts_ms_feats = [data for data in x]

        if self.with_pts_neck:
            x = self.pts_neck(x)

        if self.radar_ms_feats_distill:
            return x, pts_ms_feats
        else:
            return x, None

    def extract_radar_feat(self, radar_points, ret_coords=False):
        """Extract features of points."""
        if not self.with_pts_bbox:
            return None

        # radar visualization
        # random_sava_num = np.random.randint(0, 10000000000)
        # radar_points[0].cpu().numpy().tofile(
        #     f"./radar_vis/radapoints_{random_sava_num}.bin"
        # )

        radar_process_time_start = time.perf_counter()
        voxels, num_points, coors = self.radar_voxelization(radar_points)

        radar_pillar_features = self.radar_pillar_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        x = self.radar_middle_encoder(radar_pillar_features, coors, batch_size)
        radar_process_time_end = time.perf_counter()

        self.time_recoder.update(
            dict(radar_process_time=(radar_process_time_end - radar_process_time_start))
        )

        radar_encoder_time_start = time.perf_counter()
        if self.with_radar_backbone:
            x = self.radar_backbone(x)

            if self.radar_ms_feats_distill:
                radar_ms_feas = [data for data in x]

        if self.with_radar_neck:
            x = self.radar_neck(x)

        radar_encoder_time_end = time.perf_counter()
        self.time_recoder.update(
            dict(radar_encoder_time=(radar_encoder_time_end - radar_encoder_time_start))
        )

        if self.radar_ms_feats_distill:
            if ret_coords:
                return x, coors, radar_ms_feas
            else:
                return x, radar_ms_feas
        else:
            if ret_coords:
                return x, coors, None
            else:
                return x, None

    def extract_stereo_ref_feat(self, x):
        B, N, C, imH, imW = x.shape
        x = x.view(B * N, C, imH, imW)
        if isinstance(self.img_backbone, ResNet):
            if self.img_backbone.deep_stem:
                x = self.img_backbone.stem(x)
            else:
                x = self.img_backbone.conv1(x)
                x = self.img_backbone.norm1(x)
                x = self.img_backbone.relu(x)
            x = self.img_backbone.maxpool(x)
            for i, layer_name in enumerate(self.img_backbone.res_layers):
                res_layer = getattr(self.img_backbone, layer_name)
                x = res_layer(x)
                return x
        elif isinstance(self.img_backbone, SwinTransformer):
            x = self.img_backbone.patch_embed(x)
            hw_shape = (
                self.img_backbone.patch_embed.DH,
                self.img_backbone.patch_embed.DW,
            )
            if self.img_backbone.use_abs_pos_embed:
                x = x + self.img_backbone.absolute_pos_embed
            x = self.img_backbone.drop_after_pos(x)

            for i, stage in enumerate(self.img_backbone.stages):
                x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
                out = out.view(-1, *out_hw_shape, self.img_backbone.num_features[i])
                out = out.permute(0, 3, 1, 2).contiguous()
                return out
        elif isinstance(self.img_backbone, SwinTransformerBEVFT):
            # x = self.img_backbone.patch_embed(x)
            # hw_shape = (self.img_backbone.patch_embed.DH,
            #             self.img_backbone.patch_embed.DW)
            x, hw_shape = self.img_backbone.patch_embed(x)
            if self.img_backbone.use_abs_pos_embed:
                # x = x + self.img_backbone.absolute_pos_embed
                absolute_pos_embed = F.interpolate(
                    self.img_backbone.absolute_pos_embed, size=hw_shape, mode="bicubic"
                )
                x = x + absolute_pos_embed.flatten(2).transpose(1, 2)
            x = self.img_backbone.drop_after_pos(x)

            for i, stage in enumerate(self.img_backbone.stages):
                x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
                out = out.view(-1, *out_hw_shape, self.img_backbone.num_features[i])
                out = out.permute(0, 3, 1, 2).contiguous()
                return out
        else:
            raise NotImplemented

    def prepare_bev_feat(
        self,
        img,
        sensor2keyego,
        ego2global,
        intrin,
        post_rot,
        post_tran,
        bda,
        mlp_input,
        feat_prev_iv,
        k2s_sensor,
        extra_ref_frame,
    ):
        if extra_ref_frame:
            stereo_feat = self.extract_stereo_ref_feat(img)
            return None, None, stereo_feat

        img_encoder_time_start = time.perf_counter()
        x, stereo_feat = self.image_encoder(img, stereo=True)
        img_encoder_time_end = time.perf_counter()
        self.time_recoder.update(
            dict(img_encoder_times=(img_encoder_time_end - img_encoder_time_start))
        )

        metas = dict(
            k2s_sensor=k2s_sensor,
            intrins=intrin,
            post_rots=post_rot,
            post_trans=post_tran,
            frustum=self.img_view_transformer.cv_frustum.to(x),
            cv_downsample=4,
            downsample=self.img_view_transformer.downsample,
            grid_config=self.img_view_transformer.grid_config,
            cv_feat_list=[feat_prev_iv, stereo_feat],
        )

        ivt_time_start = time.perf_counter()
        bev_feat, depth = self.img_view_transformer(
            [x, sensor2keyego, ego2global, intrin, post_rot, post_tran, bda, mlp_input],
            metas,
        )
        ivt_time_end = time.perf_counter()
        self.time_recoder.update(dict(ivt_time=(ivt_time_end - ivt_time_start)))
        if self.pre_process:
            bev_feat = self.pre_process_net(bev_feat)[0]
        return bev_feat, depth, stereo_feat

    def extract_img_feat(
        self, img, img_metas, pred_prev=False, sequential=False, **kwargs
    ):

        if sequential:
            # Todo
            assert False

        (
            imgs,
            sensor2keyegos,
            ego2globals,
            intrins,
            post_rots,
            post_trans,
            bda,
            curr2adjsensor,
        ) = self.prepare_inputs(img, stereo=True)

        """Extract features of images."""
        bev_feat_list = []
        depth_key_frame = None
        feat_prev_iv = None

        for fid in range(self.num_frame - 1, -1, -1):
            img, sensor2keyego, ego2global, intrin, post_rot, post_tran = (
                imgs[fid],
                sensor2keyegos[fid],
                ego2globals[fid],
                intrins[fid],
                post_rots[fid],
                post_trans[fid],
            )
            key_frame = fid == 0
            extra_ref_frame = fid == self.num_frame - self.extra_ref_frames
            if key_frame or self.with_prev:
                if self.align_after_view_transfromation:
                    sensor2keyego, ego2global = sensor2keyegos[0], ego2globals[0]
                mlp_input = self.img_view_transformer.get_mlp_input(
                    sensor2keyegos[0], ego2globals[0], intrin, post_rot, post_tran, bda
                )
                inputs_curr = (
                    img,
                    sensor2keyego,
                    ego2global,
                    intrin,
                    post_rot,
                    post_tran,
                    bda,
                    mlp_input,
                    feat_prev_iv,
                    curr2adjsensor[fid],
                    extra_ref_frame,
                )
                if key_frame:
                    bev_feat, depth, feat_curr_iv = self.prepare_bev_feat(*inputs_curr)
                    depth_key_frame = depth
                else:
                    with torch.no_grad():
                        bev_feat, depth, feat_curr_iv = self.prepare_bev_feat(
                            *inputs_curr
                        )
                if not extra_ref_frame:
                    bev_feat_list.append(bev_feat)
                feat_prev_iv = feat_curr_iv
        if pred_prev:
            # Todo
            assert False
        if not self.with_prev:
            bev_feat_key = bev_feat_list[0]
            if len(bev_feat_key.shape) == 4:
                b, c, h, w = bev_feat_key.shape
                bev_feat_list = [
                    torch.zeros(
                        [b, c * (self.num_frame - self.extra_ref_frames - 1), h, w]
                    ).to(bev_feat_key),
                    bev_feat_key,
                ]
            else:
                b, c, z, h, w = bev_feat_key.shape
                bev_feat_list = [
                    torch.zeros(
                        [b, c * (self.num_frame - self.extra_ref_frames - 1), z, h, w]
                    ).to(bev_feat_key),
                    bev_feat_key,
                ]
        if self.align_after_view_transfromation:
            for adj_id in range(self.num_frame - 2):
                bev_feat_list[adj_id] = self.shift_feature(
                    bev_feat_list[adj_id],
                    [sensor2keyegos[0], sensor2keyegos[self.num_frame - 2 - adj_id]],
                    bda,
                )
        bev_feat = torch.cat(bev_feat_list, dim=1)

        bev_neck_time_start = time.perf_counter()
        x = self.bev_encoder(bev_feat)
        bev_neck_time_end = time.perf_counter()
        self.time_recoder.update(
            dict(bev_necK_time=(bev_neck_time_end - bev_neck_time_start))
        )
        return [x], depth_key_frame

    def fusion_img_radar_bev(self, img_bev, radar_bev, **kwargs) -> list:
        img_bev = (
            img_bev[0]
            if isinstance(img_bev, list) or isinstance(img_bev, tuple)
            else img_bev
        )
        radar_bev = radar_bev[0] if isinstance(radar_bev, list) else radar_bev

        radar_points = kwargs.get("radar_points")
        # output = self.reduce_conv(torch.cat([img_bev, radar_bev], dim=1))

        det_input_dict = {}

        if self.rc_bev_fusion is not None:
            output = self.rc_bev_fusion(img_bev, radar_bev)
            if not isinstance(output, dict):
                det_input_dict.update({"det_feats": [output]})
            else:
                det_input_dict.update(output)

        return det_input_dict

    def extract_feat(self, points, radar, img, img_metas, **kwargs):
        """Extract features from images and points."""

        img_feats, depth = self.extract_img_feat(img, img_metas)
        # img_feats, depth=super(BEVStereo4D)
        radar_feats, radar_ms_feats = self.extract_radar_feat(radar)

        pts_feats, pts_ms_feats = self.extract_pts_feat(points, img_feats, img_metas)

        fusion_time_start = time.perf_counter()
        det_input_feats = self.fusion_img_radar_bev(
            img_feats, radar_feats, pts_feats=pts_feats
        )
        fusion_time_end = time.perf_counter()
        self.time_recoder.update(
            dict(fusion_time=(fusion_time_end - fusion_time_start))
        )

        # base_path = "/mnt/data/exps/D3PD/d3pd/out/v3-feats-out"
        # feats_to_img(det_input_feats["det_feats"], base_path=base_path, suffix="fusion")
        # feats_to_img(img_feats, base_path=base_path, suffix="img_feats")

        # feats_to_img(
        #     det_input_feats.get("student_sampling_feats"),
        #     base_path=base_path,
        #     suffix="img_feats_sampling",
        # )
        # feats_to_img(radar_feats, base_path=base_path, suffix="radar_feats")
        # feats_to_img(pts_feats, base_path=base_path, suffix="pts_feats")

        if radar_ms_feats and pts_ms_feats:
            det_input_feats.update(dict(radar_ms_feats=radar_ms_feats))
            det_input_feats.update(dict(pts_ms_feats=pts_ms_feats))

        return img_feats, pts_feats, radar_feats, depth, det_input_feats

    def forward_distill_loss(
        self,
        teacher_bev_feats=None,
        teacher_sampling_pos=None,
        teacher_sampling_feats=None,
        tea_resp_bboxes=None,
        student_bev_feats=None,
        student_samping_pos=None,
        student_samping_feats=None,
        stu_resp_bboxes=None,
        pts_ms_feats=None,
        radar_ms_feats=None,
        radar_feats=None,
        rc_fusion_feats=None,
        gt_bboxes_3d=None,
        bda_mat=None,
        **kwargs,
    ):
        losses = {}
        ivt_cfg = {}
        # ivt_cfg.update(
        #     {
        #         "dx": self.img_view_transformer.dx,
        #         "bx": self.img_view_transformer.bx,
        #         "nx": self.img_view_transformer.nx,
        #     }
        # )

        ivt_cfg.update(
            {
                "dx": self.img_view_transformer.grid_interval,
                "bx": (
                    self.img_view_transformer.grid_lower_bound
                    + self.img_view_transformer.grid_interval
                )
                / 2.0,
                "nx": self.img_view_transformer.grid_size,
            }
        )

        if self.sparse_feats_distill_img_bev is not None:

            # ================lidar distill img-bev================

            loss_inter_channel = self.sparse_feats_distill_img_bev(
                student_bev_feats,
                teacher_bev_feats,
                gt_bboxes_list=gt_bboxes_3d,
                ivt_cfg=ivt_cfg,
            )

            if isinstance(loss_inter_channel, tuple):
                losses.update({"loss_inter_channel_img_bev": loss_inter_channel[0]})
            else:
                losses.update({"loss_inter_channel_img_bev": loss_inter_channel})

        if self.sparse_feats_distill_radar_bev is not None:

            # ================lidar distill radar-bev================
            if self.sparse_feats_distill_radar_bev is not None:
                loss_inter_channel = self.sparse_feats_distill_radar_bev(
                    radar_feats, teacher_bev_feats, gt_bboxes_3d, ivt_cfg=ivt_cfg
                )  # A self-distillation scheme is used here to migrate the features of the img-bev network to the network's own radar-bev

            if isinstance(loss_inter_channel, tuple):
                losses.update({"loss_inter_channel_radar_bev": loss_inter_channel[0]})
            else:
                losses.update({"loss_inter_channel_radar_bev": loss_inter_channel})

        if rc_fusion_feats is not None and self.det_feats_distill is not None:
            loss_det_feats_distill = self.det_feats_distill(
                rc_fusion_feats, teacher_bev_feats
            )
            losses.update({"loss_det_feats_distill": loss_det_feats_distill})

        if teacher_sampling_pos is not None and self.sampling_pos_distill is not None:
            loss_sampling_pos_distill = self.sampling_pos_distill(
                student_samping_pos, teacher_sampling_pos
            )
            losses.update({"loss_sampling_pos_distill": loss_sampling_pos_distill})

        if (
            teacher_sampling_feats is not None
            and self.sampling_feats_distill is not None
        ):
            loss_sampling_warp_distill = self.sampling_feats_distill(
                student_samping_feats, teacher_sampling_feats
            )
            losses.update({"loss_sampling_feats_distill": loss_sampling_warp_distill})

        if self.dcdistill_loss is not None:

            if self.ret_sum:
                loss_dc_distill = self.dcdistill_loss(
                    tea_resp_bboxes, stu_resp_bboxes, gt_bboxes_3d
                )
                losses.update({"loss_dc_distill": loss_dc_distill})
            else:
                loss_dc_reg_distill, loss_dc_cls_distill = self.dcdistill_loss(
                    tea_resp_bboxes, stu_resp_bboxes, gt_bboxes_3d
                )
                losses.update({"loss_dc_reg_distill": loss_dc_reg_distill})
                losses.update({"loss_dc_cls_distill": loss_dc_cls_distill})

        # Self-learning mask focused distillation
        if self.smfd_distill_loss is not None:
            loss_smfd_distill, auto_mask = self.smfd_distill_loss(
                rc_fusion_feats,
                teacher_bev_feats,
                gt_bboxes_3d,
                stu_resp_bboxes,
                bda_mat=bda_mat,
            )
            losses.update(dict(loss_smfd_distill=loss_smfd_distill))

            # forward_distill_loss_input_cfg=
        if self.radar_ms_feats_distill:
            radar_ms_distill_loss = self.radar_ms_feats_distill(
                radar_ms_feats, pts_ms_feats, gt_bboxes_3d=gt_bboxes_3d
            )
            losses.update(dict(loss_radar_ms=radar_ms_distill_loss))

        if self.heatmap_aug_distill_loss is not None:
            loss_heatmap_aug = self.heatmap_aug_distill_loss(
                stu_resp_bboxes, tea_resp_bboxes, auto_mask
            )

            losses.update(dict(loss_heatmap_aug=loss_heatmap_aug))

        return losses

    def forward_train(
        self,
        points=None,
        radar=None,
        img_metas=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        gt_labels=None,
        gt_bboxes=None,
        img_inputs=None,
        proposals=None,
        gt_bboxes_ignore=None,
        **kwargs,
    ):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        img_feats, pts_feats, radar_feats, depth, feats_wrap = self.extract_feat(
            points, radar=radar, img=img_inputs, img_metas=img_metas, **kwargs
        )

        from mmdet3d.models.utils.self_print import print2file

        print2file(self.time_recoder, "d3pd_time_recoder")

        pts_feats_kd = pts_feats[0]
        img_feats_kd = img_feats[0]
        radar_feats_kd = radar_feats[0]

        gt_depth = kwargs["gt_depth"]
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        losses = dict(loss_depth=loss_depth)
        losses_pts = self.forward_pts_train(
            img_feats, gt_bboxes_3d, gt_labels_3d, img_metas, gt_bboxes_ignore
        )
        losses.update(losses_pts)

        teacher_sampling_pos = feats_wrap.get("teacher_sampling_pos")
        teacher_sampling_feats = feats_wrap.get("teacher_sampling_feats")
        student_sampling_pos = feats_wrap.get("student_sampling_pos")
        student_sampling_feats = feats_wrap.get("student_sampling_feats")
        radar_ms_feats = feats_wrap.get("radar_ms_feats")
        pts_ms_feats = feats_wrap.get("pts_ms_feats")
        det_feats = feats_wrap.get("det_feats")
        det_feats_kd = det_feats[0]

        teacher_outs = (
            self.pts_bbox_head_tea(pts_feats) if self.distillation is not None else None
        )
        student_outs = self.pts_bbox_head(det_feats)

        distill_losses = self.forward_distill_loss(
            teacher_bev_feats=pts_feats_kd,
            student_bev_feats=img_feats_kd,
            teacher_sampling_pos=teacher_sampling_pos,
            student_samping_pos=student_sampling_pos,
            teacher_sampling_feats=teacher_sampling_feats,
            student_samping_feats=student_sampling_feats,
            tea_resp_bboxes=teacher_outs,
            stu_resp_bboxes=student_outs,
            pts_ms_feats=pts_ms_feats,
            radar_ms_feats=radar_ms_feats,
            radar_feats=radar_feats_kd,
            rc_fusion_feats=det_feats_kd,
            gt_bboxes_3d=gt_bboxes_3d,
        )

        losses.update(distill_losses)

        return losses

    def forward_test(
        self, points=None, radar=None, img_metas=None, img_inputs=None, **kwargs
    ):
        """
        Args:
            points (list[torch.Tensor]): the outer list indicates test-time
                augmentations and inner torch.Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            img (list[torch.Tensor], optional): the outer
                list indicates test-time augmentations and inner
                torch.Tensor should have a shape NxCxHxW, which contains
                all images in the batch. Defaults to None.
        """
        for var, name in [(img_inputs, "img_inputs"), (img_metas, "img_metas")]:
            if not isinstance(var, list):
                raise TypeError("{} must be a list, but got {}".format(name, type(var)))

        num_augs = len(img_inputs)
        if num_augs != len(img_metas):
            raise ValueError(
                "num of augmentations ({}) != num of image meta ({})".format(
                    len(img_inputs), len(img_metas)
                )
            )

        if not isinstance(img_inputs[0][0], list):
            img_inputs = [img_inputs] if img_inputs is None else img_inputs
            points = [points] if points is None else points
            radar = [radar] if radar is None else radar
            return self.simple_test(
                points[0], radar[0], img_metas[0], img_inputs[0], **kwargs
            )
        else:
            return self.aug_ttest(None, img_metas[0], img_inputs[0], **kwargs)

    def simple_test(self, points, radar, img_metas, img=None, rescale=False, **kwargs):
        """Test function without augmentaiton."""
        _, _, _, _, det_input_feats = self.extract_feat(
            points, radar=radar, img=img, img_metas=img_metas, **kwargs
        )

        img_feats = det_input_feats.get("det_feats")

        # base_path = "/mnt/data/exps/D3PD/d3pd/out/v3-feats-out"
        # feats_to_img(img_feats, base_path=base_path, suffix="img_feats")

        bbox_list = [dict() for _ in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict["pts_bbox"] = pts_bbox
        return bbox_list
