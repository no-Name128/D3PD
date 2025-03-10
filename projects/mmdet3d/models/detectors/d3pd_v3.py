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


@DETECTORS.register_module()
class D3PD_V3(BEVStereo4D):
    def __init__(
        self,
        teacher_pretrained=None,
        student_pretrained=None,
        vovnet_pretrained=None,
        load_with_teacher_head=None,
        freeze_img_backbone=False,
        # img_view_transformer=None,
        # img_bev_encoder_backbone=None,
        # img_bev_encoder_neck=None,
        radar_voxel_layer=None,
        radar_pillar_encoder=None,
        radar_middle_encoder=None,
        radar_backbone=None,
        radar_neck=None,
        bi_dire_fusion=None,
        middle_radar_aug=None,
        rc_bev_fusion=None,
        pts_bbox_head_tea=None,
        distillation=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.student_pretrained = student_pretrained
        self.teacher_pretrained = teacher_pretrained
        self.vovnet_pretrained = vovnet_pretrained
        self.load_with_teacher_head = load_with_teacher_head
        self.freeze_img_backbone = freeze_img_backbone

        self.bi_dire_fusion = bi_dire_fusion
        self.middle_radar_aug = middle_radar_aug
        self.rc_bev_fusion = rc_bev_fusion

        self.distillation = distillation

        self._init_fusion_module()

        self._init_teacher_model(pts_bbox_head_tea, **kwargs)
        self._init_student_model()
        # self._init_img_transformation(
        #     img_view_transformer, img_bev_encoder_backbone, img_bev_encoder_neck
        # )
        self._init_radar_net(
            radar_voxel_layer,
            radar_pillar_encoder,
            radar_middle_encoder,
            radar_backbone,
            radar_neck,
        )
        # self._init_img_backbone(close=student_pretrained is not None)
        self._init_img_backbone(close=self.freeze_img_backbone)
        self._init_distill_module()

    def _init_fusion_module(self):
        self.bi_dire_fusion = (
            builder.build_neck(self.bi_dire_fusion) if self.bi_dire_fusion else None
        )
        self.middle_radar_aug = (
            builder.build_neck(self.middle_radar_aug) if self.middle_radar_aug else None
        )
        self.rc_bev_fusion = (
            builder.build_neck(self.rc_bev_fusion) if self.rc_bev_fusion else None
        )

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

        self.sampling_pos_distill = (
            build_loss(self.distillation.sampling_pos_distill)
            if "sampling_pos_distill" in self.distillation.keys()
            else None
        )
        self.sampling_feats_distill = (
            build_loss(self.distillation.sampling_feats_distill)
            if "sampling_feats_distill" in self.distillation.keys()
            else None
        )

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
        voxels, num_points, coors = self.radar_voxelization(radar_points)

        radar_pillar_features = self.radar_pillar_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        x = self.radar_middle_encoder(radar_pillar_features, coors, batch_size)

        if self.with_radar_backbone:
            x = self.radar_backbone(x)

            if self.radar_ms_feats_distill:
                radar_ms_feas = [data for data in x]

        if self.with_radar_neck:
            x = self.radar_neck(x)

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

    # def feats_post_process(self, img_feats, radar_feats, pts_feats=None):
    #     if self.bi_dire_fusion:
    #         img_feats, radar_feats = self.bi_dire_fusion(img_feats[0], radar_feats[0])

    #     if self.middle_radar_aug:
    #         radar_feats = self.middle_radar_aug(img_feats, radar_feats)

    #     det_input_feats = (
    #         self.rc_bev_fusion(img_feats, radar_feats)
    #         if self.rc_bev_fusion is not None
    #         else dict()
    #     )

    #     return [img_feats], radar_feats, det_input_feats

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

        if self.rc_bev_fusion:
            output = self.rc_bev_fusion(img_bev, radar_bev)
            if not isinstance(output, dict):
                det_input_dict.update({"det_feats": [output]})
            else:
                det_input_dict.update(output)

        return det_input_dict

    def extract_feat(self, points, radar, img, img_metas):
        """Extract features from images and points."""

        img_feats, depth = self.extract_img_feat(img, img_metas)
        # img_feats, depth=super(BEVStereo4D)
        radar_feats, radar_ms_feats = self.extract_radar_feat(radar)

        pts_feats, pts_ms_feats = self.extract_pts_feat(points, img_feats, img_metas)

        det_input_feats = self.fusion_img_radar_bev(
            img_feats, radar_feats, pts_feats=pts_feats
        )

        base_path = "/mnt/data/exps/D3PD/d3pd/out/v3-feats-out"
        feats_to_img(det_input_feats["det_feats"], base_path=base_path, suffix="fusion")
        feats_to_img(img_feats, base_path=base_path, suffix="img_feats")

        feats_to_img(
            det_input_feats.get("student_sampling_feats"),
            base_path=base_path,
            suffix="img_feats_sampling",
        )
        feats_to_img(radar_feats, base_path=base_path, suffix="radar_feats")
        feats_to_img(pts_feats, base_path=base_path, suffix="pts_feats")

        raise RuntimeError

        # feats_to_img(radar_feats, base_path=base_path, suffix="radar_feats_aug")

        if radar_ms_feats and pts_ms_feats:
            det_input_feats.update(dict(radar_ms_feats=radar_ms_feats))
            det_input_feats.update(dict(pts_ms_feats=pts_ms_feats))

        return img_feats, pts_feats, radar_feats, depth, det_input_feats

    def forward_pts_train(
        self,
        pts_feats,
        gt_bboxes_3d,
        gt_labels_3d,
        img_metas,
        gt_bboxes_ignore=None,
    ) -> tuple:
        """Forward function for point cloud branch.

        Args:
            pts_feats (_type_): Features of point cloud branch
            gt_bboxes_3d (_type_): Ground truth
                boxes for each sample.
            gt_labels_3d (_type_): Ground truth labels for
                boxes of each sampole
            img_metas (_type_): Meta information of samples.
            gt_bboxes_ignore (_type_, optional): Ground truth
                boxes to be ignored. Defaults to None.. Defaults to None.

        Returns:
            tuple: _description_
        """

        outs = self.pts_bbox_head(pts_feats)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)

        return losses, outs

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

        if self.sampling_pos_distill is not None and teacher_sampling_pos is not None:
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
        img_metas=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        radar=None,
        gt_labels=None,
        gt_bboxes=None,
        img_inputs=None,
        proposals=None,
        gt_bboxes_ignore=None,
        **kwargs,
    ):

        img_feats, pts_feats, radar_feats, depth, feats_wrap = self.extract_feat(
            points, radar=radar, img=img_inputs, img_metas=img_metas
        )

        teacher_outs = (
            self.pts_bbox_head_tea(pts_feats) if self.distillation is not None else None
        )

        teacher_sampling_pos = feats_wrap.get("teacher_sampling_pos")
        teacher_sampling_feats = feats_wrap.get("teacher_sampling_feats")
        student_sampling_pos = feats_wrap.get("student_sampling_pos")
        student_sampling_feats = feats_wrap.get("student_sampling_feats")
        radar_ms_feats = feats_wrap.get("radar_ms_feats")
        pts_ms_feats = feats_wrap.get("pts_ms_feats")
        det_feats = feats_wrap.get("det_feats")

        # base_path = "/mnt/data/exps/D3PD/d3pd/out"
        # feats_to_img(det_feats[0], base_path, "fusion")

        # raise RuntimeError

        losses = dict()

        gt_depth = kwargs["gt_depth"]
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        losses.update(dict(loss_depth=loss_depth))

        img_feats_kd = img_feats[0] if isinstance(img_feats, list) else img_feats
        if pts_feats:
            pts_feats_kd = pts_feats[0]

        # assert det_input_feats["det_feats"].size(1) == 64
        losses_pts, student_outs = self.forward_pts_train(
            det_feats,
            gt_bboxes_3d,
            gt_labels_3d,
            img_metas,
            gt_bboxes_ignore,
        )

        losses.update(losses_pts)

        ret_losses_dict = self.forward_distill_loss(
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
            radar_feats=radar_feats[0],
            rc_fusion_feats=det_feats[0],
            gt_bboxes_3d=gt_bboxes_3d,
        )

        losses.update(ret_losses_dict)

        # adaptive_weight_list = [
        #     "loss_depth",
        #     "loss_radar_ms",
        #     "loss_dc_reg_distill",
        #     "loss_dc_cls_distill",
        #     "loss_smfd_distill",
        # ]
        # for key in adaptive_weight_list:
        #     adaptive_weight = (
        #         losses[key] / losses["loss_inter_channel_img_bev"]
        #     ).detach()
        #     losses.update({key: losses[key] / adaptive_weight})

        return losses

    def simple_test(self, points, radar, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        img_feats, _, _, _, det_input_feats = self.extract_feat(
            points, radar, img=img, img_metas=img_metas
        )
        det_feats = img_feats  # det_input_feats["det_feats"]
        bbox_list = [dict() for _ in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(det_feats, img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict["pts_bbox"] = pts_bbox
        return bbox_list

    def forward_test(
        self, points=None, radar=None, img_metas=None, img_inputs=None, **kwargs
    ):
        for var, name in [(img_inputs, "img_inputs"), (img_metas, "img_metas")]:
            if not isinstance(var, list):
                raise TypeError("{} must be a list, but got {}".format(name, type(var)))

        if not isinstance(img_inputs[0][0], list):
            img_inputs = [img_inputs] if img_inputs is None else img_inputs
            points = [points] if points is None else points
            radar = [radar] if radar is None else radar
            return self.simple_test(
                points[0], radar[0], img_metas[0], img_inputs[0], **kwargs
            )
        else:
            return self.aug_test(None, img_metas[0], img_inputs[0], **kwargs)

