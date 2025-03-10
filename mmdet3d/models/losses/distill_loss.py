import torch.nn as nn
import torch.nn.functional as F
import torch
from ..builder import LOSSES, build_loss
from ..utils.gaussian_utils import calculate_box_mask_gaussian
from torch import distributed as dist
from mmdet.core import multi_apply, build_bbox_coder
from mmdet3d.core import draw_heatmap_gaussian, gaussian_radius
from mmdet3d.core.bbox.coders.centerpoint_bbox_coders import (
    CenterPointBBoxCoder as bboxcoder,
)
from mmdet.models.losses import QualityFocalLoss
from mmcv.runner import force_fp32
import torchsort
import torchvision
import numpy as np


def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1

    return dist.get_world_size()


def reduce_sum(tensor):
    world_size = get_world_size()
    if world_size < 2:
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def reduce_mean(tensor):
    return reduce_sum(tensor) / float(get_world_size())


def _sigmoid(x):
    # y = torch.clamp(x.sigmoid(), min=1e-3, max=1 - 1e-3)
    y = x.sigmoid()
    return y


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def response_list_wrap(response_lists: list):
    """response_list_wrap

    Args:
        response_lists (list): Pack list content

    Returns:
        list: Return the new list of packaged integration
    """
    assert len(response_lists) == 1

    tmp_resp = []
    for response in response_lists:
        tmp_resp.append(response[0])

    return tmp_resp


@LOSSES.register_module()
class QualityFocalLoss_(nn.Module):
    """
    input[B,M,C] not sigmoid
    target[B,M,C], sigmoid
    """

    def __init__(self, beta=2.0):

        super(QualityFocalLoss_, self).__init__()
        self.beta = beta

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        pos_normalizer=torch.tensor(1.0),
    ):

        pred_sigmoid = torch.sigmoid(input)
        scale_factor = pred_sigmoid - target

        # pred_sigmoid = torch.sigmoid(input)
        # scale_factor = input - target
        loss = F.binary_cross_entropy_with_logits(input, target, reduction="none") * (
            scale_factor.abs().pow(self.beta)
        )
        loss /= torch.clamp(pos_normalizer, min=1.0)
        return loss


@LOSSES.register_module()
class SimpleL1(nn.Module):
    def __init__(self, criterion="L1", student_ch=256, teacher_ch=512):
        super().__init__()
        self.criterion = criterion
        if criterion == "L1":
            self.criterion_loss = nn.L1Loss(reduce=False)
        elif criterion == "SmoothL1":
            self.criterion_loss = nn.SmoothL1Loss(reduce=False)
        elif criterion == "MSE":
            self.criterion_loss = nn.MSELoss(reduction="none")

        if student_ch != teacher_ch:
            self.align = nn.Conv2d(student_ch, teacher_ch, kernel_size=1)

    def forward(self, feats1, feats2, *args, **kwargs):
        if self.criterion == "MSE":
            feats1 = self.align(feats1) if getattr(self, "align", None) else feats1
            losses = self.criterion_loss(feats1, feats2).mean()
            return losses
        else:
            return self.criterion_loss(feats1, feats2)


@LOSSES.register_module()
class Relevance_Distillation(nn.Module):
    def __init__(self, bs=2, bn_dim=512, lambd=0.0051):
        super().__init__()
        self.bs = bs
        self.bn_dim = bn_dim
        self.lambd = lambd

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(self.bn_dim, affine=False)

        self.align = nn.Conv2d(256, 512, kernel_size=1)

    def forward(self, student_bev, teacher_bev, *args, **kwargs):
        student_bev = self.align(student_bev)

        student_bev = student_bev.flatten(2)
        teacher_bev = teacher_bev.flatten(2)

        # empirical cross-correlation matrix
        c = self.bn(student_bev).flatten(1).T @ self.bn(teacher_bev).flatten(1)

        # sum the cross-correlation matrix between all gpus
        c.div_(self.bs)
        if self.bs == 1:
            pass
        else:
            torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag
        return loss


@LOSSES.register_module()
class FeatureLoss(nn.Module):
    """PyTorch version of `Masked Generative Distillation`

    Args:
        student_channels(int): Number of channels in the student's feature map.
        teacher_channels(int): Number of channels in the teacher's feature map.
        name (str): the loss name of the layer
        alpha_mgd (float, optional): Weight of dis_loss. Defaults to 0.00002
        lambda_mgd (float, optional): masked ratio. Defaults to 0.65
    """

    def __init__(
        self,
        student_channels,
        teacher_channels,
        name,
        alpha_mgd=0.00002,
        lambda_mgd=0.65,
    ):
        super(FeatureLoss, self).__init__()
        self.alpha_mgd = alpha_mgd
        self.lambda_mgd = lambda_mgd
        self.name = name

        if student_channels != teacher_channels:
            self.align = nn.Conv2d(
                student_channels, teacher_channels, kernel_size=1, stride=1, padding=0
            )
        else:
            self.align = None

        self.generation = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1),
        )

    def forward(self, preds_S, preds_T, **kwargs):
        """Forward function.
        Args:
            preds_S(Tensor): Bs*C*H*W, student's feature map
            preds_T(Tensor): Bs*C*H*W, teacher's feature map
        """
        assert preds_S.shape[-2:] == preds_T.shape[-2:]

        if self.align is not None:
            preds_S = self.align(preds_S)

        loss = self.get_dis_loss(preds_S, preds_T) * self.alpha_mgd

        return loss

    def get_dis_loss(self, preds_S, preds_T):
        loss_mse = nn.MSELoss(reduction="sum")
        N, C, H, W = preds_T.shape

        device = preds_S.device
        mat = torch.rand((N, 1, H, W)).to(device)
        mat = torch.where(mat > 1 - self.lambda_mgd, 0, 1).to(device)

        masked_fea = torch.mul(preds_S, mat)
        new_fea = self.generation(masked_fea)

        dis_loss = loss_mse(new_fea, preds_T) / N

        return dis_loss


@LOSSES.register_module()
class FeatureLoss_InnerClip(nn.Module):
    def __init__(
        self,
        x_sample_num=24,
        y_sample_num=24,
        inter_keypoint_weight=1,
        inter_channel_weight=10,
        enlarge_width=1.6,
        embed_channels=[256, 512],
        inner_feats_distill=None,
    ):
        super().__init__()
        self.x_sample_num = x_sample_num
        self.y_sample_num = y_sample_num
        self.inter_keypoint_weight = inter_keypoint_weight
        self.inter_channel_weight = inter_channel_weight
        self.enlarge_width = enlarge_width

        self.img_view_transformer = None

        self.embed_channels = embed_channels

        self.imgbev_embed = nn.Sequential(
            nn.Conv2d(
                embed_channels[0],
                embed_channels[1],
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(embed_channels[1]),
        )

        self.inner_feats_loss = (
            build_loss(inner_feats_distill) if inner_feats_distill is not None else None
        )

    def get_gt_sample_grid(self, corner_points2d):
        dH_x, dH_y = corner_points2d[0] - corner_points2d[1]
        dW_x, dW_y = corner_points2d[0] - corner_points2d[2]
        raw_grid_x = (
            torch.linspace(
                corner_points2d[0][0], corner_points2d[1][0], self.x_sample_num
            )
            .view(1, -1)
            .repeat(self.y_sample_num, 1)
        )
        raw_grid_y = (
            torch.linspace(
                corner_points2d[0][1], corner_points2d[2][1], self.y_sample_num
            )
            .view(-1, 1)
            .repeat(1, self.x_sample_num)
        )
        raw_grid = torch.cat((raw_grid_x.unsqueeze(2), raw_grid_y.unsqueeze(2)), dim=2)
        raw_grid_x_offset = (
            torch.linspace(0, -dW_x, self.x_sample_num)
            .view(-1, 1)
            .repeat(1, self.y_sample_num)
        )
        raw_grid_y_offset = (
            torch.linspace(0, -dH_y, self.y_sample_num)
            .view(1, -1)
            .repeat(self.x_sample_num, 1)
        )
        raw_grid_offset = torch.cat(
            (raw_grid_x_offset.unsqueeze(2), raw_grid_y_offset.unsqueeze(2)), dim=2
        )
        grid = raw_grid + raw_grid_offset  # X_sample,Y_sample,2
        grid[:, :, 0] = torch.clip(
            (
                (
                    grid[:, :, 0]
                    - (
                        self.img_view_transformer["bx"][0].to(grid.device)
                        - self.img_view_transformer["dx"][0].to(grid.device) / 2.0
                    )
                )
                / self.img_view_transformer["dx"][0].to(grid.device)
                / (self.img_view_transformer["nx"][0].to(grid.device) - 1)
            )
            * 2.0
            - 1.0,
            min=-1.0,
            max=1.0,
        )
        grid[:, :, 1] = torch.clip(
            (
                (
                    grid[:, :, 1]
                    - (
                        self.img_view_transformer["bx"][1].to(grid.device)
                        - self.img_view_transformer["dx"][1].to(grid.device) / 2.0
                    )
                )
                / self.img_view_transformer["dx"][1].to(grid.device)
                / (self.img_view_transformer["nx"][1].to(grid.device) - 1)
            )
            * 2.0
            - 1.0,
            min=-1.0,
            max=1.0,
        )

        return grid.unsqueeze(0)

    def get_inner_feat(self, gt_bboxes_3d, img_feats, pts_feats):
        """Use grid to sample features of key points"""
        device = img_feats.device
        dtype = img_feats[0].dtype

        img_feats_sampled_list = []
        pts_feats_sampled_list = []

        for sample_ind in torch.arange(len(gt_bboxes_3d)):
            img_feat = img_feats[sample_ind].unsqueeze(0)  # 1,C,H,W
            pts_feat = pts_feats[sample_ind].unsqueeze(0)  # 1,C,H,W

            bbox_num, corner_num, point_num = gt_bboxes_3d[sample_ind].corners.shape

            for bbox_ind in torch.arange(bbox_num):
                if self.enlarge_width > 0:
                    gt_sample_grid = self.get_gt_sample_grid(
                        gt_bboxes_3d[sample_ind]
                        .enlarged_box(self.enlarge_width)
                        .corners[bbox_ind][[0, 2, 4, 6], :-1]
                    ).to(device)
                else:
                    gt_sample_grid = self.get_gt_sample_grid(
                        gt_bboxes_3d[sample_ind].corners[bbox_ind][[0, 2, 4, 6], :-1]
                    ).to(
                        device
                    )  # 1,sample_y,sample_x,2

                img_feats_sampled_list.append(
                    F.grid_sample(
                        img_feat,
                        grid=gt_sample_grid,
                        align_corners=False,
                        mode="bilinear",
                    )
                )  # 'bilinear')) #all_bbox_num,C,y_sample,x_sample
                pts_feats_sampled_list.append(
                    F.grid_sample(
                        pts_feat,
                        grid=gt_sample_grid,
                        align_corners=False,
                        mode="bilinear",
                    )
                )  # 'bilinear')) #all_bbox_num,C,y_sample,x_sample

        return torch.cat(img_feats_sampled_list, dim=0), torch.cat(
            pts_feats_sampled_list, dim=0
        )

    @force_fp32(apply_to=("img_feats_kd", "pts_feats_kd"))
    def get_inter_channel_loss(self, img_feats_kd, pts_feats_kd):
        """Calculate the inter-channel similarities, guide the student keypoint features to mimic the channel-wise relationships of the teacher`s"""

        C_img = img_feats_kd.shape[1]
        C_pts = pts_feats_kd.shape[1]
        N = self.x_sample_num * self.y_sample_num

        img_feats_kd = img_feats_kd.view(-1, C_img, N).matmul(
            img_feats_kd.view(-1, C_img, N).permute(0, 2, 1)
        )  # -1,N,N
        pts_feats_kd = pts_feats_kd.view(-1, C_pts, N).matmul(
            pts_feats_kd.view(-1, C_pts, N).permute(0, 2, 1)
        )

        img_feats_kd = F.normalize(img_feats_kd, dim=2)
        pts_feats_kd = F.normalize(pts_feats_kd, dim=2)

        loss_inter_channel = F.mse_loss(img_feats_kd, pts_feats_kd, reduction="none")
        loss_inter_channel = loss_inter_channel.sum(-1)
        loss_inter_channel = loss_inter_channel.mean()
        loss_inter_channel = self.inter_channel_weight * loss_inter_channel
        return loss_inter_channel

    def forward(self, student_feats, teacher_feats, gt_bboxes_list, **kwargs):

        self.img_view_transformer = kwargs.get("ivt_cfg")

        if student_feats.size(1) != teacher_feats.size(1):
            student_feats = self.imgbev_embed(student_feats)

        if self.inter_keypoint_weight > 0 or self.inter_channel_weight > 0:
            img_feats_kd, pts_feats_kd = self.get_inner_feat(
                gt_bboxes_list, student_feats, teacher_feats
            )

        if self.inner_feats_loss:
            return self.inner_feats_loss(img_feats_kd, pts_feats_kd)

        # if self.inter_keypoint_weight > 0:
        #     loss_inter_keypoint = self.get_inter_keypoint_loss(
        #         img_feats_kd, pts_feats_kd
        #     )
        #     losses.update({"loss_inter_keypoint_img_bev": loss_inter_keypoint})

        if self.inter_channel_weight > 0:
            loss_inter_channel = self.get_inter_channel_loss(img_feats_kd, pts_feats_kd)

        return loss_inter_channel


@LOSSES.register_module()
class FeatureLoss_Affinity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, student_feats, teacher_feats, *args, **kwargs):
        student_feats = [student_feats]
        teacher_feats = [teacher_feats]

        feature_ditill_loss = 0.0

        resize_shape = student_feats[-1].shape[-2:]
        if isinstance(student_feats, list):
            for i in range(len(student_feats)):
                feature_target = teacher_feats[i].detach()
                feature_pred = student_feats[i]

                B, C, H, W = student_feats[-1].shape

                if student_feats[-1].size(-1) != teacher_feats[-1].size(-1):
                    feature_pred_down = F.interpolate(
                        feature_pred, size=resize_shape, mode="bilinear"
                    )
                    feature_target_down = F.interpolate(
                        feature_target, size=resize_shape, mode="bilinear"
                    )
                else:
                    feature_pred_down = feature_pred
                    feature_target_down = feature_target

                feature_target_down = feature_target_down.reshape(B, C, -1)
                depth_affinity = torch.bmm(
                    feature_target_down.permute(0, 2, 1), feature_target_down
                )

                feature_pred_down = feature_pred_down.reshape(B, C, -1)
                rgb_affinity = torch.bmm(
                    feature_pred_down.permute(0, 2, 1), feature_pred_down
                )

                feature_ditill_loss = (
                    feature_ditill_loss
                    + F.l1_loss(rgb_affinity, depth_affinity, reduction="mean") / B
                )

        else:
            raise NotImplementedError

        return feature_ditill_loss


@LOSSES.register_module()
class FeatureLoss_Coefficient(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def corrcoef(self, target, pred):
        pred_n = pred - pred.mean()
        target_n = target - target.mean()
        pred_n = pred_n / pred_n.norm()
        target_n = target_n / target_n.norm()
        return (pred_n * target_n).sum()

    def forward(
        self,
        pred,
        target,
        regularization="l2",
        regularization_strength=1.0,
    ):
        pred = [pred.clone()]
        target = [target.clone()]
        spearman_loss = 0.0

        resize_shape = pred[-1].shape[-2:]  # save training time

        if isinstance(pred, list):
            for i in range(len(pred)):
                feature_target = target[i]
                feature_pred = pred[i]
                B, C, H, W = feature_pred.shape

                feature_pred_down = F.interpolate(
                    feature_pred, size=resize_shape, mode="bilinear"
                )
                feature_target_down = F.interpolate(
                    feature_target, size=resize_shape, mode="bilinear"
                )

                # if feature_pred.size(-1) != feature_target.size(-1):

                #     feature_pred_down = F.interpolate(
                #         feature_pred, size=resize_shape, mode="bilinear"
                #     )
                #     feature_target_down = F.interpolate(
                #         feature_target, size=resize_shape, mode="bilinear"
                #     )
                # else:
                #     feature_pred_down = feature_pred
                #     feature_target_down = feature_target

                feature_pred_down = feature_pred_down.reshape(B, -1)
                feature_target_down = feature_target_down.reshape(B, -1)

                feature_pred_down = torchsort.soft_rank(
                    feature_pred_down,
                    regularization=regularization,
                    regularization_strength=regularization_strength,
                )
                spearman_loss += 1 - self.corrcoef(
                    feature_target_down, feature_pred_down / feature_pred_down.shape[-1]
                )

        return spearman_loss


@LOSSES.register_module()
class Radar_MSDistilll(nn.Module):
    def __init__(self, num_layers=2, each_layer_loss_cfg=[]):
        super().__init__()
        self.num_layers = num_layers
        assert num_layers == len(each_layer_loss_cfg)
        for idx, cfg_dict in enumerate(each_layer_loss_cfg):
            setattr(self, f"layer_loss_{idx}", build_loss(cfg_dict))

    def unify_feat_size(self, student_feat, teacher_feat):
        bs, s_c, s_w, s_h = student_feat.shape
        bs, t_c, t_w, t_h = teacher_feat.shape
        if s_w == t_w:
            return student_feat, teacher_feat
        else:
            teacher_feat = F.interpolate(teacher_feat, (s_w, s_h), mode="bilinear")
            return student_feat, teacher_feat

    def forward(self, radar_ms_feats, pts_ms_feats, gt_bboxes_3d=None):
        assert isinstance(radar_ms_feats, list)
        losses = 0.0

        for idx in range(self.num_layers):
            tmp_radar_feats, tmp_pts_feats = self.unify_feat_size(
                radar_ms_feats[idx], pts_ms_feats[idx]
            )

            if gt_bboxes_3d is not None:
                losses += getattr(self, f"layer_loss_{idx}")(
                    tmp_radar_feats, tmp_pts_feats, gt_bboxes_3d
                )[0]
            else:
                losses += getattr(self, f"layer_loss_{idx}")(
                    tmp_radar_feats, tmp_pts_feats, gt_bboxes_3d
                )

        return losses / self.num_layers


@LOSSES.register_module()
class InfoMax(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x1, x2):
        import pdb

        pdb.set_trace()
        x1 = x1 / (torch.norm(x1, p=2, dim=1, keepdim=True) + 1e-10)
        x2 = x2 / (torch.norm(x2, p=2, dim=1, keepdim=True) + 1e-10)
        bs = x1.size(0)
        s = torch.matmul(x1, x2.permute(1, 0))
        mask_joint = torch.eye(bs).cuda()
        mask_marginal = 1 - mask_joint

        Ej = (s * mask_joint).mean()
        Em = torch.exp(s * mask_marginal).mean()
        # decoupled comtrastive learning?!!!!
        # infomax_loss = - (Ej - torch.log(Em)) * self.alpha
        infomax_loss = -(Ej - torch.log(Em))  # / Em
        return infomax_loss


@LOSSES.register_module()
class HeatMapAug(nn.Module):
    def __init__(self):
        super().__init__()

    @force_fp32(apply_to=("stu_pred", "tea_pred"))
    def forward(self, stu_pred, tea_pred, fg_map):

        num_task = len(stu_pred)
        kl_loss = 0
        for task_id in range(num_task):
            student_pred = stu_pred[task_id][0]["heatmap"].sigmoid()
            teacher_pred = tea_pred[task_id][0]["heatmap"].sigmoid()
            fg_map = fg_map.unsqueeze(1)
            task_kl_loss = F.binary_cross_entropy(student_pred, teacher_pred.detach())
            task_kl_loss = torch.sum(task_kl_loss * fg_map) / torch.sum(fg_map)
            kl_loss += task_kl_loss

        return kl_loss


@LOSSES.register_module()
class Dc_ResultDistill(nn.Module):
    def __init__(
        self,
        pc_range=[],
        voxel_size=[],
        out_size_scale=8,
        ret_sum=False,
        loss_weight_reg=10,
        loss_weight_cls=10,
        max_cls=True,
    ):
        super().__init__()

        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.out_size_scale = out_size_scale
        self.ret_sum = ret_sum
        self.loss_weight_reg = loss_weight_reg
        self.loss_weight_cls = loss_weight_cls
        self.max_cls = max_cls

    def forward(self, resp_lidar, resp_fuse, gt_boxes):
        """Dc_ResultDistill forward.

        Args:
            resp_lidar (_type_):
            resp_fuse (_type_):
            gt_boxes (_type_):

        Returns:
            _type_: _description_
        """

        tmp_resp_lidar = []
        tmp_resp_fuse = []
        for res_lidar, res_fuse in zip(resp_lidar, resp_fuse):
            tmp_resp_lidar.append(res_lidar[0])
            tmp_resp_fuse.append(res_fuse[0])

        tmp_gt_boxes = []
        for bs_idx in range(len(gt_boxes)):

            gt_bboxes_3d = torch.cat(
                (gt_boxes[bs_idx].gravity_center, gt_boxes[bs_idx].tensor[:, 3:]), dim=1
            )

            tmp_gt_boxes.append(gt_bboxes_3d)

        cls_lidar = []
        reg_lidar = []
        cls_fuse = []
        reg_fuse = []

        # criterion = nn.L1Loss(reduce=False)
        # criterion_cls = QualityFocalLoss_()

        criterion = nn.SmoothL1Loss(reduce=False)
        criterion_cls = nn.L1Loss(reduce=False)

        for task_id, task_out in enumerate(tmp_resp_lidar):
            cls_lidar.append(task_out["heatmap"])
            cls_fuse.append(_sigmoid(tmp_resp_fuse[task_id]["heatmap"] / 2))
            reg_lidar.append(
                torch.cat(
                    [
                        task_out["reg"],
                        task_out["height"],
                        task_out["dim"],
                        task_out["rot"],
                        task_out["vel"],
                        # task_out["iou"],
                    ],
                    dim=1,
                )
            )
            reg_fuse.append(
                torch.cat(
                    [
                        tmp_resp_fuse[task_id]["reg"],
                        tmp_resp_fuse[task_id]["height"],
                        tmp_resp_fuse[task_id]["dim"],
                        tmp_resp_fuse[task_id]["rot"],
                        tmp_resp_fuse[task_id]["vel"],
                        # resp_fuse[task_id]["iou"],
                    ],
                    dim=1,
                )
            )
        cls_lidar = torch.cat(cls_lidar, dim=1)
        reg_lidar = torch.cat(reg_lidar, dim=1)
        cls_fuse = torch.cat(cls_fuse, dim=1)
        reg_fuse = torch.cat(reg_fuse, dim=1)

        if self.max_cls:
            cls_lidar_max, _ = torch.max(cls_lidar, dim=1)
            cls_fuse_max, _ = torch.max(cls_fuse, dim=1)
        else:
            _, _, ht_h, ht_w = cls_fuse.shape
            cls_lidar_max = cls_lidar.flatten(2).permute(0, 2, 1)
            cls_fuse_max = cls_fuse.flatten(2).permute(0, 2, 1)

        gaussian_mask = calculate_box_mask_gaussian(
            reg_lidar.shape,
            tmp_gt_boxes,
            self.pc_range,
            self.voxel_size,
            self.out_size_scale,
        )

        # # diff_reg = criterion(reg_lidar, reg_fuse)
        # # diff_cls = criterion(cls_lidar_max, cls_fuse_max)
        # diff_reg = criterion_reg(reg_lidar, reg_fuse)

        # cls_lidar_max = torch.sigmoid(cls_lidar_max)
        # diff_cls = criterion_cls(
        #     cls_fuse_max, cls_lidar_max
        # )  # Compared with directly using the L1 loss constraint, replace it with bce focal loss and change the position?

        # weight = gaussian_mask.sum()
        # weight = reduce_mean(weight)

        # # diff_cls = criterion_cls(cls_lidar_max, cls_fuse_max)
        # diff_reg = torch.mean(diff_reg, dim=1)
        # diff_reg = diff_reg * gaussian_mask
        # loss_reg_distill = torch.sum(diff_reg) / (weight + 1e-4)

        # if not self.max_cls:
        #     diff_cls = diff_cls
        #     loss_cls_distill = diff_cls.sum() * 2
        # else:
        #     diff_cls = diff_cls * gaussian_mask
        #     loss_cls_distill = torch.sum(diff_cls) / (weight + 1e-4)

        diff_reg = criterion(reg_lidar, reg_fuse)

        diff_cls = criterion_cls(cls_lidar_max, cls_fuse_max)
        diff_reg = torch.mean(diff_reg, dim=1)
        diff_reg = diff_reg * gaussian_mask
        diff_cls = diff_cls * gaussian_mask
        weight = gaussian_mask.sum()
        weight = reduce_mean(weight)
        loss_reg_distill = torch.sum(diff_reg) / (weight + 1e-4)
        loss_cls_distill = torch.sum(diff_cls) / (weight + 1e-4)

        if self.ret_sum:

            loss_det_distill = self.loss_weight * (loss_reg_distill + loss_cls_distill)

            return loss_det_distill
        else:

            return (
                self.loss_weight_reg * loss_reg_distill,
                self.loss_weight_cls * loss_cls_distill,
            )


@LOSSES.register_module()
class SelfLearningMFD(nn.Module):
    def __init__(
        self,
        bev_shape=[128, 128],
        pc_range=[-51.2, -51.2, -5, 51.2, 51.2, 3],
        voxel_size=[0.1, 0.1, 0.1],
        score_threshold=0.7,
        add_stu_decode_bboxes=False,
        loss_weight=1e-2,
        bbox_coder=None,
        shape_resize_times=None,
        student_channels=256,
        teacher_channels=512,
    ):
        super().__init__()
        self.bev_shape = bev_shape
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        if bbox_coder:
            self.shape_resize_times = bbox_coder.out_size_factor
        elif shape_resize_times:
            self.shape_resize_times = shape_resize_times
        else:
            self.shape_resize_times = 8
        self.score_threshold = score_threshold
        self.add_stu_decode_bboxes = add_stu_decode_bboxes
        self.loss_weight = loss_weight

        self.align = nn.Conv2d(
            student_channels, teacher_channels, kernel_size=1, padding=0
        )
        # nn.BatchNorm2d(teacher_channels),

        if bbox_coder is not None:
            self.bbox_coder = build_bbox_coder(bbox_coder)

    def pred2bboxes(self, preds_dict):
        """SelfLearningMFD-pred2bboxes forward

        Args:
            pred_bboxes (list):
            [task1[[task_head_reg,task_head_heatmap,...,]],
            task2...,
            taskn]
        """

        for task_id, pred_data in enumerate(preds_dict):
            bath_tmp = []
            batch_size = pred_data[0]["heatmap"].shape[0]
            batch_heatmap = pred_data[0]["heatmap"].sigmoid()
            batch_reg = pred_data[0]["reg"]
            batch_hei = pred_data[0]["height"]

            # denormalization
            batch_dim = torch.exp(pred_data[0]["dim"])

            batch_rot_sin = pred_data[0]["rot"][:, 0].unsqueeze(1)
            batch_rot_cos = pred_data[0]["rot"][:, 1].unsqueeze(1)

            if "vel" in pred_data[0].keys():
                batch_vel = pred_data[0]["vel"]

            bath_tmp.append(batch_heatmap)
            bath_tmp.append(batch_rot_sin)
            bath_tmp.append(batch_rot_cos)
            bath_tmp.append(batch_hei)
            bath_tmp.append(batch_dim)
            bath_tmp.append(batch_vel)
            bath_tmp.append(batch_reg)

            bboxes_decode = self.bbox_coder.decode(*bath_tmp)

        return bboxes_decode

    def aug_boxes(self, gt_boxes, rot_mat):
        rot_lim = (-22.5, 22.5)
        scale_lim = (0.95, 1.05)
        rotate_angle = np.random.uniform(*rot_lim)
        scale_ratio = np.random.uniform(*scale_lim)
        gt_boxes[:, :3] = (rot_mat @ gt_boxes[:, :3].unsqueeze(-1)).squeeze(-1)
        gt_boxes[:, 3:6] *= scale_ratio
        gt_boxes[:, 6] += rotate_angle
        gt_boxes[:, 6] = 2 * torch.asin(torch.tensor(1.0)) - gt_boxes[:, 6]
        gt_boxes[:, 6] = -gt_boxes[:, 6]
        gt_boxes[:, 7:] = (rot_mat[:2, :2] @ gt_boxes[:, 7:].unsqueeze(-1)).squeeze(-1)
        gt_boxes[:, :3] = gt_boxes[:, :3]

        return gt_boxes

    # @force_fp32(
    #     apply_to=(
    #         "student_bev_feat",
    #         "teacher_bev_feat",
    #     )
    # )
    def forward(
        self,
        student_bev_feat,
        teacher_bev_feat,
        gt_bboxes_list=None,
        masks_bboxes=None,
        bda_mat=None,
        **kwargs,
    ):
        """SelfLearningMFD forward.

        Args:
            student_bev_feats (torch.tensor): Calculate student feats
            teacher_bev_feats (torch.tensor): Calculate teacher feats
            masks_bboxes (list): Self-learning mask detection
            gt_bboxes_list (list): [LiDARInstance3DBoxes(gravity_center+tensor(num_objs,9))]

        Returns:
            dict: _description_
        """

        bs = student_bev_feat.size(0)

        if student_bev_feat.size(1) != teacher_bev_feat.size(1):
            student_bev_feat = self.align(student_bev_feat)

        if masks_bboxes is not None and self.add_stu_decode_bboxes:
            student_pred_bboxes_list = self.pred2bboxes(
                masks_bboxes
            )  # list of task : each shape of [(num_bboxes,9)]

            # if bda_mat is not None:
            #     student_pred_bboxes_list = self.aug_boxes(student_pred_bboxes_list)

        bev_feat_shape = torch.tensor(self.bev_shape)
        voxel_size = torch.tensor(self.voxel_size)
        feature_map_size = bev_feat_shape[:2]

        if gt_bboxes_list is not None:
            device = student_bev_feat.device

            gt_bboxes_list = [
                torch.cat(
                    (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1
                ).to(device)
                for gt_bboxes in gt_bboxes_list
            ]

            if self.add_stu_decode_bboxes:
                for idx, data in enumerate(student_pred_bboxes_list):

                    bboxes_dim = data["bboxes"].size(1)

                    scroes_mask = data["scores"] > self.score_threshold

                    new_select_by_mask = data["bboxes"][scroes_mask, :]

                    gt_bboxes_list[idx] = torch.cat(
                        [gt_bboxes_list[idx], new_select_by_mask], dim=0
                    )

            fg_map = student_bev_feat.new_zeros(
                (len(gt_bboxes_list), feature_map_size[1], feature_map_size[0])
            )

            for idx in range(len(gt_bboxes_list)):
                num_objs = gt_bboxes_list[idx].shape[0]

                for k in range(num_objs):
                    width = gt_bboxes_list[idx][k][3]
                    length = gt_bboxes_list[idx][k][4]
                    width = width / voxel_size[0] / self.shape_resize_times
                    length = length / voxel_size[1] / self.shape_resize_times

                    if width > 0 and length > 0:
                        radius = gaussian_radius((length, width), min_overlap=0.1)
                        radius = max(1, int(radius))

                        # be really careful for the coordinate system of
                        # your box annotation.
                        x, y, z = (
                            gt_bboxes_list[idx][k][0],
                            gt_bboxes_list[idx][k][1],
                            gt_bboxes_list[idx][k][2],
                        )

                        coor_x = (
                            (x - self.pc_range[0])
                            / voxel_size[0]
                            / self.shape_resize_times
                        )
                        coor_y = (
                            (y - self.pc_range[1])
                            / voxel_size[1]
                            / self.shape_resize_times
                        )

                        center = torch.tensor(
                            [coor_x, coor_y], dtype=torch.float32, device=device
                        )
                        center_int = center.to(torch.int32)

                        # throw out not in range objects to avoid out of array
                        # area when creating the heatmap
                        if not (
                            0 <= center_int[0] < feature_map_size[0]
                            and 0 <= center_int[1] < feature_map_size[1]
                        ):
                            continue

                        draw_heatmap_gaussian(fg_map[idx], center_int, radius)

        if fg_map is None:
            fg_map = student_bev_feat.new_ones(
                (student_bev_feat.shape[0], feature_map_size[1], feature_map_size[0])
            )

        if bs > 1:
            fg_map = fg_map.unsqueeze(1)

        fit_loss = F.mse_loss(student_bev_feat, teacher_bev_feat, reduction="none")
        fit_loss = torch.sum(fit_loss * fg_map) / torch.sum(fg_map)

        return fit_loss * self.loss_weight, fg_map

    # def forward(
    #     self, student_bev_feats, teacher_bev_feats, masks_bboxes, gt_bboxes_list
    # ):
    #     """SelfLearningMFD forward.

    #     Args:
    #         student_bev_feats (torch.tensor): _description_
    #         teacher_bev_feats (torch.tensor): _description_
    #         masks_bboxes (): _description_
    #         gt_bboxes_list (_type_): _description_
    #     """
    #     assert isinstance(masks_bboxes, list)
    #     ret_dict = multi_apply(
    #         self._forward,
    #         student_bev_feats,
    #         teacher_bev_feats,
    #         masks_bboxes,
    #         gt_bboxes_list,
    #     )
