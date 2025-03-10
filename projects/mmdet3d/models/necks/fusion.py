import torch
from torch import nn
from inplace_abn import InPlaceABNSync
import torch.nn.functional as F
from .. import builder
from mmcv.cnn.bricks import ConvModule, build_conv_layer
from mmdet3d.models.necks.cbam import CBAM
from mmdet3d.models.utils.self_print import feats_to_img
import einops
from ..model_utils.convmixer import ConvMixer


class LayerNormProxy(nn.Module):

    def __init__(self, dim):

        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):

        x = einops.rearrange(x, "b c h w -> b h w c")
        x = self.norm(x)
        return einops.rearrange(x, "b h w c -> b c h w")


@builder.NECKS.register_module()
class SamplingWarpFusion(nn.Module):
    def __init__(
        self, features=256, reduce_ch=dict(in_channels=512, in_channels2=512), **kwargs
    ):
        super(SamplingWarpFusion, self).__init__()

        self.delta_gen1 = nn.Sequential(
            nn.Conv2d(reduce_ch.in_channels, features, kernel_size=1, bias=False),
            InPlaceABNSync(features),
            nn.Conv2d(features, 2, kernel_size=3, padding=1, bias=False),
        )

        self.tea_delta_gen = nn.Sequential(
            nn.Conv2d(reduce_ch.in_channels, features, kernel_size=1, bias=False),
            InPlaceABNSync(features),
            nn.Conv2d(features, 2, kernel_size=3, padding=1, bias=False),
        )

        if "out_conv_cfg" in kwargs.keys():
            self.output_conv = ConvModule(**kwargs["out_conv_cfg"])
            # self.output_conv = nn.Conv2d(
            #     reduce_ch.in_channels2,
            #     kwargs["out_channels"],
            #     kernel_size=1,
            #     stride=1,
            #     bias=False,
            # )
        else:
            self.output_conv = None

        self.delta_gen1[2].weight.data.zero_()
        self.tea_delta_gen[2].weight.data.zero_()

        self.ret_sampling_feats = kwargs["ret_sampling_feats"]

        if "aug_radar_conv" in kwargs.keys():
            self.aug_radar_conv = True

            hidden_feats = 320
            kwargs["aug_radar_conv"].update(dict(out_channels=hidden_feats))
            self.aug_radar_conv1 = ConvModule(**kwargs["aug_radar_conv"])
            hidden_feats2 = 256
            kwargs["aug_radar_conv"].update(
                dict(in_channels=hidden_feats, out_channels=hidden_feats2)
            )
            self.aug_radar_conv2 = ConvModule(**kwargs["aug_radar_conv"])
            kwargs["aug_radar_conv"].update(dict(in_channels=hidden_feats2))
            self.aug_radar_conv3 = ConvModule(**kwargs["aug_radar_conv"])
        else:
            self.aug_radar_conv = None

        if "bi_dir" in kwargs.keys():
            self.bi_dir = kwargs["bi_dir"]

            # self.delta_gen2 = nn.Sequential(
            #     nn.Conv2d(self.bi_dir.in_channels, features, kernel_size=1, bias=False),
            #     InPlaceABNSync(features),
            #     nn.Conv2d(features, 2, kernel_size=3, padding=1, bias=False),
            # )

            self.teacher_reduce_channels = nn.Conv2d(
                self.bi_dir.in_channels,
                reduce_ch.in_channels,
                kernel_size=1,
            )  # ch:: 512->320
            self.teacher_output_reduce = nn.Conv2d(
                reduce_ch.in_channels, features, kernel_size=1
            )  # ch: 320->256

            if "bi_weight" in self.bi_dir.keys():
                self.bi_weight_fusion = builder.build_neck(self.bi_dir.bi_weight_fusion)
            else:
                self.bi_weight_fusion = None
        else:
            self.bi_weight_fusion = None

    def bilinear_interpolate_torch_gridsample2(self, input, size, delta=0):
        out_h, out_w = size
        n, c, h, w = input.shape
        s = 2.0
        norm = (
            torch.tensor([[[[(out_w - 1) / s, (out_h - 1) / s]]]])
            .type_as(input)
            .to(input.device)
        )
        w_list = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h_list = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((h_list.unsqueeze(2), w_list.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + delta.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid, align_corners=True)
        return output

    def teacher_sampling_warp(self, teach_feats, delta=None):
        """teacher_sampling_warp

        Args:
            teach_feats (_type_): _description_

        Returns:
            list: teacher_sampling_pos,teacher_sampling_feats
        """

        h, w = teach_feats.size(2), teach_feats.size(3)
        tea_delta = self.tea_delta_gen(teach_feats)

        if delta is not None:
            sampling_feats = self.bilinear_interpolate_torch_gridsample2(
                teach_feats, (h, w), delta
            )

            return None, sampling_feats

        sampling_feats = self.bilinear_interpolate_torch_gridsample2(
            teach_feats, (h, w), tea_delta
        )

        return tea_delta, sampling_feats

    def aug_radar_first(self, feats1, feats2):
        x = torch.cat((feats1, feats2), dim=1)
        x = self.aug_radar_conv1(x)
        x = self.aug_radar_conv2(x)
        x = self.aug_radar_conv3(x)
        return x

    # def forward(self, low_stage, high_stage, teacher_bev=None, **kwargs) -> list:
    #     """SamplingWarpFusion: Computes a list return value for use in distillation calculations.

    #     Args:
    #         low_stage (_type_): img_bev feats.
    #         high_stage (_type_): radar_bev feats.
    #         teacher_bev (_type_, optional): point cloud sparse encoder feats. Defaults to None.

    #     Returns:
    #         list: (student_sampling_feats,fusion_concat_feats,student_sampling_pos,teacher_sampling_pos)
    #     """

    #     ret_feats = dict()
    #     h, w = low_stage.size(2), low_stage.size(3)

    #     concat = torch.cat((low_stage, high_stage), 1)
    #     delta1 = self.delta_gen1(concat)
    #     sampling_feats = self.bilinear_interpolate_torch_gridsample2(
    #         low_stage, (h, w), delta1
    #     )  # _,256,_,_

    #     if self.ret_sampling_feats:
    #         ret_feats.update(dict(student_sampling_feats=sampling_feats))

    #     if self.aug_radar_conv:
    #         high_stage = self.aug_radar_first(sampling_feats, high_stage)

    #     if self.bi_weight_fusion is not None:
    #         low_stage, high_stage = self.bi_weight_fusion(low_stage, high_stage)

    #     concat = torch.cat((low_stage, high_stage), 1)

    #     if self.output_conv is not None:
    #         concat = self.output_conv(concat)

    #     ret_feats.update(dict(det_feats=concat))
    #     # ret_feats.append(concat)

    #     # Sample position distillation return
    #     ret_feats.update(dict(student_sampling_pos=delta1))
    #     # ret_feats.append(delta1)
    #     if teacher_bev is not None:
    #         _teacher_bev = teacher_bev.clone()
    #         delta2, teacher_sampleing_feats = self.teacher_sampling_warp(_teacher_bev)
    #         # ret_feats.append(delta2)
    #         # ret_feats.append(teacher_sampleing_feats)
    #         ret_feats.update(dict(teacher_sampling_pos=delta2))
    #         ret_feats.update(dict(teacher_sampling_feats=teacher_sampleing_feats))

    #     return ret_feats

    def forward(self, low_stage, high_stage, teacher_bev=None, **kwargs) -> list:

        if self.bi_weight_fusion is not None:
            low_stage, high_stage = self.bi_weight_fusion(low_stage, high_stage)

        ret_feats = dict()
        h, w = high_stage.size(2), high_stage.size(3)

        concat = torch.cat((low_stage, high_stage), 1)
        delta1 = self.delta_gen1(concat)
        ret_feats.update(dict(student_sampling_pos=delta1))
        sampling_feats = self.bilinear_interpolate_torch_gridsample2(
            low_stage, (h, w), delta1
        )  # _,256,_,_

        if self.ret_sampling_feats:
            ret_feats.update(dict(student_sampling_feats=sampling_feats))

        if self.aug_radar_conv:
            high_stage = self.aug_radar_first(sampling_feats, high_stage)

        concat = torch.cat((low_stage, sampling_feats), 1)

        if self.output_conv is not None:
            concat = self.output_conv(concat)

        ret_feats.update(dict(det_feats=concat))
        # ret_feats.append(concat)

        # Sample position distillation return
        if teacher_bev is not None:
            _teacher_bev = teacher_bev.clone()
            delta2, teacher_sampleing_feats = self.teacher_sampling_warp(_teacher_bev)
            ret_feats.update(dict(teacher_sampling_pos=delta2))
            ret_feats.update(dict(teacher_sampling_feats=teacher_sampleing_feats))

        return ret_feats


@builder.NECKS.register_module()
class SamplingWarpFusion_V2(SamplingWarpFusion):
    def __init__(self, low_feats_channels, hight_feats_channels, **kwargs):
        super(SamplingWarpFusion, self).__init__()

        self.ret_sampling_pos = kwargs.get("ret_sampling_pos")
        self.ret_sampling_feats = kwargs.get("ret_sampling_feats")

        concat_channels = low_feats_channels + hight_feats_channels
        hidden_channels = concat_channels // 2

        self.delta_gen1 = nn.Sequential(
            nn.Conv2d(concat_channels, hidden_channels, kernel_size=1, bias=False),
            InPlaceABNSync(hidden_channels),
            nn.Conv2d(hidden_channels, 2, kernel_size=3, padding=1, bias=False),
        )
        self.tea_delta_gen = nn.Sequential(
            nn.Conv2d(concat_channels, hidden_channels, kernel_size=1, bias=False),
            InPlaceABNSync(hidden_channels),
            nn.Conv2d(hidden_channels, 2, kernel_size=3, padding=1, bias=False),
        )

        if "out_conv_cfg" in kwargs.keys():
            self.output_conv = ConvModule(**kwargs["out_conv_cfg"])
        else:
            self.output_conv = None

        self.delta_gen1[2].weight.data.zero_()
        self.tea_delta_gen[2].weight.data.zero_()

    def forward(self, low_stage, high_stage, teacher_bev=None, **kwargs):

        ret_feats = dict()
        h, w = low_stage.size(2), low_stage.size(3)

        concat = torch.cat((low_stage, high_stage), 1)
        delta1 = self.delta_gen1(concat)
        ret_feats.update(dict(student_sampling_pos=delta1))
        sampling_feats = self.bilinear_interpolate_torch_gridsample2(
            low_stage, (h, w), delta1
        )  # _,256,_,_

        if self.ret_sampling_feats:
            ret_feats.update(dict(student_sampling_feats=sampling_feats))

        concat = torch.cat((low_stage, sampling_feats), 1)

        if self.output_conv is not None:
            concat = self.output_conv(concat)

        ret_feats.update(dict(det_feats=concat))

        # Sample position distillation return
        if teacher_bev is not None:
            _teacher_bev = teacher_bev.clone()
            delta2, teacher_sampleing_feats = self.teacher_sampling_warp(_teacher_bev)
            ret_feats.update(dict(teacher_sampling_pos=delta2))
            ret_feats.update(dict(teacher_sampling_feats=teacher_sampleing_feats))

        return ret_feats


@builder.NECKS.register_module()
class SamplingFusion(SamplingWarpFusion_V2):

    def __init__(
        self,
        low_feats_channels,
        hight_feats_channels,
        out_conv_cfg=dict(type=None),
        **kwargs,
    ):
        super().__init__(low_feats_channels, hight_feats_channels, **kwargs)
        self.low_feats_channels = low_feats_channels
        self.hight_feats_channels = hight_feats_channels

        cfg_type = out_conv_cfg.get("type")
        if cfg_type == "cbam".upper():
            self.out_conv = builder.build_neck(out_conv_cfg)
        elif cfg_type == "cbr":
            self.out_conv = nn.Sequential(
                nn.Conv2d(
                    low_feats_channels + hight_feats_channels,
                    hight_feats_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(hight_feats_channels),
            )
        else:
            self.out_conv = nn.Sequential(
                nn.Conv2d(
                    self.low_feats_channels * 2 + self.hight_feats_channels,
                    self.low_feats_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(self.low_feats_channels),
            )

    def forward(self, low_stage, high_stage, **kwargs):

        h, w = low_stage.size(2), low_stage.size(3)

        concat = torch.cat((low_stage, high_stage), 1)
        delta1 = self.delta_gen1(concat)

        sampling_feats = self.bilinear_interpolate_torch_gridsample2(
            low_stage, (h, w), delta1
        )  # _,160,_,_

        concat = torch.cat((sampling_feats, high_stage), 1)

        if self.out_conv is not None:
            concat = self.out_conv(concat)

        return concat


@builder.NECKS.register_module()
class DenseRadarAug(SamplingFusion):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        concat_channels = self.low_feats_channels + self.hight_feats_channels
        hidden_channels = concat_channels // 2
        self.delta_gen1 = nn.Sequential(
            nn.Conv2d(concat_channels, hidden_channels, kernel_size=1, bias=False),
            InPlaceABNSync(hidden_channels),
            nn.Conv2d(hidden_channels, 2, kernel_size=3, padding=1, bias=False),
        )

        self.low_feats_sampling_aug = nn.Sequential(
            nn.Conv2d(
                self.low_feats_channels,
                concat_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            # nn.BatchNorm2d(concat_channels),
            # nn.ReLU(),
            nn.Conv2d(
                concat_channels,
                self.low_feats_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(self.low_feats_channels),
            # nn.ReLU(),
        )
        self.high_feats_sampling_aug = nn.Sequential(
            nn.Conv2d(
                self.hight_feats_channels,
                concat_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            # nn.BatchNorm2d(concat_channels),
            # nn.ReLU(),
            nn.Conv2d(
                concat_channels,
                self.hight_feats_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(self.hight_feats_channels),
            # nn.ReLU(),
        )

    def forward(self, low_stage, high_stage, **kwargs):

        h, w = low_stage.size(2), low_stage.size(3)

        concat = torch.cat((low_stage, high_stage), 1)
        delta1 = self.delta_gen1(concat)

        sampling_feats_img_bev = self.bilinear_interpolate_torch_gridsample2(
            low_stage, (h, w), delta1
        )  # _,256,_,_
        sampling_feats_img_bev = self.low_feats_sampling_aug(sampling_feats_img_bev)

        sampling_feats_radar_bev = self.bilinear_interpolate_torch_gridsample2(
            high_stage, (h, w), delta1
        )  # _,224,_,_
        sampling_feats_radar_bev = self.high_feats_sampling_aug(
            sampling_feats_radar_bev
        )

        concat = torch.cat(
            (sampling_feats_img_bev, sampling_feats_radar_bev, low_stage), 1
        )

        if self.out_conv is not None:
            concat = self.out_conv(concat)

        return dict(det_feats=concat)


@builder.NECKS.register_module()
class SimpleAttenFusion(nn.Module):
    def __init__(self, imgbev_ch=256, radarbev_ch=64, **kwargs):
        super(SimpleAttenFusion, self).__init__()

        concat_ch = imgbev_ch + radarbev_ch

        self.reduce_conv = ConvModule(
            concat_ch,
            imgbev_ch,
            kernel_size=1,
            padding=0,
            norm_cfg=dict(type="BN", eps=1e-3, momentum=0.01),
            act_cfg=dict(type="ReLU"),
            inplace=False,
        )

        self.atten = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(imgbev_ch, imgbev_ch, kernel_size=1, stride=1),
            nn.Sigmoid(),
        )

    def forward(self, img_bev_feats, radar_bev_feats, **kwargs):

        output = self.reduce_conv(torch.cat([img_bev_feats, radar_bev_feats], dim=1))
        output = output * self.atten(output)

        return output


@builder.NECKS.register_module()
class SpatialProbAtten(nn.Module):
    def __init__(self, img_bev_channels=256, radar_bev_channle=256, r=4, divide=False):
        super().__init__()

        concat_channels = img_bev_channels + radar_bev_channle
        hidden_channles = concat_channels // r

        self.weight_b1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(concat_channels, hidden_channles, kernel_size=1),
            # nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channles, img_bev_channels, kernel_size=1),
        )

        self.weight_b2 = nn.Sequential(
            nn.Conv2d(concat_channels, hidden_channles, kernel_size=1),
            # nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channles, radar_bev_channle, kernel_size=1),
        )

        self.sig = nn.Sigmoid()
        self.divide = divide

    def forward(self, img_bev, radar_bev):
        concat = torch.cat([img_bev, radar_bev], dim=1)

        w1 = self.weight_b1(concat)
        w2 = self.weight_b2(concat)
        weight = self.sig(w1 + w2)

        if self.divide:

            ret_feats = img_bev + radar_bev
            ret_feats = torch.divide(ret_feats, (1 - weight))
        else:
            radar_bev_aug = (1 - weight) * radar_bev
            img_bev_aug = weight * img_bev
            ret_feats = img_bev_aug + radar_bev_aug

        return ret_feats


@builder.NECKS.register_module()
class BiDirectionWeightFusion(nn.Module):
    def __init__(self, img_channels, radar_channels):
        super().__init__()

        self.img_channels = img_channels
        self.radar_channels = radar_channels

        weight_channels = img_channels + radar_channels
        self.img_bev_weight = nn.Conv2d(
            weight_channels, self.img_channels, 3, 1, padding=1
        )
        self.radar_bev_weight = nn.Conv2d(
            weight_channels, self.radar_channels, 3, 1, padding=1
        )
        # self.img_bev_weight = nn.Conv2d(weight_channels, 1, 3, 1, padding=1)
        # self.radar_bev_weight = nn.Conv2d(weight_channels, 1, 3, 1, padding=1)
        self.sig1 = nn.Sigmoid()
        self.sig2 = nn.Sigmoid()

    def forward(self, img_bev, radar_bev):
        concat = torch.cat([img_bev, radar_bev], dim=1)

        # img_bev_weight = self.sig1(self.img_bev_weight(concat))
        # radar_bev_weight = self.sig2(self.radar_bev_weight(concat))

        img_bev_weight = torch.sigmoid(self.img_bev_weight(concat))
        radar_bev_weight = torch.sigmoid(self.radar_bev_weight(concat))

        assert img_bev_weight.size(2) == radar_bev_weight.size(2)

        img_bev = img_bev * img_bev_weight
        radar_bev = radar_bev * radar_bev_weight

        return img_bev, radar_bev


@builder.NECKS.register_module()
class DualWeight_Fusion(nn.Module):
    def __init__(self, in_channels=256):
        super().__init__()
        self.branch_1_out = nn.Conv2d(2, 1, kernel_size=3, padding=1)
        self.branch_1_out_7 = nn.Conv2d(2, 1, kernel_size=7, dilation=3, padding=9)

        self.branch_2 = nn.Conv2d(
            in_channels, in_channels // 2, kernel_size=3, padding=1
        )
        self.branch_2_cbam = CBAM(in_channels // 2)
        self.branch_2_out = nn.Conv2d(in_channels // 2, 1, kernel_size=1)

    def forward(self, bev_feats):
        branch_1_max, _ = torch.max(bev_feats, dim=1, keepdim=True)
        branch_1_avg = torch.mean(bev_feats, dim=1, keepdim=True)
        branch_1 = torch.cat([branch_1_max, branch_1_avg], dim=1)
        _branch_1 = branch_1
        branch_1 = self.branch_1_out(_branch_1)
        branch_1 = torch.mean(
            branch_1 + self.branch_1_out_7(_branch_1), dim=1
        ).unsqueeze(1)

        branch_2 = self.branch_2(bev_feats)
        branch_2 = self.branch_2_cbam(branch_2)
        branch_2 = self.branch_2_out(branch_2)

        attenion = torch.sigmoid(branch_1 + branch_2)

        # feats_to_img(
        #     attenion,
        #     "/mnt/data/exps/DenseRadar/out/v1_feats_out",
        #     suffix="dualweight_atten",
        # )

        bev_feats = bev_feats * attenion

        return bev_feats


@builder.NECKS.register_module()
class RC_BEV_Fusion(nn.Module):
    def __init__(
        self, img_bev_channels=256, radar_bev_channle=256, process=[], **kwargs
    ):
        super().__init__()

        self.img_bev_channels = img_bev_channels

        self.process_cfgs = process
        self.SpatialProbAtten = builder.build_neck(process[0])

        self.DualWeight_Fusion = builder.build_neck(process[1])

        # self.output_conv = nn.Sequential(
        #     nn.Conv2d(img_bev_channels, img_bev_channels, kernel_size=1),
        #     # nn.BatchNorm2d(img_bev_channels),
        #     # nn.ReLU(inplace=True),
        # )
        self.output_conv = CBAM(img_bev_channels)

    def forward(self, img_bev, radar_bev):
        # base_path = "/mnt/data/exps/DenseRadar/out/v1-sparse-feats-distill"
        # feats_to_img(img_bev, base_path=base_path,suffix='imgbev')
        # feats_to_img(radar_bev, base_path=base_path,suffix='radarbev')

        output = self.SpatialProbAtten(img_bev, radar_bev)
        output = self.DualWeight_Fusion(output)

        # output = self.output_conv(output)

        # feats_to_img(output, base_path=base_path,suffix='fusionbev')

        return output


@builder.NECKS.register_module()
class RC_BEV_Fusion_Sampling(nn.Module):
    def __init__(
        self,
        img_bev_channels=256,
        radar_bev_channle=256,
        **kwargs,
    ):
        super(RC_BEV_Fusion_Sampling, self).__init__(**kwargs)

        self.img_bev_channels = img_bev_channels
        self.radar_bev_channle = radar_bev_channle

        in_channels = self.img_bev_channels + self.radar_bev_channle
        # self.SpatialProbAtten = SpatialProbAtten(
        #     img_bev_channels=self.img_bev_channels,
        #     radar_bev_channle=self.radar_bev_channle,
        #     divide=True,
        # )
        self.DualWeight_Fusion = DualWeight_Fusion(in_channels=in_channels)

        hidden_channels = self.img_bev_channels // 2
        # self.sampling_pos_gen = nn.Sequential(
        #     nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
        #     InPlaceABNSync(hidden_channels),
        #     nn.Conv2d(hidden_channels, 2, kernel_size=3, padding=1, bias=False),
        # )

        # self.output_conv = nn.Sequential(
        #     nn.Conv2d(img_bev_channels, img_bev_channels, kernel_size=1),
        #     # nn.BatchNorm2d(img_bev_channels),
        #     # nn.ReLU(inplace=True),
        # )
        self.conv_offset_img_bev = nn.Sequential(
            nn.Conv2d(
                img_bev_channels,
                img_bev_channels,
                3,
                3,
                3 // 2,
                groups=img_bev_channels,
            ),
            LayerNormProxy(img_bev_channels),
            nn.GELU(),
            nn.Conv2d(img_bev_channels, 2, 1, 1, 0, bias=False),
        )
        self.conv_offset_radar_bev = nn.Sequential(
            nn.Conv2d(
                radar_bev_channle,
                radar_bev_channle,
                3,
                3,
                3 // 2,
                groups=radar_bev_channle,
            ),
            LayerNormProxy(radar_bev_channle),
            nn.GELU(),
            nn.Conv2d(radar_bev_channle, 2, 1, 1, 0, bias=False),
        )

        self.mixer = ConvMixer(in_channels, in_channels, depth=3)
        self.output_conv = ConvMixer(in_channels, self.img_bev_channels, 3)

    def bilinear_interpolate_torch_gridsample2(self, input, size, delta=None):
        out_h, out_w = size
        n, c, h, w = input.shape
        s = 2.0
        # norm = (
        #     torch.tensor([[[[(out_w - 1) / s, (out_h - 1) / s]]]])
        #     .type_as(input)
        #     .to(input.device)
        # )
        w_list = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h_list = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((h_list.unsqueeze(2), w_list.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        if delta is not None:
            grid = grid + delta.permute(0, 2, 3, 1)  # / norm

        output = F.grid_sample(input, grid, align_corners=True)
        return output

    # def offset_sampling(self, img_bev, radar_bev):

    #     return offset

    def forward(self, img_bev, radar_bev):

        h, w = img_bev.size(2), img_bev.size(3)
        new_cat = torch.cat([img_bev, radar_bev], dim=1)
        # import time

        # sampling_time_start = time.perf_counter()
        # delta1 = self.sampling_pos_gen(new_cat)

        img_offset = self.conv_offset_img_bev(img_bev)
        radar_offset = self.conv_offset_radar_bev(radar_bev)
        offset = (img_offset + radar_offset) / 2
        offset = F.interpolate(offset, (img_bev.size(2), img_bev.size(3)))

        sampling_feats = self.bilinear_interpolate_torch_gridsample2(
            img_bev, (h, w), delta=offset
        )  # _,256,_,_

        # sampling_time_end = time.perf_counter()
        # from mmdet3d.models.utils.self_print import print2file
        # print2file(
        #     dict(sampling_times=(sampling_time_end - sampling_time_start)),
        #     "sampling-times",
        # )

        ret_dict = {}
        output = self.mixer(new_cat)
        output = self.DualWeight_Fusion(new_cat)
        output = self.output_conv(output)

        ret_dict.update(dict(det_feats=[output]))
        ret_dict.update(dict(student_sampling_feats=sampling_feats))

        return ret_dict
