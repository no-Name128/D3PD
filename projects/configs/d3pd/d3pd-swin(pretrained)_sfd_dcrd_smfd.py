_base_ = [
    "../_base_/datasets/nus-3d.py",
    "../_base_/default_runtime.py",
    # "../_base_/schedules/cyclic_20e.py",
]
# Global
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
# For nuScenes we usually do 10-class detection
class_names = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]

data_config = {
    "cams": [
        "CAM_FRONT_LEFT",
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK_LEFT",
        "CAM_BACK",
        "CAM_BACK_RIGHT",
    ],
    "Ncams": 6,
    "input_size": (256, 704),
    "src_size": (900, 1600),
    # Augmentation
    "resize": (-0.06, 0.11),
    "rot": (-5.4, 5.4),
    "flip": True,
    "crop_h": (0.0, 0.0),
    "resize_test": -0.00,
}

# Model
grid_config = {
    "x": [-51.2, 51.2, 0.8],
    "y": [-51.2, 51.2, 0.8],
    "z": [-5, 3, 8],
    "depth": [1.0, 60.0, 0.5],
}

voxel_size = [0.1, 0.1, 0.2]

numC_Trans = 80

multi_adj_frame_id_cfg = (1, 1 + 1, 1)


# radar configuration,  x y z rcs vx_comp vy_comp x_rms y_rms vx_rms vy_rms
radar_use_dims = [0, 1, 2, 8, 9, 18]
radar_voxel_size = [0.8, 0.8, 8]
# radar_voxel_size = [0.6, 0.6, 8]
radar_max_voxels_times = 3

bev_h_ = 256 // 2
bev_w_ = 256 // 2

out_size_factor = 8
sweeps_num = 9


teacher_pretrained = "/mnt/data/exps/DenseRadar/ckpts/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220810_030004-9061688e.pth"

student_pretrained = "/mnt/data/exps/DenseRadar/ckpts/bevdet-stereo-geomim-512x1408.pth"

vovnet_pretrained = "/mnt/data/exps/DenseRadar/ckpts/depth_pretrained_v99-3jlw0p36-20210423_010520-model_final-remapped.pth"

model = dict(
    type="D3PD_V3",
    teacher_pretrained=teacher_pretrained,
    # student_pretrained=student_pretrained,
    # vovnet_pretrained=vovnet_pretrained,
    load_stu_with_tea=True,
    align_after_view_transfromation=True,
    num_adj=len(range(*multi_adj_frame_id_cfg)),
    img_backbone=dict(
        type="SwinTransformerBEVFT",
        with_cp=False,
        convert_weights=False,
        depths=[2, 2, 18, 2],
        drop_path_rate=0.2,
        embed_dims=128,
        init_cfg=dict(
            checkpoint="/mnt/data/exps/ckpt/geomim_base.pth",
            type="Pretrained",
        ),
        num_heads=[4, 8, 16, 32],
        out_indices=[2, 3],
        patch_norm=True,
        window_size=[16, 16, 16, 8],
        use_abs_pos_embed=True,
        return_stereo_feat=True,
        output_missing_index_as_none=False,
    ),
    img_neck=dict(
        type="FPN_LSS",
        in_channels=512 + 1024,
        out_channels=512,
        extra_upsample=None,
        input_feature_index=(0, 1),
        scale_factor=2,
    ),
    img_view_transformer=dict(
        type="LSSViewTransformerBEVStereo",
        grid_config=grid_config,
        input_size=data_config["input_size"],
        in_channels=512,
        out_channels=numC_Trans,
        sid=True,
        depthnet_cfg=dict(use_dcn=False, aspp_mid_channels=96, stereo=True, bias=5.0),
        downsample=16,
        loss_depth_weight=1,
    ),
    img_bev_encoder_backbone=dict(
        type="CustomResNet",
        numC_input=numC_Trans * (len(range(*multi_adj_frame_id_cfg)) + 1),
        num_channels=[numC_Trans * 2, numC_Trans * 4, numC_Trans * 8],
    ),
    img_bev_encoder_neck=dict(
        type="FPN_LSS", in_channels=numC_Trans * 8 + numC_Trans * 2, out_channels=256
    ),
    pre_process=dict(
        type="CustomResNet",
        with_cp=False,
        numC_input=numC_Trans,
        num_layer=[
            2,
        ],
        num_channels=[
            numC_Trans,
        ],
        stride=[
            1,
        ],
        backbone_output_ids=[
            0,
        ],
    ),
    # ====================radar feature processing=======================
    radar_voxel_layer=dict(
        max_num_points=10,
        voxel_size=radar_voxel_size,
        max_voxels=(
            30000 * radar_max_voxels_times,
            40000 * radar_max_voxels_times,
        ),
        point_cloud_range=point_cloud_range,
        deterministic=False,
    ),
    radar_pillar_encoder=dict(
        type="PillarFeatureNet",
        in_channels=6,
        feat_channels=[64],
        with_distance=False,
        voxel_size=radar_voxel_size,
        point_cloud_range=point_cloud_range,
    ),
    radar_middle_encoder=dict(
        type="PointPillarsScatter", in_channels=64, output_shape=(128, 128)
    ),
    radar_backbone=dict(
        type="SECOND",
        in_channels=64,
        norm_cfg=dict(type="BN", eps=1e-3, momentum=0.01),
        layer_nums=[3, 5],
        layer_strides=[1, 2],
        out_channels=[64, 128],
    ),
    radar_neck=dict(
        type="SECONDFPN",
        in_channels=[64, 128],
        upsample_strides=[1, 2],
        out_channels=[128, 128],
        norm_cfg=dict(type="BN", eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type="deconv", bias=False),
        use_conv_for_no_stride=True,
    ),
    # ====================lidar processing=======================
    pts_voxel_layer=dict(
        point_cloud_range=point_cloud_range,
        max_num_points=10,
        voxel_size=voxel_size,
        max_voxels=(90000, 120000),
    ),
    pts_voxel_encoder=dict(type="HardSimpleVFE", num_features=5),
    pts_middle_encoder=dict(
        type="SparseEncoder",
        in_channels=5,
        sparse_shape=[41, 1024, 1024],
        output_channels=128,
        order=("conv", "norm", "act"),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type="basicblock",
    ),
    pts_backbone=dict(
        type="SECOND",
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type="BN", eps=1e-3, momentum=0.01),
        conv_cfg=dict(type="Conv2d", bias=False),
    ),
    pts_neck=dict(
        type="SECONDFPN",
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type="BN", eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type="deconv", bias=False),
        use_conv_for_no_stride=True,
    ),
    ######################Fusion RC part#########################3
    rc_bev_fusion=dict(
        type="RC_BEV_Fusion_Sampling",
        low_feats_channels=256,
        hight_feats_channels=256,
        process=[dict(type="SpatialProbAtten"), dict(type="DualWeight_Fusion")],
    ),
    distillation=dict(
        sparse_feats_distill=dict(
            img_bev_distill=dict(
                type="FeatureLoss_InnerClip",
                inter_channel_weight=10,
            ),
            # radar_bev_distill=dict(
            #     # type="FeatureLoss_Affinity"
            #     type="FeatureLoss_InnerClip",
            #     x_sample_num=50,
            #     y_sample_num=50,
            #     inter_keypoint_weight=100,
            #     inter_channel_weight=1.0,
            #     enlarge_width=1.6,
            #     embed_channels=[256, 512],
            #     # inner_feats_distill=dict(type="FeatureLoss_Affinity"),
            # ),
            radar_ms_distill=dict(
                type="Radar_MSDistilll",
                num_layers=2,
                each_layer_loss_cfg=[
                    dict(
                        type="FeatureLoss",
                        student_channels=64,
                        teacher_channels=128,
                        name="each_layer_loss_cfg-1",
                        alpha_mgd=0.00002,
                        lambda_mgd=0.65,
                    ),
                    dict(
                        type="FeatureLoss",
                        student_channels=128,
                        teacher_channels=256,
                        name="each_layer_loss_cfg-2",
                        alpha_mgd=0.00002,
                        lambda_mgd=0.65,
                    ),
                ],
            ),
        ),
        sampling_pos_distill=dict(type="SimpleL1"),
        sampling_feats_distill=dict(
            type="FeatureLoss",
            student_channels=256,
            teacher_channels=512,
            name="sampling_disitll",
            alpha_mgd=0.00002,
            lambda_mgd=0.65,
        ),
        # det_feats_distill=dict(type="FeatureLoss_Affinity"),
        det_result_distill=dict(
            type="Dc_ResultDistill",
            pc_range=point_cloud_range,
            voxel_size=voxel_size,
            out_size_scale=8,
            ret_sum=False,
            loss_weight_reg=1,
            loss_weight_cls=10,
            # max_cls=False,
        ),
        mask_bev_feats_distill=dict(
            type="SelfLearningMFD",
            bev_shape=(bev_h_, bev_w_),
            pc_range=point_cloud_range,
            voxel_size=voxel_size,
            score_threshold=0.6,
            add_stu_decode_bboxes=False,
            loss_weight=0.01,
            bbox_coder=dict(
                type="CenterPointBBoxCoder",
                pc_range=point_cloud_range[:2],
                # post_center_range=[-51.2, -51.2, -10.0, 51.2, 51.2, 10.0],
                post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                max_num=500,
                score_threshold=0.1,
                out_size_factor=out_size_factor,
                voxel_size=voxel_size[:2],
                code_size=9,
            ),
            student_channels=256 * 3,
            teacher_channels=512,
        ),
        heatmap_aug_distill=dict(type="HeatMapAug"),
    ),
    pts_bbox_head=dict(
        type="CenterHead",
        task_specific=True,
        in_channels=768,
        tasks=[
            dict(num_class=1, class_names=["car"]),
            dict(num_class=2, class_names=["truck", "construction_vehicle"]),
            dict(num_class=2, class_names=["bus", "trailer"]),
            dict(num_class=1, class_names=["barrier"]),
            dict(num_class=2, class_names=["motorcycle", "bicycle"]),
            dict(num_class=2, class_names=["pedestrian", "traffic_cone"]),
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)
        ),
        share_conv_channel=64,
        bbox_coder=dict(
            type="CenterPointBBoxCoder",
            pc_range=point_cloud_range[:2],
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=out_size_factor,
            voxel_size=voxel_size[:2],
            code_size=9,
        ),
        separate_head=dict(type="SeparateHead", init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type="GaussianFocalLoss", reduction="mean"),
        loss_bbox=dict(type="L1Loss", reduction="mean", loss_weight=0.25),
        norm_bbox=True,
    ),
    pts_bbox_head_tea=dict(
        type="CenterHead",
        in_channels=sum([256, 256]),
        tasks=[
            dict(num_class=1, class_names=["car"]),
            dict(num_class=2, class_names=["truck", "construction_vehicle"]),
            dict(num_class=2, class_names=["bus", "trailer"]),
            dict(num_class=1, class_names=["barrier"]),
            dict(num_class=2, class_names=["motorcycle", "bicycle"]),
            dict(num_class=2, class_names=["pedestrian", "traffic_cone"]),
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)
        ),
        share_conv_channel=64,
        bbox_coder=dict(
            type="CenterPointBBoxCoder",
            pc_range=point_cloud_range[:2],
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            code_size=9,
        ),
        separate_head=dict(type="SeparateHead", init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type="GaussianFocalLoss", reduction="mean"),
        loss_bbox=dict(type="L1Loss", reduction="mean", loss_weight=0.25),
        norm_bbox=True,
    ),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            point_cloud_range=point_cloud_range,
            grid_size=[1024, 1024, 40],
            voxel_size=voxel_size,
            out_size_factor=out_size_factor,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        )
    ),
    test_cfg=dict(
        pts=dict(
            pc_range=point_cloud_range[:2],
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=out_size_factor,
            voxel_size=voxel_size[:2],
            pre_max_size=1000,
            post_max_size=83,
            # Scale-NMS
            nms_type=["rotate", "rotate", "rotate", "circle", "rotate", "rotate"],
            nms_thr=[0.2, 0.2, 0.2, 0.2, 0.2, 0.5],
            nms_rescale_factor=[
                1.0,
                [0.7, 0.7],
                [0.4, 0.55],
                1.1,
                [1.0, 1.0],
                [4.5, 9.0],
            ],
        )
    ),
)


# dataset_type = "CustomNuScenesDataset"
dataset_type = "NuScenesDataset_D3PD"
data_root = "data/nuscenes/"
file_client_args = dict(backend="disk")

# bda_aug_conf = dict(
#     rot_lim=(-22.5, 22.5), scale_lim=(0.95, 1.05), flip_dx_ratio=0.5, flip_dy_ratio=0.5
# )

bda_aug_conf = dict(
    rot_lim=(-22.5, 22.5), scale_lim=(0.95, 1.05), flip_dx_ratio=0.5, flip_dy_ratio=0.5
)


train_pipeline = [
    dict(
        type="PrepareImageInputs",
        is_train=True,
        data_config=data_config,
        sequential=True,
        file_client_args=file_client_args,
    ),
    dict(
        type="LoadAnnotationsBEVDepth", bda_aug_conf=bda_aug_conf, classes=class_names
    ),
    dict(
        type="LoadRadarPointsMultiSweeps",
        load_dim=18,
        sweeps_num=6,
        # sweeps_num=4,
        use_dim=radar_use_dims,
        max_num=1200,
    ),
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args,
    ),
    dict(type="PointToMultiViewDepth_D3PD", downsample=1, grid_config=grid_config),
    dict(type="ObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="ObjectNameFilter", classes=class_names),
    dict(type="DefaultFormatBundle3D_W_Radar", class_names=class_names),
    dict(
        type="CustomCollect3D",
        keys=[
            "gt_bboxes_3d",
            "gt_labels_3d",
            "img_inputs",
            "radar",
            "points",
            "gt_depth",
        ],
    ),
    # dict(type="DefaultFormatBundle3D", class_names=class_names),
    # dict(
    #     type="Collect3D",
    #     keys=["img_inputs", "gt_bboxes_3d", "gt_labels_3d", "gt_depth"],
    # ),
]

test_pipeline = [
    dict(
        type="PrepareImageInputs",
        data_config=data_config,
        sequential=True,
        file_client_args=file_client_args,
    ),
    dict(
        type="LoadAnnotationsBEVDepth", bda_aug_conf=bda_aug_conf, classes=class_names
    ),
    dict(
        type="LoadRadarPointsMultiSweeps",
        load_dim=18,
        sweeps_num=4,
        use_dim=radar_use_dims,
        max_num=1200,
    ),
    # dict(type="LoadAnnotations_D3PD"),
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args,
    ),
    # dict(
    #     type="LoadPointsFromMultiSweeps",
    #     sweeps_num=4,
    #     use_dim=[0, 1, 2, 3, 4],
    #     file_client_args=file_client_args,
    #     pad_empty_sweeps=True,
    #     remove_close=True,
    # ),
    # dict(type="BEVAug", bda_aug_conf=bda_aug_conf, classes=class_names),
    dict(
        type="MultiScaleFlipAug3D",
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type="DefaultFormatBundle3D_W_Radar",
                class_names=class_names,
                with_label=False,
            ),
            dict(type="CustomCollect3D", keys=["img_inputs", "radar", "points"]),
            # dict(
            #     type="DefaultFormatBundle3D", class_names=class_names, with_label=False
            # ),
            # dict(type="Collect3D", keys=["points", "img_inputs"]),
        ],
    ),
]

input_modality = dict(
    use_lidar=False, use_camera=True, use_radar=True, use_map=False, use_external=False
)

share_data_config = dict(
    type=dataset_type,
    classes=class_names,
    modality=input_modality,
    stereo=True,
    img_info_prototype="bevdet4d",
    multi_adj_frame_id_cfg=multi_adj_frame_id_cfg,
)

test_data_config = dict(
    pipeline=test_pipeline,
    ann_file=data_root + "bevdetv3-nuscenes_infos_val.pkl",
)

data = dict(
    samples_per_gpu=1,  # with 32 GPU
    workers_per_gpu=4,
    dataset_scale=4,
    train=dict(
        type="CBGSDataset",
        dataset=dict(
            data_root=data_root,
            ann_file=data_root + "bevdetv3-nuscenes_infos_train.pkl",
            pipeline=train_pipeline,
            classes=class_names,
            test_mode=False,
            use_valid_flag=True,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d="LiDAR",
        ),
    ),
    val=test_data_config,
    test=test_data_config,
)

for key in ["val", "test"]:
    data[key].update(share_data_config)
data["train"]["dataset"].update(share_data_config)


# data = dict(
#     samples_per_gpu=1,
#     workers_per_gpu=4,
#     dataset_scale=4,
#     train=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file=data_root + "bevdetv3-nuscenes_infos_train.pkl",
#         pipeline=train_pipeline,
#         classes=class_names,
#         modality=input_modality,
#         test_mode=False,
#         use_valid_flag=True,
#         # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
#         # and box_type_3d='Depth' in sunrgbd and scannet dataset.
#         box_type_3d="LiDAR",
#         stereo=True,
#         img_info_prototype="bevdet4d",
#         multi_adj_frame_id_cfg=multi_adj_frame_id_cfg,
#     ),
#     val=dict(
#         type=dataset_type,
#         pipeline=test_pipeline,
#         classes=class_names,
#         modality=input_modality,
#         ann_file=data_root + "bevdetv3-nuscenes_infos_val.pkl",
#         stereo=True,
#         img_info_prototype="bevdet4d",
#         multi_adj_frame_id_cfg=multi_adj_frame_id_cfg,
#     ),
#     test=dict(
#         type=dataset_type,
#         pipeline=test_pipeline,
#         classes=class_names,
#         modality=input_modality,
#         ann_file=data_root + "bevdetv3-nuscenes_infos_val.pkl",
#         stereo=True,
#         img_info_prototype="bevdet4d",
#         multi_adj_frame_id_cfg=multi_adj_frame_id_cfg,
#     ),
# )

# fp16 = dict(loss_scale='dynamic')


# Optimizer
# optimizer = dict(type='AdamW', lr=2e-4, weight_decay=1e-2)
optimizer = dict(
    type="AdamW",
    lr=0.0002,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    constructor="LayerDecayOptimizerConstructor",
    paramwise_cfg=dict(
        num_layers=24,
        layer_decay_rate=0.85,
        custom_keys={
            "absolute_pos_embed": dict(decay_mult=0.0),
            "relative_position_bias_table": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
        },
    ),
)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy="step",
    warmup="linear",
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[
        20,
    ],
)
runner = dict(type="EpochBasedRunner", max_epochs=20)

# custom_hooks = [
#     dict(
#         type="MEGVIIEMAHook",
#         init_updates=10560,
#         priority="NORMAL",
#     ),
#     dict(
#         type="SequentialControlHook",
#         temporal_start_epoch=2,
#     ),
#     dict(
#         # we use syncbn to prevent loss divergency
#         type="SyncbnControlHook",
#         syncbn_start_epoch=2,
#     ),
# ]

evaluation = dict(interval=1)

load_from = teacher_pretrained

work_dir = "d3pd/v3.0/swin(pretrained)"
