import itertools
import logging

from det3d.builder import build_box_coder
from det3d.utils.config_tool import get_downsample_factor

# norm_cfg = dict(type='SyncBN', eps=1e-3, momentum=0.01)
norm_cfg = None

tasks = [
    dict(num_class=1, class_names=["car"]),
    dict(num_class=1, class_names=["pedestrian"]),
    dict(num_class=2, class_names=["motorcycle", "bicycle"]),
    dict(num_class=1, class_names=["other_vehicle"]),
    dict(num_class=2, class_names=["bus", "truck"]),
]

class_names = list(itertools.chain(*[t["class_names"] for t in tasks]))

# training and testing settings
target_assigner = dict(
    type="iou",
    anchor_generators=[
        dict(
            type="anchor_generator_range",
            sizes=[1.93, 4.75, 1.72],
            anchor_ranges=[-100.8, -100.8, -0.86, 100.8, 100.8, -0.86],
            rotations=[0, 1.57],
            matched_threshold=0.6,
            unmatched_threshold=0.45,
            class_name="car",
        ),
        dict(
            type="anchor_generator_range",
            sizes=[0.77, 0.81, 1.78],
            anchor_ranges=[-100.8, -100.8, -0.81, 100.8, 100.8, -0.81],
            rotations=[0, 1.57],
            matched_threshold=0.55,
            unmatched_threshold=0.4,
            class_name="pedestrian",
        ),
        dict(
            type="anchor_generator_range",
            sizes=[0.97, 2.36, 1.60],
            anchor_ranges=[-100.8, -100.8, -0.9, 100.8, 100.8, -0.9],
            rotations=[0, 1.57],
            matched_threshold=0.55,
            unmatched_threshold=0.4,
            class_name="motorcycle",
        ),
        dict(
            type="anchor_generator_range",
            sizes=[0.64, 1.76, 1.46],
            anchor_ranges=[-100.8, -100.8, -1.04, 100.8, 100.8, -1.04],
            rotations=[0, 1.57],
            matched_threshold=0.55,
            unmatched_threshold=0.4,
            class_name="bicycle",
        ),
        dict(
            type="anchor_generator_range",
            sizes=[2.79, 8.2, 3.24],
            anchor_ranges=[-100.8, -100.8, -0.08, 100.8, 100.8, -0.08],
            rotations=[0, 1.57],
            matched_threshold=0.55,
            unmatched_threshold=0.4,
            class_name="other_vehicle",
        ),
        dict(
            type="anchor_generator_range",
            sizes=[2.94, 12.5, 3.43],
            anchor_ranges=[-100.8, -100.8, -0.015, 100.8, 100.8, -0.015],
            rotations=[0, 1.57],
            matched_threshold=0.6,
            unmatched_threshold=0.45,
            class_name="bus",
        ),
        dict(
            type="anchor_generator_range",
            sizes=[2.83, 10.2, 3.44],
            anchor_ranges=[-100.8, -100.8, -0.015, 100.8, 100.8, -0.015],
            rotations=[0, 1.57],
            matched_threshold=0.6,
            unmatched_threshold=0.45,
            class_name="truck",
        ),
    ],
    sample_positive_fraction=-1,
    sample_size=512,
    region_similarity_calculator=dict(type="nearest_iou_similarity",),
    pos_area_threshold=-1,
    tasks=tasks,
)

box_coder = dict(
    type="ground_box3d_coder", n_dim=7, linear_dim=False, encode_angle_vector=False,
)

# model settings
model = dict(
    type="VoxelNet",
    pretrained=None,
    reader=dict(
        type="VoxelFeatureExtractorV3", num_input_features=3, norm_cfg=norm_cfg,
    ),
    backbone=dict(
        type="SpMiddleResNetFHD", num_input_features=3, ds_factor=8, norm_cfg=norm_cfg,
    ),
    neck=dict(
        type="RPN",
        layer_nums=[5, 5],
        ds_layer_strides=[1, 2],
        ds_num_filters=[128, 256],
        us_layer_strides=[1, 2],
        us_num_filters=[256, 256],
        num_input_features=256,
        norm_cfg=norm_cfg,
        logger=logging.getLogger("RPN"),
    ),
    bbox_head=dict(
        # type='RPNHead',
        type="MultiGroupHead",
        mode="3d",
        in_channels=sum([256, 256]),
        norm_cfg=norm_cfg,
        tasks=tasks,
        weights=[1,],
        box_coder=build_box_coder(box_coder),
        encode_background_as_zeros=True,
        loss_norm=dict(
            type="NormByNumPositives", pos_cls_weight=1.0, neg_cls_weight=2.0,
        ),
        loss_cls=dict(type="SigmoidFocalLoss", alpha=0.25, gamma=2.0, loss_weight=1.0,),
        use_sigmoid_score=True,
        loss_bbox=dict(
            type="WeightedSmoothL1Loss",
            sigma=3.0,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            codewise=True,
            loss_weight=1.0,
        ),
        encode_rad_error_by_sin=True,
        loss_aux=dict(
            type="WeightedSoftmaxClassificationLoss",
            name="direction_classifier",
            loss_weight=0.2,
        ),
        direction_offset=0.785,
    ),
)

assigner = dict(
    box_coder=box_coder,
    target_assigner=target_assigner,
    out_size_factor=get_downsample_factor(model),
    debug=False,
)

train_cfg = dict(assigner=assigner)

test_cfg = dict(
    nms=dict(
        use_rotate_nms=True,
        use_multi_class_nms=False,
        nms_pre_max_size=1000,
        nms_post_max_size=80,
        nms_iou_threshold=0.2,
    ),
    score_threshold=0.1,
    post_center_limit_range=[-110, -110, -6, 110, 110, 2],
    max_per_img=500,
)

# dataset settings
dataset_type = "LyftDataset"
data_root = "/data/Datasets/LYFT/trainval"

db_sampler = dict(
    type="GT-AUG",
    enable=True,
    db_info_path="/data/Datasets/LYFT/trainval/dbinfos_train.pkl",
    sample_groups=[
        dict(car=1),
        dict(pedestrian=4),
        dict(motorcycle=4),
        dict(bicycle=4),
        dict(other_vehicle=2),
        dict(bus=5),
        dict(truck=5),
    ],
    db_prep_steps=[
        dict(
            filter_by_min_num_points=dict(
                car=5,
                pedestrian=5,
                motorcycle=5,
                bicycle=5,
                other_vehicle=5,
                bus=5,
                truck=5,
            )
        ),
        dict(filter_by_difficulty=[-1],),
    ],
    global_random_rotation_range_per_object=[0, 0],
    rate=1.0,
)
train_preprocessor = dict(
    mode="train",
    shuffle_points=True,
    gt_loc_noise=[0.0, 0.0, 0.0],
    gt_rot_noise=[0.0, 0.0],
    global_rot_noise=[-0.3925, 0.3925],
    global_scale_noise=[0.95, 1.05],
    global_rot_per_obj_range=[0, 0],
    global_trans_noise=[0.2, 0.2, 0.2],
    remove_points_after_sample=False,
    gt_drop_percentage=0.0,
    gt_drop_max_keep_points=15,
    remove_unknown_examples=False,
    remove_environment=False,
    db_sampler=db_sampler,
    class_names=class_names,
)

val_preprocessor = dict(
    mode="val",
    shuffle_points=False,
    remove_environment=False,
    remove_unknown_examples=False,
)

voxel_generator = dict(
    range=[-100.8, -100.8, -4.0, 100.8, 100.8, 2.0],
    voxel_size=[0.1, 0.1, 0.15],
    max_points_in_voxel=10,
    max_voxel_num=80000,
)

train_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type),
    dict(type="LoadPointCloudAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=train_preprocessor),
    dict(type="Voxelization", cfg=voxel_generator),
    dict(type="AssignTarget", cfg=train_cfg["assigner"]),
    dict(type="Reformat"),
]
test_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type),
    dict(type="LoadPointCloudAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=val_preprocessor),
    dict(type="Voxelization", cfg=voxel_generator),
    dict(type="AssignTarget", cfg=train_cfg["assigner"]),
    dict(type="Reformat"),
]

train_anno = "/data/Datasets/LYFT/trainval/lyft_info_train.pkl"
val_anno = "/data/Datasets/LYFT/trainval/lyft_info_val.pkl"
test_anno = None

data = dict(
    samples_per_gpu=6,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=train_anno,
        ann_file=train_anno,
        class_names=class_names,
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=val_anno,
        ann_file=val_anno,
        class_names=class_names,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=test_anno,
        ann_file=test_anno,
        class_names=class_names,
        pipeline=test_pipeline,
    ),
)

# optimizer
optimizer = dict(
    type="adam", amsgrad=0.0, wd=0.01, fixed_wd=True, moving_average=False,
)
"""training hooks """
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy in training hooks
lr_config = dict(
    type="one_cycle", lr_max=0.01, moms=[0.95, 0.85], div_factor=10.0, pct_start=0.4,
)

checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
        # dict(type='TensorboardLoggerHook')
    ],
)
# yapf:enable
# runtime settings
total_epochs = 20
device_ids = range(8)
dist_params = dict(backend="nccl", init_method="env://")
log_level = "INFO"
work_dir = "/data/Outputs/det3d_Outputs/SECOND_LYFT"
load_from = None
resume_from = None
workflow = [("train", 1), ("val", 1)]
