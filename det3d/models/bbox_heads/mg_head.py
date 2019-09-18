import logging
from collections import defaultdict
from enum import Enum

import numpy as np
import torch
from det3d.core import box_torch_ops
from det3d.models.builder import build_loss
from det3d.models.losses import metrics
from det3d.torchie.cnn import constant_init, kaiming_init
from det3d.torchie.trainer import load_checkpoint
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.batchnorm import _BatchNorm

from .. import builder
from ..losses import accuracy
from ..registry import HEADS


def one_hot_f(tensor, depth, dim=-1, on_value=1.0, dtype=torch.float32):
    tensor_onehot = torch.zeros(
        *list(tensor.shape), depth, dtype=dtype, device=tensor.device
    )
    tensor_onehot.scatter_(dim, tensor.unsqueeze(dim).long(), on_value)
    return tensor_onehot


def add_sin_difference(boxes1, boxes2):
    rad_pred_encoding = torch.sin(boxes1[..., -1:]) * torch.cos(boxes2[..., -1:])
    rad_tg_encoding = torch.cos(boxes1[..., -1:]) * torch.sin(boxes2[..., -1:])
    boxes1 = torch.cat([boxes1[..., :-1], rad_pred_encoding], dim=-1)
    boxes2 = torch.cat([boxes2[..., :-1], rad_tg_encoding], dim=-1)
    return boxes1, boxes2


def _get_pos_neg_loss(cls_loss, labels):
    # cls_loss: [N, num_anchors, num_class]
    # labels: [N, num_anchors]
    batch_size = cls_loss.shape[0]
    if cls_loss.shape[-1] == 1 or len(cls_loss.shape) == 2:
        cls_pos_loss = (labels > 0).type_as(cls_loss) * cls_loss.view(batch_size, -1)
        cls_neg_loss = (labels == 0).type_as(cls_loss) * cls_loss.view(batch_size, -1)
        cls_pos_loss = cls_pos_loss.sum() / batch_size
        cls_neg_loss = cls_neg_loss.sum() / batch_size
    else:
        cls_pos_loss = cls_loss[..., 1:].sum() / batch_size
        cls_neg_loss = cls_loss[..., 0].sum() / batch_size
    return cls_pos_loss, cls_neg_loss


def get_direction_target(anchors, reg_targets, one_hot=True, dir_offset=0.0):
    batch_size = reg_targets.shape[0]
    anchors = anchors.view(batch_size, -1, anchors.shape[-1])
    rot_gt = reg_targets[..., -1] + anchors[..., -1]
    dir_cls_targets = ((rot_gt - dir_offset) > 0).long()
    if one_hot:
        dir_cls_targets = one_hot_f(dir_cls_targets, 2, dtype=anchors.dtype)
    return dir_cls_targets


def smooth_l1_loss(pred, gt, sigma):
    def _smooth_l1_loss(pred, gt, sigma):
        sigma2 = sigma ** 2
        cond_point = 1 / sigma2
        x = pred - gt
        abs_x = torch.abs(x)

        in_mask = abs_x < cond_point
        out_mask = 1 - in_mask

        in_value = 0.5 * (sigma * x) ** 2
        out_value = abs_x - 0.5 / sigma2

        value = in_value * in_mask.type_as(in_value) + out_value * out_mask.type_as(
            out_value
        )
        return value

    value = _smooth_l1_loss(pred, gt, sigma)
    loss = value.mean(dim=1).sum()
    return loss


def smooth_l1_loss_detectron2(input, target, beta: float, reduction: str = "none"):
    """
    Smooth L1 loss defined in the Fast R-CNN paper as:
                  | 0.5 * x ** 2 / beta   if abs(x) < beta
    smoothl1(x) = |
                  | abs(x) - 0.5 * beta   otherwise,
    where x = input - target.
    Smooth L1 loss is related to Huber loss, which is defined as:
                | 0.5 * x ** 2                  if abs(x) < beta
     huber(x) = |
                | beta * (abs(x) - 0.5 * beta)  otherwise
    Smooth L1 loss is equal to huber(x) / beta. This leads to the following
    differences:
     - As beta -> 0, Smooth L1 loss converges to L1 loss, while Huber loss
       converges to a constant 0 loss.
     - As beta -> +inf, Smooth L1 converges to a constant 0 loss, while Huber loss
       converges to L2 loss.
     - For Smooth L1 loss, as beta varies, the L1 segment of the loss has a constant
       slope of 1. For Huber loss, the slope of the L1 segment is beta.
    Smooth L1 loss can be seen as exactly L1 loss, but with the abs(x) < beta
    portion replaced with a quadratic function such that at abs(x) = beta, its
    slope is 1. The quadratic segment smooths the L1 loss near x = 0.
    Args:
        input (Tensor): input tensor of any shape
        target (Tensor): target value tensor with the same shape as input
        beta (float): L1 to L2 change point.
            For beta values < 1e-5, L1 loss is computed.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        The loss with the reduction option applied.
    Note:
        PyTorch's builtin "Smooth L1 loss" implementation does not actually
        implement Smooth L1 loss, nor does it implement Huber loss. It implements
        the special case of both in which they are equal (beta=1).
        See: https://pytorch.org/docs/stable/nn.html#torch.nn.SmoothL1Loss.
     """
    if beta < 1e-5:
        # if beta == 0, then torch.where will result in nan gradients when
        # the chain rule is applied due to pytorch implementation details
        # (the False branch "0.5 * n ** 2 / 0" has an incoming gradient of
        # zeros, rather than "no gradient"). To avoid this issue, we define
        # small values of beta to be exactly l1 loss.
        loss = torch.abs(input - target)
    else:
        n = torch.abs(input - target)
        cond = n < beta
        loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss


def create_loss(
    loc_loss_ftor,
    cls_loss_ftor,
    box_preds,
    cls_preds,
    cls_targets,
    cls_weights,
    reg_targets,
    reg_weights,
    num_class,
    encode_background_as_zeros=True,
    encode_rad_error_by_sin=True,
    bev_only=False,
    box_code_size=9,
):
    batch_size = int(box_preds.shape[0])

    if bev_only:
        box_preds = box_preds.view(batch_size, -1, box_code_size - 2)
    else:
        box_preds = box_preds.view(batch_size, -1, box_code_size)

    if encode_background_as_zeros:
        cls_preds = cls_preds.view(batch_size, -1, num_class)
    else:
        cls_preds = cls_preds.view(batch_size, -1, num_class + 1)

    cls_targets = cls_targets.squeeze(-1)
    one_hot_targets = one_hot_f(cls_targets, depth=num_class + 1, dtype=box_preds.dtype)
    if encode_background_as_zeros:
        one_hot_targets = one_hot_targets[..., 1:]

    if encode_rad_error_by_sin:
        # sin(a - b) = sinacosb-cosasinb
        box_preds, reg_targets = add_sin_difference(box_preds, reg_targets)

    loc_losses = loc_loss_ftor(box_preds, reg_targets, weights=reg_weights)  # [N, M]
    cls_losses = cls_loss_ftor(
        cls_preds, one_hot_targets, weights=cls_weights
    )  # [N, M]

    return loc_losses, cls_losses


class LossNormType(Enum):
    NormByNumPositives = "norm_by_num_positives"
    NormByNumExamples = "norm_by_num_examples"
    NormByNumPosNeg = "norm_by_num_pos_neg"
    DontNorm = "dont_norm"


@HEADS.register_module
class Head(nn.Module):
    def __init__(
        self,
        num_input,
        num_pred,
        num_cls,
        use_dir=False,
        num_dir=0,
        header=True,
        name="",
        focal_loss_init=False,
        **kwargs,
    ):
        super(Head, self).__init__(**kwargs)
        self.use_dir = use_dir

        self.conv_box = nn.Conv2d(num_input, num_pred, 1)
        self.conv_cls = nn.Conv2d(num_input, num_cls, 1)

        if self.use_dir:
            self.conv_dir = nn.Conv2d(num_input, num_dir, 1)

    def forward(self, x):
        ret_list = []
        box_preds = self.conv_box(x).permute(0, 2, 3, 1).contiguous()
        cls_preds = self.conv_cls(x).permute(0, 2, 3, 1).contiguous()
        ret_dict = {"box_preds": box_preds, "cls_preds": cls_preds}
        if self.use_dir:
            dir_preds = self.conv_dir(x).permute(0, 2, 3, 1).contiguous()
            ret_dict["dir_cls_preds"] = dir_preds

        return ret_dict


@HEADS.register_module
class RegHead(nn.Module):
    def __init__(
        self,
        mode="z",
        in_channels=sum([128,]),
        norm_cfg=None,
        tasks=None,
        name="rpn",
        logger=None,
        **kwargs,
    ):
        super(RegHead, self).__init__()

        num_classes = [len(t["class_names"]) for t in tasks]
        self.class_names = [t["class_names"] for t in tasks]

        if norm_cfg is None:
            norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        self._norm_cfg = norm_cfg

        num_preds = []
        for idx, num_class in enumerate(num_classes):
            num_preds.append(2)  # h, z, gp

        self.tasks = nn.ModuleList()
        num_input = in_channels
        for task_id, num_pred in enumerate(num_preds):
            conv_box = nn.Conv2d(num_input, num_pred, 1)
            self.tasks.append(conv_box)

        self.crop_cfg = kwargs.get("crop_cfg", None)
        self.z_mode = kwargs.get("z_type", "top")
        self.iou_loss = kwargs.get("iou_loss", False)

    def forward(self, x):

        ret_dicts = []
        for task in self.tasks:
            out = task(x)
            out = F.max_pool2d(out, kernel_size=out.size()[2:])
            ret_dicts.append(out.permute(0, 2, 3, 1).contiguous())

        return ret_dicts

    def loss(self, example, preds, **kwargs):
        batch_size_dev = example["targets"].shape[0]
        rets = []
        for task_id, task_pred in enumerate(preds):
            zg = example["targets"][:, 2:3]
            hg = example["targets"][:, 3:4]
            gg = example["targets"][:, 4:5]
            gp = example["ground_plane"].view(-1, 1)

            zt = task_pred[:, :, :, 0:1].view(-1, 1)
            ht = task_pred[:, :, :, 1:2].view(-1, 1)

            height_loss = smooth_l1_loss(ht, hg, 3.0) / batch_size_dev

            height_a = self.crop_cfg.anchor.height
            z_center_a = self.crop_cfg.anchor.center

            if self.z_mode == "top":
                z_top_a = z_center_a + height_a / 2
                gt = z_top_a + zt - (height_a + ht) - gp
                z_loss = smooth_l1_loss(zt, zg, 3.0) / batch_size_dev
                # IoU Loss
                yg_top, yg_down = zg + z_top_a, zg + z_top_a - (hg + height_a)
                yp_top, yp_down = zt + z_top_a, zt + z_top_a - (ht + height_a)
            elif self.z_mode == "center":
                gt = z_center_a + zt - (height_a + ht) / 2.0 - gp
                z_loss = smooth_l1_loss(zt, zg, 3.0) / batch_size_dev
                # IoU Loss
                yg_top, yg_down = (
                    zg + z_center_a + (hg + height_a) / 2.0,
                    zg + z_center_a - (hg + height_a) / 2.0,
                )
                yp_top, yp_down = (
                    zt + z_center_a + (ht + height_a) / 2.0,
                    zt + z_center_a - (ht + height_a) / 2.0,
                )

            gp_loss = smooth_l1_loss(gt, gg, 3.0) / batch_size_dev

            h_intersect = torch.min(yp_top, yg_top) - torch.max(yp_down, yg_down)
            iou = h_intersect / (hg + height_a + ht + height_a - h_intersect)
            iou[iou < 0] = 0.0
            iou[iou > 1] = 1.0
            iou_loss = (1 - iou).sum() / batch_size_dev

            ret = dict(
                loss=z_loss + height_loss + gp_loss,
                z_loss=z_loss,
                height_loss=height_loss,
                gp_loss=gp_loss,
            )

            if self.iou_loss:
                ret["loss"] += iou_loss
                ret.update(dict(iou_loss=iou_loss,))
            rets.append(ret)
        """convert batch-key to key-batch
        """
        rets_merged = defaultdict(list)
        for ret in rets:
            for k, v in ret.items():
                rets_merged[k].append(v)

        return rets_merged

    def predict(self, example, preds, stage_one_outputs):
        rets = []
        for task_id, task_pred in enumerate(preds):
            if self.z_mode == "top":
                height_a = self.crop_cfg.anchor.height
                z_top_a = self.crop_cfg.anchor.center + height_a / 2
                # z top
                task_pred[:, :, :, 1] += height_a
                task_pred[:, :, :, 0] += z_top_a - task_pred[:, :, :, 1] / 2.0
            elif self.z_mode == "center":
                height_a = self.crop_cfg.anchor.height
                z_center_a = self.crop_cfg.anchor.center
                # regress h and z
                task_pred[:, :, :, 0] += z_center_a
                task_pred[:, :, :, 1] += height_a

            rets.append(task_pred.view(-1, 2))

        # Merge branches results
        num_tasks = len(rets)
        ret_list = []
        # len(rets) == task num
        # len(rets[0]) == batch_size
        num_preds = len(rets)
        num_samples = len(rets[0])

        ret_list = []
        for i in range(num_samples):
            ret = torch.cat(rets)
            ret_list.append(ret)

        cnt = 0
        for idx, sample in enumerate(stage_one_outputs):
            for box in sample["box3d_lidar"]:
                box[2] = ret_list[0][cnt, 0]
                box[5] = ret_list[0][cnt, 1]

                cnt += 1

        return stage_one_outputs


@HEADS.register_module
class MultiGroupHead(nn.Module):
    def __init__(
        self,
        mode="3d",
        in_channels=[128,],
        norm_cfg=None,
        tasks=[],
        weights=[],
        num_classes=[1,],
        box_coder=None,
        with_cls=True,
        with_reg=True,
        reg_class_agnostic=False,
        encode_background_as_zeros=True,
        loss_norm=dict(
            type="NormByNumPositives", pos_class_weight=1.0, neg_class_weight=1.0,
        ),
        loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0,),
        use_sigmoid_score=True,
        loss_bbox=dict(type="SmoothL1Loss", beta=1.0, loss_weight=1.0,),
        encode_rad_error_by_sin=True,
        loss_aux=None,
        direction_offset=0.0,
        name="rpn",
        logger=None,
    ):
        super(MultiGroupHead, self).__init__()

        assert with_cls or with_reg

        num_classes = [len(t["class_names"]) for t in tasks]
        self.class_names = [t["class_names"] for t in tasks]
        self.num_anchor_per_locs = [2 * n for n in num_classes]

        self.box_coder = box_coder
        box_code_sizes = [box_coder.n_dim] * len(num_classes)

        self.with_cls = with_cls
        self.with_reg = with_reg
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.reg_class_agnostic = reg_class_agnostic
        self.encode_rad_error_by_sin = encode_rad_error_by_sin
        self.encode_background_as_zeros = encode_background_as_zeros
        self.use_sigmoid_score = use_sigmoid_score
        self.box_n_dim = self.box_coder.n_dim

        self.loss_cls = build_loss(loss_cls)
        self.loss_reg = build_loss(loss_bbox)
        if loss_aux is not None:
            self.loss_aux = build_loss(loss_aux)

        self.loss_norm = loss_norm

        if not logger:
            logger = logging.getLogger("MultiGroupHead")
        self.logger = logger

        self.dcn = None
        self.zero_init_residual = False

        self.use_direction_classifier = loss_aux is not None
        if loss_aux:
            self.direction_offset = direction_offset

        self.bev_only = True if mode == "bev" else False

        num_clss = []
        num_preds = []
        num_dirs = []

        for num_c, num_a, box_cs in zip(
            num_classes, self.num_anchor_per_locs, box_code_sizes
        ):
            if self.encode_background_as_zeros:
                num_cls = num_a * num_c
            else:
                num_cls = num_a * (num_c + 1)
            num_clss.append(num_cls)

            if self.bev_only:
                num_pred = num_a * (box_cs - 2)
            else:
                num_pred = num_a * box_cs
            num_preds.append(num_pred)

            if self.use_direction_classifier:
                num_dir = num_a * 2
                num_dirs.append(num_dir)
            else:
                num_dir = None

        logger.info(
            f"num_classes: {num_classes}, num_preds: {num_preds}, num_dirs: {num_dirs}"
        )

        self.tasks = nn.ModuleList()
        for task_id, (num_pred, num_cls) in enumerate(zip(num_preds, num_clss)):
            self.tasks.append(
                Head(
                    in_channels,
                    num_pred,
                    num_cls,
                    use_dir=self.use_direction_classifier,
                    num_dir=num_dirs[task_id]
                    if self.use_direction_classifier
                    else None,
                    header=False,
                )
            )

        logger.info("Finish MultiGroupHead Initialization")

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

            if self.dcn is not None:
                for m in self.modules():
                    if isinstance(m, Bottleneck) and hasattr(m, "conv2_offset"):
                        constant_init(m.conv2_offset, 0)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        constant_init(m.norm3, 0)
                    elif isinstance(m, BasicBlock):
                        constant_init(m.norm2, 0)
        else:
            raise TypeError("pretrained must be a str or None")

    def forward(self, x):
        ret_dicts = []
        for task in self.tasks:
            ret_dicts.append(task(x))

        return ret_dicts

    def prepare_loss_weights(
        self,
        labels,
        loss_norm=dict(
            type="NormByNumPositives", pos_cls_weight=1.0, neg_cls_weight=1.0,
        ),
        dtype=torch.float32,
    ):
        loss_norm_type = getattr(LossNormType, loss_norm["type"])
        pos_cls_weight = loss_norm["pos_cls_weight"]
        neg_cls_weight = loss_norm["neg_cls_weight"]

        cared = labels >= 0
        # cared: [N, num_anchors]
        positives = labels > 0
        negatives = labels == 0
        negative_cls_weights = negatives.type(dtype) * neg_cls_weight
        cls_weights = negative_cls_weights + pos_cls_weight * positives.type(dtype)
        reg_weights = positives.type(dtype)
        if loss_norm_type == LossNormType.NormByNumExamples:
            num_examples = cared.type(dtype).sum(1, keepdim=True)
            num_examples = torch.clamp(num_examples, min=1.0)
            cls_weights /= num_examples
            bbox_normalizer = positives.sum(1, keepdim=True).type(dtype)
            reg_weights /= torch.clamp(bbox_normalizer, min=1.0)
        elif loss_norm_type == LossNormType.NormByNumPositives:
            pos_normalizer = positives.sum(1, keepdim=True).type(dtype)
            reg_weights /= torch.clamp(pos_normalizer, min=1.0)
            cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        elif loss_norm_type == LossNormType.NormByNumPosNeg:
            pos_neg = torch.stack([positives, negatives], dim=-1).type(dtype)
            normalizer = pos_neg.sum(1, keepdim=True)  # [N, 1, 2]
            cls_normalizer = (pos_neg * normalizer).sum(-1)  # [N, M]
            cls_normalizer = torch.clamp(cls_normalizer, min=1.0)
            # cls_normalizer will be pos_or_neg_weight/num_pos_or_neg
            normalizer = torch.clamp(normalizer, min=1.0)
            reg_weights /= normalizer[:, 0:1, 0]
            cls_weights /= cls_normalizer
        elif loss_norm_type == LossNormType.DontNorm:  # support ghm loss
            pos_normalizer = positives.sum(1, keepdim=True).type(dtype)
            reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        else:
            raise ValueError(f"unknown loss norm type. available: {list(LossNormType)}")
        return cls_weights, reg_weights, cared

    def loss(self, example, preds_dicts, **kwargs):

        voxels = example["voxels"]
        num_points = example["num_points"]
        coors = example["coordinates"]
        batch_anchors = example["anchors"]
        batch_size_device = batch_anchors[0].shape[0]

        rets = []
        for task_id, preds_dict in enumerate(preds_dicts):
            losses = dict()

            num_class = self.num_classes[task_id]

            box_preds = preds_dict["box_preds"]
            cls_preds = preds_dict["cls_preds"]

            labels = example["labels"][task_id]
            if kwargs.get("mode", False):
                reg_targets = example["reg_targets"][task_id][:, :, [0, 1, 3, 4, 6]]
                reg_targets_left = example["reg_targets"][task_id][:, :, [2, 5]]
            else:
                reg_targets = example["reg_targets"][task_id]

            cls_weights, reg_weights, cared = self.prepare_loss_weights(
                labels, loss_norm=self.loss_norm, dtype=torch.float32,
            )
            cls_targets = labels * cared.type_as(labels)
            cls_targets = cls_targets.unsqueeze(-1)

            loc_loss, cls_loss = create_loss(
                self.loss_reg,
                self.loss_cls,
                box_preds,
                cls_preds,
                cls_targets,
                cls_weights,
                reg_targets,
                reg_weights,
                num_class,
                self.encode_background_as_zeros,
                self.encode_rad_error_by_sin,
                bev_only=self.bev_only,
                box_code_size=self.box_n_dim,
            )

            loc_loss_reduced = loc_loss.sum() / batch_size_device
            loc_loss_reduced *= self.loss_reg._loss_weight
            cls_pos_loss, cls_neg_loss = _get_pos_neg_loss(cls_loss, labels)
            cls_pos_loss /= self.loss_norm["pos_cls_weight"]
            cls_neg_loss /= self.loss_norm["neg_cls_weight"]
            cls_loss_reduced = cls_loss.sum() / batch_size_device
            cls_loss_reduced *= self.loss_cls._loss_weight

            loss = loc_loss_reduced + cls_loss_reduced

            if self.use_direction_classifier:
                dir_targets = get_direction_target(
                    example["anchors"][task_id],
                    reg_targets,
                    dir_offset=self.direction_offset,
                )
                dir_logits = preds_dict["dir_cls_preds"].view(batch_size_device, -1, 2)
                weights = (labels > 0).type_as(dir_logits)
                weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
                dir_loss = self.loss_aux(dir_logits, dir_targets, weights=weights)
                dir_loss = dir_loss.sum() / batch_size_device
                loss += dir_loss * self.loss_aux._loss_weight

                # losses['loss_aux'] = dir_loss

            loc_loss_elem = [
                loc_loss[:, :, i].sum() / batch_size_device
                for i in range(loc_loss.shape[-1])
            ]
            ret = {
                "loss": loss,
                "cls_pos_loss": cls_pos_loss.detach().cpu(),
                "cls_neg_loss": cls_neg_loss.detach().cpu(),
                "dir_loss_reduced": dir_loss.detach().cpu()
                if self.use_direction_classifier
                else None,
                "cls_loss_reduced": cls_loss_reduced.detach().cpu().mean(),
                "loc_loss_reduced": loc_loss_reduced.detach().cpu().mean(),
                "loc_loss_elem": [elem.detach().cpu() for elem in loc_loss_elem],
                "num_pos": (labels > 0)[0].sum(),
                "num_neg": (labels == 0)[0].sum(),
            }

            # self.rpn_acc.clear()
            # losses['acc'] = self.rpn_acc(
            #     example['labels'][task_id],
            #     cls_preds,
            #     cared,
            # )

            # losses['pr'] = {}
            # self.rpn_pr.clear()
            # prec, rec = self.rpn_pr(
            #     example['labels'][task_id],
            #     cls_preds,
            #     cared,
            # )
            # for i, thresh in enumerate(self.rpn_pr.thresholds):
            #     losses["pr"][f"prec@{int(thresh*100)}"] = float(prec[i])
            #     losses["pr"][f"rec@{int(thresh*100)}"] = float(rec[i])

            rets.append(ret)
        """convert batch-key to key-batch
        """
        rets_merged = defaultdict(list)
        for ret in rets:
            for k, v in ret.items():
                rets_merged[k].append(v)

        return rets_merged

    def predict(self, example, preds_dicts, test_cfg, **kwargs):
        """start with v1.6.0, this function don't contain any kitti-specific code.
        Returns:
            predict: list of pred_dict.
            pred_dict: {
                box3d_lidar: [N, 7] 3d box.
                scores: [N]
                label_preds: [N]
                metadata: meta-data which contains dataset-specific information.
                    for kitti, it contains image idx (label idx),
                    for nuscenes, sample_token is saved in it.
            }
        """
        voxels = example["voxels"]
        num_points = example["num_points"]
        coors = example["coordinates"]
        batch_anchors = example["anchors"]
        batch_size_device = batch_anchors[0].shape[0]
        rets = []
        for task_id, preds_dict in enumerate(preds_dicts):
            batch_size = batch_anchors[task_id].shape[0]

            if "metadata" not in example or len(example["metadata"]) == 0:
                meta_list = [None] * batch_size
            else:
                meta_list = example["metadata"]

            batch_task_anchors = example["anchors"][task_id].view(
                batch_size, -1, self.box_n_dim
            )

            if "anchors_mask" not in example:
                batch_anchors_mask = [None] * batch_size
            else:
                batch_anchors_mask = example["anchors_mask"][task_id].view(
                    batch_size, -1
                )

            batch_box_preds = preds_dict["box_preds"]
            batch_cls_preds = preds_dict["cls_preds"]

            if self.bev_only:
                box_ndim = self.box_n_dim - 2
            else:
                box_ndim = self.box_n_dim

            if kwargs.get("mode", False):
                batch_box_preds_base = batch_box_preds.view(batch_size, -1, box_ndim)
                batch_box_preds = batch_task_anchors.clone()
                batch_box_preds[:, :, [0, 1, 3, 4, 6]] = batch_box_preds_base
            else:
                batch_box_preds = batch_box_preds.view(batch_size, -1, box_ndim)

            num_class_with_bg = self.num_classes[task_id]

            if not self.encode_background_as_zeros:
                num_class_with_bg = self.num_classes[task_id] + 1

            batch_cls_preds = batch_cls_preds.view(batch_size, -1, num_class_with_bg)

            batch_reg_preds = self.box_coder.decode_torch(
                batch_box_preds[:, :, : self.box_coder.code_size], batch_task_anchors
            )

            if self.use_direction_classifier:
                batch_dir_preds = preds_dict["dir_cls_preds"]
                batch_dir_preds = batch_dir_preds.view(batch_size, -1, 2)
            else:
                batch_dir_preds = [None] * batch_size
            rets.append(
                self.get_task_detections(
                    task_id,
                    num_class_with_bg,
                    test_cfg,
                    batch_cls_preds,
                    batch_reg_preds,
                    batch_dir_preds,
                    batch_anchors_mask,
                    meta_list,
                )
            )
        # Merge branches results
        num_tasks = len(rets)
        ret_list = []
        # len(rets) == task num
        # len(rets[0]) == batch_size
        num_preds = len(rets)
        num_samples = len(rets[0])

        ret_list = []
        for i in range(num_samples):
            ret = {}
            for k in rets[0][i].keys():
                if k in ["box3d_lidar", "scores"]:
                    ret[k] = torch.cat([ret[i][k] for ret in rets])
                elif k in ["label_preds"]:
                    flag = 0
                    for j, num_class in enumerate(self.num_classes):
                        rets[j][i][k] += flag
                        flag += num_class
                    ret[k] = torch.cat([ret[i][k] for ret in rets])
                elif k == "metadata":
                    # metadata
                    ret[k] = rets[0][i][k]
            ret_list.append(ret)

        return ret_list

    def get_task_detections(
        self,
        task_id,
        num_class_with_bg,
        test_cfg,
        batch_cls_preds,
        batch_reg_preds,
        batch_dir_preds=None,
        batch_anchors_mask=None,
        meta_list=None,
    ):
        predictions_dicts = []
        post_center_range = test_cfg.post_center_limit_range
        if len(post_center_range) > 0:
            post_center_range = torch.tensor(
                post_center_range,
                dtype=batch_reg_preds.dtype,
                device=batch_reg_preds.device,
            )

        for box_preds, cls_preds, dir_preds, a_mask, meta in zip(
            batch_reg_preds,
            batch_cls_preds,
            batch_dir_preds,
            batch_anchors_mask,
            meta_list,
        ):
            if a_mask is not None:
                box_preds = box_preds[a_mask]
                cls_preds = cls_preds[a_mask]

            box_preds = box_preds.float()
            cls_preds = cls_preds.float()

            if self.use_direction_classifier:
                if a_mask is not None:
                    dir_preds = dir_preds[a_mask]
                dir_labels = torch.max(dir_preds, dim=-1)[1]

            if self.encode_background_as_zeros:
                # this don't support softmax
                assert self.use_sigmoid_score is True
                total_scores = torch.sigmoid(cls_preds)
            else:
                # encode background as first element in one-hot vector
                if self.use_sigmoid_score:
                    total_scores = torch.sigmoid(cls_preds)[..., 1:]
                else:
                    total_scores = F.softmax(cls_preds, dim=-1)[..., 1:]

            # Apply NMS in birdeye view
            if test_cfg.nms.use_rotate_nms:
                nms_func = box_torch_ops.rotate_nms
            else:
                nms_func = box_torch_ops.nms

            feature_map_size_prod = (
                batch_reg_preds.shape[1] // self.num_anchor_per_locs[task_id]
            )

            if test_cfg.nms.use_multi_class_nms:
                assert self.encode_background_as_zeros is True
                boxes_for_nms = box_preds[:, [0, 1, 3, 4, -1]]
                if not test_cfg.nms.use_rotate_nms:
                    box_preds_corners = box_torch_ops.center_to_corner_box2d(
                        boxes_for_nms[:, :2], boxes_for_nms[:, 2:4], boxes_for_nms[:, 4]
                    )
                    boxes_for_nms = box_torch_ops.corner_to_standup_nd(
                        box_preds_corners
                    )

                selected_boxes, selected_labels, selected_scores = [], [], []
                selected_dir_labels = []

                scores = total_scores
                boxes = boxes_for_nms
                selected_per_class = []
                score_threshs = [test_cfg.score_threshold] * self.num_classes[task_id]
                pre_max_sizes = [test_cfg.nms.nms_pre_max_size] * self.num_classes[
                    task_id
                ]
                post_max_sizes = [test_cfg.nms.nms_post_max_size] * self.num_classes[
                    task_id
                ]
                iou_thresholds = [test_cfg.nms.nms_iou_threshold] * self.num_classes[
                    task_id
                ]

                for class_idx, score_thresh, pre_ms, post_ms, iou_th in zip(
                    range(self.num_classes[task_id]),
                    score_threshs,
                    pre_max_sizes,
                    post_max_sizes,
                    iou_thresholds,
                ):
                    self._nms_class_agnostic = False
                    if self._nms_class_agnostic:
                        class_scores = total_scores.view(
                            feature_map_size_prod, -1, self.num_classes[task_id]
                        )[..., class_idx]
                        class_scores = class_scores.contiguous().view(-1)
                        class_boxes_nms = boxes.view(-1, boxes_for_nms.shape[-1])
                        class_boxes = box_preds
                        class_dir_labels = dir_labels
                    else:
                        # anchors_range = self.target_assigner.anchors_range(class_idx)
                        anchors_range = self.target_assigners[task_id].anchors_range
                        class_scores = total_scores.view(
                            -1, self._num_classes[task_id]
                        )[anchors_range[0] : anchors_range[1], class_idx]
                        class_boxes_nms = boxes.view(-1, boxes_for_nms.shape[-1])[
                            anchors_range[0] : anchors_range[1], :
                        ]
                        class_scores = class_scores.contiguous().view(-1)
                        class_boxes_nms = class_boxes_nms.contiguous().view(
                            -1, boxes_for_nms.shape[-1]
                        )
                        class_boxes = box_preds.view(-1, box_preds.shape[-1])[
                            anchors_range[0] : anchors_range[1], :
                        ]
                        class_boxes = class_boxes.contiguous().view(
                            -1, box_preds.shape[-1]
                        )
                        if self.use_direction_classifier:
                            class_dir_labels = dir_labels.view(-1)[
                                anchors_range[0] : anchors_range[1]
                            ]
                            class_dir_labels = class_dir_labels.contiguous().view(-1)
                    if score_thresh > 0.0:
                        class_scores_keep = class_scores >= score_thresh
                        if class_scores_keep.shape[0] == 0:
                            selected_per_class.append(None)
                            continue
                        class_scores = class_scores[class_scores_keep]
                    if class_scores.shape[0] != 0:
                        if score_thresh > 0.0:
                            class_boxes_nms = class_boxes_nms[class_scores_keep]
                            class_boxes = class_boxes[class_scores_keep]
                            class_dir_labels = class_dir_labels[class_scores_keep]
                        keep = nms_func(
                            class_boxes_nms, class_scores, pre_ms, post_ms, iou_th
                        )
                        if keep.shape[0] != 0:
                            selected_per_class.append(keep)
                        else:
                            selected_per_class.append(None)
                    else:
                        selected_per_class.append(None)
                    selected = selected_per_class[-1]

                    if selected is not None:
                        selected_boxes.append(class_boxes[selected])
                        selected_labels.append(
                            torch.full(
                                [class_boxes[selected].shape[0]],
                                class_idx,
                                dtype=torch.int64,
                                device=box_preds.device,
                            )
                        )
                        if self.use_direction_classifier:
                            selected_dir_labels.append(class_dir_labels[selected])
                        selected_scores.append(class_scores[selected])
                    # else:
                    #     selected_boxes.append(torch.Tensor([], device=class_boxes.device))
                    #     selected_labels.append(torch.Tensor([], device=box_preds.device))
                    #     selected_scores.append(torch.Tensor([], device=class_scores.device))
                    #     if self.use_direction_classifier:
                    #         selected_dir_labels.append(torch.Tensor([], device=class_dir_labels.device))

                selected_boxes = torch.cat(selected_boxes, dim=0)
                selected_labels = torch.cat(selected_labels, dim=0)
                selected_scores = torch.cat(selected_scores, dim=0)
                if self.use_direction_classifier:
                    selected_dir_labels = torch.cat(selected_dir_labels, dim=0)

            else:
                # get highest score per prediction, than apply nms
                # to remove overlapped box.
                if num_class_with_bg == 1:
                    top_scores = total_scores.squeeze(-1)
                    top_labels = torch.zeros(
                        total_scores.shape[0],
                        device=total_scores.device,
                        dtype=torch.long,
                    )

                else:
                    top_scores, top_labels = torch.max(total_scores, dim=-1)

                if test_cfg.score_threshold > 0.0:
                    thresh = torch.tensor(
                        [test_cfg.score_threshold], device=total_scores.device
                    ).type_as(total_scores)
                    top_scores_keep = top_scores >= thresh
                    top_scores = top_scores.masked_select(top_scores_keep)

                if top_scores.shape[0] != 0:
                    if test_cfg.score_threshold > 0.0:
                        box_preds = box_preds[top_scores_keep]
                        if self.use_direction_classifier:
                            dir_labels = dir_labels[top_scores_keep]
                        top_labels = top_labels[top_scores_keep]
                    boxes_for_nms = box_preds[:, [0, 1, 3, 4, -1]]
                    if not test_cfg.nms.use_rotate_nms:
                        box_preds_corners = box_torch_ops.center_to_corner_box2d(
                            boxes_for_nms[:, :2],
                            boxes_for_nms[:, 2:4],
                            boxes_for_nms[:, 4],
                        )
                        boxes_for_nms = box_torch_ops.corner_to_standup_nd(
                            box_preds_corners
                        )
                    # the nms in 3d detection just remove overlap boxes.
                    selected = nms_func(
                        boxes_for_nms,
                        top_scores,
                        pre_max_size=test_cfg.nms.nms_pre_max_size,
                        post_max_size=test_cfg.nms.nms_post_max_size,
                        iou_threshold=test_cfg.nms.nms_iou_threshold,
                    )
                else:
                    selected = []
                # if selected is not None:
                selected_boxes = box_preds[selected]
                if self.use_direction_classifier:
                    selected_dir_labels = dir_labels[selected]
                selected_labels = top_labels[selected]
                selected_scores = top_scores[selected]

            # finally generate predictions.
            # self.logger.info(f"selected boxes: {selected_boxes.shape}")
            if selected_boxes.shape[0] != 0:
                # self.logger.info(f"result not none~ Selected boxes: {selected_boxes.shape}")
                box_preds = selected_boxes
                scores = selected_scores
                label_preds = selected_labels
                if self.use_direction_classifier:
                    dir_labels = selected_dir_labels
                    opp_labels = (
                        (box_preds[..., -1] - self.direction_offset) > 0
                    ) ^ dir_labels.byte()
                    box_preds[..., -1] += torch.where(
                        opp_labels,
                        torch.tensor(np.pi).type_as(box_preds),
                        torch.tensor(0.0).type_as(box_preds),
                    )
                final_box_preds = box_preds
                final_scores = scores
                final_labels = label_preds
                if post_center_range is not None:
                    mask = (final_box_preds[:, :3] >= post_center_range[:3]).all(1)
                    mask &= (final_box_preds[:, :3] <= post_center_range[3:]).all(1)
                    predictions_dict = {
                        "box3d_lidar": final_box_preds[mask],
                        "scores": final_scores[mask],
                        "label_preds": label_preds[mask],
                        "metadata": meta,
                    }
                else:
                    predictions_dict = {
                        "box3d_lidar": final_box_preds,
                        "scores": final_scores,
                        "label_preds": label_preds,
                        "metadata": meta,
                    }
            else:
                dtype = batch_reg_preds.dtype
                device = batch_reg_preds.device
                predictions_dict = {
                    "box3d_lidar": torch.zeros([0, 7], dtype=dtype, device=device),
                    "scores": torch.zeros([0], dtype=dtype, device=device),
                    "label_preds": torch.zeros(
                        [0], dtype=top_labels.dtype, device=device
                    ),
                    "metadata": meta,
                }

            predictions_dicts.append(predictions_dict)

        return predictions_dicts
