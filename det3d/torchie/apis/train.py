from __future__ import division

import re
from collections import OrderedDict, defaultdict
from functools import partial

import apex
import numpy as np
import torch
from det3d.builder import build_optimizer as build_fastai_optimizer
from det3d.builder import _create_learning_rate_scheduler

# from det3d.datasets.kitti.eval_hooks import KittiDistEvalmAPHook, KittiEvalmAPHookV2
from det3d.core import DistOptimizerHook
from det3d.datasets import DATASETS, build_dataloader
from det3d.solver.fastai_optim import OptimWrapper
from det3d.torchie.trainer import DistSamplerSeedHook, Trainer, obj_from_dict
from det3d.utils.print_utils import metric_to_str
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from .env import get_root_logger


def example_convert_to_torch(example, dtype=torch.float32, device=None) -> dict:
    assert device is not None

    example_torch = {}
    float_names = ["voxels", "bev_map"]
    for k, v in example.items():
        if k in ["anchors", "reg_targets", "reg_weights"]:
            res = []
            for kk, vv in v.items():
                res.append(torch.tensor(vv).to(device, non_blocking=True))
                # vv = np.array(vv)
                # res.append(torch.tensor(vv, dtype=torch.float32,
                #                         device=device))
            example_torch[k] = res
        elif k in float_names:
            # slow when directly provide fp32 data with dtype=torch.half
            example_torch[k] = v.to(device, non_blocking=True)
            # example_torch[k] = torch.tensor(v,
            #                                 dtype=torch.float32,
            #                                 device=device)
        elif k in ["coordinates", "num_points"]:
            example_torch[k] = v.to(device, non_blocking=True)
            # example_torch[k] = torch.tensor(v,
            #                                 dtype=torch.int32,
            #                                 device=device)
        elif k == "labels":
            res = []
            for kk, vv in v.items():
                # vv = np.array(vv)
                res.append(torch.tensor(vv).to(device, non_blocking=True))
            example_torch[k] = res
        elif k == "points":
            example_torch[k] = v.to(device, non_blocking=True)
            # example_torch[k] = torch.tensor(v,
            #                                 dtype=torch.float,
            #                                 device=device)
        elif k in ["anchors_mask"]:
            res = []
            for kk, vv in v.items():
                res.append(torch.tensor(vv).to(device, non_blocking=True))
            example_torch[k] = res
        elif k == "calib":
            calib = {}
            for k1, v1 in v.items():
                # calib[k1] = torch.tensor(v1, dtype=dtype, device=device)
                calib[k1] = torch.tensor(v1).to(device, non_blocking=True)
            example_torch[k] = calib
        elif k == "num_voxels":
            example_torch[k] = v.to(device, non_blocking=True)
            # example_torch[k] = torch.tensor(v,
            #                                 dtype=torch.int64,
            #                                 device=device)
        else:
            example_torch[k] = v

    return example_torch


def example_to_device(example, device=None, non_blocking=False) -> dict:
    assert device is not None

    example_torch = {}
    float_names = ["voxels", "bev_map"]
    for k, v in example.items():
        if k in ["anchors", "anchors_mask", "reg_targets", "reg_weights", "labels"]:
            example_torch[k] = [res.to(device, non_blocking=non_blocking) for res in v]
        elif k in [
            "voxels",
            "bev_map",
            "coordinates",
            "num_points",
            "points",
            "num_voxels",
        ]:
            example_torch[k] = v.to(device, non_blocking=non_blocking)
        elif k == "calib":
            calib = {}
            for k1, v1 in v.items():
                # calib[k1] = torch.tensor(v1, dtype=dtype, device=device)
                calib[k1] = torch.tensor(v1).to(device, non_blocking=non_blocking)
            example_torch[k] = calib
        else:
            example_torch[k] = v

    return example_torch


def parse_losses(losses):
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError("{} is not a tensor or list of tensors".format(loss_name))

    loss = sum(_value for _key, _value in log_vars.items() if "loss" in _key)

    log_vars["loss"] = loss
    for name in log_vars:
        log_vars[name] = log_vars[name].item()

    return loss, log_vars


def parse_second_losses(losses):

    log_vars = OrderedDict()
    loss = sum(losses["loss"])
    for loss_name, loss_value in losses.items():
        if loss_name == "loc_loss_elem":
            log_vars[loss_name] = [[i.item() for i in j] for j in loss_value]
        else:
            log_vars[loss_name] = [i.item() for i in loss_value]

    return loss, log_vars


def batch_processor(model, data, train_mode, **kwargs):

    if "local_rank" in kwargs:
        device = torch.device(kwargs["local_rank"])
    else:
        device = None

    # data = example_convert_to_torch(data, device=device)
    example = example_to_device(data, device, non_blocking=False)

    del data

    if train_mode:
        losses = model(example, return_loss=True)
        loss, log_vars = parse_second_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(example["anchors"][0])
        )
        return outputs
    else:
        return model(example, return_loss=False)


def flatten_model(m):
    return sum(map(flatten_model, m.children()), []) if len(list(m.children())) else [m]


def get_layer_groups(m):
    return [nn.Sequential(*flatten_model(m))]


def build_optimizer(model, optimizer_cfg):
    """Build optimizer from configs.
    Args:
        model (:obj:`nn.Module`): The model with parameters to be optimized.
        optimizer_cfg (dict): The config dict of the optimizer.
            Positional fields are:
                - type: class name of the optimizer.
                - lr: base learning rate.
            Optional fields are:
                - any arguments of the corresponding optimizer type, e.g.,
                  weight_decay, momentum, etc.
                - paramwise_options: a dict with 3 accepted fileds
                  (bias_lr_mult, bias_decay_mult, norm_decay_mult).
                  `bias_lr_mult` and `bias_decay_mult` will be multiplied to
                  the lr and weight decay respectively for all bias parameters
                  (except for the normalization layers), and
                  `norm_decay_mult` will be multiplied to the weight decay
                  for all weight and bias parameters of normalization layers.
    Returns:
        torch.optim.Optimizer: The initialized optimizer.
    """
    if hasattr(model, "module"):
        model = model.module

    optimizer_cfg = optimizer_cfg.copy()
    paramwise_options = optimizer_cfg.pop("paramwise_options", None)
    # if no paramwise option is specified, just use the global setting
    if paramwise_options is None:
        return obj_from_dict(
            optimizer_cfg, torch.optim, dict(params=model.parameters())
        )
    else:
        assert isinstance(paramwise_options, dict)
        # get base lr and weight decay
        base_lr = optimizer_cfg["lr"]
        base_wd = optimizer_cfg.get("weight_decay", None)
        # weight_decay must be explicitly specified if mult is specified
        if (
            "bias_decay_mult" in paramwise_options
            or "norm_decay_mult" in paramwise_options
        ):
            assert base_wd is not None
        # get param-wise options
        bias_lr_mult = paramwise_options.get("bias_lr_mult", 1.0)
        bias_decay_mult = paramwise_options.get("bias_decay_mult", 1.0)
        norm_decay_mult = paramwise_options.get("norm_decay_mult", 1.0)
        # set param-wise lr and weight decay
        params = []
        for name, param in model.named_parameters():
            param_group = {"params": [param]}
            if not param.requires_grad:
                # FP16 training needs to copy gradient/weight between master
                # weight copy and model weight, it is convenient to keep all
                # parameters here to align with model.parameters()
                params.append(param_group)
                continue

            # for norm layers, overwrite the weight decay of weight and bias
            # TODO: obtain the norm layer prefixes dynamically
            if re.search(r"(bn|gn)(\d+)?.(weight|bias)", name):
                if base_wd is not None:
                    param_group["weight_decay"] = base_wd * norm_decay_mult
            # for other layers, overwrite both lr and weight decay of bias
            elif name.endswith(".bias"):
                param_group["lr"] = base_lr * bias_lr_mult
                if base_wd is not None:
                    param_group["weight_decay"] = base_wd * bias_decay_mult
            # otherwise use the global settings

            params.append(param_group)

        optimizer_cls = getattr(torch.optim, optimizer_cfg.pop("type"))
        return optimizer_cls(params, **optimizer_cfg)


def train_detector(model, dataset, cfg, distributed=False, validate=False, logger=None):
    if logger is None:
        logger = get_root_logger(cfg.log_level)

    # start training
    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    data_loaders = [
        build_dataloader(
            ds, cfg.data.samples_per_gpu, cfg.data.workers_per_gpu, dist=distributed
        )
        for ds in dataset
    ]

    total_steps = cfg.total_epochs * len(data_loaders[0])
    # print(f"total_steps: {total_steps}")

    if hasattr(cfg.optimizer, 'TYPE'):
        assert cfg.optimizer.TYPE in ('rms_prop', 'momentum', 'adam')
        optimizer = build_fastai_optimizer(cfg.optimizer, model)
    else:
        assert hasattr(cfg.optimizer, 'type')
        optimizer = build_optimizer(model, cfg.optimizer)
    if hasattr(cfg.lr_config, 'type'):
        lr_scheduler = _create_learning_rate_scheduler(
            optimizer, cfg.lr_config, total_steps
        )
        cfg.lr_config = None
    else:
        lr_scheduler = None

    # put model on gpus
    if distributed:
        model = apex.parallel.convert_syncbn_model(model)
        model = DistributedDataParallel(
            model.cuda(cfg.local_rank),
            device_ids=[cfg.local_rank],
            output_device=cfg.local_rank,
            # broadcast_buffers=False,
            find_unused_parameters=True,
        )
    else:
        model = model.cuda()

    logger.info(f"model structure: {model}")

    trainer = Trainer(
        model, batch_processor, optimizer, lr_scheduler, cfg.work_dir, cfg.log_level
    )

    if distributed:
        optimizer_config = DistOptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    trainer.register_training_hooks(
        cfg.lr_config, optimizer_config, cfg.checkpoint_config, cfg.log_config
    )

    if distributed:
        trainer.register_hook(DistSamplerSeedHook())

    # # register eval hooks
    # if validate:
    #     val_dataset_cfg = cfg.data.val
    #     eval_cfg = cfg.get('evaluation', {})
    #     dataset_type = DATASETS.get(val_dataset_cfg.type)
    #     trainer.register_hook(
    #         KittiEvalmAPHookV2(val_dataset_cfg, **eval_cfg))

    if cfg.resume_from:
        trainer.resume(cfg.resume_from)
    elif cfg.load_from:
        trainer.load_checkpoint(cfg.load_from)

    trainer.run(data_loaders, cfg.workflow, cfg.total_epochs, local_rank=cfg.local_rank)
