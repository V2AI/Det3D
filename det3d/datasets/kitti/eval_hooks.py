import os
import os.path as osp

import numpy as np
import torch
import torch.distributed as dist
from det3d import datasets, torchie
from det3d.torchie.parallel import collate_kitti, scatter
from det3d.torchie.trainer import Hook
from det3d.utils.dist.dist_common import (
    all_gather,
    get_rank,
    get_world_size,
    is_main_process,
    synchronize,
)
from pycocotools.cocoeval import COCOeval
from torch.utils.data import Dataset

from det3d.core import fast_eval_recall, results2json, eval_map


class KittiDistEvalmAPHook(Hook):
    def __init__(self, dataset, interval=1):
        if isinstance(dataset, Dataset):
            self.dataset = dataset
        elif isinstance(dataset, dict):
            self.dataset = datasets.build_dataset(dataset, {"test_mode": True})
        else:
            raise TypeError(
                "dataset must be a Dataset object or a dict, not {}".format(
                    type(dataset)
                )
            )
        self.interval = interval

    def after_train_epoch(self, trainer):
        if not self.every_n_epochs(trainer, self.interval):
            return
        trainer.model.eval()
        results = [None for _ in range(len(self.dataset))]
        detections = {}
        if trainer.rank == 0:
            prog_bar = torchie.ProgressBar(len(self.dataset))
        for idx in range(trainer.rank, len(self.dataset), trainer.world_size):
            data = self.dataset[idx]
            data_gpu = scatter(
                collate_kitti([data], samples_per_gpu=1), [torch.cuda.current_device()]
            )[0]

            # compute output
            with torch.no_grad():
                output = trainer.model(data_gpu, return_loss=False)

                token = output["metadata"]["token"]
                for k, v in output.items():
                    if k not in [
                        "metadata",
                    ]:
                        output[k] = v.to(cpu_device)
                detections.update(
                    {token: output,}
                )

            detections[idx] = result

            batch_size = trainer.world_size
            if trainer.rank == 0:
                for _ in range(batch_size):
                    prog_bar.update()

        all_predictions = all_gather(detections)

        if trainer.rank != 0:
            return

        predictions = {}
        for p in all_predictions:
            predictions.update(p)

        result_dict, _ = self.dataset.evaluation(predictions, None)

        for k, v in result_dict["results"].items():
            print(f"Evaluation {k}: {v}")


class KittiEvalmAPHookV2(Hook):
    def __init__(self, dataset, interval=1):
        if isinstance(dataset, Dataset):
            self.dataset = dataset
        elif isinstance(dataset, dict):
            self.dataset = datasets.build_dataset(dataset, {"test_mode": True})
        else:
            raise TypeError(
                "dataset must be a Dataset object or a dict, not {}".format(
                    type(dataset)
                )
            )
        self.interval = interval

    def after_train_epoch(self, trainer):
        if not self.every_n_epochs(trainer, self.interval):
            return
        """
        prepare dataloader
        """
        data_loader = datasets.loader.build_dataloader(
            self.dataset, batch_size=8, workers_per_gpu=4, dist=False, shuffle=False
        )

        trainer.model.eval()
        detections = {}

        prog_bar = torchie.ProgressBar(len(data_loader.dataset))

        cpu_device = torch.device("cpu")
        for i, batch in enumerate(data_loader):
            with torch.no_grad():
                outputs = trainer.batch_processor(
                    trainer.model, batch, train_mode=False, local_rank=trainer.rank,
                )
            for output in outputs:
                token = output["metadata"]["token"]
                for k, v in output.items():
                    if k not in [
                        "metadata",
                    ]:
                        output[k] = v.to(cpu_device)
                detections.update(
                    {token: output,}
                )
                prog_bar.update()

        result_dict, _ = self.dataset.evaluation(detections, None)

        for k, v in result_dict["results"].items():
            print(f"Evaluation {k}: {v}")
