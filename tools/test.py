import argparse
import logging
import os
import os.path as osp
import shutil
import tempfile

import torch
import torch.distributed as dist
from det3d import torchie
from det3d.core import coco_eval, results2json
from det3d.datasets import build_dataloader, build_dataset
from det3d.datasets.kitti import kitti_common as kitti
from det3d.datasets.kitti.eval import get_official_eval_result
from det3d.datasets.utils.kitti_object_eval_python.evaluate import (
    evaluate as kitti_evaluate,
)
from det3d.models import build_detector
from det3d.torchie.apis import init_dist
from det3d.torchie.apis.train import example_convert_to_torch
from det3d.torchie.parallel import MegDataParallel, MegDistributedDataParallel
from det3d.torchie.trainer import get_dist_info, load_checkpoint
from det3d.torchie.trainer.trainer import example_to_device
from det3d.utils.dist.dist_common import (
    all_gather,
    get_rank,
    get_world_size,
    is_main_process,
    synchronize,
)
from tqdm import tqdm


def test(
    dataloader, model, save_dir="", device="cuda", distributed=False,
):

    if distributed:
        model = model.module

    dataset = dataloader.dataset

    device = torch.device(device)
    num_devices = get_world_size()

    detections = compute_on_dataset(model, dataloader, device)
    synchronize()

    predictions = _accumulate_predictions_from_multiple_gpus(detections)

    if not is_main_process():
        return

    return dataset.evaluation(predictions, str(save_dir))


def compute_on_dataset(model, data_loader, device, timer=None, show=False):
    model.eval()
    results_dict = []
    cpu_device = torch.device("cpu")

    results_dict = {}
    prog_bar = torchie.ProgressBar(len(data_loader.dataset))
    for i, batch in enumerate(data_loader):
        # example = example_convert_to_torch(batch, device=device)
        example = example_to_device(batch, device=device)
        with torch.no_grad():
            outputs = model(example, return_loss=False, rescale=not show)
            for output in outputs:
                token = output["metadata"]["token"]
                for k, v in output.items():
                    if k not in [
                        "metadata",
                    ]:
                        output[k] = v.to(cpu_device)
                results_dict.update(
                    {token: output,}
                )
                prog_bar.update()

    return results_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return

    predictions = {}
    for p in all_predictions:
        predictions.update(p)

    return predictions


def parse_args():
    parser = argparse.ArgumentParser(description="MegDet test detector")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument("--out", help="output result file")
    parser.add_argument(
        "--json_out", help="output result file name without extension", type=str
    )
    parser.add_argument(
        "--eval",
        type=str,
        nargs="+",
        choices=["proposal", "proposal_fast", "bbox", "segm", "keypoints"],
        help="eval types",
    )
    parser.add_argument("--show", action="store_true", help="show results")
    parser.add_argument("--txt_result", action="store_true", help="save txt")
    parser.add_argument("--tmpdir", help="tmp dir for writing some results")
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    assert args.out or args.show or args.json_out, (
        "Please specify at least one operation (save or show the results) "
        'with the argument "--out" or "--show" or "--json_out"'
    )

    if args.out is not None and not args.out.endswith((".pkl", ".pickle")):
        raise ValueError("The output file must be a pkl file.")

    if args.json_out is not None and args.json_out.endswith(".json"):
        args.json_out = args.json_out[:-5]

    cfg = torchie.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    # cfg.model.pretrained = None
    # cfg.data.test.test_mode = True
    cfg.data.val.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == "none":
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    # dataset = build_dataset(cfg.data.test)
    dataset = build_dataset(cfg.data.val)
    data_loader = build_dataloader(
        dataset,
        batch_size=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
    )

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)

    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if "CLASSES" in checkpoint["meta"]:
        model.CLASSES = checkpoint["meta"]["CLASSES"]
    else:
        model.CLASSES = dataset.CLASSES

    model = MegDataParallel(model, device_ids=[0])
    result_dict, detections = test(
        data_loader, model, save_dir=None, distributed=distributed
    )

    for k, v in result_dict["results"].items():
        print(f"Evaluation {k}: {v}")

    rank, _ = get_dist_info()
    if args.out and rank == 0:
        print("\nwriting results to {}".format(args.out))
        torchie.dump(detections, args.out)

    if args.txt_result:
        res_dir = os.path.join(os.getcwd(), "predictions")
        for dt in detections:
            with open(
                os.path.join(res_dir, "%06d.txt" % int(dt["metadata"]["token"])), "w"
            ) as fout:
                lines = kitti.annos_to_kitti_label(dt)
                for line in lines:
                    fout.write(line + "\n")

        ap_result_str, ap_dict = kitti_evaluate(
            "/data/Datasets/KITTI/Kitti/object/training/label_2",
            res_dir,
            label_split_file="/data/Datasets/KITTI/Kitti/ImageSets/val.txt",
            current_class=0,
        )

        print(ap_result_str)


if __name__ == "__main__":
    main()
