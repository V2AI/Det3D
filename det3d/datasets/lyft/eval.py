import numpy as np
import numba

from det3d.datasets.utils.eval import calculate_iou_partly
from det3d.datasets.utils.eval import prepare_data
from det3d.datasets.utils.eval import compute_statistics_jit


def clean_data(gt_anno, dt_anno, current_cls_name, difficulty=None):
    MIN_HEIGHT = [40, 25, 25]
    MAX_OCCLUSION = [0, 1, 2]
    MAX_TRUNCATION = [0.15, 0.3, 0.5]
    dc_bboxes, ignored_gt, ignored_dt = [], [], []
    num_gt = len(gt_anno["name"])
    num_dt = len(dt_anno["name"])
    num_valid_gt = 0
    for i in range(num_gt):
        gt_name = gt_anno["name"][i].lower()
        valid_class = -1
        if gt_name == current_cls_name:
            valid_class = 1
        else:
            valid_class = -1
        ignore = False
        if valid_class == 1 and not ignore:
            ignored_gt.append(0)
            num_valid_gt += 1
        else:
            ignored_gt.append(-1)
    for i in range(num_dt):
        if dt_anno["name"][i] == current_cls_name:
            valid_class = 1
        else:
            valid_class = -1
        if valid_class == 1:
            ignored_dt.append(0)
        else:
            ignored_dt.append(-1)

    return num_valid_gt, ignored_gt, ignored_dt, dc_bboxes


def get_lyft_eval_result(gt_annos, dt_annos, classes):
    metric = 2
    gt_num = len(gt_annos)
    rets = calculate_iou_partly(
        gt_annos, dt_annos, metric, num_parts=70, z_axis=2, z_center=0.5
    )

    overlaps, parted_overlaps, total_dt_num, total_gt_num = rets

    info = np.zeros((10, gt_num, len(classes), 3))
    scores = np.zeros((10, gt_num), dtype=np.float32)
    for m, current_class in enumerate(classes):
        rets = prepare_data(
            gt_annos, dt_annos, current_class, difficulty=None, clean_data=clean_data
        )
        (
            gt_datas_list,
            dt_datas_list,
            ignored_gts,
            ignored_dets,
            dontcares,
            total_dc_num,
            total_num_valid_gt,
        ) = rets
        for k, threshold in enumerate(np.arange(0.5, 1, 0.05)):
            for i in range(gt_num):
                rets = compute_statistics_jit(
                    overlaps[i],
                    gt_datas_list[i],
                    dt_datas_list[i],
                    ignored_gts[i],
                    ignored_dets[i],
                    dontcares,
                    metric,
                    min_overlap=threshold,
                    thresh=0.0,
                    compute_fp=True,
                    compute_aos=False,
                )
                tp, fp, fn, _, _ = rets
                info[k, i, m, :] = tp, fp, fn

    import pdb

    pdb.set_trace()  # XXX BREAKPOINT
    info = info.sum(axis=2)
    for i in range(10):
        for j in range(gt_num):
            tps, fps, fns = info[i, j]
            scores[i, j] = 0 if tps == 0 else tps / (tps + fps + fns)
    final_score = scores.mean()
    print(f"LYFT Evaluation Score: {final_score}")
    return f"LYFT Evaluation Score: {final_score}"
