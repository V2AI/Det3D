#!/bin/bash
TASK_DESC=$1
DATE_WITH_TIME=`date "+%Y%m%d-%H%M%S"`
OUT_DIR=/mnt/lustre/zhubenjin/logs/Det3D_Outputs

NUSC_CBGS_WORK_DIR=$OUT_DIR/NUSC_CBGS_$TASK_DESC\_$DATE_WITH_TIME
LYFT_CBGS_WORK_DIR=$OUT_DIR/LYFT_CBGS_$TASK_DESC\_$DATE_WITH_TIME
SECOND_WORK_DIR=$OUT_DIR/SECOND_$TASK_DESC\_$DATE_WITH_TIME
PP_WORK_DIR=$OUT_DIR/PointPillars_$TASK_DESC\_$DATE_WITH_TIME

if [ ! $TASK_DESC ] 
then
    echo "TASK_DESC must be specified."
    echo "Usage: train.sh task_description"
    exit $E_ASSERT_FAILED
fi

gpu8="srun --partition=vi_irdc --mpi=pmi2 --gres=gpu:8 -n8 --cpus-per-task=8 --ntasks-per-node=8 --job-name=node1gpu8 --kill-on-bad-exit=1"
debug="srun --partition=vi_irdc --mpi=pmi2 --gres=gpu:1 -n1 --cpus-per-task=8 --ntasks-per-node=1 --job-name=debug --kill-on-bad-exit=1"

# Voxelnet
# python -m torch.distributed.launch --nproc_per_node=8 ./tools/train.py examples/second/configs/kitti_car_vfev3_spmiddlefhd_rpn1_mghead_syncbn.py --work_dir=$SECOND_WORK_DIR
# $gpu8 python ./tools/train.py --launcher slurm examples/cbgs/configs/nusc_all_vfev3_spmiddleresnetfhd_rpn2_mghead_syncbn.py --work_dir=$NUSC_CBGS_WORK_DIR
$gpu8 python ./tools/dist_test.py --launcher slurm examples/cbgs/configs/nusc_all_vfev3_spmiddleresnetfhd_rpn2_mghead_syncbn.py --work_dir=/mnt/lustre/zhubenjin/logs/Det3D_Outputs/NUSC_CBGS_nusc_cbgs_baseline_20210808-144525 --checkpoint=/mnt/lustre/zhubenjin/logs/Det3D_Outputs/NUSC_CBGS_nusc_cbgs_baseline_20210808-144525/epoch_10.pth
# $debug python ./tools/train.py examples/cbgs/configs/nusc_all_vfev3_spmiddleresnetfhd_rpn2_mghead_syncbn.py --work_dir=$NUSC_CBGS_WORK_DIR
# python -m torch.distributed.launch --nproc_per_node=8 ./tools/train.py examples/second/configs/lyft_all_vfev3_spmiddleresnetfhd_rpn2_mghead_syncbn.py --work_dir=$LYFT_CBGS_WORK_DIR

# PointPillars
# python -m torch.distributed.launch --nproc_per_node=8 ./tools/train.py ./examples/point_pillars/configs/original_pp_mghead_syncbn_kitti.py --work_dir=$PP_WORK_DIR
