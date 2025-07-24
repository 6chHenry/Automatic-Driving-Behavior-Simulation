#!/bin/sh
export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2

DATA_SPLIT=validation # training, validation, testing


source /data0/miniconda3/etc/profile.d/conda.sh
conda activate catk
python \
  -m src.data_preprocess \
  --split $DATA_SPLIT \
  --num_workers 12 \
  --input_dir /data0/datasets/waymo_open_dataset_motion_v_1_3_0 \
  --output_dir ./scratch/cache/SMART