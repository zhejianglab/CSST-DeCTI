#!/bin/bash
DATA_DIR='/mnt/nas/mzh/project/dataset/HST_F814W'
PRED_DIR='/mnt/nas/mzh/project/dataset/HST_F814W_prediction'
LOG_PATH='/mnt/nas/mzh/project/log'
TRAIN=1
PLATEFRORM=1

if [[ $RANK -eq '0' ]]; then
MASTER_ADDR="localhost"
fi

mkdir -p $LOG_PATH/$0

if [[ $PLATEFRORM -eq 1 ]]; then
    CMD="torchrun --nnodes=${WORLD_SIZE} \
            --nproc_per_node=${TQ_GPU_NUM}\
            --rdzv_backend=c10d \
            --rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT} "
else
    CMD="torchrun"
fi


# #######train
if [[ $TRAIN -eq 1 ]]; then
            $CMD \
            main.py \
            --is_training 1 \
            --model 'DeCTIAbla' \
            --data_path ${DATA_DIR} \
            --prediction_path ${PRED_DIR} \
            --redivide_files 0 \
            --seq_len_perchannel 2048 \
            --img_width_perchannel 4096 \
            --half_plane 0 \
            --left_quarter -1 \
            --obs_year 2012 \
            --log_path ${LOG_PATH} \
            --log_sfolder $0 \
            --loaded_chpt_sfolder '' \
            --config_subpath /mnt/nas/mzh/project/DeCTI/config/remove_j92t \
            --num_workers 10 \
            --train_epochs 50 \
            --batch_size 2 \
            --learning_rate 0.0001 \
            --loss 'mse' \
            --pct_start 0.3 \
            --patience 10000 \
            --window_size 64 \
            --abla_rpe 1 \
            --abla_ape 1 \
            --abla_residual 1 \
            --abla_patch_size 1 > "$LOG_PATH/$0/$0.txt"
else
# #######test
            $CMD \
            --is_training 0 \
            --model 'DeCTIAbla' \
            --data_path ${DATA_DIR} \
            --prediction_path ${PRED_DIR} \
            --redivide_files 0 \
            --seq_len_perchannel 2048 \
            --img_width_perchannel 4096 \
            --half_plane 0 \
            --left_quarter 0 \
            --obs_year 2012 \
            --log_path ${LOG_PATH} \
            --log_sfolder $0 \
            --loaded_chpt_sfolder '' \
            --config_subpath config/remove_j92t \
            --num_workers 10 \
            --train_epochs 50 \
            --batch_size 64 \
            --learning_rate 0.0001 \
            --loss 'mse' \
            --pct_start 0.3 \
            --patience 10000 \
            --window_size 64 \
            --abla_rpe 1 \
            --abla_ape 1 \
            --abla_residual 1 \
            --abla_patch_size 1 > "$LOG_PATH/$0/$0 predict.txt"
fi