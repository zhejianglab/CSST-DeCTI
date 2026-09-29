#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${DATA_DIR:?Set DATA_DIR to the directory containing the HST FITS pairs}"

MODE="${MODE:-train}"
PRED_DIR="${PRED_DIR:-${ROOT_DIR}/predictions}"
LOG_PATH="${LOG_PATH:-${ROOT_DIR}/runs}"
CONFIG_DIR="${CONFIG_DIR:-${ROOT_DIR}/config/multi_year}"
RUN_NAME="${RUN_NAME:-decti_adaptive}"
CHECKPOINT_RUN="${CHECKPOINT_RUN:-}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
BATCH_SIZE="${BATCH_SIZE:-64}"

if [[ "${MODE}" != "train" && "${MODE}" != "infer" ]]; then
    echo "MODE must be 'train' or 'infer'." >&2
    exit 2
fi

IS_TRAINING=1
if [[ "${MODE}" == "infer" ]]; then
    IS_TRAINING=0
fi

mkdir -p "${LOG_PATH}/${RUN_NAME}"
cd "${ROOT_DIR}"

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" main.py \
    --is_training "${IS_TRAINING}" \
    --model DeCTIMPE \
    --data_path "${DATA_DIR}" \
    --prediction_path "${PRED_DIR}" \
    --redivide_files 0 \
    --seq_len_perchannel 2048 \
    --img_width_perchannel 4096 \
    --half_plane 0 \
    --left_quarter 0 \
    --log_path "${LOG_PATH}" \
    --log_sfolder "${RUN_NAME}" \
    --loaded_chpt_sfolder "${CHECKPOINT_RUN}" \
    --config_subpath "${CONFIG_DIR}" \
    --num_workers 10 \
    --train_epochs 50 \
    --batch_size "${BATCH_SIZE}" \
    --learning_rate 0.0005 \
    --loss mse \
    --pct_start 0.3 \
    --patience 10000 \
    --window_size 64 \
    --abla_rpe 1 \
    --abla_ape 1 \
    --abla_residual 1 \
    --abla_patch_size 1 \
    --multi_ape 4 \
    --multi_rpe 4 \
    2>&1 | tee "${LOG_PATH}/${RUN_NAME}/${MODE}.log"
