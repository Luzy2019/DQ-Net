#!/usr/bin/env bash
# Train the DQ teacher for B2-Z1 (using the visual_wholebody b2_z1 low-level policy).
# Usage:
#   RUN_NAME=test01 ./run_b2z1_DQ_teacher.sh
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${DQWBC_PYTHON:-/opt/conda/envs/dqwbc/bin/python}"
env_prefix="$(cd "$(dirname "${python_bin}")/.." && pwd)"

export LD_LIBRARY_PATH="${env_prefix}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${repo_dir}/DQ_high-level:${repo_dir}/third_party/isaacgym/python"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/tmp/torch_extensions}"

timesteps="${TIMESTEPS:-120000}"
experiment_dir="${EXPERIMENT_DIR:-DQ_teacher/b2z1}"
wandb_name="${WANDB_NAME:-b2z1-dqteacher_01}"
gpu="${GPU_ID:-0}"

cd "${repo_dir}/DQ_high-level"

CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u train_multistate_DQ_teacher.py \
    --rl_device cuda:0 \
    --sim_device cuda:0 \
    --timesteps "${timesteps}" \
    --task B2Z1PickMulti \
    --experiment_dir "${experiment_dir}" \
    --wandb_name "${wandb_name}" \
    --roboinfo \
    --observe_gait_commands \
    --small_value_set_zero \
    --rand_control \
    --headless \
    "$@"