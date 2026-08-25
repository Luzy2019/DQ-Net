#!/usr/bin/env bash
# Play/eval the DQ teacher for B2-Z1.
# Usage:
#   CHECKPOINT=DQ_teacher/b2z1/b2z1-dqteacher_01/checkpoints/xxx.pt ./run_b2z1_DQ_teacher_play.sh
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${DQWBC_PYTHON:-/opt/conda/envs/dqwbc/bin/python}"
env_prefix="$(cd "$(dirname "${python_bin}")/.." && pwd)"

export LD_LIBRARY_PATH="${env_prefix}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${repo_dir}/DQ_high-level:${repo_dir}/third_party/isaacgym/python"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/tmp/torch_extensions}"

checkpoint="${CHECKPOINT:?Set CHECKPOINT to the teacher checkpoint .pt path}"
gpu="${GPU_ID:-0}"

cd "${repo_dir}/DQ_high-level"

CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u play_multistate_DQ_teacher.py \
    --rl_device cuda:0 \
    --sim_device cuda:0 \
    --task B2Z1PickMulti \
    --checkpoint "${checkpoint}" \
    --roboinfo \
    --observe_gait_commands \
    --small_value_set_zero \
    --rand_control \
    --headless \
    "$@"