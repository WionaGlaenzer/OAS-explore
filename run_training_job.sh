#!/bin/bash
# This script is executed by Slurm on the compute node(s)

# Stop on errors, treat unset variables as errors, pipelines fail on first error
set -e -u -o pipefail

# --- Environment Modules & CUDA ---
echo "Loading modules..."
# Deactivate any potentially active virtualenv from submission node before loading modules/activating new one
# (Only needed if you activate the venv *before* submitting) - commenting out for now
# if command -v deactivate &> /dev/null; then
#     echo "Deactivating existing Python environment..."
#     deactivate
# fi

# Source system profile if necessary for module command (sometimes needed)
# . /etc/profile

# Load necessary modules
module load eth_proxy || echo "Warning: eth_proxy module not found or failed to load."
module load stack/2024-06 gcc/12.2.0 python/3.11.6 cuda/12.1.1 ninja/1.11.1 || { echo "Error: Failed to load required modules."; exit 1; }

# Verify CUDA setup and explicitly set CUDA_HOME if needed
export CUDA_HOME=${CUDA_HOME:-/REDACTED/PATHre/stacks/2024-06/spack/opt/spack/linux-ubuntu22.04-x86_64_v3/gcc-12.2.0/cuda-12.1.1-5znnrjb5x5xr26nojxp3yhh6v77il7ie}
echo "CUDA_HOME: $CUDA_HOME"
echo "Checking nvcc version..."
nvcc --version || echo "Warning: nvcc command not found."
echo "Checking GPU status..."
nvidia-smi || echo "Warning: nvidia-smi command not found or GPUs not available."

# --- Python Virtual Environment ---
# VENV_PATH should be exported by the submission script via --export=ALL
if [[ -z "${VENV_PATH}" ]]; then
  echo "Error: VENV_PATH environment variable not set. Check submission script."
  exit 1
fi
if [[ ! -f "${VENV_PATH}" ]]; then
  echo "Error: Python environment activation script not found at: ${VENV_PATH}"
  exit 1
fi
echo "Activating Python venv: ${VENV_PATH}"
source "${VENV_PATH}"

# --- WANDB Setup ---
# WANDB_LOG_DIR should be exported by the submission script
if [[ -z "${WANDB_LOG_DIR}" ]]; then
  echo "Error: WANDB_LOG_DIR environment variable not set."
  exit 1
fi
export WANDB_DIR="${WANDB_LOG_DIR}" # W&B uses WANDB_DIR
mkdir -p "${WANDB_DIR}" # Ensure directory exists
echo "WANDB_DIR set to: ${WANDB_DIR}"

# --- Create other necessary directories ---
# MODEL_DIR should be exported
if [[ -z "${MODEL_DIR}" ]]; then
  echo "Error: MODEL_DIR environment variable not set."
  exit 1
fi
echo "Ensuring model directory exists: ${MODEL_DIR}"
mkdir -p "${MODEL_DIR}"

# --- Verify Essential Variables ---
: "${TRAIN_FILE:?Error: TRAIN_FILE not set}"
: "${TEST_FILE:?Error: TEST_FILE not set}"
: "${VAL_FILE:?Error: VAL_FILE not set}"
: "${CACHE_DIR:?Error: CACHE_DIR not set}"
: "${MODEL_NAME:?Error: MODEL_NAME not set}"
: "${DONE_FILE_PATH:?Error: DONE_FILE_PATH not set}"
: "${TOKENIZER_PATH:?Error: TOKENIZER_PATH not set}"
: "${DEEPSPEED_CONFIG_PATH:?Error: DEEPSPEED_CONFIG_PATH not set}"

echo "--- Starting Training Script ---"
echo "Train file: ${TRAIN_FILE}"
echo "Test file: ${TEST_FILE}"
echo "Val file: ${VAL_FILE}"
echo "Cache dir: ${CACHE_DIR}"
echo "Model dir: ${MODEL_DIR}"
echo "Model name: ${MODEL_NAME}"
echo "Done file: ${DONE_FILE_PATH}"
echo "Tokenizer: ${TOKENIZER_PATH}"
echo "DeepSpeed Cfg: ${DEEPSPEED_CONFIG_PATH}"

# --- Execute the Training Command ---
echo "Running command: python -m torch.distributed.run train_model.py ..."
python -m torch.distributed.run \
    --nproc_per_node=$SLURM_GPUS_ON_NODE \
    train_model.py \
    "${TRAIN_FILE}" \
    "${TEST_FILE}" \
    "${VAL_FILE}" \
    "${CACHE_DIR}" \
    "${MODEL_DIR}" \
    "${MODEL_NAME}" \
    "${DONE_FILE_PATH}" \
    "${TOKENIZER_PATH}" \
    --deepspeed_config "${DEEPSPEED_CONFIG_PATH}"

# Capture exit code
EXIT_CODE=$?

# --- Job Completion ---
if [ $EXIT_CODE -eq 0 ]; then
  echo "--- Training Job Completed Successfully ---"
else
  echo "--- Training Job Failed (Exit Code: $EXIT_CODE) ---"
fi

# Deactivate virtual environment
if command -v deactivate &> /dev/null; then
    echo "Deactivating Python environment."
    deactivate
fi

exit $EXIT_CODE