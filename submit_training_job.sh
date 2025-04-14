#!/bin/bash

# --- Configuration ---
# Define paths and names here for easy modification.
# Using export makes these variables available to the sbatch command's environment,
# and Slurm can then pass them to the job script environment.
#export PROJECT_BASE="/cluster/project/reddy/wglaenzer/final_training_testing_val_data/Soto-HIP1"
export PROJECT_BASE="/cluster/project/reddy/wglaenzer/final_training_testing_val_data/Soto-HIP1"
export MODEL_DIR="${PROJECT_BASE}/model"
export WANDB_LOG_DIR="${MODEL_DIR}/wandb_logs" # W&B log directory
export VENV_PATH="/cluster/project/reddy/wglaenzer/environments/training/bin/activate"
export CODE_BASE="/cluster/home/wglaenzer/Coding/plm_training_pipeline"
export ASSETS_DIR="${CODE_BASE}/assets"
export TOKENIZER_PATH="${ASSETS_DIR}/antibody-tokenizer"
export DEEPSPEED_CONFIG_PATH="${ASSETS_DIR}/deepspeed_config.json"
export CACHE_DIR="/cluster/scratch/wglaenzer/huggingface_cache"
export MODEL_NAME="HIP1_model"
export DONE_FILE_PATH="${MODEL_DIR}/.done"
export TRAIN_FILE="${PROJECT_BASE}/training.txt"
export TEST_FILE="${PROJECT_BASE}/test.txt"
export VAL_FILE="${PROJECT_BASE}/validation.txt"

# --- Slurm Configuration ---
JOB_NAME="train_${MODEL_NAME}"

# Create necessary directories ahead of time (optional, job script can also do it)
mkdir -p "${MODEL_DIR}"
mkdir -p "${WANDB_LOG_DIR}"

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

export TRITON_CACHE_DIR=/scratch/${SLURM_JOB_USER}/triton_cache/${SLURM_JOB_ID} # Example using scratch and making it job-specific
# Or: export TRITON_CACHE_DIR=/tmp/${SLURM_JOB_USER}/triton_cache/${SLURM_JOB_ID}
mkdir -p $TRITON_CACHE_DIR

# Load necessary modules
module load eth_proxy || echo "Warning: eth_proxy module not found or failed to load."
module load stack/2024-06 gcc/12.2.0 python/3.11.6 cuda/12.1.1 ninja/1.11.1 || { echo "Error: Failed to load required modules."; exit 1; }

# Verify CUDA setup and explicitly set CUDA_HOME if needed
export CUDA_HOME=${CUDA_HOME:-/cluster/software/stacks/2024-06/spack/opt/spack/linux-ubuntu22.04-x86_64_v3/gcc-12.2.0/cuda-12.1.1-5znnrjb5x5xr26nojxp3yhh6v77il7ie}
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
# --- Submit Job ---
# Use --export=ALL to pass the current environment (including variables defined above with export)
# Alternatively, be specific: --export=PROJECT_BASE,MODEL_DIR,WANDB_LOG_DIR,...
sbatch -A es_reddy --job-name="${JOB_NAME}" \
    -n 1 \
    --cpus-per-task=6 \
    --gpus=6 \
    --gres=gpumem:6gb \
    --mem-per-cpu=20gb \
    --time=24:00:00 \
    --export=ALL \
    --wrap="python -m torch.distributed.run \
    --nproc_per_node=\$SLURM_GPUS_ON_NODE \
    train_model.py \
    \"${TRAIN_FILE}\" \
    \"${TEST_FILE}\" \
    \"${VAL_FILE}\" \
    \"${CACHE_DIR}\" \
    \"${MODEL_DIR}\" \
    \"${MODEL_NAME}\" \
    \"${DONE_FILE_PATH}\" \
    \"${TOKENIZER_PATH}\" \
    "/cluster/home/wglaenzer/Coding/plm_training_pipeline/assets/deepspeed_config.json" \
    --deepspeed_config \"${DEEPSPEED_CONFIG_PATH}\""

# Check if sbatch command succeeded
if [ $? -eq 0 ]; then
    echo "Job ${JOB_NAME} submitted successfully."
else
    echo "Error submitting job ${JOB_NAME}."
    exit 1
fi

echo "----------------------------"