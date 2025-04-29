#!/bin/bash

# --- Configuration ---
# Define paths and names here for easy modification.
# Using export makes these variables available to the sbatch command's environment,
# and Slurm can then pass them to the job script environment.
#export PROJECT_BASE="/REDACTED/PATH"

export VENV_PATH="/REDACTED/PATH"
export CODE_BASE="/REDACTED/PATH"
export ASSETS_DIR="${CODE_BASE}/assets"
export TOKENIZER_PATH="${ASSETS_DIR}/antibody-tokenizer"
export CACHE_DIR="/REDACTED/PATH"

# --- Slurm Configuration ---
JOB_NAME="test_${MODEL_NAME}"

# --- Model Configuration ---
MODEL_PATH="/REDACTED/PATH"

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

#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- Submit Job ---
# Use --export=ALL to pass the current environment (including variables defined above with export)
# Alternatively, be specific: --export=PROJECT_BASE,MODEL_DIR,WANDB_LOG_DIR,...
sbatch -A es_reddy --job-name="${JOB_NAME}" \
    -n 1 \
    --cpus-per-task=1 \
    --gpus=1 \
    --gres=gpumem:10gb \
    --mem-per-cpu=10gb \
    --time=1:00:00 \
    --export=ALL \
    --wrap="python model_assessment/inference_comparison_test.py --model_path $MODEL_PATH"

# Check if sbatch command succeeded
if [ $? -eq 0 ]; then
    echo "Job ${JOB_NAME} submitted successfully."
else
    echo "Error submitting job ${JOB_NAME}."
    exit 1
fi

echo "----------------------------"