#!/bin/bash

# --- Configuration ---
# Define paths and names here for easy modification.
# Using export makes these variables available to the sbatch command's environment,
# and Slurm can then pass them to the job script environment via --export=ALL.
export VENV_PATH="/REDACTED/PATH"
export CODE_BASE="/REDACTED/PATH"
export TOKENIZER_PATH="${CODE_BASE}/assets/antibody-tokenizer"
export CACHE_DIR="/REDACTED/PATH"
export TRANSFORMERS_CACHE=$CACHE_DIR
export HF_HOME=$CACHE_DIR
export HF_DATASETS_CACHE=$CACHE_DIR

# Model to evaluate
export MODEL_PATH="/REDACTED/PATH"

# Inference Script Path
export PYTHON_SCRIPT="${CODE_BASE}/model_assessment/deepspeed_inference_comparison.py"

# Script Parameters (exported so they are available in the wrap environment)
export EVAL_BATCH_SIZE=16
export USE_FP16="--fp16" # Set to "--fp16" to enable, "" to disable

# --- Slurm Configuration ---
# Extract model name for job name (optional)
MODEL_NAME=$(basename "${MODEL_PATH}")
JOB_NAME="ds_inf_${MODEL_NAME}"

# Slurm Resource Allocation (these will be passed to sbatch)
ACCOUNT="es_reddy"
CPUS_PER_TASK=4
NUM_GPUS=3
MEM_PER_CPU="15gb"
TIME_LIMIT="1:00:00"

# --- Environment Setup (Performed BEFORE sbatch) ---
echo "Loading modules..."
module load eth_proxy || echo "Warning: eth_proxy module not found or failed to load."
module load stack/2024-06 gcc/12.2.0 python/3.11.6 cuda/12.1.1 ninja/1.11.1 || { echo "Error: Failed to load required modules on submission node."; exit 1; }

# Verify CUDA setup and explicitly set CUDA_HOME (for the submission env, might be different in job)
export CUDA_HOME=${CUDA_HOME:-/REDACTED/PATHre/stacks/2024-06/spack/opt/spack/linux-ubuntu22.04-x86_64_v3/gcc-12.2.0/cuda-12.1.1-5znnrjb5x5xr26nojxp3yhh6v77il7ie}
echo "CUDA_HOME (Submission Env): $CUDA_HOME"
# These checks run on the submission node, nvidia-smi will likely fail here
echo "Checking nvcc version (Submission Env)..."
nvcc --version || echo "Warning: nvcc command not found on submission node."
echo "Checking GPU status (Submission Env)..."
nvidia-smi || echo "Warning: nvidia-smi failed on submission node (expected)."

# Set up Triton Cache Path using $USER (SLURM vars not available yet)
# Note: The directory inside the job will use SLURM vars from the --wrap command if needed,
# but we create the base dir here.
export TRITON_CACHE_DIR_BASE="/scratch/$USER/triton_cache" # Base path
mkdir -p "$TRITON_CACHE_DIR_BASE"
echo "Triton Cache Base Dir: $TRITON_CACHE_DIR_BASE"
# The actual job will likely use /scratch/$SLURM_JOB_USER/triton_cache/$SLURM_JOB_ID if the python script is configured that way

# --- Python Virtual Environment (Activated BEFORE sbatch) ---
if [[ -z "${VENV_PATH}" ]]; then
  echo "Error: VENV_PATH environment variable not set."
  exit 1
fi
if [[ ! -f "${VENV_PATH}" ]]; then
  echo "Error: Python environment activation script not found at: ${VENV_PATH}"
  exit 1
fi
echo "Activating Python venv: ${VENV_PATH}"
source "${VENV_PATH}"

# --- Submit Job ---
echo "Submitting DeepSpeed inference job: ${JOB_NAME}"
echo "Model: ${MODEL_PATH}"
echo "Script: ${PYTHON_SCRIPT}"

# Construct the command arguments for the Python script dynamically
# These variables are expanded *before* sbatch runs.
CMD_ARGS="--model_path \"${MODEL_PATH}\" \
          --tokenizer_path \"${TOKENIZER_PATH}\" \
          --eval_batch_size \"${EVAL_BATCH_SIZE}\" \
          ${USE_FP16} \
          --tensor_parallel_size \$SLURM_GPUS_ON_NODE" # Use Slurm var here

# Construct the full command string for --wrap
# $SLURM_GPUS_ON_NODE will be evaluated *inside* the Slurm job environment
WRAP_CMD="deepspeed --num_gpus \$SLURM_GPUS_ON_NODE \"${PYTHON_SCRIPT}\" ${CMD_ARGS}"

echo "Wrap command will be: ${WRAP_CMD}"

# sbatch command with only the execution command in --wrap
sbatch -A "${ACCOUNT}" --job-name="${JOB_NAME}" \
    -n 1 \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --gpus="${NUM_GPUS}" \
    --mem-per-cpu="${MEM_PER_CPU}" \
    --time="${TIME_LIMIT}" \
    --output="slurm-${JOB_NAME}-%j.out" \
    --error="slurm-${JOB_NAME}-%j.err" \
    --export=ALL \
    --wrap="${WRAP_CMD}"

# Check if sbatch command succeeded locally
if [ $? -eq 0 ]; then
    echo "Job ${JOB_NAME} submitted successfully."
else
    echo "Error submitting job ${JOB_NAME}."
    exit 1
fi

echo "----------------------------"
