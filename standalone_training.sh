#!/bin/bash

# Replace this placeholder with the actual path where you want W&B logs
WANDB_BASE_DIR="/cluster/project/reddy/wglaenzer/final_training_testing_val_data/Soto-HIP1/model/wandb_logs" # Example path
source /cluster/project/reddy/wglaenzer/environments/training/bin/activate
mkdir -p /cluster/project/reddy/wglaenzer/final_training_testing_val_data/Soto-HIP1/model
#RPORT=$((RANDOM % 55535 + 10000))
# Construct the command string for --wrap
# Use semicolons to separate commands.
# Escape '$' for variables that should be expanded inside the job (e.g., \$RPORT, \$CUDA_HOME).
# Escape '"' where nested quotes are needed inside the wrapped string (e.g., \" for echo, \" for endpoint).
COMMAND_TO_WRAP="echo 'CUDA_HOME is set to: \$CUDA_HOME'; \
echo \"Using rendezvous port: \$RPORT\"; \
export WANDB_DIR='${WANDB_BASE_DIR}'; \
torchrun --nproc_per_node=auto --rdzv-backend=c10d --rdzv-endpoint=localhost:26262 \
  train_model.py \
  /cluster/project/reddy/wglaenzer/final_training_testing_val_data/Soto-HIP1/training.txt \
  /cluster/project/reddy/wglaenzer/final_training_testing_val_data/Soto-HIP1/validation.txt \
  /cluster/project/reddy/wglaenzer/final_training_testing_val_data/Soto-HIP1/test.txt \
  /cluster/scratch/wglaenzer/huggingface_cache \
  /cluster/project/reddy/wglaenzer/final_training_testing_val_data/Soto-HIP1/model \
  HIP1_model \
  /cluster/project/reddy/wglaenzer/final_training_testing_val_data/Soto-HIP1/model/.done \
  /cluster/home/wglaenzer/Coding/plm_training_pipeline/assets/antibody-tokenizer \
  /cluster/home/wglaenzer/Coding/plm_training_pipeline/assets/deepspeed_config.json"

# Submit the job using sbatch and the constructed command string
sbatch -A es_reddy -n 1 --cpus-per-task=1 --mem-per-cpu=20048 --gpus=1 --gres=gpumem:1gb \
--time=01:00:00 --mem-per-cpu=100g --job-name="train_HIP1" \
--tasks=1 \
--wrap="$COMMAND_TO_WRAP"

echo "Job submitted."