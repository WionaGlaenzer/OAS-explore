#!/bin/bash

# Replace this placeholder with the actual path where you want W&B logs
WANDB_BASE_DIR="/REDACTED/PATHroject/reddy/REDACTED/PATHnal_training_testing_val_data/Soto-HIP1/model/wandb_logs" # Example path
source /REDACTED/PATHroject/reddy/REDACTED/PATHnvironments/training/bin/activate
mkdir -p /REDACTED/PATHroject/reddy/REDACTED/PATHnal_training_testing_val_data/Soto-HIP1/model
#RPORT=$((RANDOM % 55535 + 10000))
# Construct the command string for --wrap
# Use semicolons to separate commands.
# Escape '$' for variables that should be expanded inside the job (e.g., \$RPORT, \$CUDA_HOME).
# Escape '"' where nested quotes are needed inside the wrapped string (e.g., \" for echo, \" for endpoint).
COMMAND_TO_WRAP="#!/bin/bash
echo 'CUDA_HOME is set to: \$CUDA_HOME'; \
echo \"Using rendezvous port: \$RPORT\"; \
export WANDB_DIR='${WANDB_BASE_DIR}'; \
. /etc/profile; \
module load eth_proxy; \
module load stack/2024-06 gcc/12.2.0; \
module load python/3.11.6 cuda/12.1.1 ninja/1.11.1; \
export CUDA_HOME=/REDACTED/PATHre/stacks/2024-06/spack/opt/spack/linux-ubuntu22.04-x86_64_v3/gcc-12.2.0/cuda-12.1.1-5znnrjb5x5xr26nojxp3yhh6v77il7ie; \
source /REDACTED/PATHroject/reddy/REDACTED/PATHnvironments/training/bin/activate
python -m torch.distributed.run --nproc_per_node 6 \
  train_model.py \
  /REDACTED/PATHroject/reddy/REDACTED/PATHnal_training_testing_val_data/Soto-HIP1/training.txt \
  /REDACTED/PATHroject/reddy/REDACTED/PATHnal_training_testing_val_data/Soto-HIP1/test.txt \
  /REDACTED/PATHroject/reddy/REDACTED/PATHnal_training_testing_val_data/Soto-HIP1/validation.txt \
  /REDACTED/PATHratch/REDACTED/PATHngface_cache \
  /REDACTED/PATHroject/reddy/REDACTED/PATHnal_training_testing_val_data/Soto-HIP1/model \
  HIP1_model \
  /REDACTED/PATHroject/reddy/REDACTED/PATHnal_training_testing_val_data/Soto-HIP1/model/.done \
  /REDACTED/PATHnzer/Coding/plm_training_pipeline/assets/antibody-tokenizer \
  /REDACTED/PATHnzer/Coding/plm_training_pipeline/assets/deepspeed_config.json
  --deepspeed /REDACTED/PATHnzer/Coding/plm_training_pipeline/assets/deepspeed_config.json "

# Submit the job using sbatch and the constructed command string
sbatch -A es_reddy -n 1 --cpus-per-task=6 --gpus=6 --gres=gpumem:20gb --mem-per-cpu=20048  \
--time=96:00:00  --job-name="train_HIP1" \
--tasks=1 \
--wrap="$COMMAND_TO_WRAP"

echo "Job submitted."