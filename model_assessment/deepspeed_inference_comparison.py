import pandas as pd
import numpy as np
import math
import os
import logging
import argparse
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    AutoConfig, # Use AutoConfig for flexibility
    RobertaTokenizer,
    RobertaForMaskedLM,
    DataCollatorForLanguageModeling,
)
import datasets
import deepspeed # Import deepspeed

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="DeepSpeed Inference Evaluation")
parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model checkpoint directory")
parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the tokenizer directory")
parser.add_argument("--eval_batch_size", type=int, default=16, help="Per-device evaluation batch size")
parser.add_argument("--fp16", action='store_true', help="Enable FP16 inference")
parser.add_argument("--tensor_parallel_size", type=int, default=-1, help="Degree of tensor parallelism (TP). -1 means use world_size.")
# Add dataset locations here or keep the hardcoded dict
# parser.add_argument("--dataset_locations", type=str, help="Path to a file listing dataset names and paths (e.g., JSON)")

# OLD WAY:
# args = parser.parse_args()

# NEW WAY: Use parse_known_args()
args, unknown_args = parser.parse_known_args()
# 'args' will contain the arguments you defined (model_path, etc.)
# 'unknown_args' will contain ['--local_rank=0'] or similar, which we can ignore.
# Your script should still get the local rank via os.getenv('LOCAL_RANK', '0')
# because the deepspeed launcher sets that environment variable too.

# You can optionally log the unknown args if curious:
# import logging # make sure logging is configured
# logging.debug(f"Unknown arguments ignored by argparse: {unknown_args}")

# --- Distributed Setup ---
# This part remains the same, relying on environment variables set by the launcher
local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))


if world_size > 1:
    print(f"Initializing distributed environment: rank {local_rank}/{world_size}")
    dist.init_process_group(backend='nccl') # Initialize PyTorch distributed
    torch.cuda.set_device(local_rank) # Crucial: Set device for this process
else:
    print("Running on a single device.")

# --- Logging ---
logging.basicConfig(level=logging.INFO if local_rank == 0 else logging.WARN) # Log INFO only on rank 0
datasets.disable_progress_bar()
logging.info(f"Script arguments: {args}")

# --- Configuration ---
tokenizer_path = args.tokenizer_path
model_path = args.model_path
eval_batch_size = args.eval_batch_size
use_fp16 = args.fp16
tp_size = world_size if args.tensor_parallel_size == -1 else args.tensor_parallel_size
if world_size > 1 and tp_size != world_size:
     logging.warning(f"Running with world_size={world_size} but tensor_parallel_size={tp_size}. Ensure this is intended.")
elif world_size == 1 and tp_size > 1:
     logging.warning(f"Running on single device but tensor_parallel_size={tp_size}. Setting tp_size=1.")
     tp_size = 1

model_short_name = os.path.basename(os.path.normpath(model_path))
logging.info(f"Evaluating model: {model_short_name}")
logging.info(f"Using Tokenizer: {tokenizer_path}")
logging.info(f"FP16 enabled: {use_fp16}")
logging.info(f"Tensor Parallelism Size (tp_size): {tp_size}")
logging.info(f"Per-device eval batch size: {eval_batch_size}")


# --- Define paths to your PRE-TOKENIZED dataset FOLDERS ---
pre_tokenized_dataset_locations = {
    "test_HIP1": "/insert/dataset/path",
    "test_HIP2": "/insert/dataset/path",
    "test_HIP3": "/insert/dataset/path",
    "test_OAS": "/insert/dataset/path"
}

# --- Tokenizer and Collator ---
logging.info(f"Loading tokenizer from: {tokenizer_path}")
tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)

collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

# --- Model Loading ---
# Load configuration first if needed, otherwise from_pretrained handles it
# config = AutoConfig.from_pretrained(model_path)
logging.info(f"Loading model from: {model_path}")
# Load model onto CPU first if using TP > 1 or large model to avoid OOM on rank 0 before sharding
model = RobertaForMaskedLM.from_pretrained(model_path)

# --- DeepSpeed Inference Initialization ---
logging.info("Initializing DeepSpeed Inference Engine...")
inference_dtype = torch.half if use_fp16 else torch.float # Or model.dtype if pre-loaded correctly

# Note: DeepSpeed init_inference expects the model to be on the correct device
# *before* calling init_inference if TP=1 or if you aren't using its checkpoint loading.
# For TP>1, it handles sharding. Let's keep it on CPU initially.
# model = model.to(local_rank) # Move model to device BEFORE init_inference if TP=1

ds_engine = deepspeed.init_inference(
    model=model,
    tensor_parallel={"tp_size": tp_size}, # Use TP size
    dtype=inference_dtype,
    replace_with_kernel_inject=True, # Enable optimized kernels for supported models (like RoBERTa)
    # checkpoint=None, # Set if using DS checkpoint loading instead of from_pretrained
    # injection_policy=... # Needed only for unsupported models
    # config_params=... # Pass deepspeed config dict here if needed (usually not for init_inference)
)
model = ds_engine.module # Use the wrapped model
logging.info(f"DeepSpeed Inference Engine initialized. Model is on device: {model.device}") # Should report correct device(s)

# --- Prediction Loop ---
results = {}

logging.info("Starting evaluation loop...")
for dataset_name, dataset_path in pre_tokenized_dataset_locations.items():
    if local_rank == 0:
        logging.info(f"--- Processing dataset: {dataset_name} from {dataset_path} ---")

    if not os.path.isdir(dataset_path):
        logging.warning(f"Dataset path not found: {dataset_path}. Skipping.")
        continue

    try:
        # Load dataset (only needs to be done once, potentially load on rank 0 and broadcast?)
        # For simplicity here, each rank loads it. Ensure filesystem is shared and performs well.
        logging.debug(f"Rank {local_rank} loading dataset from disk: {dataset_path}")
        full_dataset = datasets.load_from_disk(dataset_path)
        logging.debug(f"Rank {local_rank} loaded dataset splits: {list(full_dataset.keys())}")

        if "test" not in full_dataset:
            logging.warning(f"'test' split not found in {dataset_path}. Skipping dataset {dataset_name}.")
            continue

        test_split = full_dataset["test"]
        if local_rank == 0:
             logging.info(f"Using 'test' split with {len(test_split)} examples.")
             logging.info(f"Test split features: {test_split.features}")

        # Check required columns
        required_cols = {'input_ids', 'attention_mask'}
        if not required_cols.issubset(test_split.column_names):
            logging.error(f"Dataset {dataset_name} missing required columns. Expected {required_cols}, found {test_split.column_names}. Skipping.")
            continue

        # --- Create DataLoader for Distributed Evaluation ---
        # DistributedSampler divides the dataset among ranks
        sampler = DistributedSampler(test_split, num_replicas=world_size, rank=local_rank, shuffle=False)
        # DataLoader handles batching and uses the sampler + collator
        dataloader = DataLoader(
            test_split,
            sampler=sampler,
            batch_size=eval_batch_size,
            collate_fn=collator, # Apply MLM masking here
            num_workers=2, # Optional: speed up data loading
            pin_memory=True, # Optional: speed up host-to-device transfer
        )

        # --- Manual Evaluation ---
        model.eval() # Set model to evaluation mode
        total_loss = 0.0
        num_batches = 0

        if local_rank == 0:
            logging.info(f"Running evaluation on {dataset_name}...")

        with torch.no_grad(): # Disable gradient calculations
            for batch in dataloader:
                # Move batch to the correct device for this rank
                # The collator produces torch tensors, move the whole dict's values
                try:
                    batch = {k: v.to(local_rank) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                except AttributeError:
                    logging.error(f"Failed to move batch to device {local_rank}. Batch keys: {batch.keys()}")
                    # Handle cases where collator might return non-tensors if error occurs
                    continue

                # Forward pass - model should handle data on correct device due to DeepSpeed wrapper
                outputs = model(**batch)
                loss = outputs.loss # The loss is usually pre-averaged over the batch for HF models

                # Check if loss is valid
                if loss is not None and not torch.isnan(loss):
                    total_loss += loss.item() # Accumulate loss on this rank
                    num_batches += 1
                else:
                    logging.warning(f"Rank {local_rank} encountered None or NaN loss in batch. Skipping batch.")

                logging.debug(f"Rank {local_rank}, Batch {num_batches}, Loss: {loss.item() if loss is not None else 'N/A'}")


        # --- Aggregate Loss Across Ranks ---
        if num_batches > 0:
            avg_loss_local = total_loss / num_batches
            loss_tensor = torch.tensor(avg_loss_local, device=local_rank)

            if world_size > 1:
                # Average the average batch losses from each rank
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                dist.barrier() # Ensure all ranks have finished reduction

            final_avg_loss = loss_tensor.item() # Get the globally averaged loss

            # Store results (only meaningful on rank 0 after reduction)
            if local_rank == 0:
                metrics = {'eval_loss': final_avg_loss, 'eval_num_batches': num_batches * world_size} # Approx total batches
                results[dataset_name] = metrics
                logging.info(f"--- Results for {dataset_name} ---")
                print(f"Metrics: {metrics}")
                try:
                    perplexity = math.exp(final_avg_loss)
                    print(f"Perplexity: {perplexity:.4f}")
                except OverflowError:
                    print("Perplexity calculation overflowed (loss might be too high)")
                print("-" * 30)
        elif local_rank == 0: # Handle case where no batches were processed on any rank
             logging.warning(f"No valid batches processed for dataset {dataset_name}. Skipping results.")
             print(f"--- No results for {dataset_name} (no valid batches) ---")
             print("-" * 30)


    except Exception as e:
        logging.error(f"Rank {local_rank}: Error processing dataset {dataset_name} at {dataset_path}: {e}", exc_info=True) # Log traceback
        if world_size > 1:
            dist.barrier() # Ensure all ranks wait if one errors
        continue # Skip to the next dataset

# --- Final Summary (Rank 0) ---
if local_rank == 0:
    logging.info("--- Overall Evaluation Results ---")
    for dataset_name, metrics in results.items():
        print(f"Dataset: {dataset_name}, Metrics: {metrics}")
        if 'eval_loss' in metrics:
            try:
                perplexity = math.exp(metrics['eval_loss'])
                print(f"  Perplexity: {perplexity:.4f}")
            except OverflowError:
                 print("  Perplexity calculation overflowed.")

# --- Cleanup Distributed ---
if world_size > 1:
    dist.destroy_process_group()

logging.info(f"Rank {local_rank}: Evaluation process finished.")