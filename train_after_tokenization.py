# -*- coding: utf-8 -*-
import sys
import os
import torch # Import torch early for distributed checks
from transformers import (
    RobertaConfig,
    RobertaTokenizer,
    RobertaForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    logging as hf_logging, # Import logging submodule
)
from datasets import load_dataset, load_from_disk
import datasets
import wandb

# --- Argument Parsing ---
# This needs to happen on all processes as they are launched with the same args
if len(sys.argv) < 10:
    print("Usage: python script.py <training_file> <test_file> <val_file> <cache_dir> <out_dir> <model_name> <flag_file> <tokenizer_path> <deepspeed_config>")
    sys.exit(1)

training_file = sys.argv[1]
test_file = sys.argv[2]
val_file = sys.argv[3]
cache_dir = sys.argv[4]
out_dir = sys.argv[5]
model_name = sys.argv[6]
flag_file = sys.argv[7]
tokenizer_path = sys.argv[8]
deepspeed_config = sys.argv[9]

# --- Determine Rank ---
# DeepSpeed/torchrun sets LOCAL_RANK environment variable. Default to 0 if not set (for single-process runs)
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
is_main_process = local_rank == 0

# --- Configure Logging ---
# Only log INFO messages from the main process to reduce noise
# Log warnings/errors from all processes
log_level = hf_logging.INFO if is_main_process else hf_logging.WARNING
datasets.utils.logging.set_verbosity(log_level)
hf_logging.set_verbosity(log_level)
hf_logging.enable_default_handler()
hf_logging.enable_explicit_format()

# Disable datasets progress bar on all processes
datasets.disable_progress_bar()

# --- W&B Environment Variable Setup ---
# Set these on all processes; Trainer/Wandb usually handle reporting correctly
os.environ["WANDB_PROJECT"] = "project_tanuki"
os.environ["WANDB_ENTITY"] = "wiona-glaenzer-eth-z-rich"
os.environ["WANDB_LOG_MODEL"] = "checkpoint" # Can be 'true', 'false', or 'checkpoint'

# --- Print the configuration (Only Rank 0) ---
if is_main_process:
    print("--- Configuration ---")
    print(f"Training File: {training_file}")
    print(f"Test File: {test_file}")
    print(f"Validation File: {val_file}")
    print(f"Cache Directory: {cache_dir}")
    print(f"Output Directory: {out_dir}")
    print(f"Model Name: {model_name}")
    print(f"Flag File: {flag_file}")
    print(f"Tokenizer Path: {tokenizer_path}")
    print(f"Deepspeed Config: {deepspeed_config}")
    print(f"Running with local_rank: {local_rank}")
    print("--------------------")

# --- Tokenizer and Collator ---
# Needed on all processes
try:
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )
    if is_main_process:
        print("Tokenizer and Collator loaded successfully.")
except Exception as e:
    print(f"Rank {local_rank}: Error loading tokenizer or creating collator: {e}")
    sys.exit(1) # Exit if essential components fail

# --- Load and Tokenize Dataset ---
# NOTE: Ideally, dataset loading and tokenization should be done *once*
# and saved, then loaded by all processes. The current approach has each
# process load and tokenize the *entire* dataset, which is inefficient
# for large datasets but simpler to implement from your original script.
# `load_dataset` is generally safe with multiple processes accessing the cache.
# The `.map` operation here runs independently on each process.
try:
    if is_main_process:
        print("Loading Pre-tokenized Datasets...")

    # Each process loads the already tokenized dataset
    tokenized_dataset = load_from_disk(f"/cluster/project/reddy/wglaenzer/final_training_testing_val_data/Soto-All/tokenized")

    if is_main_process:
        print("Pre-tokenized dataset loaded.")
        print(f"Train dataset size: {len(tokenized_dataset['train'])}")
        print(f"Eval dataset size: {len(tokenized_dataset['eval'])}")
        print(f"Test dataset size: {len(tokenized_dataset['test'])}")

except Exception as e:
    print(f"Rank {local_rank}: Error loading pre-tokenized dataset: {e}")
    sys.exit(1)


# --- Model Configuration ---
# Needed on all processes
antiberta_config = {
    "num_hidden_layers": 12, #originally 12
    "num_attention_heads": 12,
    "hidden_size": 768, #originally 768
    "d_ff": 3072,
    "vocab_size": 25,
    "max_len": 150,
    "max_position_embeddings": 152,
    "batch_size": 96, # This is PER DEVICE batch size
    "max_steps": 156250, #originally 225000
    "weight_decay": 0.01,
    "peak_learning_rate": 0.0001,
    "gradient_accumulation_steps": 1,
}
if is_main_process:
    print("\n--- Model Hyperparameters ---")
    print(antiberta_config)
    print("---------------------------\n")

# --- Initialise Model ---
# Needs to happen on all processes for DeepSpeed to wrap correctly
try:
    model_config = RobertaConfig(
        vocab_size=antiberta_config.get("vocab_size"),
        hidden_size=antiberta_config.get("hidden_size"),
        max_position_embeddings=antiberta_config.get("max_position_embeddings"),
        num_hidden_layers=antiberta_config.get("num_hidden_layers", 12),
        num_attention_heads=antiberta_config.get("num_attention_heads", 12),
        type_vocab_size=1,
    )
    model = RobertaForMaskedLM(model_config)

    if is_main_process:
        print(f"Model Parameter Count: {model.num_parameters():,}")
        print("Model structure initialization complete (on all processes).")
except Exception as e:
    print(f"Rank {local_rank}: Error initializing model: {e}")
    sys.exit(1)

# --- Training Arguments ---
# Needed on all processes. Trainer uses local_rank internally.
# Ensure deepspeed_config path is correct and accessible by all nodes/processes
args = TrainingArguments(
    disable_tqdm=True,
    output_dir=out_dir,
    overwrite_output_dir=True, # Be cautious with this in distributed settings
    do_train=True,
    do_eval=True,
    per_device_train_batch_size=antiberta_config.get("batch_size", 32),
    per_device_eval_batch_size=antiberta_config.get("batch_size", 32),
    gradient_accumulation_steps=antiberta_config.get("gradient_accumulation_steps", 1),
    learning_rate=antiberta_config.get("peak_learning_rate", 1e-4),
    adam_beta1=0.9,
    adam_beta2=0.98,
    adam_epsilon=1e-6,
    weight_decay=antiberta_config.get("weight_decay", 0.01),
    warmup_steps=int(antiberta_config.get("max_steps", 1000) * 0.06), # Calculate warmup steps
    max_steps=antiberta_config.get("max_steps", 1000),
    logging_steps=100,
    eval_strategy="steps",
    eval_steps=2500, # Match logging/save steps potentially
    save_steps=5000,
    #save_total_limit=5, # Optional: limit number of checkpoints
    fp16=True, # Ensure DeepSpeed config also handles fp16/bf16 settings
    # --- Deepspeed ---
    deepspeed=deepspeed_config,
    # --- W&B Integration ---
    report_to="wandb", # Enable W&B logging
    run_name=f"{model_name}-run-{wandb.util.generate_id()}", # Unique run name needed only on rank 0? Trainer handles this.
    logging_dir=f"{out_dir}/logs", # Directory for logs (like TensorBoard if used)
    # load_best_model_at_end=True, # Requires eval_strategy and metric
    # metric_for_best_model="eval_loss",
    # --- Distributed Training Specific ---
    local_rank=local_rank, # Pass local rank explicitly if needed (usually inferred)
    seed=42,
    # --- Logging Control ---
    # Log level already set globally for transformers/datasets logging
    # log_on_each_node=False, # Log metrics only on the main process (default is True)
)

# --- Initialize Trainer ---
# Needs to happen on all processes
if is_main_process:
    print("Initializing Trainer...")
try:
    trainer = Trainer(
        model=model,
        args=args,
        data_collator=collator,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["eval"],
        tokenizer=tokenizer, # Pass tokenizer for auto-saving
    )
    # Now we can use trainer's rank check
    is_main_process = trainer.is_world_process_zero() # Update based on Trainer's perspective
except Exception as e:
    print(f"Rank {local_rank}: Error initializing Trainer: {e}")
    # Perform necessary cleanup if possible/needed before exiting
    sys.exit(1)


# --- Start Training ---
if is_main_process:
    print("Starting Training...")

try:
    trainer.train()
    if is_main_process:
        print("Training finished.")
except Exception as e:
    print(f"Rank {local_rank}: Error during training: {e}")
    # Handle potential cleanup or checkpoint saving on error if needed
    sys.exit(1) # Or attempt recovery/logging

# --- Save Final Model ---
# Trainer's save_model is generally rank-aware and saves only from rank 0
# by default when using DeepSpeed.
if is_main_process:
    print(f"Saving final model to {out_dir}/{model_name}...")
try:
    # Ensure the directory exists (only main process needs to create it)
    if is_main_process:
        os.makedirs(f"{out_dir}/{model_name}", exist_ok=True)
    # Let Trainer handle the distributed saving logic
    trainer.save_model(f"{out_dir}/{model_name}")
    # Barrier to ensure saving is complete before rank 0 proceeds (optional, save_model might handle this)
    # torch.distributed.barrier()
    if is_main_process:
        print("Model saved.")
except Exception as e:
    print(f"Rank {local_rank}: Error saving model: {e}")
    # Exit or log error

# --- Create Flag File (Only Rank 0) ---
if is_main_process:
    print(f"Creating flag file: {flag_file}")
    try:
        with open(flag_file, 'w') as f:
            f.write("Training completed.")
        print("Flag file created.")
    except Exception as e:
        print(f"Rank {local_rank}: Error creating flag file: {e}")

# --- Finish W&B Run ---
# Typically only needed on the main process, or handled by Trainer's callbacks
if is_main_process:
    wandb.finish()
    print("Wandb run finished.")

if is_main_process:
    print("Script execution completed successfully on main process.")

# All processes exit here