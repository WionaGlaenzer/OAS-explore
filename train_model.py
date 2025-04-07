import sys
import os
from transformers import (
    RobertaConfig,
    RobertaTokenizer,
    RobertaForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
import datasets
import wandb

datasets.disable_progress_bar() 

# --- Argument Parsing ---
training_file = sys.argv[1]
test_file = sys.argv[2]
val_file = sys.argv[3]
cache_dir = sys.argv[4]
out_dir = sys.argv[5]
model_name = sys.argv[6]
flag_file = sys.argv[7]
tokenizer_path = sys.argv[8]
deepspeed_config = sys.argv[9]

# --- W&B Environment Variable Setup ---
os.environ["WANDB_PROJECT"] = "project_tanuki"
os.environ["WANDB_ENTITY"] = "wiona-glaenzer-eth-z-rich"
os.environ["WANDB_LOG_MODEL"] = "checkpoint-HIP1"

# --- Print the configuration ---
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
print("--------------------")

# --- Tokenizer and Collator ---
tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)
collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

# --- Load and Tokenize Dataset ---
print("Loading and Tokenizing Datasets...")
text_datasets = {
    "train": [training_file],
    "eval": [test_file],
    "test": [val_file]
}

dataset = load_dataset("text", data_files=text_datasets, cache_dir=cache_dir) #change this to your own cache directory
tokenized_dataset = dataset.map(
    lambda z: tokenizer(
        z["text"],
        padding="max_length",
        truncation=True,
        max_length=150,
        return_special_tokens_mask=True,
    ),
    batched=True,
    num_proc=1,
    remove_columns=["text"],
)

print("Dataset processing complete.")
print(f"Train dataset size: {len(tokenized_dataset['train'])}")
print(f"Eval dataset size: {len(tokenized_dataset['eval'])}")
print(f"Test dataset size: {len(tokenized_dataset['test'])}")

# --- Model Configuration ---
antiberta_config = {
    "num_hidden_layers": 12, #originally 12
    "num_attention_heads": 12,
    "hidden_size": 768, #originally 768
    "d_ff": 3072,
    "vocab_size": 25,
    "max_len": 150,
    "max_position_embeddings": 152,
    "batch_size": 96,
    "max_steps": 117188, #originally 225000
    "weight_decay": 0.01,
    "peak_learning_rate": 0.0001,
}
print("\n--- Model Hyperparameters ---")
print(antiberta_config)
print("---------------------------\n")

# --- Initialise Model ---
model_config = RobertaConfig(
    vocab_size=antiberta_config.get("vocab_size"),
    hidden_size=antiberta_config.get("hidden_size"),
    max_position_embeddings=antiberta_config.get("max_position_embeddings"),
    num_hidden_layers=antiberta_config.get("num_hidden_layers", 12),
    num_attention_heads=antiberta_config.get("num_attention_heads", 12),
    type_vocab_size=1,
)
model = RobertaForMaskedLM(model_config)
print(f"Model Parameter Count: {model.num_parameters():,}")
print("Model initialization complete.")

# --- Training Arguments ---
args = TrainingArguments(
    #disable_tqdm=True, #added to disable the progress bar
    output_dir=out_dir,
    overwrite_output_dir=True,
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
    warmup_steps=100,
    max_steps=antiberta_config.get("max_steps", 1000), #originally 225000
    save_steps=2500,#originally 2500
    eval_strategy="steps",
    logging_steps=100, #originally 2500
    fp16=True, #comment this out for training on CPU
    seed=42,
    deepspeed=deepspeed_config,
    # --- W&B Integration ---
    report_to="wandb", # Enable W&B logging
    run_name=f"{model_name}-run-{wandb.util.generate_id()}", # Unique run name
    logging_dir=f"{out_dir}/logs", # Keep TensorBoard logs alongside W&B
    # load_best_model_at_end=True, # Optional: Reload best model based on eval metric
    # metric_for_best_model="eval_loss", # Optional: Metric to determine the best model
)

# --- Initialize Trainer ---
print("Initializing Trainer...")
trainer = Trainer(
    model=model,
    args=args,
    data_collator=collator,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["eval"]
)

# --- Start Training ---
print("Starting Training...")
trainer.train()
print("Training finished.")

# --- Save Final Model ---
print(f"Saving final model to {out_dir}/{model_name}...")
trainer.save_model(f"{out_dir}/{model_name}")
print("Model saved.")

# --- Create Flag File ---
with open(flag_file, 'w') as f:
    f.write("Training completed.")

# --- Finish W&B Run ---
wandb.finish() # Explicitly finish the run