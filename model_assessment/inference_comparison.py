import pandas as pd
import numpy as np
from transformers import (
    RobertaConfig,
    RobertaTokenizer,
    # RobertaModel, # Not strictly needed if using ForMaskedLM
    RobertaForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
# from datasets import load_dataset # Keep for load_from_disk
import datasets
import os
import logging
import json

# Configure logging to show info level messages
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
datasets.disable_progress_bar() # Keep progress bars disabled if desired

# --- Configuration ---

# Initialize the tokenizer (still needed for the data collator)
# Make sure this is the *same* tokenizer used to pre-tokenize your datasets
tokenizer_path = assets/antibody-tokenizer
logging.info(f"Loading tokenizer from: {tokenizer_path}")
tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)

# --- Define paths to your PRE-TOKENIZED dataset FOLDERS ---
# Each path should be a directory containing the saved dataset (e.g., created with dataset.save_to_disk())
# and should contain 'train', 'eval', and 'test' splits internally.
# We are interested in the 'test' split here.
pre_tokenized_dataset_locations = {
    "test_heavy_chain": "/REDACTED/PATHroject/reddy/REDACTED/PATHretraining/heavy_chain/tokenized",
    "test_human": "/REDACTED/PATHroject/reddy/REDACTED/PATHretraining/human/tokenized",
    "test_human_heavy_chain": "/REDACTED/PATHroject/reddy/REDACTED/PATHretraining/human_heavy_chain/tokenized",
    "test_human_light_chain": "/REDACTED/PATHroject/reddy/REDACTED/PATHretraining/human_light_chain/tokenized",
    "test_human_mouse": "/REDACTED/PATHroject/reddy/REDACTED/PATHretraining/human_mouse/tokenized",
    "test_light_chain": "/REDACTED/PATHroject/reddy/REDACTED/PATHretraining/light_chain/tokenized",
    "test_mouse": "/REDACTED/PATHroject/reddy/REDACTED/PATHretraining/mouse/tokenized",
    "test_mouse_heavy_chain": "/REDACTED/PATHroject/reddy/REDACTED/PATHretraining/mouse_heavy_chain/tokenized",
    "test_mouse_light_chain": "/REDACTED/PATHroject/reddy/REDACTED/PATHretraining/mouse_light_chain/tokenized",
}

# --- Model Loading ---
model_path = "/REDACTED/PATHroject/reddy/REDACTED/PATHrcentage_comparison/500k/model/checkpoint-17370" # Specify the path to trained model
logging.info(f"Loading model: {model_path}")
model = RobertaForMaskedLM.from_pretrained(model_path)
model_name = model_path.split("/")[-2] # Extract model name from path
print(f"Model name: {model_name}")
# --- Data Collator ---
# Needed to handle MLM masking dynamically during evaluation/prediction
collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

# --- Training Arguments (used by Trainer, some args might be less relevant for prediction) ---
# Consider adjusting batch size for evaluation if needed
eval_batch_size = 96 # Or adjust based on memory

args = TrainingArguments(
    output_dir=f"/REDACTED/PATHratch/REDACTED/PATHname}_evaluation", # Directory for prediction outputs/logs if any
    # overwrite_output_dir=True, # Be cautious with this during prediction
    per_device_eval_batch_size=eval_batch_size,
    logging_steps=100, # Log frequency during prediction if needed
    # --- Args less relevant for prediction-only ---
    # per_device_train_batch_size=96,
    # max_steps=234375,
    # save_steps=2500,
    # adam_beta2=0.98,
    # adam_epsilon=1e-6,
    # weight_decay=0.01,
    # warmup_steps=10000,
    # learning_rate=1e-4,
    # gradient_accumulation_steps=1,
    # evaluation_strategy="steps", # Can set to "no" if only predicting
    evaluation_strategy="no", # No evaluation needed during predict() call itself
    fp16=True, # Use FP16 for prediction if GPU supports it and model was trained with it
    seed=42,
    disable_tqdm=True, # Keep progress bars disabled if desired
    dataloader_num_workers=2, # Optional: speed up data loading
    remove_unused_columns=False, # Important: Keep all columns from the pre-tokenized dataset for the collator
)

# --- Initialize Trainer (once) ---
trainer = Trainer(
    model=model,
    args=args,
    data_collator=collator,
    # train_dataset=None, # Not needed for prediction
    # eval_dataset=None,  # Prediction dataset is passed to predict()
    # compute_metrics=None # Define if you want specific metrics calculated by Trainer
)

# --- Prediction Loop ---
results = {}

logging.info("Starting prediction loop...")
for dataset_name, dataset_path in pre_tokenized_dataset_locations.items():
    logging.info(f"--- Processing dataset: {dataset_name} from {dataset_path} ---")

    if not os.path.isdir(dataset_path):
        logging.warning(f"Dataset path not found or not a directory: {dataset_path}. Skipping.")
        continue

    try:
        # Load the pre-tokenized dataset from disk
        logging.info(f"Loading dataset from disk: {dataset_path}")
        full_dataset = datasets.load_from_disk(dataset_path)
        logging.info(f"Loaded dataset splits: {list(full_dataset.keys())}")

        # Check if 'test' split exists
        if "test" not in full_dataset:
            logging.warning(f"'test' split not found in {dataset_path}. Skipping dataset {dataset_name}.")
            continue

        # Select the test split
        test_split = full_dataset["test"]
        logging.info(f"Using 'test' split with {len(test_split)} examples.")
        logging.info(f"Test split features: {test_split.features}")

        # Ensure required columns are present (optional but good practice)
        required_cols = {'input_ids', 'attention_mask'}
        if not required_cols.issubset(test_split.column_names):
             logging.error(f"Dataset {dataset_name} is missing required columns. Expected {required_cols}, found {test_split.column_names}. Skipping.")
             continue

        # Run prediction
        logging.info(f"Running prediction on {dataset_name}...")
        predictions = trainer.predict(test_split)

        # Store and print results
        # predictions object contains: predictions (logits), label_ids (if labels were present), metrics
        results[dataset_name] = predictions.metrics # Store metrics like perplexity ('eval_loss')
        logging.info(f"--- Results for {dataset_name} ---")
        # print(f"Raw Prediction Output: {predictions}") # Can be very verbose
        print(f"Metrics: {predictions.metrics}")
        # Perplexity is often calculated as exp(loss) for language models
        if 'eval_loss' in predictions.metrics:
             perplexity = np.exp(predictions.metrics['eval_loss'])
            logging.info(f"Perplexity: {perplexity:.4f}")
        print("-" * 30)


    except Exception as e:
        logging.error(f"Error processing dataset {dataset_name} at {dataset_path}: {e}")
        continue # Skip to the next dataset in case of error

logging.info("--- Overall Prediction Results ---")
for dataset_name, metrics in results.items():
    print(f"Dataset: {dataset_name}, Metrics: {metrics}")
    if 'eval_loss' in metrics:
        perplexity = pd.np.exp(metrics['eval_loss'])
        print(f"  Perplexity: {perplexity:.4f}")

logging.info("Prediction process finished.")

with open("prediction_results.json", "w") as f:
    json.dump(results, f, indent=4)

logging.info("Prediction results saved to prediction_results.json")