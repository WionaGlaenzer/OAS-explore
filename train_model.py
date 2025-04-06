# Training script modified form Leem et al.
import sys
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
import os

datasets.disable_progress_bar() 

training_file = sys.argv[1]
test_file = sys.argv[2]
val_file = sys.argv[3]
cache_dir = sys.argv[4]
out_dir = sys.argv[5]
model_name = sys.argv[6]
flag_file = sys.argv[7]
tokenizer_path = sys.argv[8]
deepspeed_config = sys.argv[9]

# Initialise the tokeniser
tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)

# Initialise the data collator, which is necessary for batching
collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

# Load the dataset
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

# These are the cofigurations used for pre-training
antiberta_config = {
    "num_hidden_layers": 12, #originally 12
    "num_attention_heads": 12,
    "hidden_size": 768, #originally 768
    "d_ff": 3072,
    "vocab_size": 25,
    "max_len": 150,
    "max_position_embeddings": 152,
    "batch_size": 96,
    "max_steps": 1000, #originally 225000
    "weight_decay": 0.01,
    "peak_learning_rate": 0.0001,
}

# Initialise the model
model_config = RobertaConfig(
    vocab_size=antiberta_config.get("vocab_size"),
    hidden_size=antiberta_config.get("hidden_size"),
    max_position_embeddings=antiberta_config.get("max_position_embeddings"),
    num_hidden_layers=antiberta_config.get("num_hidden_layers", 12),
    num_attention_heads=antiberta_config.get("num_attention_heads", 12),
    type_vocab_size=1,
)
model = RobertaForMaskedLM(model_config)

# Construct training arguments
args = TrainingArguments(
    #disable_tqdm=True, #added to disable the progress bar
    output_dir=out_dir,
    overwrite_output_dir=True,
    per_device_train_batch_size=antiberta_config.get("batch_size", 32),
    per_device_eval_batch_size=antiberta_config.get("batch_size", 32),
    max_steps=1000, #originally 225000
    save_steps=2500,#originally 2500
    logging_steps=2500, #originally 2500
    logging_dir=f"{out_dir}/logs",
    adam_beta2=0.98,
    adam_epsilon=1e-6,
    weight_decay=0.01,
    warmup_steps=100,
    learning_rate=1e-4,
    gradient_accumulation_steps=antiberta_config.get("gradient_accumulation_steps", 1),
    fp16=True, #comment this out for training on CPU
    evaluation_strategy="steps",
    seed=42,
    deepspeed=deepspeed_config,
)

### Setup the HuggingFace Trainer
trainer = Trainer(
    model=model,
    args=args,
    data_collator=collator,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["eval"]
)

trainer.train()
trainer.save_model(f"{out_dir}/{model_name}")

# make flag file
with open(flag_file, 'a'):
    os.utime(flag_file, None)