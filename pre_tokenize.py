from datasets import load_dataset, DatasetDict
from transformers import RobertaTokenizer
import os
import sys

training_file = sys.argv[1]
test_file = sys.argv[2]
val_file = sys.argv[3]
cache_dir = sys.argv[4]
out_dir = sys.argv[5]
tokenizer_path = sys.argv[6]

tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)
tokenized_path = os.path.join(out_dir, "tokenized")
if not os.path.exists(tokenized_path):
    os.makedirs(tokenized_path)

text_datasets = {
    "train": [training_file],
    "eval": [test_file],
    "test": [val_file]
}

# Load dataset
dataset = load_dataset("text", data_files=text_datasets, cache_dir=cache_dir)

tokenized_dataset = dataset.map(
        lambda z: tokenizer(
            z["text"],
            padding="max_length",
            truncation=True,
            max_length=150,
            return_special_tokens_mask=True,
        ),
        batched=True,
        num_proc=12,
        remove_columns=["text"],
)

# Save to disk for reuse
tokenized_dataset.save_to_disk(out_dir)
print(f"Tokenized datasets saved to: {out_dir}")
