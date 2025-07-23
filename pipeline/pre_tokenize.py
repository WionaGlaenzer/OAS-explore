from datasets import load_dataset, DatasetDict, Dataset
from transformers import RobertaTokenizer
from tqdm import tqdm
import os
import sys

# Process arguments
training_file = sys.argv[1]
val_file = sys.argv[2]
test_file = sys.argv[3]
cache_dir = sys.argv[4]
out_dir = sys.argv[5]
tokenizer_path = sys.argv[6]

# Load tokenizer and construct output path
tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)
tokenized_path = os.path.join(out_dir, "tokenized")
os.makedirs(tokenized_path, exist_ok=True)

# Define the datasets for streaming
text_datasets = {
    "train": training_file,
    "eval": val_file,
    "test": test_file
}

# Set tokenizer config
max_length = 150
buffer_size = 1000000  # How many examples to batch before writing

def tokenize_and_save(split_name, file_path):
    """
    Reads in one split of the data and saves it at the specified path.

    Args:
        split_name (str): Name of the dataset split (train, eval, test).
        file_path (str): Path to the dataset file.
    """
    print(f"Streaming and tokenizing: {split_name}")
    
    stream = load_dataset("text", data_files={split_name: file_path}, cache_dir=cache_dir, split=split_name, streaming=True)
    buffer = []

    save_split_path = os.path.join(tokenized_path, split_name)
    os.makedirs(save_split_path, exist_ok=True)
    
    shard_id = 0
    total = 0

    for example in tqdm(stream, desc=f"Processing {split_name}"):
        tokens = tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_special_tokens_mask=True,
        )
        buffer.append(tokens)
        total += 1

        # If buffer is full, save it
        if len(buffer) >= buffer_size:
            dataset = Dataset.from_list(buffer)
            dataset.save_to_disk(os.path.join(save_split_path, f"shard_{shard_id}"))
            buffer = []
            shard_id += 1

    # Save any remaining examples
    if buffer:
        dataset = Dataset.from_list(buffer)
        dataset.save_to_disk(os.path.join(save_split_path, f"shard_{shard_id}"))

    print(f"{split_name} done. Total examples: {total}")


# Run for each split
tokenize_and_save("train", training_file)
tokenize_and_save("eval", val_file)
tokenize_and_save("test", test_file)
