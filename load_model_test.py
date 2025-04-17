from safetensors.torch import load_file
import torch

try:
    tensors = load_file("/REDACTED/PATH")
    print("Checkpoint is OK ✅")
    print(f"Keys in checkpoint: {list(tensors.keys())}")
except Exception as e:
    print("Checkpoint appears to be corrupted ❌")
    print(f"Error: {e}")

"""
try:
    # Use the specific file path from the error log for the rank that failed (rank 0 here)
    file_path = '/REDACTED/PATH'
    data = torch.load(file_path, map_location='cpu', weights_only=False)
    print("File loaded successfully (basic check).")
    # You could add more checks here, like iterating through keys/values if it's a dict
except Exception as e:
    print(f"Failed to load file: {e}")
    # This will likely reproduce your utf-8 error if the file is corrupt in that way.
"""