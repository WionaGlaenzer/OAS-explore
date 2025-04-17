from safetensors.torch import load_file
import torch

try:
    tensors = load_file("/cluster/project/reddy/wglaenzer/final_training_testing_val_data/Soto-HIP1/model/checkpoint-156250/model.safetensors")
    print("Checkpoint is OK ✅")
    print(f"Keys in checkpoint: {list(tensors.keys())}")
except Exception as e:
    print("Checkpoint appears to be corrupted ❌")
    print(f"Error: {e}")

"""
try:
    # Use the specific file path from the error log for the rank that failed (rank 0 here)
    file_path = '/cluster/project/reddy/wglaenzer/final_training_testing_val_data/Soto-HIP2/model/checkpoint-2500/global_step2500/zero_pp_rank_3_mp_rank_00_optim_states.pt'
    data = torch.load(file_path, map_location='cpu', weights_only=False)
    print("File loaded successfully (basic check).")
    # You could add more checks here, like iterating through keys/values if it's a dict
except Exception as e:
    print(f"Failed to load file: {e}")
    # This will likely reproduce your utf-8 error if the file is corrupt in that way.
"""