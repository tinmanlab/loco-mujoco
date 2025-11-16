#!/usr/bin/env python3
import pickle

CHECKPOINT_PATH = "/home/tinman/loco-mujoco/my_amp_training/outputs/2025-10-31/02-01-09/AMPJax_saved.pkl"

print("Loading checkpoint...")
with open(CHECKPOINT_PATH, 'rb') as f:
    checkpoint = pickle.load(f)

print("\nCheckpoint keys:")
print(checkpoint.keys())

print("\nCheckpoint structure:")
for key in checkpoint.keys():
    print(f"\n{key}:")
    if hasattr(checkpoint[key], 'keys'):
        print(f"  Sub-keys: {list(checkpoint[key].keys())[:10]}")  # First 10 keys
    else:
        print(f"  Type: {type(checkpoint[key])}")
