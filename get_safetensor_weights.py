import json
import os
from safetensors import safe_open
from pathlib import Path

def load_sharded_safetensors(model_dir):
    """Load metadata from sharded safetensors files"""
    model_dir = Path(model_dir)
    index_file = model_dir / "model.safetensors.index.json"

    # Load the index file
    with open(index_file) as f:
        index_data = json.load(f)

    weight_map = index_data["weight_map"]  # Maps tensor names to shard files
    return weight_map, model_dir

def get_tensor(tensor_name, weight_map, model_dir):
    """Get a specific tensor by name"""
    if tensor_name not in weight_map:
        available = list(weight_map.keys())[:10]  # Show first 10
        raise KeyError(f"Tensor '{tensor_name}' not found. Available: {available}...")

    shard_file = weight_map[tensor_name]
    shard_path = model_dir / shard_file

    with safe_open(shard_path, framework="pt", device="cpu") as f:
        return f.get_tensor(tensor_name)

def list_available_tensors(weight_map):
    """List all available tensor names"""
    return list(weight_map.keys())

# Usage example
model_directory = "/path/to/your/model"
weight_map, model_dir = load_sharded_safetensors(model_directory)

# List available tensors
available_tensors = list_available_tensors(weight_map)
print(f"Found {len(available_tensors)} tensors")
print("First 10 tensors:", available_tensors[:10])

# Load and inspect specific weights
tensor_name = "model.layers.0.self_attn.q_proj.weight"  # Example tensor name
try:
    weight = get_tensor(tensor_name, weight_map, model_dir)
    print(f"\n{tensor_name}:")
    print(f"Shape: {weight.shape}")
    print(f"Dtype: {weight.dtype}")
    print(f"Min: {weight.min().item():.6f}, Max: {weight.max().item():.6f}")
    print(f"Mean: {weight.mean().item():.6f}, Std: {weight.std().item():.6f}")
except KeyError as e:
    print(e)

