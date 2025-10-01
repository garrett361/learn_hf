import json
from pathlib import Path

from safetensors import safe_open


class SafetensorWeights:
    def __init__(self, model_dir):
        """Initialize from a model directory containing sharded safetensors"""
        self.model_dir = Path(model_dir)
        self.index_file = self.model_dir / "model.safetensors.index.json"

        # Load the index file
        with open(self.index_file) as f:
            index_data = json.load(f)

        self.weight_map = index_data["weight_map"]  # Maps tensor names to shard files

    def __iter__(self):
        """Iterate over tensor names"""
        return iter(self.weight_map.keys())

    def __getitem__(self, tensor_name):
        """Get a specific tensor by name using bracket notation, loaded on CPU"""
        if tensor_name not in self.weight_map:
            available = list(self.weight_map.keys())[:10]  # Show first 10
            raise KeyError(
                f"Tensor '{tensor_name}' not found. Available: {available}..."
            )

        shard_file = self.weight_map[tensor_name]
        shard_path = self.model_dir / shard_file

        with safe_open(shard_path, framework="pt", device="cpu") as f:
            return f.get_tensor(tensor_name)

    def __len__(self):
        """Return the number of available tensors"""
        return len(self.weight_map)
