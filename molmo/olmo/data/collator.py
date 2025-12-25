from typing import Dict, Any, List

import numpy as np
import torch

numpy_to_torch_dtype_dict = {
    np.dtype("bool"): torch.bool,
    np.dtype("uint8"): torch.uint8,
    np.dtype("int8"): torch.int8,
    np.dtype("int16"): torch.int16,
    np.dtype("int32"): torch.int32,
    np.dtype("int64"): torch.int64,
    np.dtype("float16"): torch.float16,
    np.dtype("float32"): torch.float32,
    np.dtype("float64"): torch.float64,
    np.dtype("complex64"): torch.complex64,
    np.dtype("complex128"): torch.complex128,
}


def _collate(tensors, max_sequence_length=None, dtype=None, pad=False, pad_value=-1):
    if pad == "to_max":
        max_len = max_sequence_length
        tensor = [x for x in tensors if x is not None][0]
        arr = np.full([len(tensors), max_len] + list(tensor.shape[1:]), pad_value,
                      dtype=dtype or tensor.dtype)
    else:
        max_len = max((0 if x is None else x.shape[0]) for x in tensors)
        if max_sequence_length:
            max_len = min(max_len, max_sequence_length)
        elif pad is not None:
            raise NotImplementedError(pad)

        arr = np.full([len(tensors), max_len] + list(tensors[0].shape[1:]), pad_value,
                      dtype=dtype or tensors[0].dtype)

    for ix, tensor in enumerate(tensors):
        if tensor is not None:
            arr[ix, :len(tensor)] = tensor[:max_len]
    return torch.from_numpy(arr)


class MMCollator:
    """Converts list of examples from our datasets into a tensor batch"""

    TEXT_KEYS = ["input_tokens", "target_tokens", "loss_masks", "subsegment_ids", "position_ids"]
    IMAGE_KEYS = ["images", "image_masks", "image_input_idx",]
    TRAJECTORY_KEYS = ["trajectory_target", "proprio_state", "expert_type"]

    def __init__(self, max_sequence_length=None, include_metadata=True, pad=None,
                 max_crops=None):
        """
        :param max_sequence_length: truncate examples longer than this length
        :param include_metadata: whether to include the metadata in the out batch
        :param pad: how to pad the tensors
        :param max_crops: max number of crops to use if padding to the max sequence length
        """
        if pad:
            assert max_sequence_length is not None and max_crops is not None
        self.max_sequence_length = max_sequence_length
        self.max_crops = max_crops
        self.include_metadata = include_metadata
        self.pad = pad

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        assert len(batch) > 0, "Given an empty batch"
        keys = batch[0].keys()
        out = {}
        for key in self.TEXT_KEYS:
            # If one examples has subsegment_ids, all examples need it so with ones
            # matching the input tokens
            if any(key in ex for ex in batch):
                if key == "subsegment_ids":
                    for ex in batch:
                        if "subsegment_ids" not in ex:
                            ex["subsegment_ids"] = np.ones_like(ex["input_tokens"])

                dtype = np.float32 if key == "loss_masks" else np.int64
                out[key] = _collate(
                    [ex.get(key) for ex in batch], self.max_sequence_length, dtype, pad=self.pad)

        for key in self.IMAGE_KEYS:
            if any(key in ex for ex in batch):
                out[key] = _collate([ex.get(key) for ex in batch], self.max_crops, pad=self.pad)
        
        # Handle trajectory targets and proprioception states
        # In multi-expert mode (shared mode with human+robot), dimensions may vary per sample
        # - trajectory_target: 2D tensor (action_horizon x action_dim)
        # - proprio_state: 1D tensor (proprio_dim)
        # - expert_type: scalar (0=human, 1=robot)
        for key in self.TRAJECTORY_KEYS:
            if any(key in ex for ex in batch):
                trajectory_tensors = []
                for ex in batch:
                    if key in ex:
                        traj = ex[key]
                        if isinstance(traj, np.ndarray):
                            traj = torch.from_numpy(traj)
                        elif isinstance(traj, (int, float)):
                            # Handle scalar values like expert_type
                            traj = torch.tensor(traj)
                        trajectory_tensors.append(traj)
                if trajectory_tensors:
                    # Check if all tensors have the same shape (for stacking)
                    # Case 1: Scalar tensors (like expert_type) - dim() == 0
                    if trajectory_tensors[0].dim() == 0:
                        out[key] = torch.stack(trajectory_tensors, dim=0)
                    
                    # Case 2: All tensors have the same shape - simple stack
                    elif all(t.shape == trajectory_tensors[0].shape for t in trajectory_tensors):
                        out[key] = torch.stack(trajectory_tensors, dim=0)
                    
                    # Case 3: 1D tensors with different sizes (proprio_state in shared mode)
                    elif trajectory_tensors[0].dim() == 1:
                        # Find max dimension
                        max_dim = max(t.shape[0] for t in trajectory_tensors)
                        
                        # Pad each tensor to max dimension
                        padded_tensors = []
                        dim_mask = []  # Track valid dimensions per sample
                        for t in trajectory_tensors:
                            padded = torch.zeros(max_dim, dtype=t.dtype)
                            padded[:t.shape[0]] = t
                            padded_tensors.append(padded)
                            dim_mask.append(t.shape[0])
                        
                        out[key] = torch.stack(padded_tensors, dim=0)
                        # Store actual dimension per sample for masking
                        out[f"{key}_dims"] = torch.tensor(dim_mask)
                    
                    # Case 4: 2D+ tensors with different shapes (trajectory_target in shared mode)
                    else:
                        # Find max dimensions across all axes
                        max_dims = list(trajectory_tensors[0].shape)
                        for t in trajectory_tensors[1:]:
                            for i in range(len(max_dims)):
                                max_dims[i] = max(max_dims[i], t.shape[i])
                        
                        # Pad each tensor to max dimensions
                        padded_tensors = []
                        action_dim_mask = []  # Track valid action_dim (last dim) per sample
                        for t in trajectory_tensors:
                            # Create padded tensor filled with zeros
                            padded = torch.zeros(max_dims, dtype=t.dtype)
                            # Copy original tensor into padded tensor
                            slices = [slice(0, s) for s in t.shape]
                            padded[slices] = t
                            padded_tensors.append(padded)
                            # Store actual action_dim for this sample (last dimension)
                            action_dim_mask.append(t.shape[-1])
                        
                        out[key] = torch.stack(padded_tensors, dim=0)
                        # Store action_dim per sample for loss masking
                        out[f"{key}_dims"] = torch.tensor(action_dim_mask)
        
        out["input_ids"] = out.pop("input_tokens")
        if "target_tokens" in out:
            out["labels"] = out.pop("target_tokens")
        if self.include_metadata:
            out["metadata"] = [ex.get("metadata", {}) for ex in batch]
        return out
