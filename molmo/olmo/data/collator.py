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
    # Keys always present for trajectory training
    TRAJECTORY_KEYS_REQUIRED = ["trajectory_target", "proprio_state", "expert_type"]
    # Keys only present for robot data (human samples won't have these)
    TRAJECTORY_KEYS_OPTIONAL = ["robot_actions"]

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
        out = {}
        
        # Text keys
        for key in self.TEXT_KEYS:
            if any(key in ex for ex in batch):
                if key == "subsegment_ids":
                    for ex in batch:
                        if "subsegment_ids" not in ex:
                            ex["subsegment_ids"] = np.ones_like(ex["input_tokens"])
                dtype = np.float32 if key == "loss_masks" else np.int64
                out[key] = _collate(
                    [ex.get(key) for ex in batch], self.max_sequence_length, dtype, pad=self.pad)

        # Image keys
        for key in self.IMAGE_KEYS:
            if any(key in ex for ex in batch):
                out[key] = _collate([ex.get(key) for ex in batch], self.max_crops, pad=self.pad)
        
        # Required trajectory keys (always present in all samples)
        for key in self.TRAJECTORY_KEYS_REQUIRED:
            if not any(key in ex for ex in batch):
                continue
            
            tensors = []
            for ex in batch:
                t = ex[key]
                if isinstance(t, np.ndarray):
                    t = torch.from_numpy(t)
                elif isinstance(t, (int, float)):
                    t = torch.tensor(t)
                tensors.append(t)
            
            # Scalar (expert_type)
            if tensors[0].dim() == 0:
                out[key] = torch.stack(tensors, dim=0)
            # All same shape - simple stack
            elif all(t.shape == tensors[0].shape for t in tensors):
                out[key] = torch.stack(tensors, dim=0)
            # Different shapes - pad to max
            else:
                max_dims = list(tensors[0].shape)
                for t in tensors[1:]:
                    for i in range(len(max_dims)):
                        max_dims[i] = max(max_dims[i], t.shape[i])
                
                padded = []
                dims = []
                for t in tensors:
                    p = torch.zeros(max_dims, dtype=t.dtype)
                    slices = [slice(0, s) for s in t.shape]
                    p[slices] = t
                    padded.append(p)
                    dims.append(t.shape[-1])
                out[key] = torch.stack(padded, dim=0)
                out[f"{key}_dims"] = torch.tensor(dims)
        
        # Optional trajectory keys (only present for robot samples)
        for key in self.TRAJECTORY_KEYS_OPTIONAL:
            if not any(key in ex for ex in batch):
                continue
            
            # Get first valid tensor to determine shape/dtype
            first = None
            for ex in batch:
                if key in ex:
                    t = ex[key]
                    if isinstance(t, np.ndarray):
                        t = torch.from_numpy(t)
                    first = t
                    break
            
            # Collect tensors (None for missing samples)
            tensors = []
            for ex in batch:
                if key in ex:
                    t = ex[key]
                    if isinstance(t, np.ndarray):
                        t = torch.from_numpy(t)
                    tensors.append(t)
                else:
                    tensors.append(None)
            
            # Find max dims across valid tensors
            max_dims = list(first.shape)
            for t in tensors:
                if t is not None:
                    for i in range(len(max_dims)):
                        max_dims[i] = max(max_dims[i], t.shape[i])
            
            # Pad all tensors (zeros for missing samples)
            padded = []
            dims = []
            for t in tensors:
                if t is None:
                    padded.append(torch.zeros(max_dims, dtype=first.dtype))
                    dims.append(0)  # 0 indicates missing
                else:
                    p = torch.zeros(max_dims, dtype=t.dtype)
                    slices = [slice(0, s) for s in t.shape]
                    p[slices] = t
                    padded.append(p)
                    dims.append(t.shape[-1])
            out[key] = torch.stack(padded, dim=0)
            out[f"{key}_dims"] = torch.tensor(dims)
        
        out["input_ids"] = out.pop("input_tokens")
        if "target_tokens" in out:
            out["labels"] = out.pop("target_tokens")
        if self.include_metadata:
            out["metadata"] = [ex.get("metadata", {}) for ex in batch]
        return out
