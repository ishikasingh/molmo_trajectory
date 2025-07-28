import json
import numpy as np
import datasets
from PIL import Image
from olmo.data.dataset import Dataset
from pathlib import Path

class HandPositioningDataset(Dataset):
    def __init__(self, data_path, split="train", keep_in_memory=False):
        """
        data_path: Path to your dataset
        split: "train", "validation", or "test"
        note: the order for each keypoint is:
        [left hand, left thumb, left index, left middle, left ring, left pinky, right hand, right thumb, right index, right middle, right ring, right pinky]
        """
        self.split = split
        self.data_path = data_path
        # Load your data - this depends on your data format
        self.data = self._load_data(data_path, split)
        
    def _load_data(self, data_path, split):
        """
        Load your data from whatever format you have.
        Should return a list/dataset where each item contains:
        - image: path to image or PIL Image
        - instruction: language instruction text
        - hand_positions: dict with hand keypoints
        """
        # Try to load from HuggingFace dataset first
        if isinstance(data_path, str) and not data_path.endswith('.json'):
            # Assume it's a HuggingFace dataset path
            dataset = datasets.load_from_disk(data_path)
            return dataset[split] if split in dataset else dataset
        else:
            raise ValueError(f"Invalid data path: {data_path}")
    
    def __len__(self):
        return len(self.data)
    
    def get(self, item, rng):
        example = self.data[item]
        
        # Load image - handle both PIL Image objects and file paths
        if "image" in example:
            image_data = example["image"]
        elif "image_path" in example:
            image_data = example["image_path"]
            image_data = Path(self.data_path).parent / "affordance_images" / image_data
        else:
            raise ValueError(f"Invalid image data format: {example}")
        
        if isinstance(image_data, str):
            image = Image.open(image_data)
        elif hasattr(image_data, 'mode'):  # PIL Image object
            image = image_data
        else:
            # Assume it's already in the right format
            image = image_data
        
        # Process hand positions
        hand_positions = self._process_hand_positions(example["hand_positions"])
        
        # Safely get metadata with fallbacks
        try:
            metadata = example["metadata"] if "metadata" in example else {}
        except (KeyError, TypeError):
            metadata = {}
        
        # Safe access to optional fields
        def safe_get(obj, key, default):
            try:
                if hasattr(obj, 'get'):
                    return obj.get(key, default)
                elif key in obj:
                    return obj[key]
                else:
                    return default
            except (KeyError, TypeError, AttributeError):
                print(f"KeyError: {key} not found in {obj}")
                return default
        
        return {
            "image": image,
            "message_list": [
                {
                    "label": example["instruction"],
                    "points": hand_positions,
                    "point_scale": 100,  # Our coordinates are already in percentage (0-100)
                    "style": "affordance"
                }
            ],
            "metadata": {
                "image": image,  # Add this line - put image in metadata too
                "image_path": safe_get(example, "image_path", ""),
                "hand_data": example["hand_positions"],  # Keep original for debugging
                "video_id": safe_get(metadata, "video_id", ""),
                "frame_idx": safe_get(metadata, "frame_idx", 0),
                "image_size": safe_get(metadata, "image_size", [1920, 1080])
            }
        }
    
    def _process_hand_positions(self, hand_data):
        """
        Convert hand position data to the expected format for molmo.
        
        Input format from HuggingFace dataset:
        {
            "points": [hand_data1, hand_data2, ...],  # Raw hand data for each detected hand
            "labels": ["left_hand", "right_hand", ...]  # Hand type labels
        }
        
        Where each hand_data contains the actual keypoint coordinates.
        
        Returns: numpy array of shape (N, 2) where N is number of keypoints
        """
        points_list = []
        if isinstance(hand_data, dict) and "points" in hand_data:
            # New format from HuggingFace dataset
            
            # Extract coordinate points from each hand's data
            for hand_keypoints in hand_data["points"]:
                # If hand_keypoints is already a list of coordinates
                if len(hand_keypoints) == 2 and isinstance(hand_keypoints[0], (int, float)):
                    # Single coordinate pair
                    points_list.append([float(hand_keypoints[0]), float(hand_keypoints[1])])
                else:
                    raise ValueError(f"Invalid hand keypoints format: {hand_keypoints}")
        else:
            raise ValueError(f"Invalid hand data format: {hand_data}")
            
            
        return np.array(points_list, dtype=np.float32)
