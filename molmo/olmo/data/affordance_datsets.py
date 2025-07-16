import numpy as np
import datasets
import json
from PIL import Image
from olmo.data.dataset import Dataset

class HandPositioningDataset(Dataset):
    def __init__(self, data_path, split="train", keep_in_memory=False):
        """
        data_path: Path to your dataset
        split: "train", "validation", or "test"
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
        try:
            if isinstance(data_path, str) and not data_path.endswith('.json'):
                # Assume it's a HuggingFace dataset path
                dataset = datasets.load_from_disk(data_path)
                return dataset[split] if split in dataset else dataset
        except:
            pass
            
        # Fall back to JSON loading
        json_path = f"{data_path}/{split}.json" if not data_path.endswith('.json') else data_path
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            return data
        except FileNotFoundError:
            print(f"Could not find data file at {json_path}")
            print("Please make sure your data is in the correct format or create it using create_hf_dataset()")
            return []
    
    def __len__(self):
        return len(self.data)
    
    def get(self, item, rng):
        example = self.data[item]
        
        # Load image
        if isinstance(example["image"], str):
            image = Image.open(example["image"])
        else:
            image = example["image"]
        
        # Process hand positions
        hand_positions = self._process_hand_positions(example["hand_positions"])
        
        return {
            "image": image,
            "message_list": [
                {
                    "question": example["instruction"],
                    "points": hand_positions,
                    "point_scale": 100,  # Adjust based on your coordinate system
                    "style": "affordance"  # Changed from "hand_positioning" to "affordance"
                }
            ],
            "metadata": {
                "image_path": example.get("image_path", ""),
                "hand_data": example["hand_positions"]  # Keep original for debugging
            }
        }
    
    def _process_hand_positions(self, hand_data):
        """
        Convert your hand position data to the expected format.
        
        For both hands with fingertips + wrists, you might have:
        - Left hand: 5 fingertips + 1 wrist = 6 points
        - Right hand: 5 fingertips + 1 wrist = 6 points
        - Total: 12 points
        
        Returns: numpy array of shape (N, 2) where N is number of keypoints
        """
        points = []
        
        # Example structure - adjust based on your data format
        if "left_hand" in hand_data:
            # Add left hand points (fingertips + wrist)
            for fingertip in hand_data["left_hand"]["fingertips"]:
                points.append([fingertip["x"], fingertip["y"]])
            points.append([hand_data["left_hand"]["wrist"]["x"], 
                          hand_data["left_hand"]["wrist"]["y"]])
        
        if "right_hand" in hand_data:
            # Add right hand points (fingertips + wrist)
            for fingertip in hand_data["right_hand"]["fingertips"]:
                points.append([fingertip["x"], fingertip["y"]])
            points.append([hand_data["right_hand"]["wrist"]["x"], 
                          hand_data["right_hand"]["wrist"]["y"]])
        
        return np.array(points, dtype=np.float32)

    @staticmethod
    def create_hf_dataset(raw_data, output_path, splits=None):
        """
        Create a HuggingFace dataset from your raw data.
        
        Args:
            raw_data: Your raw data in whatever format you have
            output_path: Where to save the processed dataset
            splits: Optional dict like {"train": 0.8, "validation": 0.1, "test": 0.1}
        """
        # This is a template - you'll need to adapt it to your specific data format
        processed_data = []
        
        for item in raw_data:
            processed_item = {
                "image": item["image_path"],  # Adjust field names as needed
                "instruction": item["instruction"],  # Adjust field names as needed
                "hand_positions": item["hand_keypoints"],  # Adjust field names as needed
                "image_path": item["image_path"]
            }
            processed_data.append(processed_item)
        
        # Convert to HuggingFace dataset
        dataset = datasets.Dataset.from_list(processed_data)
        
        # Split if requested
        if splits:
            dataset = dataset.train_test_split(test_size=1-splits["train"], seed=42)
            train_data = dataset["train"]
            test_data = dataset["test"]
            
            if "validation" in splits:
                val_size = splits["validation"] / (splits["validation"] + splits["test"])
                test_val_split = test_data.train_test_split(test_size=val_size, seed=42)
                dataset = datasets.DatasetDict({
                    "train": train_data,
                    "test": test_val_split["train"],
                    "validation": test_val_split["test"]
                })
            else:
                dataset = datasets.DatasetDict({
                    "train": train_data,
                    "test": test_data
                })
        
        # Save to disk
        dataset.save_to_disk(output_path)
        print(f"Dataset saved to {output_path}")
        return dataset