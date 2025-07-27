import json
import numpy as np
import datasets
from PIL import Image
from olmo.data.dataset import Dataset
from pathlib import Path
from pprint import pprint
import time
import random

import torch
from huggingface_hub import HfApi

import lerobot
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

class RobotCasaHandPositioningDataset(Dataset):
    def __init__(self, max_retries=5, retry_delay=1):
        """
        data_path: Path to your dataset
        split: "train", "validation", or "test"
        max_retries: Maximum number of retry attempts for loading dataset
        retry_delay: Base delay between retries (will be randomized with jitter)
        """
        self.repo_id = "ishika/robocasa_gr1_tabletop_tasks_fingertips"
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Load metadata with retry logic
        self.metadata = LeRobotDatasetMetadata(self.repo_id)
        # Load dataset with retry logic
        self.data = self._load_with_retry(
            lambda: LeRobotDataset(self.repo_id),
            "LeRobotDataset"
        )

    def _load_with_retry(self, load_func, component_name):
        """
        Retry loading a component with exponential backoff and jitter.
        
        Args:
            load_func: Function that loads the component
            component_name: Name of the component for error messages
            
        Returns:
            The loaded component
            
        Raises:
            Exception: If all retries fail
        """
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                print(f"Attempting to load {component_name} (attempt {attempt + 1}/{self.max_retries})")
                return load_func()
                
            except Exception as e:
                last_exception = e
                print(f"Failed to load {component_name} on attempt {attempt + 1}: {str(e)}")
                
                if attempt < self.max_retries - 1:
                    # Calculate delay with exponential backoff and jitter
                    delay = self.retry_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                else:
                    print(f"All {self.max_retries} attempts to load {component_name} failed.")
                    raise last_exception
        
        # This should never be reached, but just in case
        raise last_exception
    
    def __len__(self):
        return len(self.data)
    
    def get(self, item, rng):
        example = self.data[item]
        image_tensor = example["observation.images.egoview"]  # torch [3, 256, 256]
        # Convert torch tensor to PIL Image
        image = Image.fromarray(
            (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        )

        # Get image dimensions for normalization
        img_h, img_w = image.size[1], image.size[0]  # PIL Image size is (width, height)

        hand_position_2d = example["observation.condition_2D"].reshape(12, 2)  # torch [12, 2]
        
        # Normalize the hand positions to percentage coordinates (vectorized)
        hand_positions = (hand_position_2d.numpy() / [img_w, img_h]) * 100.0

        language_instruction = example["task"]
        language_instruction = self.language_mapping(language_instruction)
        
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
                # "image": image,  # Add this line - put image in metadata too
                # "image_path": safe_get(example, "image_path", ""),
                # "hand_data": example["hand_positions"],  # Keep original for debugging
                # "video_id": safe_get(metadata, "video_id", ""),
                # "frame_idx": safe_get(metadata, "frame_idx", 0),
                # "image_size": safe_get(metadata, "image_size", [1920, 1080])
            }
        }
    def language_mapping(self, language_instruction):
        mapping_mapping = {
            "PosttrainPnPNovelFromPlacematToTieredshelfSplitA": "Pick up the object on the placemat and place it on the shelf",
            "PosttrainPnPNovelFromCuttingboardToBasketSplitA": "Pick up the object on the cutting board and place it in the basket",
            "PosttrainPnPNovelFromTrayToTieredshelfSplitA": "Pick up the object on the tray and place it on the tiered shelf",
            "PnPBottleToCabinetClose": "Pick up the bottle and place it in the cabinet",
            "PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA": "Pick up the object on the cutting board and place it in the basket",
            "PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA": "Pick up the object on the cutting board and place it in the cardboard box",
            "PosttrainPnPNovelFromPlateToPanSplitA": "Pick up the object on the plate and place it on the pan",
            "PosttrainPnPNovelFromPlacematToBasketSplitA": "Pick up the object on the placemat and place it in the basket",
            "PosttrainPnPNovelFromTrayToCardboardboxSplitA": "Pick up the object on the tray and place it in the cardboard box",
            "PosttrainPnPNovelFromCuttingboardToPotSplitA": "Pick up the object on the cutting board and place it in the pot",
            "PnPWineToCabinetClose": "Pick up the wine and place it in the cabinet",
            "PosttrainPnPNovelFromTrayToPotSplitA": "Pick up the object on the tray and place it in the pot",
            "PosttrainPnPNovelFromPlateToBowlSplitA": "Pick up the object on the plate and place it in the bowl",
            "PosttrainPnPNovelFromTrayToPlateSplitA": "Pick up the object on the tray and place it on the plate",
            "PnPMilkToMicrowaveClose": "Pick up the milk and place it in the microwave",
            "PnPPotatoToMicrowaveClose": "Pick up the potato and place it in the microwave",
            "PosttrainPnPNovelFromCuttingboardToPanSplitA": "Pick up the object on the cutting board and place it on the pan",
            "PosttrainPnPNovelFromPlacematToPlateSplitA": "Pick up the object on the placemat and place it on the plate",
            "PnPCupToDrawerClose": "Pick up the cup and place it in the drawer",
            "PnPCanToDrawerClose": "Pick up the can and place it in the drawer"
        }
        return mapping_mapping[language_instruction]