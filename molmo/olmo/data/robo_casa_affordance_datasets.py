#!/usr/bin/env python3
"""
RoboCasa Affordance Dataset for VLA Training

This module provides a dataset implementation for RoboCasa trajectory prediction.
Currently uses dummy data for debugging the training pipeline.
TODO: Implement actual data loading from RoboCasa dataset.
"""

import os
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional
from PIL import Image
from olmo.data.dataset import Dataset


class RoboCasaTrajectoryDataset(Dataset):
    """PyTorch Dataset for RoboCasa trajectory prediction training (3D flow matching only)."""
    
    # Class-level parameter for limiting examples in overfit split
    overfit_num_examples: Optional[int] = 10
    
    def __init__(
        self,
        data_dir: str = None,
        split: str = "train",
        action_chunking_horizon: int = 30,
        joint_names: Optional[List[str]] = None,
        normalize_coordinates: bool = True,
        stats_file: Optional[str] = None,
        trajectory_representation: str = "absolute",  # "absolute" or "delta"
        load_images: bool = True,
        # Dummy data configuration
        dummy_num_samples: int = 1000,
        dummy_image_size: tuple = (256, 256),
    ):
        """
        Initialize RoboCasa Trajectory Dataset (3D flow matching only).
        
        Args:
            data_dir: Path to data directory (not used in dummy mode)
            split: Dataset split ('train', 'test', 'overfit')
            action_chunking_horizon: Number of frames in action chunks
            joint_names: List of joint names to use
            normalize_coordinates: Whether to normalize coordinates
            stats_file: Path to stats file for normalization
            trajectory_representation: 'absolute' or 'delta'
            load_images: Whether to load images
            dummy_num_samples: Number of dummy samples to generate
            dummy_image_size: Size of dummy images (H, W)
        """
        self.data_dir = Path(data_dir) if data_dir is not None else None
        self.split = split
        self.action_chunking_horizon = action_chunking_horizon
        self.normalize_coordinates = normalize_coordinates
        self.stats_file = stats_file
        self.trajectory_representation = trajectory_representation
        self.overfit_num_examples = type(self).overfit_num_examples
        self.load_images = load_images
        self.dummy_image_size = dummy_image_size
        
        # Validate parameters
        assert trajectory_representation in ["absolute", "delta"], \
            f"trajectory_representation must be 'absolute' or 'delta', got {trajectory_representation}"
        
        # Load normalization stats if needed
        if self.normalize_coordinates and self.stats_file is not None and os.path.exists(self.stats_file):
            print(f"Loading trajectory normalization stats from {self.stats_file}...")
            stats = torch.load(self.stats_file)
            self.stats_mean = stats["mean"].float() if isinstance(stats["mean"], torch.Tensor) else torch.tensor(stats["mean"]).float()
            self.stats_std = stats["std"].float() if isinstance(stats["std"], torch.Tensor) else torch.tensor(stats["std"]).float()
        else:
            self.stats_mean = None
            self.stats_std = None
        
        # Default joint names matching TrajectoryDataset
        if joint_names is None:
            self.joint_names = [
                'leftThumbTip', 'leftIndexFingerTip', 'leftMiddleFingerTip', 'leftRingFingerTip', 'leftLittleFingerTip',
                'rightThumbTip', 'rightIndexFingerTip', 'rightMiddleFingerTip', 'rightRingFingerTip', 'rightLittleFingerTip'
            ]
        else:
            self.joint_names = joint_names
        
        self.num_joints = len(self.joint_names)
        self.coord_dim = 3  # Always 3D
        
        # Calculate number of samples based on split
        if self.split == "overfit" and self.overfit_num_examples is not None:
            self.num_samples = self.overfit_num_examples
        elif self.split == "test":
            self.num_samples = min(100, dummy_num_samples)
        else:
            self.num_samples = dummy_num_samples
        
        # Dummy task instructions
        self.dummy_instructions = [
            "Pick up the object on the placemat and place it on the shelf",
            "Pick up the object on the cutting board and place it in the basket",
            "Pick up the bottle and place it in the cabinet",
            "Pick up the object on the tray and place it on the plate",
            "Pick up the cup and place it in the drawer",
            "Move the object from the counter to the table",
            "Grasp the item and transfer it to the container",
            "Pick up the food item and place it in the pan",
        ]
        
        print(f"[RoboCasaTrajectoryDataset] Initialized with {self.num_samples} DUMMY samples for {split} split")
        print(f"  - action_chunking_horizon: {action_chunking_horizon}")
        print(f"  - trajectory_representation: {trajectory_representation}")
        print(f"  - num_joints: {self.num_joints}, coord_dim: {self.coord_dim}")
        print("  - WARNING: Using dummy data! Implement actual data loading for real training.")
    
    def __len__(self):
        return self.num_samples
    
    def get(self, idx, rng):
        """Get a single training example with dummy data."""
        # Use idx as seed for reproducibility
        np.random.seed(idx)
        
        # Generate dummy image
        if self.load_images:
            image = self._generate_dummy_image(idx)
        else:
            image = Image.fromarray(np.zeros((1, 1, 3), dtype=np.uint8))
        
        # Generate dummy trajectory: shape (action_chunking_horizon, num_joints, 3)
        trajectory = self._generate_dummy_trajectory(idx)
        
        # Get initial state (first frame, flattened)
        initial_state = trajectory[0].reshape(-1).astype(np.float32)
        
        # Convert to delta representation if requested
        if self.trajectory_representation == "delta":
            trajectory = self._convert_to_delta_representation(trajectory)
        
        # Get instruction
        instruction = self.dummy_instructions[idx % len(self.dummy_instructions)]
        
        # Flatten trajectory: [num_steps, num_joints, 3] -> [num_steps, num_joints*3]
        num_steps = trajectory.shape[0]
        trajectory_flattened = trajectory.reshape(num_steps, -1).astype(np.float32)
        
        return {
            'image': image,
            'state': initial_state,
            'message_list': [
                {
                    'label': instruction,
                    'points': trajectory,
                    'point_scale': None,
                    'style': 'trajectory_3d_fm',
                    'state': initial_state,
                }
            ],
            'trajectory_target': trajectory_flattened,
            'trajectory_shape': trajectory.shape,
            'expert_type': 1,  # Robot trajectory expert (for multi-expert routing)
            'metadata': {
                'image': image,
                'task_name': f'dummy_task_{idx % len(self.dummy_instructions)}',
                'frame_idx': idx,
                'output_2d_trajectory': False,
                'trajectory_representation': self.trajectory_representation,
            }
        }
    
    def _generate_dummy_image(self, idx: int) -> Image.Image:
        """Generate a dummy image with some visual features for debugging."""
        h, w = self.dummy_image_size
        image = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Background gradient
        for i in range(h):
            for j in range(w):
                image[i, j, 0] = int((i / h) * 128 + (idx % 128))
                image[i, j, 1] = int((j / w) * 128 + ((idx * 7) % 128))
                image[i, j, 2] = int(((i + j) / (h + w)) * 128 + ((idx * 13) % 128))
        
        # Add a simple "object" rectangle
        obj_x = (idx * 17) % (w - 40) + 20
        obj_y = (idx * 23) % (h - 40) + 20
        image[obj_y:obj_y+30, obj_x:obj_x+30] = [255, 200, 100]
        
        return Image.fromarray(image)
    
    def _generate_dummy_trajectory(self, idx: int) -> np.ndarray:
        """
        Generate a dummy 3D trajectory for debugging.
        
        Returns:
            np.ndarray of shape (action_chunking_horizon, num_joints, 3)
        """
        num_steps = self.action_chunking_horizon
        t = np.linspace(0, 2 * np.pi, num_steps)
        
        trajectory = np.zeros((num_steps, self.num_joints, 3), dtype=np.float32)
        
        for j in range(self.num_joints):
            phase = (idx + j) * 0.5
            amplitude = 0.1 + 0.05 * (j % 5)
            
            # 3D trajectory in camera frame (meter scale)
            start_x = 0.0 + 0.05 * (j % 5)
            start_y = 0.1 + 0.05 * (j // 5)
            start_z = 0.5 + 0.02 * j
            
            trajectory[:, j, 0] = start_x + amplitude * np.sin(t + phase)
            trajectory[:, j, 1] = start_y + amplitude * np.cos(t + phase * 0.7)
            trajectory[:, j, 2] = start_z + amplitude * 0.5 * np.sin(t * 0.5 + phase)
        
        return trajectory
    
    def _convert_to_delta_representation(self, trajectory: np.ndarray) -> np.ndarray:
        """Convert absolute positions to delta positions (velocities)."""
        deltas = trajectory[1:] - trajectory[:-1]
        last_delta = deltas[-1:]
        return np.concatenate([deltas, last_delta], axis=0)


# Alias for backwards compatibility
RobotCasaHandPositioningDataset = RoboCasaTrajectoryDataset


if __name__ == "__main__":
    print("Testing RoboCasaTrajectoryDataset with dummy data...\n")
    
    dataset = RoboCasaTrajectoryDataset(
        split="overfit",
        action_chunking_horizon=30,
        dummy_num_samples=100,
    )
    
    sample = dataset.get(0, None)
    print(f"\nSample output:")
    print(f"  Image shape: {np.array(sample['image']).shape}")
    print(f"  State shape: {sample['state'].shape}")
    print(f"  Trajectory target shape: {sample['trajectory_target'].shape}")
    print(f"  Trajectory original shape: {sample['trajectory_shape']}")
    print(f"  Style: {sample['message_list'][0]['style']}")
    print(f"  Label: {sample['message_list'][0]['label']}")
    print(f"\nDummy data generation working correctly!")
