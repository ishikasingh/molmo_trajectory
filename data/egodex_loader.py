#!/usr/bin/env python3
"""
EgoDex VLA Dataset Loader

A PyTorch dataset for loading the EgoDex dataset for Vision-Language-Action (VLA) training.
This dataset provides an index-based interface where each index corresponds to a specific
frame from a video, along with the corresponding language instruction and future trajectory.

Usage:
    dataset = EgoDexVLADataset(data_dir="/path/to/egodex", split="train", action_chunking_horizon=10)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
"""

import os
import h5py
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm


class EgoDexVLADataset(Dataset):
    """
    PyTorch Dataset for EgoDex VLA training.
    
    Each index corresponds to a specific frame from a video, providing:
    - Image: Current frame from the video
    - Instruction: Language description of the task
    - Trajectory: Future N steps of hand/camera trajectories
    
    Args:
        data_dir: Path to EgoDex dataset directory
        split: Dataset split ("train", "test", or "extra")
        action_chunking_horizon: Number of future trajectory steps to predict
        image_size: Target image size (height, width)
        joint_names: List of joint names to include in trajectory
        use_confidence_filter: Whether to filter out low-confidence poses
        confidence_threshold: Minimum confidence threshold (0-1)
        cache_frames: Whether to cache loaded frames in memory
        transform: Optional transforms to apply to the image
        output_2d_trajectory: Whether to output 2D or 3D trajectory
        normalize_2d_coordinates: Whether to normalize 2D coordinates to 0-100 scale
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        action_chunking_horizon: int = 10,
        image_size: Tuple[int, int] = (256, 256),
        joint_names: Optional[List[str]] = None,
        use_confidence_filter: bool = False,
        confidence_threshold: float = 0.5,
        cache_frames: bool = False,
        transform: Optional[transforms.Compose] = None,
        output_2d_trajectory: bool = False,
        normalize_2d_coordinates: bool = True,  # New parameter
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.action_chunking_horizon = action_chunking_horizon
        self.image_size = image_size
        self.use_confidence_filter = use_confidence_filter
        self.confidence_threshold = confidence_threshold
        self.cache_frames = cache_frames
        self.transform = transform
        self.output_2d_trajectory = output_2d_trajectory
        self.normalize_2d_coordinates = normalize_2d_coordinates  # New parameter
        
        # Default joint names for VLA training
        if joint_names is None:
            # self.joint_names = ['leftHand', 'rightHand', 'camera']
            # self.joint_names = ['leftHand', 'leftThumbTip', 'leftIndexFingerTip', 'leftMiddleFingerTip', 'leftRingFingerTip', 'leftLittleFingerTip',
            #                'rightHand', 'rightThumbTip', 'rightIndexFingerTip', 'rightMiddleFingerTip', 'rightRingFingerTip', 'rightLittleFingerTip',
            #                'camera']
            self.joint_names = ['leftThumbTip', 'leftIndexFingerTip',
                           'rightThumbTip', 'rightIndexFingerTip', ]
        else:
            self.joint_names = joint_names
            
        # Frame cache for faster loading
        self.frame_cache = {} if cache_frames else None
        
        # Build index mapping
        print(f"Building index mapping for {split} split...")
        self.index_mapping = self._build_index_mapping()
        print(f"Loaded {len(self.index_mapping)} samples from {split} split")
        
    def _build_index_mapping(self) -> List[Dict]:
        """Build a flat index mapping across all videos and frames."""
        index_mapping = []
        
        # Determine split directory
        if self.split == "train":
            # Training data is split across multiple parts
            split_dirs = []
            for part in ["part1", "part2", "part3", "part4", "part5"]:
                part_dir = self.data_dir / part
                if part_dir.exists():
                    split_dirs.append(part_dir)
        elif self.split == "test":
            split_dirs = [self.data_dir / "test"]
        else:
            raise ValueError(f"Invalid split: {self.split}. Must be 'train' or 'test'")
        
        # Process each split directory
        for split_dir in split_dirs:
            if not split_dir.exists():
                print(f"Warning: {split_dir} does not exist, skipping...")
                continue
                
            print(f"Processing {split_dir}...")
            
            # Find all task directories within this split
            task_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
            print(f"Found {len(task_dirs)} task directories in {split_dir}")
            
            for task_dir in tqdm(task_dirs, desc=f"Processing {split_dir.name}"):
                task_name = task_dir.name
                print(f"  Processing task: {task_name}")
                
                # Find all video files in this task directory
                video_files = list(task_dir.glob("*.mp4"))
                print(f"    Found {len(video_files)} video files in task {task_name}")
                
                for video_file in video_files:
                    hdf5_file = video_file.with_suffix('.hdf5')
                    if not hdf5_file.exists():
                        print(f"    Warning: No corresponding HDF5 file for {video_file}")
                        continue
                    
                    # Get video properties
                    cap = cv2.VideoCapture(str(video_file))
                    if not cap.isOpened():
                        print(f"    Warning: Could not open video {video_file}")
                        continue
                        
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    # fps = cap.get(cv2.CAP_PROP_FPS)
                    fps = 30
                    cap.release()
                    
                    if frame_count < self.action_chunking_horizon:
                        print(f"    Warning: Video {video_file} has only {frame_count} frames, skipping...")
                        continue
                    
                    # Check HDF5 file for trajectory data
                    try:
                        with h5py.File(hdf5_file, 'r') as f:
                            # Verify required joints exist
                            missing_joints = []
                            for joint in self.joint_names:
                                if f'transforms/{joint}' not in f:
                                    missing_joints.append(joint)
                            
                            if missing_joints:
                                print(f"    Warning: Missing joints {missing_joints} in {hdf5_file}")
                                continue
                                
                            # Get trajectory length
                            trajectory_length = f[f'transforms/{self.joint_names[0]}'].shape[0]
                            
                            # Add each valid frame as a separate index
                            for frame_idx in range(trajectory_length - self.action_chunking_horizon):
                                index_mapping.append({
                                    'video_path': str(video_file),
                                    'hdf5_path': str(hdf5_file),
                                    'frame_idx': frame_idx,
                                    'task_name': task_name,  # Use the actual task directory name
                                    'episode_id': video_file.stem,
                                    'total_frames': frame_count,
                                    'trajectory_length': trajectory_length
                                })
                                
                    except Exception as e:
                        print(f"    Error processing {hdf5_file}: {e}")
                        continue
        
        return index_mapping
    
    def __len__(self):
        return len(self.index_mapping)
    
    def __getitem__(self, idx):
        if idx >= len(self.index_mapping):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.index_mapping)}")
            
        mapping = self.index_mapping[idx]
        
        # Load image frame
        image = self._load_frame(mapping['video_path'], mapping['frame_idx'])
        
        # Load trajectory data (current + future steps)
        trajectory = self._load_trajectory(
            mapping['hdf5_path'], 
            mapping['frame_idx'], 
            self.action_chunking_horizon
        )
        
        # Transform trajectory based on output type
        if self.output_2d_trajectory:
            # Project 3D trajectory to 2D image plane
            trajectory_2d = self._project_trajectory_to_2d(
                mapping['hdf5_path'],
                mapping['frame_idx'],
                trajectory
            )
            final_trajectory = trajectory_2d
        else:
            # Transform to camera frame (3D)
            trajectory_camera_frame = self._transform_trajectory_to_camera_frame(
                mapping['hdf5_path'],
                mapping['frame_idx'],
                trajectory
            )
            final_trajectory = trajectory_camera_frame
        
        # Load language instruction
        instruction = self._load_instruction(mapping['hdf5_path'])
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'instruction': instruction,
            'trajectory': final_trajectory,
            'metadata': {
                'task_name': mapping['task_name'],
                'episode_id': mapping['episode_id'],
                'frame_idx': mapping['frame_idx'],
                'video_path': mapping['video_path'],
                'hdf5_path': mapping['hdf5_path'],
                'output_2d_trajectory': self.output_2d_trajectory,
            }
        }
    
    def _load_frame(self, video_path: str, frame_idx: int) -> torch.Tensor:
        """Load a specific frame from video."""
        # Check cache first
        cache_key = f"{video_path}_{frame_idx}"
        if self.frame_cache is not None and cache_key in self.frame_cache:
            return self.frame_cache[cache_key]
        
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError(f"Could not load frame {frame_idx} from {video_path}")
        
        # Convert BGR to RGB and resize
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # frame = cv2.resize(frame, (self.image_size[1], self.image_size[0]))  # cv2.resize uses (width, height)
        
        # Convert to tensor
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        
        # Cache if enabled
        if self.frame_cache is not None:
            self.frame_cache[cache_key] = frame_tensor
        
        return frame_tensor
    
    def _load_trajectory(self, hdf5_path: str, start_frame: int, num_steps: int) -> torch.Tensor:
        """Load trajectory data for current and future steps."""
        with h5py.File(hdf5_path, 'r') as f:
            trajectories = {}
            confidences = {}
            
            # Load trajectories and confidences for each joint
            for joint in self.joint_names:
                if f'transforms/{joint}' in f:
                    transforms = f[f'transforms/{joint}'][start_frame:start_frame + num_steps]
                    # Extract position (x, y, z) from 4x4 transform matrices
                    positions = transforms[:, :3, 3]  # Shape: [num_steps, 3]
                    trajectories[joint] = positions
                
                # Load confidences if available
                if f'confidences/{joint}' in f:
                    confidences[joint] = f[f'confidences/{joint}'][start_frame:start_frame + num_steps]
            
            # Apply confidence filtering if enabled
            if self.use_confidence_filter and confidences:
                valid_mask = np.ones(num_steps, dtype=bool)
                for joint in self.joint_names:
                    if joint in confidences:
                        joint_valid = confidences[joint] >= self.confidence_threshold
                        valid_mask = valid_mask & joint_valid
                
                # Filter trajectories
                for joint in self.joint_names:
                    if joint in trajectories:
                        trajectories[joint] = trajectories[joint][valid_mask]
            
            # Combine into single trajectory tensor
            # Shape: [num_steps, num_joints, 3]
            trajectory_list = []
            for joint in self.joint_names:
                if joint in trajectories:
                    trajectory_list.append(trajectories[joint])
                else:
                    # Fill with zeros if joint not found
                    trajectory_list.append(np.zeros((num_steps, 3)))
            
            trajectory = np.stack(trajectory_list, axis=1)
            return torch.from_numpy(trajectory).float()
    
    def _load_instruction(self, hdf5_path: str) -> str:
        """Load language instruction from HDF5 file attributes."""
        with h5py.File(hdf5_path, 'r') as f:
            instruction = f.attrs.get('llm_description', 'No instruction available')
            if isinstance(instruction, bytes):
                instruction = instruction.decode('utf-8')
            return instruction
    
    def get_episode_info(self, idx: int) -> Dict:
        """Get detailed information about a specific sample."""
        if idx >= len(self.index_mapping):
            raise IndexError(f"Index {idx} out of range")
            
        mapping = self.index_mapping[idx]
        
        # Load additional info from HDF5
        with h5py.File(mapping['hdf5_path'], 'r') as f:
            info = {
                'task_name': mapping['task_name'],
                'episode_id': mapping['episode_id'],
                'frame_idx': mapping['frame_idx'],
                'total_frames': mapping['total_frames'],
                'trajectory_length': mapping['trajectory_length'],
                'video_path': mapping['video_path'],
                'hdf5_path': mapping['hdf5_path'],
                'instruction': self._load_instruction(mapping['hdf5_path']),
                'joint_names': self.joint_names,
                'action_chunking_horizon': self.action_chunking_horizon
            }
            
            # Add joint confidence info if available
            if 'confidences' in f:
                confidences = {}
                for joint in self.joint_names:
                    if f'confidences/{joint}' in f:
                        conf = f[f'confidences/{joint}'][mapping['frame_idx']:mapping['frame_idx'] + self.action_chunking_horizon]
                        confidences[joint] = {
                            'mean': float(np.mean(conf)),
                            'min': float(np.min(conf)),
                            'max': float(np.max(conf))
                        }
                info['confidences'] = confidences
        
        return info

    def _transform_trajectory_to_camera_frame(self, hdf5_path: str, current_frame: int, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Transform trajectory from world frame to camera frame of the current timestep.
        """
        with h5py.File(hdf5_path, 'r') as f:
            if 'transforms/camera' not in f:
                print(f"Warning: No camera transform found in {hdf5_path}")
                return trajectory
            
            camera_transform = f['transforms/camera'][current_frame]  # Shape: [4, 4]
            camera_transform = torch.from_numpy(camera_transform).float()
            
            # The camera transform is camera-to-world, so we need its inverse for world-to-camera
            camera_transform_inv = torch.inverse(camera_transform)
            
            num_steps, num_joints, _ = trajectory.shape
            
            trajectory_homo = torch.cat([
                trajectory,
                torch.ones(num_steps, num_joints, 1)
            ], dim=-1)
            
            trajectory_homo_flat = trajectory_homo.view(-1, 4)
            
            # Use the inverse transform (world-to-camera)
            transformed_homo = trajectory_homo_flat @ camera_transform_inv.T
            
            transformed_3d = transformed_homo[:, :3]
            transformed_trajectory = transformed_3d.view(num_steps, num_joints, 3)
            
            return transformed_trajectory

    def _project_trajectory_to_2d(self, hdf5_path: str, current_frame: int, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Project 3D trajectory to 2D image plane using camera intrinsics.
        
        Args:
            hdf5_path: Path to HDF5 file
            current_frame: Current frame index
            trajectory: 3D trajectory tensor of shape [num_steps, num_joints, 3]
            
        Returns:
            2D projected trajectory tensor of shape [num_steps, num_joints, 2]
            - If normalize_2d_coordinates=True: normalized to 0-100 scale
            - If normalize_2d_coordinates=False: in pixel coordinates
        """
        with h5py.File(hdf5_path, 'r') as f:
            # Get camera intrinsics
            if 'camera/intrinsic' in f and f['camera/intrinsic'][()].shape == (3, 3):
                intrinsic = f['camera/intrinsic'][()]  # Shape: [3, 3]
                # Extract image dimensions from intrinsics (principal point should be near image center)
                img_width = intrinsic[0, 2] * 2  # cx * 2 approximates width
                img_height = intrinsic[1, 2] * 2  # cy * 2 approximates height
            else:
                # Use default intrinsics if not available
                intrinsic = np.array([
                    [736.6339, 0., 960.], 
                    [0., 736.6339, 540.], 
                    [0., 0., 1.]
                ])
                img_width = 1920  # Default 1080p width
                img_height = 1080  # Default 1080p height
            
            intrinsic = torch.from_numpy(intrinsic).float()
            
            # Get camera transform at current frame
            if 'transforms/camera' not in f:
                print(f"Warning: No camera transform found in {hdf5_path}")
                return trajectory[:, :, :2]  # Return x,y coordinates only
            
            camera_transform = f['transforms/camera'][current_frame]  # Shape: [4, 4]
            camera_transform = torch.from_numpy(camera_transform).float()
            
            # Transform trajectory to camera frame first
            trajectory_camera_frame = self._transform_trajectory_to_camera_frame(
                hdf5_path, current_frame, trajectory
            )
            
            # Project 3D points to 2D image plane
            num_steps, num_joints, _ = trajectory_camera_frame.shape
            
            # Reshape for batch processing: [num_steps * num_joints, 3]
            points_3d = trajectory_camera_frame.view(-1, 3)
            
            # Direct projection using camera intrinsics
            # K @ [X, Y, Z]^T = [u, v, w]^T
            points_2d_homo = points_3d @ intrinsic.T  # Shape: [num_steps * num_joints, 3]
            
            # Convert from homogeneous to 2D coordinates
            # Avoid division by zero
            w = points_2d_homo[:, 2:3]
            w = torch.where(w == 0, torch.ones_like(w), w)  # Replace zeros with 1
            
            points_2d_pixel = points_2d_homo[:, :2] / w  # Shape: [num_steps * num_joints, 2]
            
            # Apply normalization if requested
            if self.normalize_2d_coordinates:
                # Normalize to 0-100 scale like molmo data processing
                # X-axis: 0 = left edge, 100 = right edge
                # Y-axis: 0 = top edge, 100 = bottom edge
                points_2d_normalized = torch.zeros_like(points_2d_pixel)
                points_2d_normalized[:, 0] = (points_2d_pixel[:, 0] / img_width) * 100.0   # X normalization
                points_2d_normalized[:, 1] = (points_2d_pixel[:, 1] / img_height) * 100.0  # Y normalization
                points_2d_final = points_2d_normalized
            else:
                # Keep in pixel coordinates
                points_2d_final = points_2d_pixel
            
            # Reshape back to original shape
            points_2d = points_2d_final.view(num_steps, num_joints, 2)
            
            return points_2d

    def visualize_trajectory_on_image(self, idx: int, save_path: Optional[str] = None, show_plot: bool = True) -> np.ndarray:
        """
        Debug function to visualize trajectory on the image.
        
        Args:
            idx: Sample index
            save_path: Optional path to save the visualization
            show_plot: Whether to display the plot
            
        Returns:
            Image with trajectory overlaid
        """
        if idx >= len(self.index_mapping):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.index_mapping)}")
            
        mapping = self.index_mapping[idx]
        
        # Load the original image (without transforms)
        cap = cv2.VideoCapture(mapping['video_path'])
        cap.set(cv2.CAP_PROP_POS_FRAMES, mapping['frame_idx'])
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError(f"Could not load frame {mapping['frame_idx']} from {mapping['video_path']}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        original_height, original_width = image.shape[:2]
        
        # Load trajectory data
        trajectory = self._load_trajectory(
            mapping['hdf5_path'], 
            mapping['frame_idx'], 
            self.action_chunking_horizon
        )
        
        # Get trajectory in the desired format
        if self.output_2d_trajectory:
            # Project to 2D
            trajectory_2d = self._project_trajectory_to_2d(
                mapping['hdf5_path'],
                mapping['frame_idx'],
                trajectory
            )
            trajectory_points = trajectory_2d.numpy()  # Shape: [num_steps, num_joints, 2]
        else:
            # Transform to camera frame and project to 2D for visualization
            trajectory_camera_frame = self._transform_trajectory_to_camera_frame(
                mapping['hdf5_path'],
                mapping['frame_idx'],
                trajectory
            )
            trajectory_2d = self._project_trajectory_to_2d(
                mapping['hdf5_path'],
                mapping['frame_idx'],
                trajectory
            )
            trajectory_points = trajectory_2d.numpy()  # Shape: [num_steps, num_joints, 2]
        
        # Create visualization
        vis_image = image.copy()
        
        # Define colors for different joints
        joint_colors = [
            (255, 0, 0),      # Red - leftHand
            (255, 100, 0),    # Orange - leftThumbTip
            (255, 200, 0),    # Yellow - leftIndexFingerTip
            (0, 255, 0),      # Green - leftMiddleFingerTip
            (0, 255, 100),    # Light green - leftRingFingerTip
            (0, 255, 200),    # Lighter green - leftLittleFingerTip
            (0, 0, 255),      # Blue - rightHand
            (100, 0, 255),    # Purple - rightThumbTip
            (200, 0, 255),    # Magenta - rightIndexFingerTip
            (255, 0, 255),    # Pink - rightMiddleFingerTip
            (255, 100, 255),  # Light pink - rightRingFingerTip
            (255, 200, 255),  # Lighter pink - rightLittleFingerTip
            (255, 255, 255)   # White - camera
        ]
        
        # Draw trajectory for each joint
        num_steps, num_joints, _ = trajectory_points.shape
        
        for joint_idx in range(num_joints):
            color = joint_colors[joint_idx % len(joint_colors)]
            joint_name = self.joint_names[joint_idx]
            
            # Get trajectory for this joint
            joint_trajectory = trajectory_points[:, joint_idx, :]  # Shape: [num_steps, 2]
            
            # Filter out invalid points (outside image bounds or NaN)
            valid_mask = (
                (joint_trajectory[:, 0] >= 0) & 
                (joint_trajectory[:, 0] < original_width) &
                (joint_trajectory[:, 1] >= 0) & 
                (joint_trajectory[:, 1] < original_height) &
                ~np.isnan(joint_trajectory[:, 0]) &
                ~np.isnan(joint_trajectory[:, 1])
            )
            
            if not np.any(valid_mask):
                continue
                
            valid_trajectory = joint_trajectory[valid_mask]
            
            # Draw trajectory line
            for i in range(len(valid_trajectory) - 1):
                pt1 = tuple(map(int, valid_trajectory[i]))
                pt2 = tuple(map(int, valid_trajectory[i + 1]))
                cv2.line(vis_image, pt1, pt2, color, 2)
            
            # Draw points
            for i, point in enumerate(valid_trajectory):
                pt = tuple(map(int, point))
                # Make current frame more prominent
                if i == 0:  # Current frame
                    cv2.circle(vis_image, pt, 8, color, -1)
                    cv2.circle(vis_image, pt, 10, (0, 0, 0), 2)
                else:  # Future frames
                    cv2.circle(vis_image, pt, 4, color, -1)
                    cv2.circle(vis_image, pt, 6, (0, 0, 0), 1)
        
        # Add text information
        instruction = self._load_instruction(mapping['hdf5_path'])
        cv2.putText(vis_image, f"Frame: {mapping['frame_idx']}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(vis_image, f"Task: {mapping['task_name']}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(vis_image, f"Output 2D: {self.output_2d_trajectory}", (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add instruction (truncated if too long)
        instruction_short = instruction[:50] + "..." if len(instruction) > 50 else instruction
        cv2.putText(vis_image, f"Instruction: {instruction_short}", (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add legend for joint colors
        y_offset = 200
        for i, joint_name in enumerate(self.joint_names[:5]):  # Show first 5 joints
            color = joint_colors[i % len(joint_colors)]
            cv2.putText(vis_image, f"{joint_name}:", (10, y_offset + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
            print(f"Visualization saved to: {save_path}")
        
        if show_plot:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(15, 10))
            plt.imshow(vis_image)
            plt.title(f"Trajectory Visualization - Sample {idx}")
            plt.axis('off')
            plt.show()
        
        return vis_image
    
    def debug_trajectory_projection(self, idx: int, num_samples: int = 3):
        """
        Debug function to compare 3D and 2D trajectories with visualization.
        
        Args:
            idx: Starting sample index
            num_samples: Number of samples to debug
        """
        print("=== Trajectory Projection Debug ===")
        
        for i in range(min(num_samples, len(self.index_mapping) - idx)):
            sample_idx = idx + i
            mapping = self.index_mapping[sample_idx]
            
            print(f"\nSample {sample_idx}:")
            print(f"  Task: {mapping['task_name']}")
            print(f"  Frame: {mapping['frame_idx']}")
            
            # Load trajectory data
            trajectory = self._load_trajectory(
                mapping['hdf5_path'], 
                mapping['frame_idx'], 
                self.action_chunking_horizon
            )
            
            # Get 3D camera frame trajectory
            trajectory_3d = self._transform_trajectory_to_camera_frame(
                mapping['hdf5_path'],
                mapping['frame_idx'],
                trajectory
            )
            
            # Get 2D projected trajectory
            trajectory_2d = self._project_trajectory_to_2d(
                mapping['hdf5_path'],
                mapping['frame_idx'],
                trajectory
            )
            
            print(f"  Original trajectory shape: {trajectory.shape}")
            print(f"  3D camera frame shape: {trajectory_3d.shape}")
            print(f"  2D projected shape: {trajectory_2d.shape}")
            
            # Show some sample points
            print(f"  Sample 3D points (first joint, first 3 steps):")
            print(f"    {trajectory_3d[:3, 0, :].numpy()}")
            print(f"  Sample 2D points (first joint, first 3 steps):")
            print(f"    {trajectory_2d[:3, 0, :].numpy()}")
            
            # Check for invalid projections
            invalid_2d = np.any(np.isnan(trajectory_2d.numpy()) | np.isinf(trajectory_2d.numpy()))
            if invalid_2d:
                print(f"  WARNING: Invalid 2D projections detected!")
            
            # Check bounds
            image_bounds = (0, 0, 1920, 1080)  # Assuming 1080p
            out_of_bounds = np.any(
                (trajectory_2d.numpy()[:, :, 0] < 0) | 
                (trajectory_2d.numpy()[:, :, 0] > image_bounds[2]) |
                (trajectory_2d.numpy()[:, :, 1] < 0) | 
                (trajectory_2d.numpy()[:, :, 1] > image_bounds[3])
            )
            if out_of_bounds:
                print(f"  WARNING: Some points are outside image bounds!")
            
            # Create visualization for this sample
            print(f"  Creating visualization...")
            vis_image = self.visualize_trajectory_on_image(
                sample_idx, 
                save_path=f"debug_trajectory_sample_{sample_idx}.png",
                show_plot=True
            )
            print(f"  Visualization saved as debug_trajectory_sample_{sample_idx}.png")
            
            # Show trajectory statistics
            trajectory_2d_np = trajectory_2d.numpy()
            print(f"  2D trajectory statistics:")
            print(f"    X range: [{np.min(trajectory_2d_np[:, :, 0]):.1f}, {np.max(trajectory_2d_np[:, :, 0]):.1f}]")
            print(f"    Y range: [{np.min(trajectory_2d_np[:, :, 1]):.1f}, {np.max(trajectory_2d_np[:, :, 1]):.1f}]")
            
            # Show which joints are visible
            visible_joints = []
            for joint_idx, joint_name in enumerate(self.joint_names):
                joint_points = trajectory_2d_np[:, joint_idx, :]
                valid_points = np.sum(
                    (joint_points[:, 0] >= 0) & 
                    (joint_points[:, 0] < 1920) &
                    (joint_points[:, 1] >= 0) & 
                    (joint_points[:, 1] < 1080) &
                    ~np.isnan(joint_points[:, 0]) &
                    ~np.isnan(joint_points[:, 1])
                )
                if valid_points > 0:
                    visible_joints.append(f"{joint_name}({valid_points}/{len(joint_points)})")
            
            print(f"  Visible joints: {', '.join(visible_joints)}")


def trajectory_to_xml_text(trajectory: torch.Tensor, keypoint_names: Optional[List[str]] = None) -> str:
    """
    Convert trajectory tensor to XML-based text format.
    
    Args:
        trajectory: Trajectory tensor of shape [num_steps, num_joints, 2] for 2D or [num_steps, num_joints, 3] for 3D
        keypoint_names: Optional list of keypoint names. If None, uses default names
        
    Returns:
        XML-formatted string with trajectory data for each keypoint
    """
    if keypoint_names is None:
        keypoint_names = ['leftThumbTip', 'leftIndexFingerTip', 'rightThumbTip', 'rightIndexFingerTip']
        
    num_steps, num_joints, num_dims = trajectory.shape
    
    # Round coordinates to 1 decimal place
    trajectory = torch.round(trajectory * 10) / 10
    
    xml_parts = []
    
    for joint_idx in range(num_joints):
        joint_name = keypoint_names[joint_idx] if joint_idx < len(keypoint_names) else f"joint_{joint_idx}"
        
        # Get trajectory for this joint across all time steps
        joint_trajectory = trajectory[:, joint_idx, :]  # Shape: [num_steps, num_dims]
        
        # Build coordinate attributes for this keypoint
        coord_attrs = []
        for t in range(num_steps):
            if num_dims == 3:
                # 3D trajectory: include x, y, z coordinates
                x, y, z = joint_trajectory[t, 0].item(), joint_trajectory[t, 1].item(), joint_trajectory[t, 2].item()
                coord_attrs.append(f"x{t+1}=\"{x:0.1f}\"")
                coord_attrs.append(f"y{t+1}=\"{y:0.1f}\"")
                coord_attrs.append(f"z{t+1}=\"{z:0.1f}\"")
            else:
                # 2D trajectory: include x, y coordinates
                x, y = joint_trajectory[t, 0].item(), joint_trajectory[t, 1].item()
                coord_attrs.append(f"x{t+1}=\"{x:0.1f}\"")
                coord_attrs.append(f"y{t+1}=\"{y:0.1f}\"")
        
        # Combine all coordinates for this keypoint
        coord_text = " ".join(coord_attrs)
        
        # Create XML element for this keypoint
        xml_element = f"<{joint_name} {coord_text} />"
        xml_parts.append(xml_element)
    
    return ", ".join(xml_parts)


def create_data_transforms(image_size: Tuple[int, int] = (256, 256), augment: bool = True):
    """Create data transforms for training."""
    if augment:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    return transform



def create_dataloader(
    data_dir: str,
    split: str = "train",
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 1,
    action_chunking_horizon: int = 10,
    image_size: Tuple[int, int] = (256, 256),
    augment: bool = True,
    output_2d_trajectory: bool = False,
    normalize_2d_coordinates: bool = True,  # New parameter
    **kwargs
) -> DataLoader:
    """Create a DataLoader for EgoDex VLA training."""
    
    # Create transforms
    transform = create_data_transforms(image_size, augment)
    
    # Create dataset
    dataset = EgoDexVLADataset(
        data_dir=data_dir,
        split=split,
        action_chunking_horizon=action_chunking_horizon,
        image_size=image_size,
        transform=transform,
        output_2d_trajectory=output_2d_trajectory,
        normalize_2d_coordinates=normalize_2d_coordinates,  # Pass the new parameter
        **kwargs
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    data_dir = "/home/ANT.AMAZON.COM/fanyangr/Downloads"
    
    # Create dataset
    # dataset = EgoDexVLADataset(
    #     data_dir=data_dir,
    #     split="test",
    #     action_chunking_horizon=10,
    #     image_size=(256, 256),
    #     cache_frames=False
    # )
    
    # print(f"Dataset size: {len(dataset)}")
    
    # # Test loading a sample
    # if len(dataset) > 0:
    #     sample = dataset[0]
    #     print(f"Sample keys: {sample.keys()}")
    #     print(f"Image shape: {sample['image'].shape}")
    #     print(f"Trajectory shape: {sample['trajectory'].shape}")
    #     print(f"Instruction: {sample['instruction']}")
    #     print(f"Metadata: {sample['metadata']}")
        
    #     # Get detailed info
    #     info = dataset.get_episode_info(0)
    #     print(f"Episode info: {info}")
    
    # Create dataloader
    # For normalized coordinates (0-100 scale, like molmo)
    # dataloader = create_dataloader(
    #     data_dir=data_dir,
    #     split="test",
    #     output_2d_trajectory=True,
    #     action_chunking_horizon=30,
    #     batch_size=4,
    #     normalize_2d_coordinates=True,  # Default, coordinates from 0-100
    #     output_xml_trajectory=True  # Enable XML output
    # )

    # For pixel coordinates
    dataloader = create_dataloader(
        data_dir=data_dir,
        split="test", 
        output_2d_trajectory=True,
        normalize_2d_coordinates=True,  # Raw pixel coordinates
        batch_size=2
    )
    
    # Create dataloader - simple tensor output
    # dataloader = create_dataloader(
    #     data_dir=data_dir,
    #     split="test",
    #     output_2d_trajectory=True,
    #     action_chunking_horizon=10,
    #     batch_size=2
    # )
    
    print(f"Dataloader created with {len(dataloader)} batches")
    
    # Test batch loading
    for i, batch in enumerate(dataloader):
        print(f"Batch {i}:")
        print(f"  Images: {batch['image'].shape}")
        print(f"  Trajectories: {batch['trajectory'].shape}")
        print(f"  Instructions: {len(batch['instruction'])}")
        
        # Convert first trajectory to XML using utility function
        if batch['trajectory'].shape[0] > 0:
            first_trajectory = batch['trajectory'][0]  # Shape: [num_steps, num_joints, 2/3]
            xml_output = trajectory_to_xml_text(first_trajectory)
            print(f"  XML Trajectory (first sample): {xml_output[:200]}...")
        
        if i >= 1:  # Test first few batches
            break
