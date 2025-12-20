#!/usr/bin/env python3
"""
RoboCasa Affordance Dataset for VLA Training

This module provides a dataset implementation for RoboCasa trajectory prediction.
Loads actual data from processed RoboCasa dataset (created by update_dataset_with_keypoints.py).

Dataset Structure:
    data_dir/
        task_name_1/
            1.hdf5, 1.mp4
            2.hdf5, 2.mp4
            ...
        task_name_2/
            1.hdf5, 1.mp4
            ...

HDF5 File Contents:
    - keypoint_positions_flat: (num_frames, 36) - 12 keypoints * 3 coords in world frame
    - keypoint_poses: (num_frames, 12, 4, 4) - 4x4 transformation matrices
    - camera_intrinsic_matrices: (num_frames, 3, 3)
    - camera_extrinsic_matrices: (num_frames, 4, 4)
    - states: (num_frames, state_dim) - Robot state at each frame
    - actions: (num_frames, action_dim) - Robot actions at each frame (typically 24-dim for bimanual)
    - Attributes: language_command, video_path, fps, video_frame_count, num_frames, keypoint_site_names
"""

import os
import json
import cv2
import h5py
import pickle
import time
import random
import contextlib
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional
from PIL import Image
from tqdm import tqdm
from olmo.data.dataset import Dataset


# Default keypoint names matching the RoboCasa update_dataset_with_keypoints.py output
ROBOCASA_KEYPOINT_NAMES = [
    # Right fingertips (5)
    "gripper0_right_site_R_thumb_distal_link",
    "gripper0_right_site_R_index_intermediate_link",
    "gripper0_right_site_R_middle_intermediate_link",
    "gripper0_right_site_R_ring_intermediate_link",
    "gripper0_right_site_R_pinky_intermediate_link",
    # Left fingertips (5)
    "gripper0_left_site_L_thumb_distal_link",
    "gripper0_left_site_L_index_intermediate_link",
    "gripper0_left_site_L_middle_intermediate_link",
    "gripper0_left_site_L_ring_intermediate_link",
    "gripper0_left_site_L_pinky_intermediate_link",
    # Wrists (2)
    "robot0_r_wrist_site",
    "robot0_l_wrist_site",
]

# Mapping from RoboCasa keypoints to EgoDex-style names for compatibility
KEYPOINT_NAME_MAPPING = {
    "gripper0_right_site_R_thumb_distal_link": "rightThumbTip",
    "gripper0_right_site_R_index_intermediate_link": "rightIndexFingerTip",
    "gripper0_right_site_R_middle_intermediate_link": "rightMiddleFingerTip",
    "gripper0_right_site_R_ring_intermediate_link": "rightRingFingerTip",
    "gripper0_right_site_R_pinky_intermediate_link": "rightLittleFingerTip",
    "gripper0_left_site_L_thumb_distal_link": "leftThumbTip",
    "gripper0_left_site_L_index_intermediate_link": "leftIndexFingerTip",
    "gripper0_left_site_L_middle_intermediate_link": "leftMiddleFingerTip",
    "gripper0_left_site_L_ring_intermediate_link": "leftRingFingerTip",
    "gripper0_left_site_L_pinky_intermediate_link": "leftLittleFingerTip",
    "robot0_r_wrist_site": "rightWrist",
    "robot0_l_wrist_site": "leftWrist",
}


@contextlib.contextmanager
def video_capture_context(video_path: str):
    """Context manager to ensure VideoCapture is always released, even on errors."""
    cap = None
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        yield cap
    finally:
        if cap is not None:
            try:
                cap.release()
            except:
                pass


@contextlib.contextmanager
def hdf5_file_with_retry(hdf5_path: str, max_retries: int = 3):
    """Context manager for opening HDF5 files with retry logic for NFS I/O errors."""
    last_exception = None
    file_opened = False
    
    for attempt in range(max_retries):
        try:
            f = h5py.File(hdf5_path, 'r')
            file_opened = True
            try:
                yield f
                return
            finally:
                f.close()
        except (OSError, IOError) as e:
            if not file_opened and attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 0.1 + random.uniform(0, 0.1)
                print(f"Warning: Failed to open HDF5 file {hdf5_path} (attempt {attempt + 1}/{max_retries}): {str(e)}, retrying in {wait_time:.2f}s...")
                time.sleep(wait_time)
                last_exception = e
                continue
            else:
                raise
        except Exception as e:
            raise
    
    raise OSError(f"Could not open HDF5 file {hdf5_path} after {max_retries} attempts")


class RoboCasaTrajectoryDataset(Dataset):
    """PyTorch Dataset for RoboCasa trajectory prediction training (3D flow matching only).
    Note: robocasa dataset is recorded at 20 fps, while egodex is recorded at 30 fps.
    """
    
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
        output_2d_trajectory: bool = False,
        frame_downsampling_ratio: int = 1,  # Sample every n frames
        # Train/test split ratio (for random split within each task)
        train_ratio: float = 0.9,
    ):
        """
        Initialize RoboCasa Trajectory Dataset (3D flow matching only).
        
        Args:
            data_dir: Path to data directory containing task subdirectories
            split: Dataset split ('train', 'test', 'overfit')
            action_chunking_horizon: Number of frames in action chunks
            joint_names: List of joint names to use (uses EgoDex-style names)
            normalize_coordinates: Whether to normalize coordinates
            stats_file: Path to stats file for normalization (for 3D trajectories)
            trajectory_representation: 'absolute' or 'delta'
            load_images: Whether to load images
            output_2d_trajectory: Whether to output 2D or 3D trajectory
            frame_downsampling_ratio: Downsample frames by this ratio
            train_ratio: Ratio of episodes to use for training (rest for testing)
        """
        if data_dir is None:
            data_dir = os.environ.get("ROBOCASA_DATA_DIR")
            if data_dir is None:
                raise ValueError("data_dir must be provided or ROBOCASA_DATA_DIR environment variable must be set")
        
        self.data_dir = Path(data_dir)
        self.split = split
        self.action_chunking_horizon = action_chunking_horizon
        self.normalize_coordinates = normalize_coordinates
        self.stats_file = stats_file
        self.trajectory_representation = trajectory_representation
        self.overfit_num_examples = type(self).overfit_num_examples
        self.load_images = load_images
        self.output_2d_trajectory = output_2d_trajectory
        self.frame_downsampling_ratio = frame_downsampling_ratio
        self.train_ratio = train_ratio
        
        # Validate parameters
        assert trajectory_representation in ["absolute", "delta"], \
            f"trajectory_representation must be 'absolute' or 'delta', got {trajectory_representation}"
        assert frame_downsampling_ratio >= 1, \
            f"frame_downsampling_ratio must be >= 1, got {frame_downsampling_ratio}"
        
        # Load normalization stats if needed
        if self.normalize_coordinates and not self.output_2d_trajectory:
            if self.stats_file is not None and os.path.exists(self.stats_file):
                print(f"Loading trajectory normalization stats from {self.stats_file}...")
                stats = torch.load(self.stats_file)
                self.stats_mean = stats["mean"].float() if isinstance(stats["mean"], torch.Tensor) else torch.tensor(stats["mean"]).float()
                self.stats_std = stats["std"].float() if isinstance(stats["std"], torch.Tensor) else torch.tensor(stats["std"]).float()
            else:
                self.stats_mean = None
                self.stats_std = None
        else:
            self.stats_mean = None
            self.stats_std = None
        
        # Joint names - use EgoDex-style names for compatibility
        if joint_names is None:
            # Default: 10 fingertips (matching EgoDex TrajectoryDataset)
            self.joint_names = [
                'leftThumbTip', 'leftIndexFingerTip', 'leftMiddleFingerTip', 'leftRingFingerTip', 'leftLittleFingerTip',
                'rightThumbTip', 'rightIndexFingerTip', 'rightMiddleFingerTip', 'rightRingFingerTip', 'rightLittleFingerTip'
            ]
        else:
            self.joint_names = joint_names
        
        self.num_joints = len(self.joint_names)
        self.coord_dim = 2 if self.output_2d_trajectory else 3
        
        # Build reverse mapping from EgoDex names to RoboCasa keypoint indices
        self._build_keypoint_index_mapping()
        
        # Build index mapping
        print(f"Building index mapping for {split} split...")
        self.index_mapping = self._build_index_mapping()
        
        # Apply overfit example limit
        if self.split == "overfit" and self.overfit_num_examples is not None:
            original_len = len(self.index_mapping)
            self.index_mapping = self.index_mapping[:self.overfit_num_examples]
            print(f"Limited overfit split to {len(self.index_mapping)} examples (from {original_len})")
        
        print(f"[RoboCasaTrajectoryDataset] Initialized with {len(self.index_mapping)} samples for {split} split")
        print(f"  - data_dir: {self.data_dir}")
        print(f"  - action_chunking_horizon: {action_chunking_horizon}")
        print(f"  - trajectory_representation: {trajectory_representation}")
        print(f"  - num_joints: {self.num_joints}, coord_dim: {self.coord_dim}")
        print(f"  - output_2d_trajectory: {output_2d_trajectory}")
        print(f"  - frame_downsampling_ratio: {frame_downsampling_ratio}")
    
    def _build_keypoint_index_mapping(self):
        """Build mapping from EgoDex joint names to RoboCasa keypoint indices."""
        # Reverse mapping: EgoDex name -> RoboCasa name
        egodex_to_robocasa = {v: k for k, v in KEYPOINT_NAME_MAPPING.items()}
        
        # Map EgoDex joint names to indices in keypoint_positions_flat
        self.joint_to_keypoint_idx = {}
        for joint_name in self.joint_names:
            if joint_name in egodex_to_robocasa:
                robocasa_name = egodex_to_robocasa[joint_name]
                if robocasa_name in ROBOCASA_KEYPOINT_NAMES:
                    self.joint_to_keypoint_idx[joint_name] = ROBOCASA_KEYPOINT_NAMES.index(robocasa_name)
                else:
                    print(f"Warning: Could not find RoboCasa keypoint for {joint_name}")
                    self.joint_to_keypoint_idx[joint_name] = None
            else:
                print(f"Warning: Unknown joint name {joint_name}")
                self.joint_to_keypoint_idx[joint_name] = None
    
    def _get_cache_filepath(self) -> Path:
        """Get the path to the cached index mapping file."""
        cache_filename = f"robocasa_index_mapping_{self.split}_horizon{self.action_chunking_horizon}_ratio{self.train_ratio}.pkl"
        return self.data_dir / cache_filename
    
    def _load_index_from_cache(self) -> Optional[List[Dict]]:
        """Load index mapping from cache file if it exists."""
        cache_file = self._get_cache_filepath()
        if cache_file.exists():
            print(f"Loading cached index mapping from {cache_file}...")
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            if (cached_data.get('split') == self.split and
                cached_data.get('action_chunking_horizon') == self.action_chunking_horizon and
                cached_data.get('joint_names') == self.joint_names and
                cached_data.get('train_ratio') == self.train_ratio):
                full_index = cached_data['index_mapping']
                print(f"Successfully loaded {len(full_index)} samples from cache")
                
                if self.frame_downsampling_ratio > 1:
                    filtered_index = self._apply_downsampling_filter(full_index)
                    print(f"Applied downsampling ratio {self.frame_downsampling_ratio}: {len(full_index)} -> {len(filtered_index)} samples")
                    return filtered_index
                return full_index
            else:
                print("Cache configuration mismatch, rebuilding index...")
        return None
    
    def _save_index_to_cache(self, index_mapping: List[Dict]) -> None:
        """Save index mapping to cache file."""
        cache_file = self._get_cache_filepath()
        cache_data = {
            'split': self.split,
            'action_chunking_horizon': self.action_chunking_horizon,
            'joint_names': self.joint_names,
            'train_ratio': self.train_ratio,
            'index_mapping': index_mapping,
        }
        
        print(f"Saving index mapping to cache: {cache_file}...")
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"Successfully saved {len(index_mapping)} samples to cache")
    
    def _apply_downsampling_filter(self, index_mapping: List[Dict]) -> List[Dict]:
        """Filter index mapping to keep only frames that match the downsampling pattern."""
        if self.frame_downsampling_ratio == 1:
            return index_mapping
        
        filtered = []
        for entry in index_mapping:
            if entry['frame_idx'] % self.frame_downsampling_ratio == 0:
                filtered.append(entry)
        
        return filtered
    
    def _build_index_mapping(self) -> List[Dict]:
        """Build a flat index mapping across all tasks and episodes."""
        # Try to load from cache first
        cached_index = self._load_index_from_cache()
        if cached_index is not None:
            return cached_index
        
        print("Building index mapping from scratch...")
        index_mapping = []
        
        # Find all task directories
        if not self.data_dir.exists():
            raise ValueError(f"Data directory does not exist: {self.data_dir}")
        
        task_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        
        if not task_dirs:
            print(f"Warning: No task directories found in {self.data_dir}")
            return index_mapping
        
        print(f"Found {len(task_dirs)} task directories")
        
        for task_dir in tqdm(task_dirs, desc="Processing tasks"):
            task_name = task_dir.name
            
            # Find all HDF5 files in this task directory
            hdf5_files = sorted(task_dir.glob("*.hdf5"))
            
            if not hdf5_files:
                continue
            
            # Split episodes into train/test
            num_episodes = len(hdf5_files)
            num_train = int(num_episodes * self.train_ratio)
            
            if self.split == "train":
                selected_files = hdf5_files[:num_train]
            elif self.split == "test":
                selected_files = hdf5_files[num_train:]
            elif self.split == "overfit":
                # Use first few episodes for overfitting
                selected_files = hdf5_files[:min(3, num_episodes)]
            else:
                raise ValueError(f"Invalid split: {self.split}")
            
            for hdf5_file in selected_files:
                video_file = hdf5_file.with_suffix('.mp4')
                
                if not video_file.exists():
                    print(f"Warning: No corresponding video file for {hdf5_file}")
                    continue
                
                try:
                    with h5py.File(hdf5_file, 'r') as f:
                        # Get number of frames
                        if 'keypoint_positions_flat' not in f:
                            print(f"Warning: No keypoint_positions_flat in {hdf5_file}")
                            continue
                        
                        num_frames = f['keypoint_positions_flat'].shape[0]
                        
                        if num_frames < self.action_chunking_horizon:
                            print(f"Warning: Episode {hdf5_file} has only {num_frames} frames, skipping...")
                            continue
                        
                        # Add each valid frame as a separate index
                        for frame_idx in range(num_frames - self.action_chunking_horizon):
                            index_mapping.append({
                                'video_path': str(video_file),
                                'hdf5_path': str(hdf5_file),
                                'frame_idx': frame_idx,
                                'task_name': task_name,
                                'num_frames': num_frames,
                            })
                            
                except Exception as e:
                    print(f"Error processing {hdf5_file}: {e}")
                    continue
        
        # Save to cache
        self._save_index_to_cache(index_mapping)
        
        # Apply downsampling if needed
        if self.frame_downsampling_ratio > 1:
            filtered_index = self._apply_downsampling_filter(index_mapping)
            print(f"Applied downsampling ratio {self.frame_downsampling_ratio}: {len(index_mapping)} -> {len(filtered_index)} samples")
            return filtered_index
        
        return index_mapping
    
    def __len__(self):
        return len(self.index_mapping)
    
    def get(self, idx, rng):
        """Get a single training example."""
        mapping = self.index_mapping[idx]
        
        video_path = mapping['video_path']
        hdf5_path = mapping['hdf5_path']
        frame_idx = mapping['frame_idx']
        
        # Load image
        if self.load_images:
            image = self._load_frame(video_path, frame_idx)
        else:
            image = Image.fromarray(np.zeros((1, 1, 3), dtype=np.uint8))
        
        # Load trajectory data
        trajectory = self._load_trajectory(hdf5_path, frame_idx, self.action_chunking_horizon)
        
        # Transform trajectory based on output type
        if self.output_2d_trajectory:
            final_trajectory = self._project_trajectory_to_2d(hdf5_path, frame_idx, trajectory)
        else:
            final_trajectory = self._transform_trajectory_to_camera_frame(hdf5_path, frame_idx, trajectory)
        
        # Get initial state (first frame, flattened) - always absolute position
        if isinstance(final_trajectory, torch.Tensor):
            initial_state = final_trajectory[0].numpy()
        else:
            initial_state = final_trajectory[0]
        initial_state = initial_state.reshape(-1).astype(np.float32)
        
        # Convert to delta representation if requested
        if self.trajectory_representation == "delta":
            final_trajectory = self._convert_to_delta_representation(final_trajectory)
        
        # Normalize 3D trajectory if requested
        if self.normalize_coordinates and not self.output_2d_trajectory and self.stats_mean is not None:
            num_steps, num_joints, coords = final_trajectory.shape
            mean = self.stats_mean.view(1, num_joints, coords)
            std = self.stats_std.view(1, num_joints, coords)
            final_trajectory = (final_trajectory - mean) / std
        
        # Load instruction
        instruction = self._load_instruction(hdf5_path)
        
        # Load robot actions and states (in a single file open)
        robot_actions, robot_states = self._load_robot_actions_and_states(hdf5_path, frame_idx, self.action_chunking_horizon)
        
        if isinstance(final_trajectory, torch.Tensor):
            final_trajectory = final_trajectory.numpy()
        
        # Flatten trajectory for flow matching: [num_steps, num_joints*coords]
        num_steps = final_trajectory.shape[0]
        trajectory_flattened = final_trajectory.reshape(num_steps, -1).astype(np.float32)
        
        # Determine style based on output format
        if self.output_2d_trajectory:
            style = 'trajectory_2d_text'
        else:
            style = 'trajectory_3d_fm'
        
        result = {
            'image': image,
            'state': initial_state,
            'message_list': [
                {
                    'label': instruction,
                    'points': final_trajectory,
                    'point_scale': 100 if self.normalize_coordinates and self.output_2d_trajectory else None,
                    'style': style,
                    'state': initial_state,
                }
            ],
            'trajectory_target': trajectory_flattened,
            'trajectory_shape': final_trajectory.shape,
            'expert_type': 1,  # Robot trajectory expert (for multi-expert routing)
            'metadata': {
                'image': image,
                'task_name': mapping['task_name'],
                'frame_idx': frame_idx,
                'output_2d_trajectory': self.output_2d_trajectory,
                'trajectory_representation': self.trajectory_representation,
            }
        }
        
        # Add robot actions and states (may be None if not available in HDF5)
        if robot_actions is not None:
            result['robot_actions'] = robot_actions  # Shape: [num_steps, action_dim]
            result['metadata']['action_dim'] = robot_actions.shape[-1]
        if robot_states is not None:
            result['robot_states'] = robot_states  # Shape: [num_steps, state_dim]
            result['metadata']['state_dim'] = robot_states.shape[-1]
        
        return result
    
    def _load_frame(self, video_path: str, frame_idx: int, max_retries: int = 3) -> Image.Image:
        """Load a single frame from a video file with retry logic."""
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                with video_capture_context(video_path) as cap:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    
                    if ret and frame is not None:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        return Image.fromarray(frame)
                    else:
                        if attempt < max_retries - 1:
                            wait_time = (2 ** attempt) * 0.1 + random.uniform(0, 0.1)
                            print(f"Warning: Failed to read frame {frame_idx} from {video_path} "
                                  f"(attempt {attempt + 1}/{max_retries}), retrying in {wait_time:.2f}s...")
                            time.sleep(wait_time)
                            last_exception = ValueError(f"Could not read frame {frame_idx}")
                            continue
                        else:
                            raise ValueError(f"Could not load frame {frame_idx} from {video_path}")
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 0.1 + random.uniform(0, 0.1)
                    time.sleep(wait_time)
                    last_exception = e
                    continue
                else:
                    raise ValueError(f"Could not load frame {frame_idx} from {video_path}: {str(e)}")
        
        raise ValueError(f"Could not load frame {frame_idx} from {video_path}")
    
    def _load_trajectory(self, hdf5_path: str, start_frame: int, num_steps: int, max_retries: int = 3) -> torch.Tensor:
        """
        Load trajectory data from HDF5 file.
        
        Extracts positions for the requested joints from keypoint_positions_flat.
        
        Args:
            hdf5_path: Path to the HDF5 file
            start_frame: Starting frame index
            num_steps: Number of steps to load
            
        Returns:
            torch.Tensor of shape [num_steps, num_joints, 3]
        """
        with hdf5_file_with_retry(hdf5_path, max_retries) as f:
            # Load keypoint positions: shape (num_frames, 36) = (num_frames, 12 keypoints * 3)
            keypoint_positions = f['keypoint_positions_flat'][start_frame:start_frame + num_steps]
            
            # Reshape to (num_steps, 12, 3)
            keypoint_positions = keypoint_positions.reshape(num_steps, 12, 3)
            
            # Extract only the requested joints
            trajectory_list = []
            for joint_name in self.joint_names:
                keypoint_idx = self.joint_to_keypoint_idx.get(joint_name)
                if keypoint_idx is not None:
                    trajectory_list.append(keypoint_positions[:, keypoint_idx, :])
                else:
                    # Fill with zeros for unknown joints
                    trajectory_list.append(np.zeros((num_steps, 3)))
            
            trajectory = np.stack(trajectory_list, axis=1)  # Shape: (num_steps, num_joints, 3)
            return torch.from_numpy(trajectory).float()
    
    def _load_instruction(self, hdf5_path: str, max_retries: int = 3) -> str:
        """Load instruction from HDF5 file attributes."""
        with hdf5_file_with_retry(hdf5_path, max_retries) as f:
            instruction = f.attrs.get('language_command', 'No instruction available')
            if isinstance(instruction, bytes):
                instruction = instruction.decode('utf-8')
            return instruction
    
    def _load_robot_actions_and_states(self, hdf5_path: str, start_frame: int, num_steps: int, max_retries: int = 3) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Load robot actions and states from HDF5 file in a single file open.
        
        The actions are typically 24-dimensional for bimanual dexterous manipulation:
        - Right arm/hand (12 dims): position, finger flexion
        - Left arm/hand (12 dims): position, finger flexion
        
        Args:
            hdf5_path: Path to the HDF5 file
            start_frame: Starting frame index
            num_steps: Number of steps to load
            
        Returns:
            Tuple of (actions, states) where:
            - actions: np.ndarray of shape [num_steps, action_dim] or None if not available
            - states: np.ndarray of shape [num_steps, state_dim] or None if not available
        """
        with hdf5_file_with_retry(hdf5_path, max_retries) as f:
            # Load actions if available
            actions = None
            if 'actions' in f:
                actions = f['actions'][start_frame:start_frame + num_steps].astype(np.float32)
            
            # Load states if available
            states = None
            if 'states' in f:
                states = f['states'][start_frame:start_frame + num_steps].astype(np.float32)
            
            return actions, states
    
    def _transform_trajectory_to_camera_frame(self, hdf5_path: str, current_frame: int, trajectory: torch.Tensor, max_retries: int = 3) -> torch.Tensor:
        """
        Transform trajectory from world frame to camera frame.
        
        Args:
            hdf5_path: Path to the HDF5 file
            current_frame: Current frame index
            trajectory: Trajectory tensor of shape [num_steps, num_joints, 3]
            
        Returns:
            Transformed trajectory in camera frame
        """
        with hdf5_file_with_retry(hdf5_path, max_retries) as f:
            if 'camera_extrinsic_matrices' not in f:
                print(f"Warning: No camera extrinsic matrices in {hdf5_path}")
                return trajectory
            
            # Load camera extrinsic matrix at current frame
            camera_extrinsic = f['camera_extrinsic_matrices'][current_frame]  # Shape: [4, 4]
            camera_extrinsic = torch.from_numpy(camera_extrinsic.copy()).float()
        
        # The extrinsic matrix transforms from world to camera frame
        # For MuJoCo, this should already be world-to-camera
        camera_transform_inv = torch.inverse(camera_extrinsic)
        
        num_steps, num_joints, _ = trajectory.shape
        
        # Convert to homogeneous coordinates
        trajectory_homo = torch.cat([
            trajectory,
            torch.ones(num_steps, num_joints, 1)
        ], dim=-1)
        
        trajectory_homo_flat = trajectory_homo.view(-1, 4)
        
        # Apply transform
        transformed_homo = trajectory_homo_flat @ camera_transform_inv.T
        transformed_3d = transformed_homo[:, :3]
        
        return transformed_3d.view(num_steps, num_joints, 3)
    
    def _project_trajectory_to_2d(self, hdf5_path: str, current_frame: int, trajectory: torch.Tensor, max_retries: int = 3) -> torch.Tensor:
        """
        Project 3D trajectory to 2D image plane using camera intrinsics.
        
        Args:
            hdf5_path: Path to HDF5 file
            current_frame: Current frame index
            trajectory: 3D trajectory tensor of shape [num_steps, num_joints, 3]
            
        Returns:
            2D projected trajectory tensor of shape [num_steps, num_joints, 2]
        """
        with hdf5_file_with_retry(hdf5_path, max_retries) as f:
            # Get camera intrinsics
            if 'camera_intrinsic_matrices' in f:
                intrinsic = f['camera_intrinsic_matrices'][current_frame]  # Shape: [3, 3]
                img_width = intrinsic[0, 2] * 2
                img_height = intrinsic[1, 2] * 2
            else:
                # Default intrinsics for 256x256 image
                intrinsic = np.array([
                    [128.0, 0., 128.],
                    [0., 128.0, 128.],
                    [0., 0., 1.]
                ])
                img_width, img_height = 256, 256
            
            intrinsic = torch.from_numpy(intrinsic).float()
        
        # Transform trajectory to camera frame first
        trajectory_camera_frame = self._transform_trajectory_to_camera_frame(hdf5_path, current_frame, trajectory, max_retries)
        
        num_steps, num_joints, _ = trajectory_camera_frame.shape
        points_3d = trajectory_camera_frame.view(-1, 3)
        
        # Project: K @ [X, Y, Z]^T = [u, v, w]^T
        points_2d_homo = points_3d @ intrinsic.T
        
        # Convert from homogeneous coordinates
        w = points_2d_homo[:, 2:3]
        w = torch.where(w == 0, torch.ones_like(w), w)
        points_2d_pixel = points_2d_homo[:, :2] / w
        
        # Normalize to 0-100 scale if requested
        if self.normalize_coordinates:
            points_2d_normalized = torch.zeros_like(points_2d_pixel)
            points_2d_normalized[:, 0] = (points_2d_pixel[:, 0] / img_width) * 100.0
            points_2d_normalized[:, 1] = (points_2d_pixel[:, 1] / img_height) * 100.0
            points_2d = points_2d_normalized
        else:
            points_2d = points_2d_pixel
        
        return points_2d.view(num_steps, num_joints, 2)
    
    def _convert_to_delta_representation(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Convert absolute positions to delta positions (velocities)."""
        if isinstance(trajectory, np.ndarray):
            trajectory = torch.from_numpy(trajectory)
        
        deltas = trajectory[1:] - trajectory[:-1]
        last_delta = deltas[-1:]
        return torch.cat([deltas, last_delta], dim=0)


# Alias for backwards compatibility
RobotCasaHandPositioningDataset = RoboCasaTrajectoryDataset


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test RoboCasa Trajectory Dataset")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to RoboCasa processed dataset")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test", "overfit"])
    parser.add_argument("--action_horizon", type=int, default=30)
    parser.add_argument("--output_2d", action="store_true", help="Output 2D trajectory")
    
    args = parser.parse_args()
    
    print("Testing RoboCasaTrajectoryDataset...\n")
    
    dataset = RoboCasaTrajectoryDataset(
        data_dir=args.data_dir,
        split=args.split,
        action_chunking_horizon=args.action_horizon,
        output_2d_trajectory=args.output_2d,
        frame_downsampling_ratio=10,
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset.get(0, None)
        print(f"\nSample output:")
        print(f"  Image shape: {np.array(sample['image']).shape}")
        print(f"  State shape: {sample['state'].shape}")
        print(f"  Trajectory target shape: {sample['trajectory_target'].shape}")
        print(f"  Trajectory original shape: {sample['trajectory_shape']}")
        print(f"  Style: {sample['message_list'][0]['style']}")
        print(f"  Label: {sample['message_list'][0]['label']}")
        
        # Print robot actions info if available
        if 'robot_actions' in sample:
            print(f"  Robot actions shape: {sample['robot_actions'].shape}")
            print(f"  Robot actions sample: {sample['robot_actions'][0][:5]}...")  # First 5 dims
        if 'robot_states' in sample:
            print(f"  Robot states shape: {sample['robot_states'].shape}")
        
        print(f"\nData loading working correctly!")
    else:
        print("No samples found in dataset!")
