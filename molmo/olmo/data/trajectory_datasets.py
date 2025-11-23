#!/usr/bin/env python3
"""
Trajectory Dataset for VLA Training
"""

import os
import h5py
import cv2
import numpy as np
import torch
import pickle
import time
import random
import contextlib
from pathlib import Path
from typing import Dict, List, Optional
from PIL import Image
from tqdm import tqdm
from olmo.data.dataset import Dataset


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
                pass  # Ignore errors during cleanup


@contextlib.contextmanager
def hdf5_file_with_retry(hdf5_path: str, max_retries: int = 3):
    """
    Context manager for opening HDF5 files with retry logic for NFS I/O errors.
    
    Args:
        hdf5_path: Path to the HDF5 file
        max_retries: Maximum number of retry attempts
        
    Yields:
        h5py.File object
        
    Raises:
        OSError: If file cannot be opened after max_retries attempts
    """
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
            # Only retry if file opening failed, not if error occurred during usage
            if not file_opened and attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 0.1 + random.uniform(0, 0.1)
                print(f"Warning: Failed to open HDF5 file {hdf5_path} (attempt {attempt + 1}/{max_retries}): {str(e)}, retrying in {wait_time:.2f}s...")
                time.sleep(wait_time)
                last_exception = e
                continue
            else:
                # If file was opened but error occurred during usage, don't retry
                raise
        except Exception as e:
            # For other exceptions, don't retry
            raise
    
    # Should not reach here, but just in case
    raise OSError(f"Could not open HDF5 file {hdf5_path} after {max_retries} attempts")


class TrajectoryDataset(Dataset):
    """PyTorch Dataset for trajectory prediction training."""
    
    # Class-level parameter for limiting examples in overfit split
    # Set before instantiation: TrajectoryDataset.overfit_num_examples = 1
    overfit_num_examples: Optional[int] = 10
    
    def __init__(
        self,
        data_dir: str = None,
        split: str = "train",
        action_chunking_horizon: int = 30,
        joint_names: Optional[List[str]] = None,
        output_2d_trajectory: bool = True,
        normalize_coordinates: bool = True,
        stats_file: Optional[str] = None,
        use_confidence_filter: bool = False,
        confidence_threshold: float = 0.5,
        output_format: str = "text",  # "text" or "flow_matching"
        frame_downsampling_ratio: int = 15,  # Sample every n frames (1 = no downsampling, 10 = every 10 frames)
        trajectory_representation: str = "absolute",  # "absolute" or "delta" (velocity: v_t = x_{t+1} - x_t)
        load_images: bool = True,
    ):
        # Get data directory from environment variable if not provided
        if data_dir is None:
            data_dir = os.environ.get("EGODEX_DATA_DIR")
            if data_dir is None:
                raise ValueError("data_dir must be provided or EGODEX_DATA_DIR environment variable must be set")
        
        self.data_dir = Path(data_dir)
        self.split = split
        self.action_chunking_horizon = action_chunking_horizon
        self.use_confidence_filter = use_confidence_filter
        self.confidence_threshold = confidence_threshold
        self.output_2d_trajectory = output_2d_trajectory
        self.normalize_coordinates = normalize_coordinates
        self.stats_file = stats_file
        self.output_format = output_format  # "text" or "flow_matching"
        self.frame_downsampling_ratio = frame_downsampling_ratio
        self.trajectory_representation = trajectory_representation
        self.overfit_num_examples = type(self).overfit_num_examples
        self.load_images = load_images
        
        assert output_format in ["text", "flow_matching"], f"output_format must be 'text' or 'flow_matching', got {output_format}"
        assert frame_downsampling_ratio >= 1, f"frame_downsampling_ratio must be >= 1, got {frame_downsampling_ratio}"
        assert trajectory_representation in ["absolute", "delta"], f"trajectory_representation must be 'absolute' or 'delta', got {trajectory_representation}"
        if self.overfit_num_examples is not None:
            assert self.overfit_num_examples > 0, f"overfit_num_examples must be > 0, got {self.overfit_num_examples}"
        
        if self.normalize_coordinates and not self.output_2d_trajectory:
            if self.stats_file is None:
                raise ValueError("stats_file must be provided when normalize_coordinates is True and output_2d_trajectory is False")
            print(f"Loading trajectory normalizationstats from {self.stats_file}...")
            stats = torch.load(self.stats_file)
            self.stats_mean = stats["mean"]
            self.stats_std = stats["std"]
            # Ensure stats are on CPU/float
            if isinstance(self.stats_mean, torch.Tensor):
                self.stats_mean = self.stats_mean.float()
                self.stats_std = self.stats_std.float()

        # Default joint names matching data_formatter.py TRAJECTORY_KEYPOINTS
        if joint_names is None:
            self.joint_names = [
                'leftThumbTip', 'leftIndexFingerTip', 'leftMiddleFingerTip', 'leftRingFingerTip', 'leftLittleFingerTip',
                'rightThumbTip', 'rightIndexFingerTip', 'rightMiddleFingerTip', 'rightRingFingerTip', 'rightLittleFingerTip'
            ]
        else:
            self.joint_names = joint_names
        
        print(f"Loading index mapping for {split} split...")
        self.index_mapping = self._build_index_mapping()
        
        # Apply overfit example limit if specified (only when split="overfit")
        if self.split == "overfit" and self.overfit_num_examples is not None:
            original_len = len(self.index_mapping)
            self.index_mapping = self.index_mapping[:self.overfit_num_examples]
            print(f"Limited overfit split to {len(self.index_mapping)} examples (from {original_len})")
        
        print(f"Loaded {len(self.index_mapping)} samples from {split} split")
        
    def _get_cache_filepath(self) -> Path:
        """Get the path to the cached index mapping file."""
        cache_filename = f"index_mapping_{self.split}_horizon{self.action_chunking_horizon}.pkl"
        return self.data_dir / cache_filename
    
    def _load_index_from_cache(self) -> Optional[List[Dict]]:
        """Load index mapping from cache file if it exists."""
        cache_file = self._get_cache_filepath()
        if cache_file.exists():
            print(f"Loading cached index mapping from {cache_file}...")
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            # Validate that cache matches current configuration
            if (cached_data.get('split') == self.split and
                cached_data.get('action_chunking_horizon') == self.action_chunking_horizon and
                cached_data.get('joint_names') == self.joint_names):
                full_index = cached_data['index_mapping']
                print(f"Successfully loaded {len(full_index)} samples from cache")
                
                # Apply downsampling filter if needed (only for training split)
                if self.frame_downsampling_ratio > 1 and (self.split == "train" or self.split == "overfit"):
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
            'index_mapping': index_mapping,
        }
        
        print(f"Saving index mapping to cache: {cache_file}...")
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"Successfully saved {len(index_mapping)} samples to cache")
    
    def _apply_downsampling_filter(self, index_mapping: List[Dict]) -> List[Dict]:
        """
        Filter index mapping to keep only frames that match the downsampling pattern.
        
        For each video/hdf5 file, keep frames 0, n, 2n, 3n, ... where n is the downsampling ratio.
        
        Args:
            index_mapping: Full index mapping with all frames
            
        Returns:
            Filtered index mapping with downsampled frames
        """
        if self.frame_downsampling_ratio == 1:
            return index_mapping
        
        filtered = []
        for entry in index_mapping:
            # Keep frame if frame_idx is divisible by downsampling ratio
            if entry['frame_idx'] % self.frame_downsampling_ratio == 0:
                filtered.append(entry)
        
        return filtered

    def _build_index_mapping(self) -> List[Dict]:
        """Build a flat index mapping across all videos and frames."""
        # Try to load from cache first
        cached_index = self._load_index_from_cache()
        if cached_index is not None:
            return cached_index
        
        # Build index from scratch if cache doesn't exist
        print("Building index mapping from scratch...")
        index_mapping = []
        
        if self.split == "train":
            split_dirs = []
            for part in ["part1", "part2", "part3", "part4", "part5"]:
            # for part in ["small_test"]: # NOTE: debug only
                part_dir = self.data_dir / part
                if part_dir.exists():
                    split_dirs.append(part_dir)
        elif self.split == "test":
            # split_dirs = [self.data_dir / "test"]
            split_dirs = [self.data_dir / "small_test"]
        elif self.split == "overfit":
            print("!!Using overfit split")
            split_dirs = [self.data_dir / "very_small_test"]
        else:
            raise ValueError(f"Invalid split: {self.split}")
        
        # First, collect all video files to get total count for progress bar
        all_video_files = []
        for split_dir in split_dirs:
            if not split_dir.exists():
                continue
                
            task_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
            
            for task_dir in task_dirs:
                video_files = list(task_dir.glob("*.mp4"))
                all_video_files.extend(video_files)
        
        # Process video files with progress bar
        for video_file in tqdm(all_video_files, desc="Building index mapping", unit="video"):
            hdf5_file = video_file.with_suffix('.hdf5')
            if not hdf5_file.exists():
                continue
            
            # cap = cv2.VideoCapture(str(video_file))
            # if not cap.isOpened():
            #     continue
                
            # frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # cap.release()
            
            # if frame_count < self.action_chunking_horizon:
            #     continue
            
            # Get task name from parent directory
            task_name = video_file.parent.name
            
            try:
                with h5py.File(hdf5_file, 'r') as f:
                    # missing_joints = [j for j in self.joint_names if f'transforms/{j}' not in f]
                    # if missing_joints:
                    #     continue
                        
                    trajectory_length = f[f'transforms/{self.joint_names[0]}'].shape[0]
                # Always build full index with all frames (ratio=1)
                for frame_idx in range(trajectory_length - self.action_chunking_horizon):
                    index_mapping.append({
                        'video_path': str(video_file),
                        'hdf5_path': str(hdf5_file),
                        'frame_idx': frame_idx,
                        'task_name': task_name,
                        # 'episode_id': video_file.stem,
                        # 'total_frames': frame_count,
                        # 'trajectory_length': trajectory_length
                    })
            except Exception as e:
                print(f"    Error processing {hdf5_file}: {e}")
                continue
            # time.sleep(0.05)
        
        # Save full index to cache for future use
        self._save_index_to_cache(index_mapping)
        
        # Apply downsampling filter if needed (only for training split)
        if self.frame_downsampling_ratio > 1 and (self.split == "train" or self.split == "overfit"):
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
        
        if self.load_images:
            image = self._load_frame(video_path, mapping['frame_idx'])
        else:
            # Return dummy image (1x1 pixel black image)
            image = Image.fromarray(np.zeros((1, 1, 3), dtype=np.uint8))
            
        trajectory = self._load_trajectory(hdf5_path, mapping['frame_idx'], self.action_chunking_horizon)
        
        if self.output_2d_trajectory:
            final_trajectory = self._project_trajectory_to_2d(hdf5_path, mapping['frame_idx'], trajectory)
        else:
            final_trajectory = self._transform_trajectory_to_camera_frame(hdf5_path, mapping['frame_idx'], trajectory)
        
        # Save initial state before converting to delta (if applicable)
        # The state should always be absolute position, even when predicting deltas
        if isinstance(final_trajectory, torch.Tensor):
            initial_state = final_trajectory[0].numpy()
        else:
            initial_state = final_trajectory[0]
        
        # Flatten state to match proprio_dim (num_joints * coords)
        # This ensures state is (30,) instead of (10, 3)
        initial_state = initial_state.reshape(-1).astype(np.float32)
        
        # Convert to delta representation if requested
        if self.trajectory_representation == "delta":
            final_trajectory = self._convert_to_delta_representation(final_trajectory)
        
        # Normalize 3D trajectory if requested
        if self.normalize_coordinates and not self.output_2d_trajectory:
            # final_trajectory shape: [num_steps, num_joints, 3]
            # stats shape: [num_joints * 3]
            
            num_steps, num_joints, coords = final_trajectory.shape
            
            # Reshape stats to match trajectory [1, num_joints, 3]
            mean = self.stats_mean.view(1, num_joints, coords)
            std = self.stats_std.view(1, num_joints, coords)
            
            final_trajectory = (final_trajectory - mean) / std

        instruction = self._load_instruction(hdf5_path)
        
        if isinstance(final_trajectory, torch.Tensor):
            final_trajectory = final_trajectory.numpy()
        
        # Reshape trajectory from [num_steps, num_joints, coords] -> [num_steps, num_joints*coords]
        # This maintains 2D array shape (num_steps as first dim) for proper batching
        # instead of fully flattening to 1D
        num_steps = final_trajectory.shape[0]
        trajectory_flattened_joints = final_trajectory.reshape(num_steps, -1).astype(np.float32)
        
        # Determine style based on output format and trajectory representation
        if self.output_2d_trajectory:
            if self.output_format == "flow_matching":
                raise ValueError("trajectory_2d_fm and trajectory_2d_delta_fm modes have been removed. Use trajectory_2d_text or trajectory_3d_fm instead.")
            style = 'trajectory_2d_text'
        else:
            style = 'trajectory_3d_text' if self.output_format == "text" else 'trajectory_3d_fm'
        
        return {
            'image': image,
            'state': initial_state,  # Always use absolute position for state
            'message_list': [
                {
                    'label': instruction,
                    'points': final_trajectory, # to make it compatible with the data formatter, points means trajectory
                    'point_scale': 100 if self.normalize_coordinates and self.output_2d_trajectory else None,
                    'style': style,
                    'state': initial_state,  # Include the initial robot state so it can be formatted in the prompt
                }
            ],
            'trajectory_target': trajectory_flattened_joints,  # For flow matching: shape (num_steps, num_joints*coords) = (action_horizon, action_dim)
            'trajectory_shape': final_trajectory.shape,  # Store original shape for potential reshaping later
            'metadata': {
                'image': image,
                'task_name': mapping['task_name'],
                # 'episode_id': mapping['episode_id'],
                'frame_idx': mapping['frame_idx'],
                'output_2d_trajectory': self.output_2d_trajectory,
                'trajectory_representation': self.trajectory_representation,
            }
        }
    
    def _load_frame(self, video_path: str, frame_idx: int, max_retries: int = 3) -> Image.Image:
        """
        Load a single frame from a video file with retry logic and proper resource cleanup.
        
        Args:
            video_path: Path to the video file
            frame_idx: Frame index to load
            max_retries: Maximum number of retry attempts
            
        Returns:
            PIL Image of the requested frame
            
        Raises:
            ValueError: If frame cannot be loaded after max_retries attempts
        """
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
                        # Read failed, retry with exponential backoff
                        if attempt < max_retries - 1:
                            wait_time = (2 ** attempt) * 0.1 + random.uniform(0, 0.1)
                            print(f"Warning: Failed to read frame {frame_idx} from {video_path} "
                                  f"(attempt {attempt + 1}/{max_retries}), retrying in {wait_time:.2f}s...")
                            time.sleep(wait_time)
                            last_exception = ValueError(
                                f"Could not read frame {frame_idx} from {video_path} "
                                f"(attempt {attempt + 1}/{max_retries})"
                            )
                            continue
                        else:
                            raise ValueError(
                                f"Could not load frame {frame_idx} from {video_path} "
                                f"after {max_retries} attempts"
                            )
            except Exception as e:
                # Handle any other exceptions (e.g., file not found, corrupted file)
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 0.1 + random.uniform(0, 0.1)
                    print(f"Warning: Exception loading frame {frame_idx} from {video_path}: {str(e)} "
                          f"(attempt {attempt + 1}/{max_retries}), retrying in {wait_time:.2f}s...")
                    time.sleep(wait_time)
                    last_exception = e
                    continue
                else:
                    raise ValueError(
                        f"Could not load frame {frame_idx} from {video_path} "
                        f"after {max_retries} attempts: {str(e)}"
                    ) from last_exception
        
        # Should not reach here, but just in case
        raise ValueError(
            f"Could not load frame {frame_idx} from {video_path} "
            f"after {max_retries} attempts"
        )
    
    def _load_trajectory(self, hdf5_path: str, start_frame: int, num_steps: int, max_retries: int = 3) -> torch.Tensor:
        """
        Load trajectory data from HDF5 file with retry logic for NFS I/O errors.
        
        Args:
            hdf5_path: Path to the HDF5 file
            start_frame: Starting frame index
            num_steps: Number of steps to load
            max_retries: Maximum number of retry attempts
            
        Returns:
            torch.Tensor containing trajectory data
            
        Raises:
            OSError: If file cannot be opened after max_retries attempts
        """
        with hdf5_file_with_retry(hdf5_path, max_retries) as f:
            trajectories = {}
            for joint in self.joint_names:
                if f'transforms/{joint}' in f:
                    transforms = f[f'transforms/{joint}'][start_frame:start_frame + num_steps]
                    positions = transforms[:, :3, 3]
                    trajectories[joint] = positions
            
            trajectory_list = [trajectories.get(j, np.zeros((num_steps, 3))) for j in self.joint_names]
            trajectory = np.stack(trajectory_list, axis=1)
            return torch.from_numpy(trajectory).float()
    
    def _load_instruction(self, hdf5_path: str, max_retries: int = 3) -> str:
        """
        Load instruction from HDF5 file attributes with retry logic for NFS I/O errors.
        
        Args:
            hdf5_path: Path to the HDF5 file
            max_retries: Maximum number of retry attempts
            
        Returns:
            Instruction string from HDF5 file attributes
            
        Raises:
            OSError: If file cannot be opened after max_retries attempts
        """
        with hdf5_file_with_retry(hdf5_path, max_retries) as f:
            instruction = f.attrs.get('llm_description', 'No instruction available')
            if isinstance(instruction, bytes):
                instruction = instruction.decode('utf-8')
            return instruction
    
    def _transform_trajectory_to_camera_frame(self, hdf5_path: str, current_frame: int, trajectory: torch.Tensor, max_retries: int = 3) -> torch.Tensor:
        """
        Transform trajectory to camera frame with retry logic for NFS I/O errors.
        
        Args:
            hdf5_path: Path to the HDF5 file
            current_frame: Current frame index
            trajectory: Trajectory tensor to transform
            max_retries: Maximum number of retry attempts
            
        Returns:
            Transformed trajectory tensor in camera frame
            
        Raises:
            OSError: If file cannot be opened after max_retries attempts
        """
        with hdf5_file_with_retry(hdf5_path, max_retries) as f:
            if 'transforms/camera' not in f:
                print(f"Warning: No camera transform found in {hdf5_path}")
                return trajectory
            
            camera_transform_data = f['transforms/camera'][current_frame]
            camera_transform = torch.from_numpy(camera_transform_data.copy()).float()
        
        camera_transform_inv = torch.inverse(camera_transform)
        
        num_steps, num_joints, _ = trajectory.shape
        trajectory_homo = torch.cat([trajectory, torch.ones(num_steps, num_joints, 1)], dim=-1)
        trajectory_homo_flat = trajectory_homo.view(-1, 4)
        transformed_homo = trajectory_homo_flat @ camera_transform_inv.T
        transformed_3d = transformed_homo[:, :3]
        return transformed_3d.view(num_steps, num_joints, 3)

    def _project_trajectory_to_2d(self, hdf5_path: str, current_frame: int, trajectory: torch.Tensor, max_retries: int = 3) -> torch.Tensor:
        """
        Project trajectory to 2D with retry logic for NFS I/O errors.
        
        Args:
            hdf5_path: Path to the HDF5 file
            current_frame: Current frame index
            trajectory: Trajectory tensor to project
            max_retries: Maximum number of retry attempts
            
        Returns:
            Projected 2D trajectory tensor
            
        Raises:
            OSError: If file cannot be opened after max_retries attempts
        """
        with hdf5_file_with_retry(hdf5_path, max_retries) as f:
            if 'camera/intrinsic' in f and f['camera/intrinsic'][()].shape == (3, 3):
                intrinsic = f['camera/intrinsic'][()]
                img_width = intrinsic[0, 2] * 2
                img_height = intrinsic[1, 2] * 2
            else:
                intrinsic = np.array([[736.6339, 0., 960.], [0., 736.6339, 540.], [0., 0., 1.]])
                img_width, img_height = 1920, 1080
            
            intrinsic = torch.from_numpy(intrinsic).float()
        
        # Transform trajectory to camera frame
        trajectory_camera_frame = self._transform_trajectory_to_camera_frame(hdf5_path, current_frame, trajectory, max_retries)
        
        num_steps, num_joints, _ = trajectory_camera_frame.shape
        points_3d = trajectory_camera_frame.view(-1, 3)
        points_2d_homo = points_3d @ intrinsic.T
        
        w = points_2d_homo[:, 2:3]
        w = torch.where(w == 0, torch.ones_like(w), w)
        points_2d_pixel = points_2d_homo[:, :2] / w
        
        if self.normalize_coordinates:
            points_2d_normalized = torch.zeros_like(points_2d_pixel)
            points_2d_normalized[:, 0] = (points_2d_pixel[:, 0] / img_width) * 100.0
            points_2d_normalized[:, 1] = (points_2d_pixel[:, 1] / img_height) * 100.0
            points_2d_final = points_2d_normalized
        else:
            points_2d_final = points_2d_pixel
        
        return points_2d_final.view(num_steps, num_joints, 2)

    def _convert_to_delta_representation(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Convert absolute positions to delta positions (velocities).
        
        For a trajectory of shape (num_steps, num_joints, coords), computes:
        v_t = x_{t+1} - x_t for t = 0, ..., num_steps-2
        
        The last timestep uses v_{T-1} = x_T - x_{T-1} (same as the second to last).
        This keeps the same shape as input.
        
        Args:
            trajectory: Trajectory tensor of shape (num_steps, num_joints, coords)
            
        Returns:
            Delta trajectory tensor of shape (num_steps, num_joints, coords)
        """
        # Compute deltas: v_t = x_{t+1} - x_t
        # For timesteps 0 to T-2, compute x_{t+1} - x_t
        deltas = trajectory[1:] - trajectory[:-1]  # Shape: (num_steps-1, num_joints, coords)
        
        # For the last timestep, we duplicate the last delta to maintain shape
        # This means the last action is the same as the second-to-last action
        last_delta = deltas[-1:]  # Shape: (1, num_joints, coords)
        
        # Concatenate to get the full delta trajectory
        delta_trajectory = torch.cat([deltas, last_delta], dim=0)  # Shape: (num_steps, num_joints, coords)
        
        return delta_trajectory


def build_and_save_index_offline(
    data_dir: str,
    split: str = "train",
    action_chunking_horizon: int = 30,
    joint_names: Optional[List[str]] = None,
    force_rebuild: bool = False,
):
    """
    Build and save index mapping offline for faster dataset loading.
    
    Note: This always builds the full index with all frames (downsampling_ratio=1).
    Downsampling is applied in-memory when loading the dataset.
    
    Args:
        data_dir: Path to the data directory
        split: Dataset split ('train' or 'test')
        action_chunking_horizon: Number of frames in action chunks
        joint_names: Optional list of joint names to use
        force_rebuild: If True, rebuild index even if cache exists
    """
    print(f"\n{'='*80}")
    print(f"Building index mapping offline for {split} split")
    print(f"Data directory: {data_dir}")
    print(f"Action chunking horizon: {action_chunking_horizon}")
    print(f"Note: Building full index with all frames (downsampling applied at load time)")
    print(f"{'='*80}\n")
    
    # Create a temporary dataset instance to trigger index building
    # Always use frame_downsampling_ratio=1 to build full index
    dataset = TrajectoryDataset(
        data_dir=data_dir,
        split=split,
        action_chunking_horizon=action_chunking_horizon,
        joint_names=joint_names,
        frame_downsampling_ratio=1,
    )
    
    if force_rebuild:
        cache_file = dataset._get_cache_filepath()
        if cache_file.exists():
            print(f"Removing existing cache file: {cache_file}")
            cache_file.unlink()
            # Rebuild by creating a new instance
            dataset = TrajectoryDataset(
                data_dir=data_dir,
                split=split,
                action_chunking_horizon=action_chunking_horizon,
                joint_names=joint_names,
                frame_downsampling_ratio=1,
            )
    
    print(f"\n{'='*80}")
    print(f"Index building complete!")
    print(f"Total samples: {len(dataset.index_mapping)}")
    print(f"Cache saved to: {dataset._get_cache_filepath()}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build and save trajectory dataset index mapping offline")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to data directory (defaults to EGODEX_DATA_DIR env variable)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test", "overfit"],
        help="Dataset split to build index for"
    )
    parser.add_argument(
        "--action_chunking_horizon",
        type=int,
        default=30,
        help="Number of frames in action chunks"
    )
    parser.add_argument(
        "--force_rebuild",
        action="store_true",
        default=False,
        help="Force rebuild index even if cache exists"
    )
    parser.add_argument(
        "--build_all",
        action="store_true",
        default=False,
        help="Build index for both train and test splits"
    )
    
    args = parser.parse_args()
    
    # Get data directory
    data_dir = args.data_dir
    if data_dir is None:
        data_dir = os.environ.get("EGODEX_DATA_DIR")
        if data_dir is None:
            raise ValueError("--data_dir must be provided or EGODEX_DATA_DIR environment variable must be set")
    
    if args.build_all:
        # Build both train and test splits
        for split in ["train", "test", "overfit"]:
            build_and_save_index_offline(
                data_dir=data_dir,
                split=split,
                action_chunking_horizon=args.action_chunking_horizon,
                force_rebuild=args.force_rebuild,
            )
    else:
        # Build only the specified split
        build_and_save_index_offline(
            data_dir=data_dir,
            split=args.split,
            action_chunking_horizon=args.action_chunking_horizon,
            force_rebuild=args.force_rebuild,
        )