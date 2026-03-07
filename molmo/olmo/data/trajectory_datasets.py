#!/usr/bin/env python3
"""
Trajectory Dataset for VLA Training
"""

import os
import h5py
import cv2
import numpy as np
import torch
import torch.nn.functional as F
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
        pad_action_chunk: bool = False,  # If True, pad action chunks with repeated last step when near end of trajectory
        interpolation_times: int = 1,  # If > 1, load horizon//interpolation_times steps and interpolate to horizon steps
    ):
        # Get data directory from environment variable if not provided
        if data_dir is None:
            data_dir = os.environ.get("EGODEX_DATA_DIR")
            if data_dir is None:
                raise ValueError("data_dir must be provided or EGODEX_DATA_DIR environment variable must be set")
        
        self.data_dir = Path(data_dir)
        self.split = split
        self.action_chunking_horizon = action_chunking_horizon
        self.interpolation_times = max(1, int(interpolation_times))
        self.steps_to_load = max(1, action_chunking_horizon // self.interpolation_times)
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
        self.pad_action_chunk = pad_action_chunk
        self.joint_names = [
                'leftThumbTip', 'leftIndexFingerTip', 'leftMiddleFingerTip', 'leftRingFingerTip', 'leftLittleFingerTip',
                'rightThumbTip', 'rightIndexFingerTip', 'rightMiddleFingerTip', 'rightRingFingerTip', 'rightLittleFingerTip'
            ]
        
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

            # Default joint names matching data_formatter.py TRAJECTORY_KEYPOINTS
            
            
            if joint_names is None:
                self.finger_indices = list(range(0, len(self.joint_names)))
            else:
                self.finger_indices = [self.joint_names.index(j) for j in joint_names if j in self.joint_names]

            # Select only the index fingers from mean and std
            # stats["mean"] is shape [30], reshape to (10,3), then select [1,6], then flatten to (6,)
            self.stats_mean = stats["mean"].reshape(len(self.joint_names), 3)[self.finger_indices].reshape(-1)
            self.stats_std = stats["std"].reshape(len(self.joint_names), 3)[self.finger_indices].reshape(-1)

            

            # Ensure stats are on CPU/float
            if isinstance(self.stats_mean, torch.Tensor):
                self.stats_mean = self.stats_mean.float()
                self.stats_std = self.stats_std.float()

        
        
        print(f"Loading index mapping for {split} split...")
        self.index_mapping = self._build_index_mapping()
        
        # Apply overfit example limit if specified (only when split="overfit")
        if self.split == "overfit" and self.overfit_num_examples is not None:
            original_len = len(self.index_mapping)
            self.index_mapping = self.index_mapping[:self.overfit_num_examples]
            print(f"Limited overfit split to {len(self.index_mapping)} examples (from {original_len})")
        
        print(f"Loaded {len(self.index_mapping)} samples from {split} split")
        
    def _get_cache_filepath(self) -> Path:
        """Cache path keyed by steps_to_load (actual loading horizon), not action_chunking_horizon."""
        parts = [f"index_mapping_{self.split}_horizon{self.steps_to_load}"]
        if self.pad_action_chunk:
            parts.append("padded")
        return self.data_dir / ("_".join(parts) + ".pkl")

    def _load_index_from_cache(self) -> Optional[List[Dict]]:
        """Load index mapping from cache file if it exists."""
        cache_file = self._get_cache_filepath()
        if cache_file.exists():
            print(f"Loading cached index mapping from {cache_file}...")
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            # Effective steps to load: horizon // interpolation_times (same index for same value)
            cached_steps_to_load = cached_data.get('steps_to_load')
            if cached_steps_to_load is None:
                h = cached_data.get('action_chunking_horizon')
                cached_steps_to_load = (h // max(1, cached_data.get('interpolation_times', 1))) if h is not None else None
            horizon_match = cached_steps_to_load == self.steps_to_load
            split_ok = cached_data.get('split') == self.split
            joint_names_ok = cached_data.get('joint_names') == self.joint_names
            pad_ok = cached_data.get('pad_action_chunk', False) == self.pad_action_chunk
            if split_ok and horizon_match and pad_ok: # and joint_names_ok:
                full_index = cached_data['index_mapping']
                print(f"Successfully loaded {len(full_index)} samples from cache")
                if self.frame_downsampling_ratio > 1:
                    filtered_index = self._apply_downsampling_filter(full_index)
                    print(f"Applied downsampling ratio {self.frame_downsampling_ratio}: {len(full_index)} -> {len(filtered_index)} samples")
                    return filtered_index
                return full_index
            mismatches = []
            if not split_ok:
                mismatches.append(f"split (cached={cached_data.get('split')!r}, current={self.split!r})")
            if not horizon_match:
                mismatches.append(f"steps_to_load (cached={cached_steps_to_load}, current={self.steps_to_load})")
            if not joint_names_ok:
                mismatches.append("joint_names")
            if not pad_ok:
                mismatches.append(f"pad_action_chunk (cached={cached_data.get('pad_action_chunk', False)}, current={self.pad_action_chunk})")
            print(f"Cache configuration mismatch, rebuilding index. Mismatch: {', '.join(mismatches)}")
            print(cached_data.get('joint_names'), self.joint_names)
        return None
    
    def _save_index_to_cache(self, index_mapping: List[Dict]) -> None:
        """Save index mapping to cache file. Path uses steps_to_load; dict includes action_chunking_horizon for backward compat."""
        cache_file = self._get_cache_filepath()
        cache_data = {
            'split': self.split,
            'steps_to_load': self.steps_to_load,
            'joint_names': self.joint_names,
            'pad_action_chunk': self.pad_action_chunk,
            'action_chunking_horizon': self.action_chunking_horizon,
            'interpolation_times': self.interpolation_times,
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
            split_dirs = [self.data_dir / "test"]
            # split_dirs = [self.data_dir / "small_test"]
        elif self.split == "train_pick_and_place":
            split_dirs = [self.data_dir / "part_pick_and_place"]
        elif self.split == "test_pick_and_place":
            split_dirs = [self.data_dir / "test_pick_and_place"]
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
            try: 
                if not hdf5_file.exists():
                    continue
            except Exception as e:
                print(f"    Error processing {video_file}: {e}")
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
                
                if self.pad_action_chunk:
                    # When padding is enabled, include all frames
                    # Frames near the end will have their action chunks padded with repeated last step
                    # requires at least steps_to_load frames
                    if trajectory_length < self.steps_to_load:
                        continue
                    
                    for frame_idx in range(trajectory_length):
                        index_mapping.append({
                            'video_path': str(video_file),
                            'hdf5_path': str(hdf5_file),
                            'frame_idx': frame_idx,
                            'task_name': task_name,
                            'trajectory_length': trajectory_length,  # Store for padding calculation
                        })
                else:
                    # Original behavior: only include frames with enough future steps (steps_to_load when using interpolation)
                    if trajectory_length < self.steps_to_load:
                        continue
                    
                    for frame_idx in range(trajectory_length - self.steps_to_load):
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
        
        # Apply downsampling filter if needed
        if self.frame_downsampling_ratio > 1:
            filtered_index = self._apply_downsampling_filter(index_mapping)
            print(f"Applied downsampling ratio {self.frame_downsampling_ratio}: {len(index_mapping)} -> {len(filtered_index)} samples")
            return filtered_index
        
        return index_mapping
    
    def __len__(self):
        return len(self.index_mapping)
    
    def get(self, idx):
        """Get a single training example."""
        mapping = self.index_mapping[idx]
        
        video_path = mapping['video_path']
        hdf5_path = mapping['hdf5_path']
        
        if self.load_images:
            image = self._load_frame(video_path, mapping['frame_idx'])
        else:
            # Return dummy image (1x1 pixel black image)
            image = Image.fromarray(np.zeros((1, 1, 3), dtype=np.uint8))
        
        # Get trajectory_length for padding calculation (only available when pad_action_chunk is enabled)
        trajectory_length = mapping.get('trajectory_length', None)
        trajectory = self._load_trajectory(hdf5_path, mapping['frame_idx'], self.steps_to_load, trajectory_length)
        if self.interpolation_times > 1:
            trajectory = self._interpolate_trajectory(trajectory)
        
        if self.output_2d_trajectory:
            final_trajectory = self._project_trajectory_to_2d(hdf5_path, mapping['frame_idx'], trajectory)
        else:
            final_trajectory = self._transform_trajectory_to_camera_frame(hdf5_path, mapping['frame_idx'], trajectory)

        trajectory_absolute_cam = final_trajectory
        # final_trajectory = final_trajectory[:, [1, 6], :]
        # print(f"final_trajectory shape: {final_trajectory.shape}")
        
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
        
        # Build base result dict
        result = {
            'image': image,
            'state': initial_state,  # Always use absolute position for state
            'trajectory_shape': final_trajectory.shape,  # Store original shape for potential reshaping later
            'expert_type': 0,  # Human trajectory expert (for multi-expert routing)
            'metadata': {
                'image': image,
                'task_name': mapping['task_name'],
                # 'episode_id': mapping['episode_id'],
                'frame_idx': mapping['frame_idx'],
                'output_2d_trajectory': self.output_2d_trajectory,
                'trajectory_representation': self.trajectory_representation,
                'trajectory_absolute_cam': trajectory_absolute_cam,
            }
        }
        
        # For text-based modes, use message_list (needed for text output formatting)
        # For flow matching, use top-level fields (no text output needed)
        if style in ['trajectory_2d_text', 'trajectory_3d_text']:
            result['message_list'] = [
                {
                    'label': instruction,
                    'points': final_trajectory,  # For text output formatting
                    'point_scale': 100 if self.normalize_coordinates and self.output_2d_trajectory else None,
                    'style': style,
                    'state': initial_state,  # Include the initial robot state so it can be formatted in the prompt
                }
            ]
        else:
            # Flow matching mode: use top-level fields for prompt formatting, trajectory_target for training
            result['label'] = instruction
            result['style'] = style
            result['trajectory_target'] = trajectory_flattened_joints  # For flow matching: shape (num_steps, num_joints*coords) = (action_horizon, action_dim)
        
        return result
    
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
    
    def _load_trajectory(self, hdf5_path: str, start_frame: int, num_steps: int, trajectory_length: int = None, max_retries: int = 3) -> torch.Tensor:
        """
        Load trajectory data from HDF5 file with retry logic for NFS I/O errors.
        
        Args:
            hdf5_path: Path to the HDF5 file
            start_frame: Starting frame index
            num_steps: Number of steps to load
            trajectory_length: Total length of the trajectory (for padding calculation)
            max_retries: Maximum number of retry attempts
            
        Returns:
            torch.Tensor containing trajectory data
            
        Raises:
            OSError: If file cannot be opened after max_retries attempts
        """
        with hdf5_file_with_retry(hdf5_path, max_retries) as f:
            trajectories = {}
            
            # Calculate how many steps we can actually load
            if trajectory_length is not None:
                available_steps = min(num_steps, trajectory_length - start_frame)
            else:
                available_steps = num_steps
            
            for joint in self.joint_names:
                if f'transforms/{joint}' in f:
                    transforms = f[f'transforms/{joint}'][start_frame:start_frame + available_steps]
                    positions = transforms[:, :3, 3]
                    trajectories[joint] = positions
            
            trajectory_list = [trajectories.get(j, np.zeros((available_steps, 3))) for j in self.joint_names]
            trajectory = np.stack(trajectory_list, axis=1)
            
            # Pad with repeated last step if we don't have enough steps
            if self.pad_action_chunk and available_steps < num_steps:
                padding_needed = num_steps - available_steps
                last_step = trajectory[-1:]  # Shape: (1, num_joints, 3)
                padding = np.repeat(last_step, padding_needed, axis=0)
                trajectory = np.concatenate([trajectory, padding], axis=0)
            
            return torch.from_numpy(trajectory).float()
    
    def _interpolate_trajectory(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Linearly interpolate trajectory from steps_to_load steps to action_chunking_horizon steps.
        trajectory: [steps_to_load, num_joints, 3]
        Returns: [action_chunking_horizon, num_joints, 3]
        """
        if self.interpolation_times <= 1 or trajectory.shape[0] >= self.action_chunking_horizon:
            return trajectory
        steps_in = trajectory.shape[0]
        steps_out = self.action_chunking_horizon
        device = trajectory.device
        dtype = trajectory.dtype
        if steps_out == 1:
            return trajectory[0:1].expand(1, *trajectory.shape[1:]).clone()
        idx_float = torch.linspace(0, steps_in - 1, steps_out, device=device, dtype=dtype)
        idx_low = idx_float.long().clamp(0, steps_in - 2)
        idx_high = (idx_low + 1).clamp(max=steps_in - 1)
        weight_high = (idx_float - idx_low.float()).unsqueeze(-1).unsqueeze(-1)
        low = trajectory[idx_low]
        high = trajectory[idx_high]
        interp = low + weight_high * (high - low)
        return interp
    
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

    def get_videos_info(self) -> List[Dict]:
        """
        Get information about all videos in the dataset, grouped by video path.
        
        Returns:
            List of dictionaries, each containing:
                - video_path: Path to the video file
                - hdf5_path: Path to the corresponding HDF5 file
                - task_name: Name of the task
                - num_frames: Number of available frames in this video
                - frame_indices: List of frame indices available in the dataset
        """
        from collections import defaultdict
        
        # Group frames by video path
        video_to_frames = defaultdict(list)
        for idx, mapping in enumerate(self.index_mapping):
            video_path = mapping['video_path']
            video_to_frames[video_path].append({
                'dataset_idx': idx,
                'frame_idx': mapping['frame_idx'],
                'task_name': mapping['task_name'],
                'hdf5_path': mapping['hdf5_path'],
            })
        
        # Sort frames within each video by frame_idx and build info list
        videos_info = []
        for video_path, frames in video_to_frames.items():
            frames_sorted = sorted(frames, key=lambda x: x['frame_idx'])
            videos_info.append({
                'video_path': video_path,
                'hdf5_path': frames_sorted[0]['hdf5_path'],
                'task_name': frames_sorted[0]['task_name'],
                'num_frames': len(frames_sorted),
                'frame_indices': [f['frame_idx'] for f in frames_sorted],
                '_frames_data': frames_sorted,  # Internal use for iter_video_frames
            })
        
        # Sort videos by path for consistent ordering
        videos_info.sort(key=lambda x: x['video_path'])
        
        return videos_info

    def iter_video_frames(self, 
                          video_info: Dict,
                          rng: Optional[np.random.RandomState] = None) -> List[Dict]:
        """
        Load all frames from a specific video.
        
        Note: Frame downsampling is controlled by the dataset's frame_downsampling_ratio
        parameter, so all frames in the index_mapping for this video are loaded.
        
        Args:
            video_info: Video info dictionary from get_videos_info()
            rng: Random state for dataset.get() (optional)
            
        Returns:
            List of frame data dictionaries, each containing:
                - image: numpy array of the image
                - prompt: instruction text
                - ground_truth_trajectory: trajectory points
                - point_scale: scale for points
                - style: data style
                - task_name: name of the task
                - frame_idx: original frame index in video
                - state: initial state
                - trajectory_representation: 'absolute' or 'delta'
        """
        if rng is None:
            rng = np.random.RandomState(42)
        
        frames_data = video_info.get('_frames_data', [])
        if not frames_data:
            return []
        
        result_frames = []
        for frame_info in frames_data:
            dataset_idx = frame_info['dataset_idx']
            
            # Load the example from dataset
            example_data = self.get(dataset_idx, rng=rng)
            
            # Extract data
            image = example_data["image"]
            metadata = example_data.get("metadata", {})
            
            # Handle both message_list (text-based) and top-level fields (flow matching)
            if "message_list" in example_data and example_data["message_list"] and len(example_data["message_list"]) > 0:
                # Text-based mode: read from message_list
                message_list = example_data["message_list"]
                instruction = message_list[0].get("label", "")
                gt_trajectory = message_list[0].get("points", None)
                point_scale = message_list[0].get("point_scale", None)
                style = message_list[0].get("style", "")
                state = message_list[0].get("state", None)
            else:
                # Flow matching mode: read from top-level fields
                instruction = example_data.get("label", "")
                gt_trajectory = example_data.get("trajectory_target", None)  # Use trajectory_target for flow matching
                point_scale = None  # Not used for flow matching
                style = example_data.get("style", "")
                state = example_data.get("state", None)
            
            # Convert image to numpy if needed
            if hasattr(image, 'save'):  # PIL Image
                image_np = np.array(image)
            else:
                image_np = image
            
            frame_data = {
                "image": image_np,
                "prompt": instruction,
                "ground_truth_trajectory": gt_trajectory,
                "point_scale": point_scale,
                "style": style,
                "task_name": metadata.get("task_name", "unknown"),
                "frame_idx": frame_info['frame_idx'],
                "state": state,
                "trajectory_representation": metadata.get("trajectory_representation", "absolute"),
            }
            
            result_frames.append(frame_data)
        
        return result_frames

    def sample_videos(self, num_videos: int) -> List[Dict]:
        """
        Sample videos evenly from the dataset.
        
        Args:
            num_videos: Number of videos to sample
            
        Returns:
            List of video info dictionaries (subset of get_videos_info())
        """
        all_videos = self.get_videos_info()
        
        if not all_videos:
            return []
        
        # Sample evenly across the available videos
        indices = np.linspace(0, len(all_videos) - 1, min(num_videos, len(all_videos)), dtype=int)
        
        return [all_videos[i] for i in indices]


def build_and_save_index_offline(
    data_dir: str,
    split: str = "train",
    action_chunking_horizon: int = 30,
    joint_names: Optional[List[str]] = None,
    force_rebuild: bool = False,
    pad_action_chunk: bool = False,
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
        pad_action_chunk: If True, include all frames and pad action chunks with repeated last step
    """
    print(f"\n{'='*80}")
    print(f"Building index mapping offline for {split} split")
    print(f"Data directory: {data_dir}")
    print(f"Action chunking horizon: {action_chunking_horizon}")
    print(f"Pad action chunk: {pad_action_chunk}")
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
        pad_action_chunk=pad_action_chunk,
        output_format="flow_matching",
        output_2d_trajectory=False,
        stats_file=os.environ.get("TRAJECTORY_STATS_FILE"),
        trajectory_representation="delta",
    )

    import ipdb; ipdb.set_trace()
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
                pad_action_chunk=pad_action_chunk,
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
    parser.add_argument(
        "--pad_action_chunk",
        action="store_true",
        default=False,
        help="Include all frames and pad action chunks with repeated last step when near end of trajectory"
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
                pad_action_chunk=args.pad_action_chunk,
            )
    else:
        # Build only the specified split
        build_and_save_index_offline(
            data_dir=data_dir,
            split=args.split,
            action_chunking_horizon=args.action_chunking_horizon,
            force_rebuild=args.force_rebuild,
            pad_action_chunk=args.pad_action_chunk,
        )