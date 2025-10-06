#!/usr/bin/env python3
"""
Trajectory Dataset for VLA Training
"""

import os
import h5py
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional
from PIL import Image
from tqdm import tqdm
from olmo.data.dataset import Dataset


class TrajectoryDataset(Dataset):
    """PyTorch Dataset for trajectory prediction training."""
    
    def __init__(
        self,
        data_dir: str = None,
        split: str = "train",
        action_chunking_horizon: int = 10,
        joint_names: Optional[List[str]] = None,
        output_2d_trajectory: bool = True,
        normalize_2d_coordinates: bool = True,
        use_confidence_filter: bool = False,
        confidence_threshold: float = 0.5,
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
        self.normalize_2d_coordinates = normalize_2d_coordinates
        
        # Default joint names matching data_formatter.py TRAJECTORY_KEYPOINTS
        if joint_names is None:
            self.joint_names = [
                'leftThumbTip', 'leftIndexFingerTip', 'leftMiddleFingerTip', 'leftRingFingerTip', 'leftLittleFingerTip',
                'rightThumbTip', 'rightIndexFingerTip', 'rightMiddleFingerTip', 'rightRingFingerTip', 'rightLittleFingerTip'
            ]
        else:
            self.joint_names = joint_names
        
        print(f"Building index mapping for {split} split...")
        self.index_mapping = self._build_index_mapping()
        print(f"Loaded {len(self.index_mapping)} samples from {split} split")
        
    def _build_index_mapping(self) -> List[Dict]:
        """Build a flat index mapping across all videos and frames."""
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
        
        return index_mapping
    
    def __len__(self):
        return len(self.index_mapping)
    
    def get(self, idx, rng):
        """Get a single training example."""
        mapping = self.index_mapping[idx]
        
        image = self._load_frame(mapping['video_path'], mapping['frame_idx'])
        trajectory = self._load_trajectory(mapping['hdf5_path'], mapping['frame_idx'], self.action_chunking_horizon)
        
        if self.output_2d_trajectory:
            final_trajectory = self._project_trajectory_to_2d(mapping['hdf5_path'], mapping['frame_idx'], trajectory)
        else:
            final_trajectory = self._transform_trajectory_to_camera_frame(mapping['hdf5_path'], mapping['frame_idx'], trajectory)
        
        instruction = self._load_instruction(mapping['hdf5_path'])
        
        if isinstance(final_trajectory, torch.Tensor):
            final_trajectory = final_trajectory.numpy()
        
        return {
            'image': image,
            'message_list': [
                {
                    'label': instruction,
                    'points': final_trajectory, # to make it compatible with the data formatter, points means trajectory
                    'point_scale': 100 if self.normalize_2d_coordinates and self.output_2d_trajectory else None,
                    'style': 'trajectory_2d' if self.output_2d_trajectory else 'trajectory_3d',
                }
            ],
            'metadata': {
                'image': image,
                'task_name': mapping['task_name'],
                # 'episode_id': mapping['episode_id'],
                'frame_idx': mapping['frame_idx'],
                'output_2d_trajectory': self.output_2d_trajectory,
            }
        }
    
    def _load_frame(self, video_path: str, frame_idx: int) -> Image.Image:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError(f"Could not load frame {frame_idx} from {video_path}")
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame)
    
    def _load_trajectory(self, hdf5_path: str, start_frame: int, num_steps: int) -> torch.Tensor:
        with h5py.File(hdf5_path, 'r') as f:
            trajectories = {}
            for joint in self.joint_names:
                if f'transforms/{joint}' in f:
                    transforms = f[f'transforms/{joint}'][start_frame:start_frame + num_steps]
                    positions = transforms[:, :3, 3]
                    trajectories[joint] = positions
            
            trajectory_list = [trajectories.get(j, np.zeros((num_steps, 3))) for j in self.joint_names]
            trajectory = np.stack(trajectory_list, axis=1)
            return torch.from_numpy(trajectory).float()
    
    def _load_instruction(self, hdf5_path: str) -> str:
        with h5py.File(hdf5_path, 'r') as f:
            instruction = f.attrs.get('llm_description', 'No instruction available')
            if isinstance(instruction, bytes):
                instruction = instruction.decode('utf-8')
            return instruction
    
    def _transform_trajectory_to_camera_frame(self, hdf5_path: str, current_frame: int, trajectory: torch.Tensor) -> torch.Tensor:
        with h5py.File(hdf5_path, 'r') as f:
            if 'transforms/camera' not in f:
                return trajectory
            
            camera_transform = torch.from_numpy(f['transforms/camera'][current_frame]).float()
            camera_transform_inv = torch.inverse(camera_transform)
            
            num_steps, num_joints, _ = trajectory.shape
            trajectory_homo = torch.cat([trajectory, torch.ones(num_steps, num_joints, 1)], dim=-1)
            trajectory_homo_flat = trajectory_homo.view(-1, 4)
            transformed_homo = trajectory_homo_flat @ camera_transform_inv.T
            transformed_3d = transformed_homo[:, :3]
            return transformed_3d.view(num_steps, num_joints, 3)

    def _project_trajectory_to_2d(self, hdf5_path: str, current_frame: int, trajectory: torch.Tensor) -> torch.Tensor:
        with h5py.File(hdf5_path, 'r') as f:
            if 'camera/intrinsic' in f and f['camera/intrinsic'][()].shape == (3, 3):
                intrinsic = f['camera/intrinsic'][()]
                img_width = intrinsic[0, 2] * 2
                img_height = intrinsic[1, 2] * 2
            else:
                intrinsic = np.array([[736.6339, 0., 960.], [0., 736.6339, 540.], [0., 0., 1.]])
                img_width, img_height = 1920, 1080
            
            intrinsic = torch.from_numpy(intrinsic).float()
            trajectory_camera_frame = self._transform_trajectory_to_camera_frame(hdf5_path, current_frame, trajectory)
            
            num_steps, num_joints, _ = trajectory_camera_frame.shape
            points_3d = trajectory_camera_frame.view(-1, 3)
            points_2d_homo = points_3d @ intrinsic.T
            
            w = points_2d_homo[:, 2:3]
            w = torch.where(w == 0, torch.ones_like(w), w)
            points_2d_pixel = points_2d_homo[:, :2] / w
            
            if self.normalize_2d_coordinates:
                points_2d_normalized = torch.zeros_like(points_2d_pixel)
                points_2d_normalized[:, 0] = (points_2d_pixel[:, 0] / img_width) * 100.0
                points_2d_normalized[:, 1] = (points_2d_pixel[:, 1] / img_height) * 100.0
                points_2d_final = points_2d_normalized
            else:
                points_2d_final = points_2d_pixel
            
            return points_2d_final.view(num_steps, num_joints, 2)