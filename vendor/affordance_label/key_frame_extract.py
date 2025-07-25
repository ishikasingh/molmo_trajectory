#!/usr/bin/env python3
"""
Key Frame Extraction Script

This script detects key frames based on hand velocity transitions.
Key frames are defined as frames where at least one hand transitions from 
moving (velocity > threshold) to stationary (velocity <= threshold).

Author: AI Assistant
Date: 2025-07-24
"""

import os
import json
import h5py
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from tqdm import tqdm
import argparse


class KeyFrameDetector:
    """
    A class to detect key frames based on hand velocity transitions.
    Key frames are detected when hands transition from moving to stationary.
    """
    
    def __init__(self, velocity_threshold: float = 5.0, fps: int = 30):
        """
        Initialize the key frame detector.
        
        Args:
            velocity_threshold: Maximum velocity (pixels/frame) to consider a keypoint stationary
            fps: Frames per second of the video
        """
        self.velocity_threshold = velocity_threshold
        self.fps = fps
        
    def calculate_keypoint_velocity(self, keypoints_current: Dict, keypoints_previous: Dict) -> Dict[str, float]:
        """
        Calculate velocity for each keypoint between two consecutive frames.
        
        Args:
            keypoints_current: Current frame keypoints
            keypoints_previous: Previous frame keypoints
            
        Returns:
            Dictionary with keypoint velocities
        """
        velocities = {}
        
        for keypoint_name in keypoints_current.keys():
            if keypoint_name in keypoints_previous and keypoint_name != "camera":
                # Get current and previous positions
                curr_pos = keypoints_current[keypoint_name]
                prev_pos = keypoints_previous[keypoint_name]
                
                # Skip if either position is None (missing data)
                if curr_pos is None or prev_pos is None:
                    velocities[keypoint_name] = float('inf')
                    continue
                
                # Handle different possible data structures
                if isinstance(curr_pos, list) and len(curr_pos) >= 2:
                    curr_x, curr_y = curr_pos[0], curr_pos[1]
                elif isinstance(curr_pos, dict) and 'x' in curr_pos and 'y' in curr_pos:
                    curr_x, curr_y = curr_pos['x'], curr_pos['y']
                else:
                    velocities[keypoint_name] = float('inf')  # Invalid data
                    continue
                    
                if isinstance(prev_pos, list) and len(prev_pos) >= 2:
                    prev_x, prev_y = prev_pos[0], prev_pos[1]
                elif isinstance(prev_pos, dict) and 'x' in prev_pos and 'y' in prev_pos:
                    prev_x, prev_y = prev_pos['x'], prev_pos['y']
                else:
                    velocities[keypoint_name] = float('inf')  # Invalid data
                    continue
                
                # Check for valid numeric values
                try:
                    curr_x, curr_y = float(curr_x), float(curr_y)
                    prev_x, prev_y = float(prev_x), float(prev_y)
                    
                    # Calculate Euclidean distance (velocity)
                    velocity = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
                    velocities[keypoint_name] = velocity
                except (ValueError, TypeError):
                    velocities[keypoint_name] = float('inf')  # Invalid numeric data
            else:
                velocities[keypoint_name] = float('inf')  # Missing data
                
        return velocities
    
    def is_hand_keypoint(self, keypoint_name: str) -> Tuple[bool, str]:
        """
        Check if a keypoint belongs to a hand and which hand.
        Based on JOINT_NAMES_OF_INTEREST from label_affordance.py:
        ['leftHand', 'leftThumbTip', 'leftIndexFingerTip', 'leftMiddleFingerTip', 'leftRingFingerTip', 'leftLittleFingerTip',
         'rightHand', 'rightThumbTip', 'rightIndexFingerTip', 'rightMiddleFingerTip', 'rightRingFingerTip', 'rightLittleFingerTip']
        
        Args:
            keypoint_name: Name of the keypoint
            
        Returns:
            Tuple of (is_hand_keypoint, hand_side) where hand_side is 'left' or 'right'
        """
        # Exact patterns from label_affordance.py
        left_hand_keypoints = [
            'leftHand', 'leftThumbTip', 'leftIndexFingerTip', 'leftMiddleFingerTip', 
            'leftRingFingerTip', 'leftLittleFingerTip'
        ]
        
        right_hand_keypoints = [
            'rightHand', 'rightThumbTip', 'rightIndexFingerTip', 'rightMiddleFingerTip',
            'rightRingFingerTip', 'rightLittleFingerTip'
        ]
        
        if keypoint_name in left_hand_keypoints:
            return True, 'left'
        elif keypoint_name in right_hand_keypoints:
            return True, 'right'
        
        # Fallback to pattern matching for compatibility
        keypoint_lower = keypoint_name.lower()
        if keypoint_lower.startswith('left') and ('hand' in keypoint_lower or 'finger' in keypoint_lower or 'thumb' in keypoint_lower):
            return True, 'left'
        elif keypoint_lower.startswith('right') and ('hand' in keypoint_lower or 'finger' in keypoint_lower or 'thumb' in keypoint_lower):
            return True, 'right'
            
        return False, ''
    
    def detect_keyframes_from_velocities(self, all_keypoints: List[Dict]) -> List[bool]:
        """
        Detect key frames based on hand velocity transitions.
        A frame is considered a key frame if at least one hand transitions from moving 
        (velocity > threshold) to stationary (velocity <= threshold).
        
        Args:
            all_keypoints: List of keypoint dictionaries for each frame
            
        Returns:
            List of boolean values indicating whether each frame is a key frame
        """
        if len(all_keypoints) < 3:  # Need at least 3 frames to detect transitions
            return [False] * len(all_keypoints)
            
        keyframe_flags = [False, False]  # First two frames cannot be keyframes
        
        # Hand keypoint groupings
        left_hand_keypoints = [
            'leftHand', 'leftThumbTip', 'leftIndexFingerTip', 'leftMiddleFingerTip', 
            'leftRingFingerTip', 'leftLittleFingerTip'
        ]
        
        right_hand_keypoints = [
            'rightHand', 'rightThumbTip', 'rightIndexFingerTip', 'rightMiddleFingerTip',
            'rightRingFingerTip', 'rightLittleFingerTip'
        ]
        
        # Track hand states for previous frame
        prev_left_hand_moving = None
        prev_right_hand_moving = None
        
        for frame_idx in range(2, len(all_keypoints)):  # Start from frame 2
            current_keypoints = all_keypoints[frame_idx]
            previous_keypoints = all_keypoints[frame_idx - 1]
            prev_prev_keypoints = all_keypoints[frame_idx - 2]
            
            # Calculate velocities between current and previous frame
            curr_velocities = self.calculate_keypoint_velocity(current_keypoints, previous_keypoints)
            
            # Calculate velocities between previous and frame before that
            prev_velocities = self.calculate_keypoint_velocity(previous_keypoints, prev_prev_keypoints)
            
            # Analyze left hand current state
            left_hand_velocities_curr = []
            for kp in left_hand_keypoints:
                if kp in curr_velocities and curr_velocities[kp] != float('inf'):
                    left_hand_velocities_curr.append(curr_velocities[kp])
            
            # Analyze left hand previous state
            left_hand_velocities_prev = []
            for kp in left_hand_keypoints:
                if kp in prev_velocities and prev_velocities[kp] != float('inf'):
                    left_hand_velocities_prev.append(prev_velocities[kp])
            
            # Analyze right hand current state
            right_hand_velocities_curr = []
            for kp in right_hand_keypoints:
                if kp in curr_velocities and curr_velocities[kp] != float('inf'):
                    right_hand_velocities_curr.append(curr_velocities[kp])
            
            # Analyze right hand previous state
            right_hand_velocities_prev = []
            for kp in right_hand_keypoints:
                if kp in prev_velocities and prev_velocities[kp] != float('inf'):
                    right_hand_velocities_prev.append(prev_velocities[kp])
            
            # Determine current hand states (moving vs stationary)
            left_hand_moving_curr = False
            right_hand_moving_curr = False
            left_hand_moving_prev = False
            right_hand_moving_prev = False
            
            # Current frame: hand is moving if majority of keypoints exceed threshold
            if left_hand_velocities_curr:
                moving_count = sum(1 for v in left_hand_velocities_curr if v > self.velocity_threshold)
                left_hand_moving_curr = moving_count >= len(left_hand_velocities_curr) * 0.8
            
            if right_hand_velocities_curr:
                moving_count = sum(1 for v in right_hand_velocities_curr if v > self.velocity_threshold)
                right_hand_moving_curr = moving_count >= len(right_hand_velocities_curr) * 0.8
            
            # Previous frame: hand was moving if majority of keypoints exceeded threshold
            if left_hand_velocities_prev:
                moving_count = sum(1 for v in left_hand_velocities_prev if v > self.velocity_threshold)
                left_hand_moving_prev = moving_count >= len(left_hand_velocities_prev) * 0.8
            
            if right_hand_velocities_prev:
                moving_count = sum(1 for v in right_hand_velocities_prev if v > self.velocity_threshold)
                right_hand_moving_prev = moving_count >= len(right_hand_velocities_prev) * 0.8
            
            # Detect transitions: moving -> stationary
            left_hand_transition = left_hand_moving_prev and not left_hand_moving_curr
            right_hand_transition = right_hand_moving_prev and not right_hand_moving_curr
            
            # Frame is a keyframe if at least one hand transitions from moving to stationary
            is_keyframe = left_hand_transition or right_hand_transition
            keyframe_flags.append(is_keyframe)
            
        return keyframe_flags
    
    def read_keypoints_from_json(self, json_path: str) -> List[Dict]:
        """
        Read keypoints from JSON file.
        
        Args:
            json_path: Path to the JSON file
            
        Returns:
            List of keypoint dictionaries for each frame
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Extract keypoints for each frame
        if 'frame_keypoints' in data:
            return [frame_data['keypoints_2d'] for frame_data in data['frame_keypoints']]
        else:
            raise ValueError(f"Expected 'frame_keypoints' key in JSON file: {json_path}")
    
    def read_keypoints_from_hdf5(self, hdf5_path: str) -> List[Dict]:
        """
        Read keypoints from HDF5 file using the same structure as label_affordance.py.
        
        Args:
            hdf5_path: Path to the HDF5 file
            
        Returns:
            List of keypoint dictionaries for each frame with 2D projected coordinates
        """
        # Hand keypoint names from label_affordance.py
        JOINT_NAMES_OF_INTEREST = [
            'leftHand', 'leftThumbTip', 'leftIndexFingerTip', 'leftMiddleFingerTip', 
            'leftRingFingerTip', 'leftLittleFingerTip', 'rightHand', 'rightThumbTip', 
            'rightIndexFingerTip', 'rightMiddleFingerTip', 'rightRingFingerTip', 
            'rightLittleFingerTip', 'camera'
        ]
        
        # Default camera intrinsics from label_affordance.py
        DEFAULT_INTRINSIC = np.array([
            [736.6339, 0., 960.], 
            [0., 736.6339, 540.], 
            [0., 0., 1.]
        ])
        
        keypoints_list = []
        
        with h5py.File(hdf5_path, 'r') as f:
            if 'transforms' not in f:
                raise ValueError(f"Expected 'transforms' key in HDF5 file: {hdf5_path}")
            
            keypoints_traj = f['transforms']
            
            # Load camera trajectory
            if 'camera' not in keypoints_traj:
                raise ValueError(f"Expected 'camera' trajectory in transforms: {hdf5_path}")
            
            camera_traj = keypoints_traj['camera'][:]  # [T, 4, 4]
            
            # Load keypoint trajectories for joints of interest
            joint_trajectories = {}
            for joint_name in JOINT_NAMES_OF_INTEREST:
                if joint_name in keypoints_traj and joint_name != 'camera':
                    joint_trajectories[joint_name] = keypoints_traj[joint_name][:]  # [T, 4, 4]
            
            # Convert 3D trajectories to 2D keypoints for each frame
            num_frames = len(camera_traj)
            
            for frame_idx in range(num_frames):
                frame_keypoints_2d = {}
                current_camera_pose = camera_traj[frame_idx]
                
                for joint_name, trajectory in joint_trajectories.items():
                    if frame_idx < len(trajectory):
                        # Extract 3D position from transformation matrix
                        transform_matrix = trajectory[frame_idx]  # [4, 4]
                        pos_3d = transform_matrix[:3, 3]  # Extract translation part
                        
                        # Project 3D to 2D
                        pos_2d = self.project_3d_to_2d(
                            pos_3d.reshape(1, 3), 
                            current_camera_pose, 
                            DEFAULT_INTRINSIC
                        )[0]  # Get first (and only) point
                        
                        frame_keypoints_2d[joint_name] = [float(pos_2d[0]), float(pos_2d[1])]
                    else:
                        frame_keypoints_2d[joint_name] = None
                
                keypoints_list.append(frame_keypoints_2d)
        
        return keypoints_list
    
    def project_3d_to_2d(self, points_3d: np.ndarray, camera_pose: np.ndarray, intrinsic_matrix: np.ndarray) -> np.ndarray:
        """
        Project 3D points to 2D using camera pose and intrinsic matrix.
        Based on the implementation in label_affordance.py.
        
        Args:
            points_3d: 3D points in world coordinates [N, 3]
            camera_pose: Camera pose transformation matrix [4, 4]
            intrinsic_matrix: Camera intrinsic matrix [3, 3]
        
        Returns:
            2D projected points [N, 2]
        """
        # Convert 3D points to homogeneous coordinates
        points_3d_homo = np.concatenate([points_3d, np.ones((points_3d.shape[0], 1))], axis=1)
        
        # Transform points to camera coordinate system
        # Note: camera_pose is world-to-camera transform, so we use it directly
        world_to_camera = np.linalg.inv(camera_pose)
        points_cam = (world_to_camera @ points_3d_homo.T).T
        
        # Extract 3D coordinates in camera frame
        points_cam_3d = points_cam[:, :3]
        
        # Project to 2D using intrinsic matrix
        points_2d_homo = (intrinsic_matrix @ points_cam_3d.T).T
        
        # Convert from homogeneous to 2D coordinates
        points_2d = points_2d_homo[:, :2] / points_2d_homo[:, 2:3]
        return points_2d
    
    def process_episode(self, mp4_path: str, keypoints_source: str) -> Tuple[List[bool], int]:
        """
        Process a single episode to detect key frames.
        
        Args:
            mp4_path: Path to the MP4 video file
            keypoints_source: Path to either JSON or HDF5 file containing keypoints
            
        Returns:
            Tuple of (keyframe_flags, total_frames)
        """
        # Determine file type and read keypoints
        if keypoints_source.endswith('.json'):
            keypoints_data = self.read_keypoints_from_json(keypoints_source)
        elif keypoints_source.endswith('.hdf5'):
            keypoints_data = self.read_keypoints_from_hdf5(keypoints_source)
        else:
            raise ValueError(f"Unsupported keypoints file format: {keypoints_source}")
        
        # Get video frame count for validation
        cap = cv2.VideoCapture(mp4_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # Validate that keypoints match video length
        if len(keypoints_data) != total_frames:
            print(f"Warning: Keypoints length ({len(keypoints_data)}) doesn't match video frames ({total_frames})")
            # Adjust to minimum length
            min_length = min(len(keypoints_data), total_frames)
            keypoints_data = keypoints_data[:min_length]
            total_frames = min_length
        
        # Detect key frames
        keyframe_flags = self.detect_keyframes_from_velocities(keypoints_data)
        
        return keyframe_flags, total_frames
    
    def save_keyframe_json(self, keyframe_flags: List[bool], output_path: str, episode_id: str):
        """
        Save key frame detection results to a JSON file.
        
        Args:
            keyframe_flags: List of boolean flags indicating key frames
            output_path: Path to save the JSON file
            episode_id: Episode identifier
        """
        # Convert boolean flags to binary string (like "00011001")
        binary_string = ''.join('1' if flag else '0' for flag in keyframe_flags)
        
        # Calculate transition intervals
        transition_indices = [i for i, flag in enumerate(keyframe_flags) if flag]
        transition_intervals = []
        if len(transition_indices) > 1:
            transition_intervals = [transition_indices[i+1] - transition_indices[i] 
                                   for i in range(len(transition_indices)-1)]
        
        # Create detailed output
        output_data = {
            'episode_id': episode_id,
            'total_frames': len(keyframe_flags),
            'keyframe_count': sum(keyframe_flags),
            'keyframe_binary': binary_string,
            'keyframe_indices': transition_indices,
            'transition_intervals': transition_intervals,
            'average_interval': sum(transition_intervals) / len(transition_intervals) if transition_intervals else 0,
            'detection_method': 'velocity_transition',
            'description': 'Key frames represent transitions from moving to stationary hand states',
            'parameters': {
                'velocity_threshold': self.velocity_threshold,
                'fps': self.fps,
                'detection_type': 'moving_to_stationary_transition'
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
    
    def create_annotated_video(self, input_video_path: str, keyframe_flags: List[bool], 
                             output_video_path: str, font_size: int = 50):
        """
        Create a video with text annotations showing key frames.
        
        Args:
            input_video_path: Path to the input video
            keyframe_flags: List of boolean flags indicating key frames
            output_video_path: Path to save the annotated video
            font_size: Size of the annotation text
        """
        cap = cv2.VideoCapture(input_video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Add annotation if this is a key frame
            if frame_idx < len(keyframe_flags) and keyframe_flags[frame_idx]:
                # Add text overlay
                cv2.putText(frame, 'TRANSITION', (50, font_size + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_size/30, (0, 255, 0), 3)
                cv2.putText(frame, 'KEY FRAME', (50, font_size + 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_size/40, (0, 255, 0), 2)
                
                # Add a colored border
                cv2.rectangle(frame, (0, 0), (width-1, height-1), (0, 255, 0), 5)
            
            out.write(frame)
            frame_idx += 1
        
        cap.release()
        out.release()


def find_episode_files(data_folder: str) -> List[Dict[str, str]]:
    """
    Find all episode files (MP4 and corresponding keypoint files) in the data folder.
    
    Args:
        data_folder: Path to the folder containing episode data
        
    Returns:
        List of dictionaries with file paths for each episode
    """
    data_path = Path(data_folder)
    episodes = []
    
    # Find all MP4 files
    mp4_files = list(data_path.glob("**/*.mp4"))
    
    for mp4_file in mp4_files:
        episode_id = mp4_file.stem
        
        # Look for corresponding keypoint files
        json_file = mp4_file.parent / f"{episode_id}_keypoint_projections_keypoints.json"
        hdf5_file = mp4_file.parent / f"{episode_id}.hdf5"
        
        keypoints_file = None
        if json_file.exists():
            keypoints_file = str(json_file)
        elif hdf5_file.exists():
            keypoints_file = str(hdf5_file)
        
        if keypoints_file:
            episodes.append({
                'episode_id': episode_id,
                'mp4_path': str(mp4_file),
                'keypoints_path': keypoints_file,
                'hdf5_path': str(hdf5_file) if hdf5_file.exists() else None
            })
    
    return episodes


def main():
    """
    Main function to run key frame detection on a dataset.
    """
    parser = argparse.ArgumentParser(description='Extract key frames based on hand velocity transitions (moving to stationary)')
    parser.add_argument('--data_folder', required=True, help='Path to folder containing episode data')
    parser.add_argument('--output_folder', required=True, help='Path to save output files')
    parser.add_argument('--velocity_threshold', type=float, default=5.0, 
                       help='Velocity threshold (pixels/frame) for detecting moving vs stationary states')
    parser.add_argument('--fps', type=int, default=30, help='Video frame rate')
    parser.add_argument('--create_videos', action='store_true', 
                       help='Create annotated videos showing key frames')
    parser.add_argument('--font_size', type=int, default=50, 
                       help='Font size for video annotations')
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if args.create_videos:
        video_output_path = output_path / "annotated_videos"
        video_output_path.mkdir(exist_ok=True)
    
    # Initialize detector
    detector = KeyFrameDetector(velocity_threshold=args.velocity_threshold, fps=args.fps)
    
    # Find all episodes
    episodes = find_episode_files(args.data_folder)
    print(f"Found {len(episodes)} episodes to process")
    
    # Process each episode
    for episode in tqdm(episodes, desc="Processing episodes"):
        try:
            # Detect key frames
            keyframe_flags, total_frames = detector.process_episode(
                episode['mp4_path'], 
                episode['keypoints_path']
            )
            
            # Save JSON results
            json_output_path = output_path / f"{episode['episode_id']}_keyframes.json"
            detector.save_keyframe_json(keyframe_flags, str(json_output_path), episode['episode_id'])
            
            # Create annotated video if requested
            if args.create_videos:
                video_output_file = video_output_path / f"{episode['episode_id']}_annotated.mp4"
                detector.create_annotated_video(
                    episode['mp4_path'], 
                    keyframe_flags, 
                    str(video_output_file),
                    args.font_size
                )
            
            print(f"Processed episode {episode['episode_id']}: "
                  f"{sum(keyframe_flags)}/{total_frames} key frames detected")
            
        except Exception as e:
            print(f"Error processing episode {episode['episode_id']}: {str(e)}")
            continue


if __name__ == "__main__":
    main()
