#!/usr/bin/env python3
"""
Key Frame Extraction Script

This script detects key frames based on hand keypoint velocities.
Key frames are defined as frames where at least one hand has keypoints 
with velocities below a certain threshold.

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
    A class to detect key frames based on hand keypoint velocities.
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
                
                # Calculate Euclidean distance (velocity)
                velocity = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
                velocities[keypoint_name] = velocity
            else:
                velocities[keypoint_name] = float('inf')  # Missing data
                
        return velocities
    
    def is_hand_keypoint(self, keypoint_name: str) -> Tuple[bool, str]:
        """
        Check if a keypoint belongs to a hand and which hand.
        
        Args:
            keypoint_name: Name of the keypoint
            
        Returns:
            Tuple of (is_hand_keypoint, hand_side) where hand_side is 'left' or 'right'
        """
        keypoint_lower = keypoint_name.lower()
        
        # Common hand keypoint patterns
        left_patterns = ['left_hand', 'lefthand', 'left_wrist', 'leftwrist', 'left_thumb', 'left_index', 'left_middle', 'left_ring', 'left_pinky']
        right_patterns = ['right_hand', 'righthand', 'right_wrist', 'rightwrist', 'right_thumb', 'right_index', 'right_middle', 'right_ring', 'right_pinky']
        
        for pattern in left_patterns:
            if pattern in keypoint_lower:
                return True, 'left'
                
        for pattern in right_patterns:
            if pattern in keypoint_lower:
                return True, 'right'
                
        # Generic hand patterns
        if 'hand' in keypoint_lower or 'wrist' in keypoint_lower or 'thumb' in keypoint_lower or \
           'index' in keypoint_lower or 'middle' in keypoint_lower or 'ring' in keypoint_lower or 'pinky' in keypoint_lower:
            return True, 'unknown'
            
        return False, ''
    
    def detect_keyframes_from_velocities(self, all_keypoints: List[Dict]) -> List[bool]:
        """
        Detect key frames based on hand keypoint velocities.
        
        Args:
            all_keypoints: List of keypoint dictionaries for each frame
            
        Returns:
            List of boolean values indicating whether each frame is a key frame
        """
        if len(all_keypoints) < 2:
            return [False] * len(all_keypoints)
            
        keyframe_flags = [False]  # First frame is not a keyframe (no previous frame)
        
        for frame_idx in range(1, len(all_keypoints)):
            current_keypoints = all_keypoints[frame_idx]
            previous_keypoints = all_keypoints[frame_idx - 1]
            
            # Calculate velocities for all keypoints
            velocities = self.calculate_keypoint_velocity(current_keypoints, previous_keypoints)
            
            # Check if any hand has low velocity keypoints
            left_hand_low_velocity = False
            right_hand_low_velocity = False
            
            for keypoint_name, velocity in velocities.items():
                is_hand, hand_side = self.is_hand_keypoint(keypoint_name)
                
                if is_hand and velocity <= self.velocity_threshold:
                    if hand_side == 'left':
                        left_hand_low_velocity = True
                    elif hand_side == 'right':
                        right_hand_low_velocity = True
                    elif hand_side == 'unknown':
                        # If we can't determine the hand side, consider it as both
                        left_hand_low_velocity = True
                        right_hand_low_velocity = True
            
            # Frame is a keyframe if at least one hand has low velocity
            is_keyframe = left_hand_low_velocity or right_hand_low_velocity
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
        Read keypoints from HDF5 file.
        
        Args:
            hdf5_path: Path to the HDF5 file
            
        Returns:
            List of keypoint dictionaries for each frame
        """
        keypoints_list = []
        
        with h5py.File(hdf5_path, 'r') as f:
            # Try different possible structures in HDF5
            if 'keypoints' in f:
                keypoints_data = f['keypoints'][:]
                # Convert to list of dictionaries
                for frame_keypoints in keypoints_data:
                    # This depends on the exact structure of your HDF5 file
                    # You may need to adjust this based on your data format
                    keypoints_list.append(frame_keypoints)
            elif 'frame_keypoints' in f:
                frame_keypoints = f['frame_keypoints']
                for frame_idx in range(len(frame_keypoints)):
                    keypoints_list.append(dict(frame_keypoints[frame_idx]))
            else:
                # Try to find any dataset that might contain keypoints
                def find_keypoints(name, obj):
                    if isinstance(obj, h5py.Dataset) and 'keypoint' in name.lower():
                        keypoints_list.extend(obj[:])
                
                f.visititems(find_keypoints)
                
                if not keypoints_list:
                    raise ValueError(f"Could not find keypoints data in HDF5 file: {hdf5_path}")
        
        return keypoints_list
    
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
        
        # Create detailed output
        output_data = {
            'episode_id': episode_id,
            'total_frames': len(keyframe_flags),
            'keyframe_count': sum(keyframe_flags),
            'keyframe_binary': binary_string,
            'keyframe_indices': [i for i, flag in enumerate(keyframe_flags) if flag],
            'parameters': {
                'velocity_threshold': self.velocity_threshold,
                'fps': self.fps
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
                cv2.putText(frame, 'KEY FRAME', (50, font_size + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_size/30, (0, 255, 0), 3)
                
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
    parser = argparse.ArgumentParser(description='Extract key frames based on hand keypoint velocities')
    parser.add_argument('--data_folder', required=True, help='Path to folder containing episode data')
    parser.add_argument('--output_folder', required=True, help='Path to save output files')
    parser.add_argument('--velocity_threshold', type=float, default=5.0, 
                       help='Maximum velocity (pixels/frame) for keypoint to be considered stationary')
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
