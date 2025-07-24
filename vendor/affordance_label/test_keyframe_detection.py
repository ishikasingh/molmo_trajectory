#!/usr/bin/env python3
"""
Simple test script to demonstrate key frame detection.
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path so we can import our module
sys.path.append(str(Path(__file__).parent))

from key_frame_extract import KeyFrameDetector, find_episode_files

def test_keyframe_detection():
    """
    Test the key frame detection on a small dataset.
    """
    
    # Example usage:
    # python test_keyframe_detection.py
    
    # Set your data folder path here
    data_folder = "/path/to/your/dataset"  # Update this path
    output_folder = "./keyframe_results"
    
    if not os.path.exists(data_folder):
        print(f"Data folder not found: {data_folder}")
        print("Please update the data_folder path in this script.")
        return
    
    # Initialize detector with custom parameters
    detector = KeyFrameDetector(
        velocity_threshold=3.0,  # Lower threshold = more sensitive to small movements
        fps=30
    )
    
    # Find episodes
    episodes = find_episode_files(data_folder)
    print(f"Found {len(episodes)} episodes")
    
    if len(episodes) == 0:
        print("No valid episodes found. Make sure your data folder contains:")
        print("- MP4 files (e.g., 0.mp4, 1.mp4)")
        print("- Corresponding JSON files (e.g., 0_keypoint_projections_keypoints.json)")
        print("- Or HDF5 files (e.g., 0.hdf5)")
        return
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Process first episode as test
    episode = episodes[0]
    print(f"Processing test episode: {episode['episode_id']}")
    
    try:
        # Detect key frames
        keyframe_flags, total_frames = detector.process_episode(
            episode['mp4_path'], 
            episode['keypoints_path']
        )
        
        # Save results
        json_output = os.path.join(output_folder, f"{episode['episode_id']}_keyframes.json")
        detector.save_keyframe_json(keyframe_flags, json_output, episode['episode_id'])
        
        # Create annotated video
        video_output = os.path.join(output_folder, f"{episode['episode_id']}_annotated.mp4")
        detector.create_annotated_video(
            episode['mp4_path'], 
            keyframe_flags, 
            video_output
        )
        
        # Print results
        keyframe_count = sum(keyframe_flags)
        binary_string = ''.join('1' if flag else '0' for flag in keyframe_flags)
        
        print(f"Results for episode {episode['episode_id']}:")
        print(f"  Total frames: {total_frames}")
        print(f"  Key frames: {keyframe_count}")
        print(f"  Binary string: {binary_string}")
        print(f"  Key frame indices: {[i for i, flag in enumerate(keyframe_flags) if flag]}")
        print(f"  JSON saved to: {json_output}")
        print(f"  Annotated video saved to: {video_output}")
        
    except Exception as e:
        print(f"Error processing episode: {str(e)}")


if __name__ == "__main__":
    test_keyframe_detection()
