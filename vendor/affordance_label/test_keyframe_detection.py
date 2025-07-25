#!/usr/bin/env python3
"""
Simple test script to demonstrate key frame detection with velocity transitions.
Key frames are detected when hands transition from moving to stationary.
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path so we can import our module
sys.path.append(str(Path(__file__).parent))

from key_frame_extract import KeyFrameDetector, find_episode_files

def test_single_episode(mp4_path: str, hdf5_path: str):
    """
    Test key frame detection on a single episode.
    """
    print(f"Testing single episode:")
    print(f"  MP4: {mp4_path}")
    print(f"  HDF5: {hdf5_path}")
    
    # Check if files exist
    if not os.path.exists(mp4_path):
        print(f"Error: MP4 file not found: {mp4_path}")
        return False
        
    if not os.path.exists(hdf5_path):
        print(f"Error: HDF5 file not found: {hdf5_path}")
        return False
    
    # Initialize detector with custom parameters
    detector = KeyFrameDetector(
        velocity_threshold=5.0,  # Threshold for distinguishing moving vs stationary
        fps=30
    )
    
    try:
        # Process the episode
        print("Processing episode...")
        keyframe_flags, total_frames = detector.process_episode(mp4_path, hdf5_path)
        
        # Print results
        keyframe_count = sum(keyframe_flags)
        binary_string = ''.join('1' if flag else '0' for flag in keyframe_flags)
        
        print(f"\nResults:")
        print(f"  Total frames: {total_frames}")
        print(f"  Transition key frames detected: {keyframe_count}")
        print(f"  Key frame percentage: {keyframe_count/total_frames*100:.1f}%")
        print(f"  Binary string (first 50 chars): {binary_string[:50]}{'...' if len(binary_string) > 50 else ''}")
        
        # Show transition details
        transition_frames = [i for i, flag in enumerate(keyframe_flags) if flag]
        print(f"  Transition frame indices: {transition_frames[:10]}{'...' if len(transition_frames) > 10 else ''}")
        
        if len(transition_frames) > 1:
            # Calculate intervals between transitions
            intervals = [transition_frames[i+1] - transition_frames[i] for i in range(len(transition_frames)-1)]
            avg_interval = sum(intervals) / len(intervals) if intervals else 0
            print(f"  Average interval between transitions: {avg_interval:.1f} frames ({avg_interval/30:.1f} seconds)")
        
        return True
        
    except Exception as e:
        print(f"Error processing episode: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_keyframe_detection():
    """
    Test the key frame detection on a dataset.
    """
    
    # Example usage:
    # python test_keyframe_detection.py
    
    # Set your data folder path here
    data_folder = "/path/to/your/dataset"  # Update this path
    
    if len(sys.argv) > 1:
        data_folder = sys.argv[1]
    
    if not os.path.exists(data_folder):
        print(f"Data folder not found: {data_folder}")
        print("Usage: python test_keyframe_detection.py [data_folder_path]")
        print("Please provide a valid path to your dataset folder.")
        return
    
    print(f"Testing key frame detection on dataset: {data_folder}")
    
    # Initialize detector with custom parameters
    detector = KeyFrameDetector(
        velocity_threshold=3.0,  # Threshold for detecting moving vs stationary transitions
        fps=30
    )
    
    # Find episodes
    episodes = find_episode_files(data_folder)
    print(f"Found {len(episodes)} episodes")
    
    if len(episodes) == 0:
        print("No valid episodes found. Make sure your data folder contains:")
        print("- MP4 files (e.g., 0.mp4, 1.mp4)")
        print("- Corresponding HDF5 files (e.g., 0.hdf5, 1.hdf5)")
        print("- Or JSON files (e.g., 0_keypoint_projections_keypoints.json)")
        return
    
    # Test first few episodes
    test_count = min(3, len(episodes))
    successful_tests = 0
    
    for i in range(test_count):
        episode = episodes[i]
        print(f"\n{'='*60}")
        print(f"Testing episode {i+1}/{test_count}: {episode['episode_id']}")
        print(f"{'='*60}")
        
        success = test_single_episode(episode['mp4_path'], episode['keypoints_path'])
        if success:
            successful_tests += 1
    
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Episodes tested: {test_count}")
    print(f"Successful: {successful_tests}")
    print(f"Failed: {test_count - successful_tests}")
    
    if successful_tests > 0:
        print(f"\n✅ Key frame transition detection is working correctly!")
        print(f"You can now run the full pipeline with:")
        print(f"python key_frame_extract.py --data_folder {data_folder} --output_folder ./results --create_videos")
        print(f"\nNote: Key frames represent transitions from moving to stationary states.")
    else:
        print(f"\n❌ All tests failed. Please check:")
        print(f"1. File format and structure")
        print(f"2. HDF5 contains 'transforms' with keypoint trajectories")
        print(f"3. Dependencies are installed correctly")
        print(f"4. Velocity threshold might need adjustment (try --velocity_threshold 1.0 or 10.0)")


if __name__ == "__main__":
    test_keyframe_detection()
