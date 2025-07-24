#!/usr/bin/env python3
"""
Example script showing how to customize keypoint detection patterns
and examine the structure of your specific dataset.
"""

import json
import h5py
import cv2
from pathlib import Path

def examine_keypoint_structure(file_path: str):
    """
    Examine the structure of a keypoint file to understand the data format.
    
    Args:
        file_path: Path to JSON or HDF5 file
    """
    file_path = Path(file_path)
    
    if file_path.suffix == '.json':
        examine_json_keypoints(file_path)
    elif file_path.suffix == '.hdf5':
        examine_hdf5_keypoints(file_path)
    else:
        print(f"Unsupported file format: {file_path.suffix}")

def examine_json_keypoints(json_path: Path):
    """Examine JSON keypoint file structure."""
    print(f"Examining JSON file: {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print("JSON structure:")
    print(f"  Top-level keys: {list(data.keys())}")
    
    if 'frame_keypoints' in data:
        print(f"  Number of frames: {len(data['frame_keypoints'])}")
        
        # Examine first frame
        first_frame = data['frame_keypoints'][0]
        print(f"  First frame keys: {list(first_frame.keys())}")
        
        if 'keypoints_2d' in first_frame:
            keypoints_2d = first_frame['keypoints_2d']
            print(f"  Keypoint names: {list(keypoints_2d.keys())}")
            
            # Show example keypoint data
            for i, (name, coords) in enumerate(keypoints_2d.items()):
                if i < 5:  # Show first 5 keypoints
                    print(f"    {name}: {coords}")
                elif i == 5:
                    print(f"    ... and {len(keypoints_2d) - 5} more keypoints")
                    break

def examine_hdf5_keypoints(hdf5_path: Path):
    """Examine HDF5 keypoint file structure."""
    print(f"Examining HDF5 file: {hdf5_path}")
    
    with h5py.File(hdf5_path, 'r') as f:
        print("HDF5 structure:")
        
        def print_structure(name, obj):
            if isinstance(obj, h5py.Group):
                print(f"  Group: {name}")
            elif isinstance(obj, h5py.Dataset):
                print(f"  Dataset: {name}, shape: {obj.shape}, dtype: {obj.dtype}")
        
        f.visititems(print_structure)
        
        # Check for attributes
        print(f"  Root attributes: {list(f.attrs.keys())}")
        for attr_name in f.attrs.keys():
            print(f"    {attr_name}: {f.attrs[attr_name]}")

def find_hand_keypoints_in_data(keypoints_dict: dict):
    """
    Analyze a keypoints dictionary to find hand-related keypoints.
    
    Args:
        keypoints_dict: Dictionary of keypoint names and coordinates
    """
    hand_keypoints = []
    
    for keypoint_name in keypoints_dict.keys():
        name_lower = keypoint_name.lower()
        
        # Check for hand-related keywords
        hand_keywords = ['hand', 'wrist', 'thumb', 'index', 'middle', 'ring', 'pinky', 
                        'finger', 'palm', 'left', 'right']
        
        if any(keyword in name_lower for keyword in hand_keywords):
            hand_keypoints.append(keypoint_name)
    
    return hand_keypoints

def analyze_dataset_keypoints(data_folder: str):
    """
    Analyze a dataset to understand keypoint naming patterns.
    
    Args:
        data_folder: Path to dataset folder
    """
    data_path = Path(data_folder)
    
    # Find keypoint files
    json_files = list(data_path.glob("**/*keypoints*.json"))
    hdf5_files = list(data_path.glob("**/*.hdf5"))
    
    print(f"Found {len(json_files)} JSON keypoint files")
    print(f"Found {len(hdf5_files)} HDF5 files")
    
    # Analyze first JSON file if available
    if json_files:
        print("\n" + "="*50)
        examine_keypoint_structure(json_files[0])
        
        # Analyze hand keypoints
        with open(json_files[0], 'r') as f:
            data = json.load(f)
        
        if 'frame_keypoints' in data and len(data['frame_keypoints']) > 0:
            first_frame_keypoints = data['frame_keypoints'][0]['keypoints_2d']
            hand_keypoints = find_hand_keypoints_in_data(first_frame_keypoints)
            
            print(f"\nDetected hand keypoints ({len(hand_keypoints)}):")
            for hand_kp in hand_keypoints:
                print(f"  - {hand_kp}")
    
    # Analyze first HDF5 file if available
    if hdf5_files:
        print("\n" + "="*50)
        examine_keypoint_structure(hdf5_files[0])

def customize_hand_patterns():
    """
    Example of how to customize hand keypoint detection patterns.
    """
    from key_frame_extract import KeyFrameDetector
    
    class CustomKeyFrameDetector(KeyFrameDetector):
        """Custom detector with dataset-specific hand keypoint patterns."""
        
        def is_hand_keypoint(self, keypoint_name: str):
            """
            Custom implementation for your specific keypoint naming.
            Modify this based on your dataset's keypoint names.
            """
            keypoint_lower = keypoint_name.lower()
            
            # Add your specific patterns here
            left_patterns = [
                'left_hand', 'left_wrist', 'left_thumb_tip', 'left_index_tip',
                'left_middle_tip', 'left_ring_tip', 'left_pinky_tip',
                # Add more patterns based on your dataset
            ]
            
            right_patterns = [
                'right_hand', 'right_wrist', 'right_thumb_tip', 'right_index_tip',
                'right_middle_tip', 'right_ring_tip', 'right_pinky_tip',
                # Add more patterns based on your dataset
            ]
            
            for pattern in left_patterns:
                if pattern in keypoint_lower:
                    return True, 'left'
                    
            for pattern in right_patterns:
                if pattern in keypoint_lower:
                    return True, 'right'
            
            return False, ''
    
    # Use the custom detector
    detector = CustomKeyFrameDetector(velocity_threshold=3.0, fps=30)
    return detector

def main():
    """
    Main function to analyze your dataset structure.
    """
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python examine_dataset.py <path_to_dataset_folder>")
        print("\nThis script will help you understand your dataset structure")
        print("and customize the keypoint detection patterns.")
        return
    
    data_folder = sys.argv[1]
    
    if not Path(data_folder).exists():
        print(f"Dataset folder not found: {data_folder}")
        return
    
    print(f"Analyzing dataset in: {data_folder}")
    analyze_dataset_keypoints(data_folder)
    
    print("\n" + "="*50)
    print("CUSTOMIZATION GUIDE:")
    print("1. Look at the keypoint names printed above")
    print("2. Identify which ones correspond to hands")
    print("3. Modify the hand detection patterns in key_frame_extract.py")
    print("4. Or use the CustomKeyFrameDetector example in this file")

if __name__ == "__main__":
    main()
