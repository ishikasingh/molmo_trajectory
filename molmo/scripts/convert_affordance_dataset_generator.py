import os
import json
import h5py
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Union, Optional
from tqdm import tqdm

def read_complete_file_sets(folder_path: str, recursive: bool = False) -> Dict:
    """
    Find complete sets of files with pattern: {number}.mp4, {number}_keypoint_projections_keypoints.json, {number}.hdf5
    Only returns sets that have all three files present.
    
    Args:
        folder_path: Path to the folder to scan
        recursive: Whether to search subdirectories recursively
    
    Returns:
        Dictionary with complete file sets
    """
    folder_path_obj = Path(folder_path)
    
    if not folder_path_obj.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    # Search for files
    pattern = "**/*" if recursive else "*"
    
    # Find all keypoint JSON files first (they're the anchor)
    keypoint_files = {}
    mp4_files = {}
    hdf5_files = {}
    
    for file_path in folder_path_obj.glob(pattern):
        if file_path.is_file():
            file_name = file_path.name
            
            # Check for keypoint JSON files
            if file_name.endswith('_keypoint_projections_keypoints.json'):
                # Extract the number (e.g., "0" from "0_keypoint_projections_keypoints.json")
                base_number = file_name.replace('_keypoint_projections_keypoints.json', '')
                keypoint_files[base_number] = str(file_path)
            
            # Check for MP4 files
            elif file_name.endswith('.mp4'):
                base_number = file_path.stem  # Gets "0" from "0.mp4"
                mp4_files[base_number] = str(file_path)
            
            # Check for HDF5 files
            elif file_name.endswith('.hdf5'):
                base_number = file_path.stem  # Gets "0" from "0.hdf5"
                hdf5_files[base_number] = str(file_path)
    
    # Find complete sets (all three files must be present)
    complete_sets = []
    
    for base_number in keypoint_files.keys():
        if base_number in mp4_files and base_number in hdf5_files:
            complete_sets.append({
                'base_number': base_number,
                'mp4_file': mp4_files[base_number],
                'keypoint_json': keypoint_files[base_number],
                'hdf5_file': hdf5_files[base_number]
            })
    
    # Sort by base number (convert to int for proper sorting)
    complete_sets.sort(key=lambda x: int(x['base_number']) if x['base_number'].isdigit() else float('inf'))
    
    return {
        'complete_sets': complete_sets,
        'total_complete': len(complete_sets),
        'total_keypoint_files': len(keypoint_files),
        'total_mp4_files': len(mp4_files),
        'total_hdf5_files': len(hdf5_files),
        'incomplete_sets': {
            'keypoint_only': [num for num in keypoint_files.keys() if num not in mp4_files or num not in hdf5_files],
            'missing_keypoint': [num for num in mp4_files.keys() if num not in keypoint_files],
            'missing_hdf5': [num for num in keypoint_files.keys() if num not in hdf5_files]
        }
    }

def read_file_set_generator(mp4_path: str, keypoint_json_path: str, hdf5_path: str, frame_interval: int = 1):
    """
    Generator that yields individual frames with their keypoints and metadata.
    Memory-efficient version that processes one frame at a time.
    
    Args:
        mp4_path: Path to the MP4 video file
        keypoint_json_path: Path to the keypoint JSON file
        hdf5_path: Path to the HDF5 file
        frame_interval: Process every n-th frame
        
    Yields:
        Dictionary containing frame, keypoints, language instruction, and metadata
    """
    
    # Read keypoint JSON file once
    with open(keypoint_json_path, 'r') as f:
        keypoints_data = json.load(f)
    
    # Read HDF5 file once for language instruction
    with h5py.File(hdf5_path, 'r') as f:
        language_instruction = f.attrs['llm_description']
    
    # Open video file
    cap = cv2.VideoCapture(mp4_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {mp4_path}")
    
    # Get video properties
    fps = 30
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    # Process frames one by one
    frame_idx = 0
    keypoints_list = [keypoints_data['frame_keypoints'][i]['keypoints_2d'] for i in range(len(keypoints_data['frame_keypoints']))]
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Only process frames at the specified interval
        if frame_idx % frame_interval == 0:
            # Convert BGR to RGB and then to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(frame_rgb)
            
            # Get corresponding keypoints
            if frame_idx < len(keypoints_list):
                keypoints = keypoints_list[frame_idx]
                
                yield {
                    'frame': pil_frame,
                    'keypoints': keypoints,
                    'language_instruction': language_instruction,
                    'fps': fps,
                    'duration': duration,
                    'total_frames': frame_count,
                    'frame_idx': frame_idx,
                    'metadata': {
                        'mp4_path': mp4_path,
                        'keypoint_json_path': keypoint_json_path,
                        'hdf5_path': hdf5_path
                    }
                }
        
        frame_idx += 1
    
    cap.release()

def dataset_generator(folder_path: str, recursive: bool = True, frame_interval: int = 1):
    """
    Generator that yields dataset items one by one for memory-efficient processing.
    
    Args:
        folder_path: Path to the folder containing the data files
        recursive: Whether to search subdirectories recursively
        frame_interval: Process every n-th frame
        
    Yields:
        Dictionary containing processed dataset items
    """
    # Find all complete file sets
    file_sets = read_complete_file_sets(folder_path, recursive=recursive)
    
    print(f"Found {file_sets['total_complete']} complete file sets")
    print(f"Frame interval: {frame_interval} (processing every {frame_interval}{'st' if frame_interval == 1 else 'nd' if frame_interval == 2 else 'rd' if frame_interval == 3 else 'th'} frame)")
    
    # Process each file set
    for file_set in tqdm(file_sets['complete_sets'], desc="Processing file sets", unit="set"):
        # Use generator to process frames one by one
        for frame_data in read_file_set_generator(
            file_set['mp4_file'], 
            file_set['keypoint_json'], 
            file_set['hdf5_file'],
            frame_interval
        ):
            frame = frame_data['frame']
            keypoints = frame_data['keypoints']
            
            # Process keypoints to match the expected format
            hand_positions = process_keypoints_for_dataset(
                keypoints, 
                image_width=frame.width, 
                image_height=frame.height
            )
            
            # Create dataset item
            item = {
                "image": frame,  # PIL Image
                "instruction": frame_data['language_instruction'],
                "hand_positions": hand_positions,
                "metadata": {
                    "video_id": file_set['base_number'],
                    "frame_idx": frame_data['frame_idx'],
                    "original_frame_idx": frame_data['frame_idx'],
                    "image_size": [frame.width, frame.height],
                    "fps": frame_data['fps'],
                    "duration": frame_data['duration'],
                    "total_frames": frame_data['total_frames'],
                    "frame_interval": frame_interval,
                    "mp4_file": file_set['mp4_file'],
                    "keypoint_json": file_set['keypoint_json'],
                    "hdf5_file": file_set['hdf5_file']
                }
            }
            
            yield item

def create_huggingface_dataset_memory_efficient(folder_path: str, output_path: str, recursive: bool = True, frame_interval: int = 1):
    """
    Create a HuggingFace dataset from the complete file sets using generators for memory efficiency.
    
    Args:
        folder_path: Path to the folder containing the data files
        output_path: Path where to save the HuggingFace dataset
        recursive: Whether to search subdirectories recursively
        frame_interval: Process every n-th frame (1 = every frame, 2 = every 2nd frame, etc.)
    """
    import datasets
    
    # Define the features schema for the dataset
    features = datasets.Features({
        "image": datasets.Image(),
        "instruction": datasets.Value("string"),
        "hand_positions": {
            "points": datasets.Sequence(datasets.Sequence(datasets.Value("float32"))),
            "labels": datasets.Sequence(datasets.Value("string"))
        },
        "metadata": {
            "video_id": datasets.Value("string"),
            "frame_idx": datasets.Value("int32"),
            "original_frame_idx": datasets.Value("int32"),
            "image_size": datasets.Sequence(datasets.Value("int32")),
            "fps": datasets.Value("float32"),
            "duration": datasets.Value("float32"),
            "total_frames": datasets.Value("int32"),
            "frame_interval": datasets.Value("int32"),
            "mp4_file": datasets.Value("string"),
            "keypoint_json": datasets.Value("string"),
            "hdf5_file": datasets.Value("string")
        }
    })
    
    # Create dataset from generator
    dataset = datasets.Dataset.from_generator(
        lambda: dataset_generator(folder_path, recursive, frame_interval),
        features=features
    )
    
    # Save to disk
    dataset.save_to_disk(output_path)
    print(f"Dataset saved to {output_path}")
    
    return dataset

def create_dataset_splits_memory_efficient(dataset, output_path: str, splits: dict = None):
    """
    Split the dataset into train/validation/test splits in a memory-efficient way.
    
    Args:
        dataset: The HuggingFace dataset to split
        output_path: Path to save the split dataset
        splits: Dictionary with split ratios, e.g., {"train": 0.8, "validation": 0.1, "test": 0.1}
    """
    import datasets
    
    if splits is None:
        splits = {"train": 0.8, "validation": 0.1, "test": 0.1}
    
    # First split: train vs (validation + test)
    train_val_split = dataset.train_test_split(test_size=1-splits["train"], seed=42)
    train_dataset = train_val_split["train"]
    temp_dataset = train_val_split["test"]
    
    # Second split: validation vs test
    if "validation" in splits and "test" in splits:
        val_test_ratio = splits["validation"] / (splits["validation"] + splits["test"])
        val_test_split = temp_dataset.train_test_split(test_size=1-val_test_ratio, seed=42)
        val_dataset = val_test_split["train"]
        test_dataset = val_test_split["test"]
        
        # Create final dataset dict
        final_dataset = datasets.DatasetDict({
            "train": train_dataset,
            "validation": val_dataset,
            "test": test_dataset
        })
    else:
        # Only train and test
        final_dataset = datasets.DatasetDict({
            "train": train_dataset,
            "test": temp_dataset
        })
    
    # Save split dataset
    final_dataset.save_to_disk(output_path)
    print(f"Split dataset saved to {output_path}")
    print(f"Train: {len(final_dataset['train'])} samples")
    if "validation" in final_dataset:
        print(f"Validation: {len(final_dataset['validation'])} samples")
    print(f"Test: {len(final_dataset['test'])} samples")
    
    return final_dataset

def create_dataset_example_memory_efficient():
    """
    Memory-efficient example of creating a HuggingFace dataset from your data.
    """
    
    # Step 1: Set your paths
    input_folder = "/root/sky_workdir/dataset/egodex/part2"
    # input_folder = "/root/sky_workdir/dataset/small_test"   
    output_dataset_path = "/root/sky_workdir/dataset/affordance_dataset"
    output_split_path = "/root/sky_workdir/dataset/affordance_dataset_splits"
    
    # Step 2: Create the dataset using memory-efficient approach
    print("Creating HuggingFace dataset (memory-efficient)...")
    dataset = create_huggingface_dataset_memory_efficient(
        folder_path=input_folder,
        output_path=output_dataset_path,
        recursive=True,
        frame_interval=3  # Process every 3rd frame - adjust this value as needed
    )
    
    # Step 3: Create train/validation/test splits
    print("\nCreating dataset splits...")
    split_dataset = create_dataset_splits_memory_efficient(
        dataset=dataset,
        output_path=output_split_path,
        splits={"train": 0.8, "validation": 0.1, "test": 0.1}
    )
    
    # Step 4: Verify the dataset
    print("\nDataset verification:")
    print(f"Dataset features: {dataset.features}")
    print(f"Sample item keys: {list(dataset[0].keys())}")
    print(f"Sample instruction: {dataset[0]['instruction']}")
    print(f"Sample hand_positions keys: {list(dataset[0]['hand_positions'].keys())}")
    print(f"Number of keypoints: {len(dataset[0]['hand_positions']['points'])}")
    
    return split_dataset

def process_keypoints_for_dataset(keypoints_data, image_width, image_height):
    """
    Process keypoints data to match the expected format for the dataset.
    Normalizes coordinates to 0-100 percentage scale.
    
    Args:
        keypoints_data: Raw keypoints data from JSON file
        image_width: Width of the image in pixels
        image_height: Height of the image in pixels
        
    Returns:
        Dictionary with "points" and "labels" for the dataset
    """
    points = []
    labels = []
    
    def normalize_coordinates(x, y, img_w, img_h):
        """Convert pixel coordinates to percentage (0-100)"""
        x_norm = (x / img_w) * 100.0
        y_norm = (y / img_h) * 100.0
        return [x_norm, y_norm]
    
    # Process keypoints data based on its structure
    # If keypoints_data has a structure like {"left_hand": {...}, "right_hand": {...}}
    for keypoint_label, keypoint_coords in keypoints_data.items():  
        if keypoint_label == "camera":
            continue
        normalized_coords = normalize_coordinates(
            keypoint_coords[0], keypoint_coords[1], image_width, image_height
        )
        points.append(normalized_coords)
        labels.append(keypoint_label)
    
    return {
        "points": points,
        "labels": labels
    }


if __name__ == "__main__":
    # Uncomment the function you want to run:
    
    # read_complete_file_sets_example()  # Test reading files
    # create_dataset_example()  # Create the HuggingFace dataset
    create_dataset_example_memory_efficient()  # Create the HuggingFace dataset (memory-efficient)