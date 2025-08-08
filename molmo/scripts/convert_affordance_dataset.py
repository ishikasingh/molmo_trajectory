import os
import json
import h5py
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Union, Optional
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

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

def read_file_set(mp4_path: str, keypoint_json_path: str, hdf5_path: str) -> Dict:
    """
    Read and process a complete file set (MP4, keypoint JSON, and HDF5).
    
    Args:
        mp4_path: Path to the MP4 video file
        keypoint_json_path: Path to the keypoint JSON file
        hdf5_path: Path to the HDF5 file
        
    Returns:
        Dictionary containing frames, keypoints, language instruction, and metadata
    """
    
    # Read MP4 file and extract frames
    # print(f"Reading MP4: {mp4_path}")
    cap = cv2.VideoCapture(mp4_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {mp4_path}")
    
    # Get video properties
    fps = 30
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    # Extract all frames
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB and then to PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(frame_rgb)
        frames.append(pil_frame)
    
    cap.release()
    # print(f"Extracted {len(frames)} frames from video")
    
    # Read keypoint JSON file
    # print(f"Reading keypoints: {keypoint_json_path}")
    with open(keypoint_json_path, 'r') as f:
        keypoints_data = json.load(f)
    
    # Process keypoints - this depends on your JSON structure
    # Assuming the JSON contains keypoints for each frame
    keypoints = [keypoints_data['frame_keypoints'][i]['keypoints_2d'] for i in range(len(keypoints_data['frame_keypoints']))]
    
    # Ensure keypoints list matches frame count
    assert len(keypoints) == len(frames)
    
    # Read HDF5 file
    # print(f"Reading HDF5: {hdf5_path}")
    language_instruction = ""
    metadata = {}
    
    with h5py.File(hdf5_path, 'r') as f:
        # Try to read language instruction - adjust keys based on your HDF5 structure
        language_instruction = f.attrs['llm_description']
                    
    return {
        'frames': frames,
        'keypoints': keypoints,
        'language_instruction': language_instruction,
        'fps': fps,
        'duration': duration,
        'total_frames': len(frames),
        'metadata': metadata
    }

def process_file_set_worker(file_set: Dict, images_output_path: str, frame_interval: int) -> List[Dict]:
    file_data = read_file_set(
        file_set['mp4_file'],
        file_set['keypoint_json'],
        file_set['hdf5_file']
    )

    video_images_dir = Path(images_output_path) / f"video_{file_set['base_number']}"
    video_images_dir.mkdir(exist_ok=True)

    dataset_items = []
    total_frames = len(file_data['frames'])

    for frame_idx in range(0, total_frames, frame_interval):
        frame = file_data['frames'][frame_idx]
        keypoints = file_data['keypoints'][frame_idx]

        image_filename = f"frame_{frame_idx:06d}.jpg"
        image_path = video_images_dir / image_filename
        frame.save(image_path, "JPEG", quality=95)

        relative_image_path = f"video_{file_set['base_number']}/{image_filename}"

        hand_positions = process_keypoints_for_dataset(
            keypoints,
            image_width=frame.width,
            image_height=frame.height
        )

        item = {
            "image_path": relative_image_path,
            "instruction": file_data['language_instruction'],
            "hand_positions": hand_positions,
            "metadata": {
                "video_id": file_set['base_number'],
                "frame_idx": frame_idx,
                "original_frame_idx": frame_idx,
                "image_size": [frame.width, frame.height],
                "fps": file_data['fps'],
                "duration": file_data['duration'],
                "total_frames": total_frames,
                "frame_interval": frame_interval,
                "images_base_path": str(images_output_path),
            }
        }
        dataset_items.append(item)

    return dataset_items

def create_huggingface_dataset(folder_path: str, output_path: str, images_output_path: str = None,
                               recursive: bool = True, frame_interval: int = 1, num_workers: Optional[int] = None):
    import datasets

    if images_output_path is None:
        images_output_path = Path(output_path).parent / "images"
    images_output_path = Path(images_output_path)
    images_output_path.mkdir(parents=True, exist_ok=True)

    file_sets = read_complete_file_sets(folder_path, recursive=recursive)

    print(f"Found {file_sets['total_complete']} complete file sets")
    print(f"Frame interval: {frame_interval} (processing every {frame_interval}{'st' if frame_interval == 1 else 'nd' if frame_interval == 2 else 'rd' if frame_interval == 3 else 'th'} frame)")
    print(f"Images will be saved to: {images_output_path}")

    dataset_items = []

    complete_sets = file_sets['complete_sets']
    if num_workers is None:
        try:
            import os as _os
            num_workers = max(1, min(len(complete_sets), _os.cpu_count() or 1))
        except Exception:
            num_workers = 1

    if num_workers <= 1:
        # Sequential fallback
        from tqdm import tqdm as _tqdm
        for file_set in _tqdm(complete_sets, desc="Processing file sets", unit="set"):
            items = process_file_set_worker(file_set, str(images_output_path), frame_interval)
            dataset_items.extend(items)
    else:
        # Parallel across file sets
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(process_file_set_worker, file_set, str(images_output_path), frame_interval)
                for file_set in complete_sets
            ]
            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Processing file sets (workers={num_workers})", unit="set"):
                try:
                    items = fut.result()
                    dataset_items.extend(items)
                except Exception as e:
                    print(f"Error processing a file set: {e}")

    print(f"Created {len(dataset_items)} dataset items")

    dataset = datasets.Dataset.from_list(dataset_items)
    dataset.save_to_disk(output_path)
    print(f"Dataset saved to {output_path}")
    print(f"Images saved to {images_output_path}")

    return dataset

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
def create_dataset_splits(dataset, output_path: str, splits: dict = None):
    """
    Split the dataset into train/validation/test splits.
    
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

def load_image_from_dataset_item(item, images_base_path: str = None):
    """
    Helper function to load an image from a dataset item.
    
    Args:
        item: Dataset item containing image_path
        images_base_path: Base path where images are stored. If None, uses the path from metadata
        
    Returns:
        PIL Image object
    """
    if images_base_path is None:
        images_base_path = item["metadata"]["images_base_path"]
    
    full_image_path = Path(images_base_path) / item["image_path"]
    return Image.open(full_image_path)

def create_dataset_example():
    """
    Complete example of creating a HuggingFace dataset from your data.
    """
    
    # Step 1: Set your paths
    input_folder = "/root/sky_workdir/dataset/egodex/part2"
    # input_folder = "/root/sky_workdir/dataset/small_test"   
    output_dataset_path = "/root/sky_workdir/dataset/affordance_dataset"
    output_split_path = "/root/sky_workdir/dataset/affordance_dataset_splits"
    images_output_path = "/root/sky_workdir/dataset/affordance_images"  # New: separate images directory
    
    # Step 2: Create the dataset
    print("Creating HuggingFace dataset...")
    dataset = create_huggingface_dataset(
        folder_path=input_folder,
        output_path=output_dataset_path,
        images_output_path=images_output_path,
        recursive=True,
        frame_interval=5,
        num_workers=None  # or an int like 8
    )
    
    # Step 3: Create train/validation/test splits
    print("\nCreating dataset splits...")
    split_dataset = create_dataset_splits(
        dataset=dataset,
        output_path=output_split_path,
        splits={"train": 0.95, "validation": 0.02, "test": 0.03}
    )
    
    # Step 4: Verify the dataset
    print("\nDataset verification:")
    print(f"Dataset features: {dataset.features}")
    print(f"Sample item keys: {list(dataset[0].keys())}")
    print(f"Sample instruction: {dataset[0]['instruction']}")
    print(f"Sample image_path: {dataset[0]['image_path']}")
    print(f"Sample hand_positions keys: {list(dataset[0]['hand_positions'].keys())}")
    print(f"Number of keypoints: {len(dataset[0]['hand_positions']['points'])}")
    
    # Example of how to load an image from the dataset
    print("\nExample of loading an image:")
    sample_image = load_image_from_dataset_item(dataset[0])
    print(f"Loaded image size: {sample_image.size}")
    
    return split_dataset

if __name__ == "__main__":
    # Uncomment the function you want to run:
    
    # read_complete_file_sets_example()  # Test reading files
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    create_dataset_example()  # Create the HuggingFace dataset