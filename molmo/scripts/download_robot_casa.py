import os
import subprocess
import sys
from pathlib import Path
from pprint import pprint

import torch
from huggingface_hub import HfApi

import lerobot
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.constants import HF_LEROBOT_HOME

def ensure_git_lfs():
    """Ensure git lfs is installed and initialized."""
    try:
        # Check if git lfs is available
        result = subprocess.run(['git', 'lfs', 'version'], 
                              capture_output=True, text=True, check=True)
        print(f"Git LFS version: {result.stdout.strip()}")
        
        # Initialize git lfs
        subprocess.run(['git', 'lfs', 'install'], check=True)
        print("Git LFS initialized successfully")
        
    except subprocess.CalledProcessError:
        print("Error: Git LFS is not installed or not working properly")
        print("Please install Git LFS: https://git-lfs.github.io/")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: Git is not installed or not in PATH")
        sys.exit(1)

def dataset_exists_and_valid(repo_id):
    """Check if dataset already exists and can be loaded."""
    try:
        # Try to load the dataset
        dataset = LeRobotDataset(repo_id)
        print(f"Dataset already exists and is valid. Total episodes: {len(dataset)}")
        return True, dataset
    except Exception as e:
        print(f"Dataset not found or invalid: {e}")
        return False, None

def download_dataset_with_git(repo_id, dataset_path):
    """Download dataset using git clone with LFS support."""
    if dataset_path.exists():
        print(f"Dataset directory already exists at {dataset_path}, but may be incomplete")
        # Remove incomplete dataset to re-download
        import shutil
        shutil.rmtree(dataset_path)
    
    print(f"Downloading dataset {repo_id} to {dataset_path}...")
    
    # Create parent directory if it doesn't exist
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Clone the repository
        clone_url = f"https://huggingface.co/datasets/{repo_id}"
        subprocess.run([
            'git', 'clone', clone_url, str(dataset_path)
        ], check=True)
        
        print(f"Dataset downloaded successfully to {dataset_path}")
        return dataset_path
        
    except subprocess.CalledProcessError as e:
        print(f"Error downloading dataset: {e}")
        # Clean up partial download
        if dataset_path.exists():
            import shutil
            shutil.rmtree(dataset_path)
        sys.exit(1)

def main():
    repo_id = "ishika/robocasa_gr1_tabletop_tasks_fingertips_corrected"
    
    # First, check if dataset already exists and is valid
    exists, dataset = dataset_exists_and_valid(repo_id)
    
    if exists:
        print("Skipping download - using existing dataset")
    else:
        # Use the same default path structure as LeRobotDataset
        dataset_path = HF_LEROBOT_HOME / repo_id
        
        print(f"Using LeRobot cache directory: {dataset_path}")
        
        # Ensure git lfs is set up
        ensure_git_lfs()
        
        # Download dataset using git clone
        download_dataset_with_git(repo_id, dataset_path)
        
        # Load dataset with LeRobotDataset (will use the default path automatically)
        print("Loading dataset with LeRobotDataset...")
        dataset = LeRobotDataset(repo_id)  # No need to specify root, it will use the default path
        print(f"Dataset loaded successfully. Total episodes: {len(dataset)}")
    
    # Print first example
    print("\nFirst example:")
    pprint(dataset[0].keys())

if __name__ == "__main__":
    main()