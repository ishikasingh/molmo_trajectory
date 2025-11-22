import os
import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pickle
from typing import List, Dict

from olmo.data.trajectory_datasets import TrajectoryDataset

def compute_stats(
    data_dir: str,
    split: str = "train",
    action_chunking_horizon: int = 30,
    trajectory_representation: str = "absolute",
    sample_size: int = 5000,
    output_file: str = "trajectory_stats.pt",
    batch_size: int = 32,
):
    print(f"Initializing dataset for stats calculation...")
    # Initialize dataset (we won't use __getitem__ directly to avoid image loading)
    # We set output_2d_trajectory=False as requested
    dataset = TrajectoryDataset(
        data_dir=data_dir,
        split=split,
        action_chunking_horizon=action_chunking_horizon,
        trajectory_representation=trajectory_representation,
        output_2d_trajectory=False,
        normalize_coordinates=False, # Disable normalization to compute raw stats
        frame_downsampling_ratio=1, # Use all frames for stats,
        load_images=False, # Use new flag to avoid loading images
    )
    
    total_samples = len(dataset.index_mapping)
    print(f"Total samples in dataset: {total_samples}")
    
    if sample_size > total_samples:
        print(f"Requested sample size {sample_size} > total samples. Using all samples.")
        indices = list(range(total_samples))
    else:
        print(f"Sampling {sample_size} random samples...")
        # Use a fixed seed for reproducibility
        rng = np.random.RandomState(42)
        indices = rng.choice(total_samples, size=sample_size, replace=False)
    
    # Use subset indices
    subset = torch.utils.data.Subset(dataset, indices)
    
    # Create DataLoader with multiple workers to hide latency
    # Use collate_fn=lambda x: x to get a list of dicts, or write a custom one
    loader = torch.utils.data.DataLoader(
        subset,
        batch_size=batch_size, # Process in batches
        num_workers=16,  # Increase workers for parallel I/O
        collate_fn=lambda x: x, # Simple collate (returns list of items)
        shuffle=False
    )
    
    # Lists to store processed trajectories
    all_trajectories = []
    
    print(f"Processing samples with {loader.num_workers} workers...")
    valid_samples = 0
    
    for batch in tqdm(loader, desc="Computing stats"):
        for item in batch:
            try:
                # Extract trajectory from item['message_list'][0]['points']
                # The dataset.get() method returns 'points' which is the final_trajectory
                
                # The item structure from TrajectoryDataset.get():
                # {
                #     'image': image,
                #     'message_list': [
                #         {
                #             'points': final_trajectory, 
                #             ...
                #         }
                #     ],
                #     ...
                # }
                
                final_trajectory = item['message_list'][0]['points']
                
                # Ensure it's numpy array
                if isinstance(final_trajectory, torch.Tensor):
                    final_trajectory = final_trajectory.numpy()
                
                # Reshape: [num_steps, num_joints, 3] -> [num_steps, num_joints * 3]
                num_steps = final_trajectory.shape[0]
                trajectory_flat = final_trajectory.reshape(num_steps, -1).astype(np.float32)
                
                all_trajectories.append(trajectory_flat)
                valid_samples += 1
                
            except Exception as e:
                print(f"Error processing sample: {e}")
                continue
            
    if valid_samples == 0:
        raise ValueError("No valid samples found to compute stats!")
        
    print(f"Computed stats from {valid_samples} samples.")
    
    # Stack all trajectories: [total_steps, feature_dim]
    # Each trajectory is [horizon, feature_dim]
    # We want mean/std per feature dimension
    
    # Concatenate along the time dimension (0)
    all_data = np.concatenate(all_trajectories, axis=0)
    
    print(f"Data shape for stats: {all_data.shape}")
    
    mean = np.mean(all_data, axis=0)
    std = np.std(all_data, axis=0)
    
    # Avoid division by zero
    std[std < 1e-6] = 1.0
    
    stats = {
        "mean": torch.from_numpy(mean),
        "std": torch.from_numpy(std),
        "n_samples": valid_samples,
        "trajectory_representation": trajectory_representation,
        "action_chunking_horizon": action_chunking_horizon,
        "joint_names": dataset.joint_names
    }
    
    print(f"Saving stats to {output_file}")
    torch.save(stats, output_file)
    
    print("Mean shape:", mean.shape)
    print("Std shape:", std.shape)
    print("Mean (first 5):", mean[:5])
    print("Std (first 5):", std[:5])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--output_file", type=str, default="trajectory_stats.pt")
    parser.add_argument("--sample_size", type=int, default=5000)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--action_chunking_horizon", type=int, default=30)
    parser.add_argument("--trajectory_representation", type=str, default="absolute", choices=["absolute", "delta"])
    
    args = parser.parse_args()
    
    compute_stats(
        data_dir=args.data_dir,
        split=args.split,
        action_chunking_horizon=args.action_chunking_horizon,
        trajectory_representation=args.trajectory_representation,
        sample_size=args.sample_size,
        output_file=args.output_file
    )
