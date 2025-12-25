#!/usr/bin/env python3
"""
Compute trajectory statistics for normalization.

Supports both EgoDex and RoboCasa datasets.

For RoboCasa dataset, computes stats for both trajectory and robot actions in a single file.
The output file structure:
    - Top-level keys: 'mean', 'std', 'joint_names', etc. (trajectory stats, compatible with trajectory_datasets.py)
    - Additional key: 'robot_action_stats' (dict with 'mean', 'std', etc.) - only for RoboCasa

Usage:
    # For EgoDex dataset
    python compute_trajectory_stats.py --dataset egodex --data_dir /path/to/egodex --output_file egodex_stats.pt

    # For RoboCasa dataset - computes both trajectory and robot action stats
    python compute_trajectory_stats.py --dataset robocasa --data_dir /path/to/robocasa --output_file robocasa_stats.pt
"""

import os
import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pickle
from typing import List, Dict, Optional

from olmo.data.trajectory_datasets import TrajectoryDataset
from olmo.data.robo_casa_affordance_datasets import RoboCasaTrajectoryDataset


def get_dataset_class(dataset_type: str):
    """Get the appropriate dataset class based on dataset type."""
    if dataset_type == "egodex":
        return TrajectoryDataset
    elif dataset_type == "robocasa":
        return RoboCasaTrajectoryDataset
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Must be 'egodex' or 'robocasa'")


def compute_stats(
    data_dir: str,
    dataset_type: str = "egodex",
    split: str = "train",
    action_chunking_horizon: int = 30,
    trajectory_representation: str = "absolute",
    sample_size: int = 5000,
    output_file: str = "trajectory_stats.pt",
    batch_size: int = 32,
    num_workers: int = 16,
    joint_names: Optional[List[str]] = None,
):
    """
    Compute trajectory statistics for normalization.
    
    For RoboCasa dataset, computes stats for both trajectory and robot actions.
    
    Args:
        data_dir: Path to dataset directory
        dataset_type: Type of dataset ('egodex' or 'robocasa')
        split: Dataset split ('train', 'test', 'overfit')
        action_chunking_horizon: Number of frames in action chunks
        trajectory_representation: 'absolute' or 'delta'
        sample_size: Number of samples to use for computing stats
        output_file: Output file path for stats
        batch_size: Batch size for processing
        num_workers: Number of workers for data loading
        joint_names: Optional list of joint names to use
    """
    print(f"=" * 60)
    print(f"Computing trajectory stats for {dataset_type.upper()} dataset")
    if dataset_type == "robocasa":
        print("(Computing both trajectory and robot action stats)")
    print(f"=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Split: {split}")
    print(f"Action horizon: {action_chunking_horizon}")
    print(f"Representation: {trajectory_representation}")
    print(f"Sample size: {sample_size}")
    print(f"Output file: {output_file}")
    print(f"=" * 60)
    
    # Get the appropriate dataset class
    DatasetClass = get_dataset_class(dataset_type)
    
    # Common dataset arguments
    dataset_kwargs = {
        "data_dir": data_dir,
        "split": split,
        "action_chunking_horizon": action_chunking_horizon,
        "trajectory_representation": trajectory_representation,
        "output_2d_trajectory": False,  # We need 3D for stats
        "normalize_coordinates": False,  # Disable normalization to compute raw stats
        "load_images": False,  # Avoid loading images for efficiency
    }
    
    # Add dataset-specific arguments
    if dataset_type == "egodex":
        dataset_kwargs["frame_downsampling_ratio"] = 1  # Use all frames for stats
    elif dataset_type == "robocasa":
        dataset_kwargs["frame_downsampling_ratio"] = 1
    
    # Add joint names if specified
    if joint_names is not None:
        dataset_kwargs["joint_names"] = joint_names
    
    print(f"\nInitializing {dataset_type} dataset...")
    dataset = DatasetClass(**dataset_kwargs)
    
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
    loader = torch.utils.data.DataLoader(
        subset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=lambda x: x,  # Simple collate (returns list of items)
        shuffle=False
    )
    
    # Lists to store processed trajectories and robot actions
    all_trajectories = []
    all_robot_actions = []
    
    print(f"\nProcessing samples with {loader.num_workers} workers...")
    valid_samples = 0
    valid_robot_action_samples = 0
    
    for batch in tqdm(loader, desc="Computing stats"):
        for item in batch:
            try:
                # Always extract trajectory data (for both egodex and robocasa)
                if "message_list" in item and item["message_list"] and len(item["message_list"]) > 0:
                    # Text-based mode: use points from message_list
                    final_trajectory = item['message_list'][0]['points']
                else:
                    # Flow matching mode: use trajectory_target from top-level
                    final_trajectory = item.get('trajectory_target', None)
                    if final_trajectory is None:
                        raise ValueError("No trajectory data found in item")
                
                # Ensure it's numpy array
                if isinstance(final_trajectory, torch.Tensor):
                    final_trajectory = final_trajectory.numpy()
                
                # Reshape: [num_steps, num_joints, 3] -> [num_steps, num_joints * 3]
                # Note: trajectory_target is already flattened, but message_list points may need reshaping
                if final_trajectory.ndim == 3:
                    num_steps = final_trajectory.shape[0]
                    trajectory_flat = final_trajectory.reshape(num_steps, -1).astype(np.float32)
                else:
                    # Already flattened (trajectory_target case)
                    trajectory_flat = final_trajectory.astype(np.float32)
                
                all_trajectories.append(trajectory_flat)
                valid_samples += 1
                
                # For RoboCasa dataset, also extract robot actions if available
                if dataset_type == "robocasa":
                    robot_actions = item.get('robot_actions', None)
                    if robot_actions is not None:
                        # Ensure it's numpy array
                        if isinstance(robot_actions, torch.Tensor):
                            robot_actions = robot_actions.numpy()
                        robot_actions_flat = robot_actions.astype(np.float32)
                        all_robot_actions.append(robot_actions_flat)
                        valid_robot_action_samples += 1
                
            except Exception as e:
                print(f"Error processing sample: {e}")
                continue
            
    if valid_samples == 0:
        raise ValueError("No valid samples found to compute stats!")
        
    print(f"\nComputed stats from {valid_samples} samples.")
    
    # Compute trajectory stats
    all_trajectory_data = np.concatenate(all_trajectories, axis=0)
    print(f"Trajectory data shape for stats: {all_trajectory_data.shape}")
    
    trajectory_mean = np.mean(all_trajectory_data, axis=0)
    trajectory_std = np.std(all_trajectory_data, axis=0)
    
    # Avoid division by zero
    trajectory_std[trajectory_std < 1e-8] = 1.0
    
    # Build stats dictionary - trajectory stats at top level for compatibility with trajectory_datasets.py
    stats = {
        "mean": torch.from_numpy(trajectory_mean),
        "std": torch.from_numpy(trajectory_std),
        "n_samples": valid_samples,
        "trajectory_representation": trajectory_representation,
        "action_chunking_horizon": action_chunking_horizon,
        "dataset_type": dataset_type,
        "joint_names": dataset.joint_names,
    }
    
    # Compute robot action stats for RoboCasa if available
    if dataset_type == "robocasa" and len(all_robot_actions) > 0:
        all_robot_action_data = np.concatenate(all_robot_actions, axis=0)
        print(f"Robot action data shape for stats: {all_robot_action_data.shape}")
        
        robot_action_mean = np.mean(all_robot_action_data, axis=0)
        robot_action_std = np.std(all_robot_action_data, axis=0)
        
        # Avoid division by zero
        robot_action_std[robot_action_std < 1e-8] = 1.0
        
        stats["robot_action_stats"] = {
            "mean": torch.from_numpy(robot_action_mean),
            "std": torch.from_numpy(robot_action_std),
            "n_samples": valid_robot_action_samples,
            "action_dim": robot_action_mean.shape[0],
        }
        print(f"Computed robot action stats from {valid_robot_action_samples} samples.")
    else:
        if dataset_type == "robocasa":
            print(f"Warning: No robot actions found in dataset. Only trajectory stats will be saved.")
    
    print(f"\nSaving stats to {output_file}")
    torch.save(stats, output_file)
    
    # Print summary
    print(f"\n{'=' * 60}")
    print("Stats Summary:")
    print(f"{'=' * 60}")
    
    # Trajectory stats summary
    print(f"\nTrajectory Stats:")
    print(f"  Mean shape: {trajectory_mean.shape}")
    print(f"  Std shape: {trajectory_std.shape}")
    if dataset_type == "robocasa" or dataset_type == "egodex":
        print(f"  Joint names: {dataset.joint_names}")
        print(f"\n  Mean values:")
        for i, joint in enumerate(dataset.joint_names):
            idx = i * 3
            print(f"    {joint}: x={trajectory_mean[idx]:.4f}, y={trajectory_mean[idx+1]:.4f}, z={trajectory_mean[idx+2]:.4f}")
        print(f"\n  Std values:")
        for i, joint in enumerate(dataset.joint_names):
            idx = i * 3
            print(f"    {joint}: x={trajectory_std[idx]:.4f}, y={trajectory_std[idx+1]:.4f}, z={trajectory_std[idx+2]:.4f}")
    
    # Robot action stats summary (if available)
    if dataset_type == "robocasa" and "robot_action_stats" in stats:
        print(f"\nRobot Action Stats:")
        robot_action_mean = stats["robot_action_stats"]["mean"].numpy()
        robot_action_std = stats["robot_action_stats"]["std"].numpy()
        print(f"  Mean shape: {robot_action_mean.shape}")
        print(f"  Std shape: {robot_action_std.shape}")
        print(f"  Action dimension: {robot_action_mean.shape[0]}")
        print(f"\n  Mean values (first 10 dims): {robot_action_mean[:10]}")
        print(f"  Std values (first 10 dims): {robot_action_std[:10]}")
        print(f"\n  Mean range: [{robot_action_mean.min():.4f}, {robot_action_mean.max():.4f}]")
        print(f"  Std range: [{robot_action_std.min():.4f}, {robot_action_std.max():.4f}]")
    
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute trajectory statistics for normalization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # For EgoDex dataset (fingertip trajectory)
    python compute_trajectory_stats.py --dataset egodex --data_dir /path/to/egodex

    # For RoboCasa dataset - computes both trajectory and robot action stats in one file
    python compute_trajectory_stats.py --dataset robocasa --data_dir /path/to/robocasa \\
        --output_file robocasa_stats.pt

    # With custom sample size and delta representation
    python compute_trajectory_stats.py --dataset robocasa --data_dir /path/to/data \\
        --output_file robocasa_delta_stats.pt --trajectory_representation delta --sample_size 10000

Note: For RoboCasa, trajectory stats are saved at top level (mean/std) and robot_action_stats
      is saved as an additional key. The dataset loader can choose which to use via normalize_stats_type.
        """
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="egodex",
        choices=["egodex", "robocasa"],
        help="Dataset type to compute stats for (default: egodex)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to dataset directory. If not provided, uses EGODEX_DATA_DIR or ROBOCASA_DATA_DIR env var."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output file for stats (default: {dataset}_trajectory_stats.pt)"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=5000,
        help="Number of samples to use for computing stats (default: 5000)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use (default: train)"
    )
    parser.add_argument(
        "--action_chunking_horizon",
        type=int,
        default=30,
        help="Action chunking horizon (default: 30)"
    )
    parser.add_argument(
        "--trajectory_representation",
        type=str,
        default="absolute",
        choices=["absolute", "delta"],
        help="Trajectory representation type (default: absolute)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for processing (default: 32)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Number of workers for data loading (default: 16)"
    )
    args = parser.parse_args()
    
    # Set default output file based on dataset type
    if args.output_file is None:
        args.output_file = f"{args.dataset}_trajectory_stats.pt"
    
    # Get data_dir from env if not provided
    if args.data_dir is None:
        if args.dataset == "egodex":
            args.data_dir = os.environ.get("EGODEX_DATA_DIR")
        elif args.dataset == "robocasa":
            args.data_dir = os.environ.get("ROBOCASA_DATA_DIR")
        
        if args.data_dir is None:
            env_var = "EGODEX_DATA_DIR" if args.dataset == "egodex" else "ROBOCASA_DATA_DIR"
            raise ValueError(f"--data_dir must be provided or {env_var} environment variable must be set")
    
    compute_stats(
        data_dir=args.data_dir,
        dataset_type=args.dataset,
        split=args.split,
        action_chunking_horizon=args.action_chunking_horizon,
        trajectory_representation=args.trajectory_representation,
        sample_size=args.sample_size,
        output_file=args.output_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
