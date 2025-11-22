#!/usr/bin/env python3
"""
Simple script to visualize 3D trajectories from saved .npz files.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import json


def get_finger_colors():
    """Get color mapping for different finger types."""
    return {
        'thumb': '#FF6B6B',      # Red
        'index': '#4ECDC4',      # Teal
        'middle': '#45B7D1',     # Blue
        'ring': '#96CEB4',       # Green
        'pinky': '#FFEAA7',      # Yellow
    }


def get_finger_names():
    """Get finger names in the expected order (10 keypoints: 5 per hand)."""
    return [
        'left_thumb', 'left_index', 'left_middle', 'left_ring', 'left_pinky',
        'right_thumb', 'right_index', 'right_middle', 'right_ring', 'right_pinky'
    ]


def visualize_3d_trajectory(trajectory_file: str, 
                            show_ground_truth: bool = True,
                            save_image: bool = False,
                            output_path: str = None):
    """
    Visualize 3D trajectory from saved .npz file.
    
    Args:
        trajectory_file: Path to .npz file containing trajectory data
        show_ground_truth: Whether to show ground truth trajectory if available
        save_image: Whether to save the visualization as an image
        output_path: Path to save the image (if None, uses trajectory_file with .png extension)
    """
    # Load trajectory data
    data = np.load(trajectory_file, allow_pickle=True)
    
    pred_trajectory = data['predicted_trajectory_3d']  # Shape: (num_steps, num_joints, 3)
    task_name = str(data.get('task_name', 'unknown'))
    prompt = str(data.get('prompt', ''))
    
    # Get ground truth if available
    gt_trajectory = None
    if show_ground_truth and 'ground_truth_trajectory_3d' in data:
        gt_trajectory = data['ground_truth_trajectory_3d']
    
    num_steps, num_joints, _ = pred_trajectory.shape
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    finger_colors = get_finger_colors()
    finger_names = get_finger_names()
    
    # Plot ground truth first (if available)
    if gt_trajectory is not None:
        for joint_idx in range(min(num_joints, len(finger_names))):
            finger_type = finger_names[joint_idx].replace('left_', '').replace('right_', '')
            color = finger_colors.get(finger_type, '#FFFFFF')
            
            # Extract trajectory for this joint
            joint_traj = gt_trajectory[:, joint_idx, :]  # Shape: (num_steps, 3)
            
            # Plot trajectory line
            ax.plot(joint_traj[:, 0], joint_traj[:, 1], joint_traj[:, 2],
                   color=color, alpha=0.3, linewidth=1.5, linestyle='--',
                   label=f'{finger_type} (GT)' if joint_idx < 5 else '')
            
            # Plot start point
            ax.scatter(joint_traj[0, 0], joint_traj[0, 1], joint_traj[0, 2],
                      color=color, s=50, marker='o', alpha=0.5)
            
            # Plot end point
            ax.scatter(joint_traj[-1, 0], joint_traj[-1, 1], joint_traj[-1, 2],
                      color=color, s=50, marker='s', alpha=0.5)
    
    # Plot predicted trajectories
    for joint_idx in range(min(num_joints, len(finger_names))):
        finger_type = finger_names[joint_idx].replace('left_', '').replace('right_', '')
        color = finger_colors.get(finger_type, '#FFFFFF')
        
        # Extract trajectory for this joint
        joint_traj = pred_trajectory[:, joint_idx, :]  # Shape: (num_steps, 3)
        
        # Plot trajectory line with color gradient (lighter = later in time)
        for step in range(num_steps - 1):
            alpha = 0.5 + (step / max(1, num_steps - 1)) * 0.5
            ax.plot([joint_traj[step, 0], joint_traj[step+1, 0]],
                   [joint_traj[step, 1], joint_traj[step+1, 1]],
                   [joint_traj[step, 2], joint_traj[step+1, 2]],
                   color=color, alpha=alpha, linewidth=2.5)
        
        # Plot start point
        ax.scatter(joint_traj[0, 0], joint_traj[0, 1], joint_traj[0, 2],
                  color=color, s=100, marker='o', edgecolors='black', linewidths=1.5)
        
        # Plot end point
        ax.scatter(joint_traj[-1, 0], joint_traj[-1, 1], joint_traj[-1, 2],
                  color=color, s=100, marker='s', edgecolors='black', linewidths=1.5)
        
        # Label only first 5 joints to avoid clutter
        if joint_idx < 5:
            ax.text(joint_traj[0, 0], joint_traj[0, 1], joint_traj[0, 2],
                   f' {finger_type}', fontsize=8)
    
    # Set labels and title
    ax.set_xlabel('X (mm)', fontsize=12)
    ax.set_ylabel('Y (mm)', fontsize=12)
    ax.set_zlabel('Z (mm)', fontsize=12)
    
    title = f"3D Trajectory: {task_name}"
    if prompt:
        title += f"\n{prompt[:60]}..." if len(prompt) > 60 else f"\n{prompt}"
    ax.set_title(title, fontsize=14, pad=20)
    
    # Set equal aspect ratio
    # Get the limits
    all_points = pred_trajectory.reshape(-1, 3)
    if gt_trajectory is not None:
        all_points = np.vstack([all_points, gt_trajectory.reshape(-1, 3)])
    
    max_range = np.array([all_points[:, 0].max() - all_points[:, 0].min(),
                          all_points[:, 1].max() - all_points[:, 1].min(),
                          all_points[:, 2].max() - all_points[:, 2].min()]).max() / 2.0
    
    mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
    mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
    mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Add legend (only show unique entries)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=8)
    
    # Add info text
    info_text = f"Steps: {num_steps}, Joints: {num_joints}"
    if 'trajectory_representation' in data:
        info_text += f"\nRepresentation: {data['trajectory_representation']}"
    ax.text2D(0.02, 0.98, info_text, transform=ax.transAxes,
              fontsize=9, verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save or show
    if save_image:
        if output_path is None:
            output_path = str(Path(trajectory_file).with_suffix('.png'))
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize 3D trajectories from saved .npz files"
    )
    parser.add_argument("trajectory_file", type=str,
                       help="Path to .npz file containing 3D trajectory data")
    parser.add_argument("--no-gt", action="store_true",
                       help="Don't show ground truth trajectory even if available")
    parser.add_argument("--save", action="store_true",
                       help="Save visualization as image instead of displaying")
    parser.add_argument("--output", type=str, default=None,
                       help="Output path for saved image (default: trajectory_file.png)")
    
    args = parser.parse_args()
    
    if not Path(args.trajectory_file).exists():
        print(f"Error: File not found: {args.trajectory_file}")
        return
    
    visualize_3d_trajectory(
        args.trajectory_file,
        show_ground_truth=not args.no_gt,
        save_image=args.save,
        output_path=args.output
    )


if __name__ == "__main__":
    main()

