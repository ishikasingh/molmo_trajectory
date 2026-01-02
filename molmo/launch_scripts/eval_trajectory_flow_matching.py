#!/usr/bin/env python3
"""
Evaluation script for 3D trajectory predictions.
Supports both flow matching (ODE sampling) and direct regression modes.
Trajectories are projected to 2D for visualization on images.

Evaluation modes:
- Image mode (default): Randomly samples frames and generates individual visualization images
- Video mode: Samples videos, evaluates consecutive frames, and generates output videos
"""

import argparse
import torch
import re
import os
import time
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Dict, Optional
import json
import cv2

from olmo.model import Molmo
from olmo.data.model_preprocessor import load_image
from olmo.data import build_mm_preprocessor
from olmo.config import ModelConfig
from olmo.data.trajectory_datasets import TrajectoryDataset
from olmo.data.robo_casa_affordance_datasets import RoboCasaTrajectoryDataset


def get_finger_colors() -> Dict[str, str]:
    """Get color mapping for different finger types."""
    return {
        'thumb': '#FF6B6B',      # Red
        'index': '#4ECDC4',      # Teal
        'middle': '#45B7D1',     # Blue
        'ring': '#96CEB4',       # Green
        'pinky': '#FFEAA7',      # Yellow
    }


def get_finger_names() -> List[str]:
    """Get finger names in the expected order (10 keypoints: 5 per hand)."""
    return [
        'left_thumb', 'left_index', 'left_middle', 'left_ring', 'left_pinky',
        'right_thumb', 'right_index', 'right_middle', 'right_ring', 'right_pinky'
    ]


def sanitize_filename(s: str) -> str:
    """Replace any character that is not alphanumeric or underscore with underscore."""
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', s)

def project_3d_trajectory_to_2d(trajectory_3d: np.ndarray, 
                                intrinsic: np.ndarray,
                                img_width: int, 
                                img_height: int,
                                normalize: bool = True) -> np.ndarray:
    """
    Project 3D trajectory from camera frame to 2D image coordinates.
    
    Args:
        trajectory_3d: Trajectory in camera frame of shape [num_steps, num_joints, 3]
        intrinsic: Camera intrinsic matrix of shape [3, 3]
        img_width: Image width in pixels
        img_height: Image height in pixels
        normalize: Whether to normalize to 0-100 scale
    
    Returns:
        Trajectory in 2D image space of shape [num_steps, num_joints, 2]
    """
    num_steps, num_joints, _ = trajectory_3d.shape
    
    # Reshape to [num_steps * num_joints, 3]
    points_3d = trajectory_3d.reshape(-1, 3)
    
    # Project to 2D: [x', y', z'] = K @ [x, y, z]
    points_2d_homo = points_3d @ intrinsic.T
    
    # Normalize by depth (z coordinate)
    w = points_2d_homo[:, 2:3]
    w = np.where(w == 0, 1, w)  # Avoid division by zero
    points_2d_pixel = points_2d_homo[:, :2] / w
    
    if normalize:
        # Normalize to 0-100 scale
        points_2d_normalized = np.zeros_like(points_2d_pixel)
        points_2d_normalized[:, 0] = (points_2d_pixel[:, 0] / img_width) * 100.0
        points_2d_normalized[:, 1] = (points_2d_pixel[:, 1] / img_height) * 100.0
        points_2d_final = points_2d_normalized
    else:
        points_2d_final = points_2d_pixel
    
    # Reshape back to [num_steps, num_joints, 2]
    trajectory_2d = points_2d_final.reshape(num_steps, num_joints, 2)
    return trajectory_2d


def convert_trajectory_to_pixel_coords(trajectory: np.ndarray, 
                                       img_width: int, 
                                       img_height: int,
                                       is_normalized: bool = True) -> np.ndarray:
    """
    Convert trajectory from normalized (0-100) or pixel coordinates.
    
    Args:
        trajectory: Trajectory of shape [num_steps, num_joints, 2]
        img_width: Image width in pixels
        img_height: Image height in pixels
        is_normalized: Whether the input is normalized (0-100 scale)
    
    Returns:
        Trajectory in pixel coordinates of shape [num_steps, num_joints, 2]
    """
    if is_normalized:
        # Convert from 0-100 scale to pixel coordinates
        trajectory_pixel = np.zeros_like(trajectory)
        trajectory_pixel[..., 0] = (trajectory[..., 0] / 100.0) * img_width
        trajectory_pixel[..., 1] = (trajectory[..., 1] / 100.0) * img_height
        return trajectory_pixel
    else:
        return trajectory


def visualize_trajectory_on_image(image_np: np.ndarray, 
                                  pred_trajectory: Optional[np.ndarray] = None,
                                  gt_trajectory: Optional[np.ndarray] = None,
                                  is_normalized: bool = True,
                                  prompt: Optional[str] = None) -> Image.Image:
    """
    Visualize trajectory on an image.
    
    Args:
        image_np: Input image as numpy array
        pred_trajectory: Predicted trajectory of shape [num_steps, num_joints, 2] (optional)
        gt_trajectory: Ground truth trajectory of shape [num_steps, num_joints, 2] (optional)
        is_normalized: Whether trajectories are in normalized (0-100) coordinates
        prompt: Language command/instruction to display on the image (optional)
    
    Returns:
        PIL Image with visualized trajectories
    """
    image_pil = Image.fromarray(image_np)
    draw = ImageDraw.Draw(image_pil, 'RGBA')
    h, w = image_np.shape[:2]
    
    finger_colors = get_finger_colors()
    finger_names = get_finger_names()
    
    # Convert trajectories to pixel coordinates if needed
    if pred_trajectory is not None:
        pred_trajectory_pixel = convert_trajectory_to_pixel_coords(
            pred_trajectory, w, h, is_normalized
        )
    
    if gt_trajectory is not None:
        gt_trajectory_pixel = convert_trajectory_to_pixel_coords(
            gt_trajectory, w, h, is_normalized
        )
    
    # Draw ground truth trajectories first (so predictions appear on top)
    if gt_trajectory is not None:
        num_steps, num_joints = gt_trajectory_pixel.shape[:2]
        
        for joint_idx in range(num_joints):
            finger_type = finger_names[joint_idx].replace('left_', '').replace('right_', '')
            color = finger_colors.get(finger_type, '#FFFFFF')
            
            # Draw trajectory path
            trajectory_points = []
            for step in range(num_steps):
                x, y = gt_trajectory_pixel[step, joint_idx]
                if 0 <= x < w and 0 <= y < h:  # Only add valid points
                    trajectory_points.append((x, y))
            
            # Draw lines connecting trajectory points
            if len(trajectory_points) > 1:
                # Convert hex color to RGB with alpha
                rgb_color = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
                rgba_color = rgb_color + (60,)  # More transparent for GT
                
                for i in range(len(trajectory_points) - 1):
                    draw.line([trajectory_points[i], trajectory_points[i+1]], 
                             fill=rgba_color, width=2)
            
            # Draw points along trajectory - smaller points for GT
            point_radius = h / 250
            for x, y in trajectory_points:
                draw.ellipse((x - point_radius, y - point_radius, 
                            x + point_radius, y + point_radius),
                           fill='red', outline='darkred', width=1)
    
    # Draw prediction trajectories
    if pred_trajectory is not None:
        num_steps, num_joints = pred_trajectory_pixel.shape[:2]
        
        for joint_idx in range(num_joints):
            finger_type = finger_names[joint_idx].replace('left_', '').replace('right_', '')
            color = finger_colors.get(finger_type, '#FFFFFF')
            
            # Draw trajectory path
            trajectory_points = []
            for step in range(num_steps):
                x, y = pred_trajectory_pixel[step, joint_idx]
                if 0 <= x < w and 0 <= y < h:  # Only add valid points
                    trajectory_points.append((x, y))
            
            # Draw lines connecting trajectory points
            if len(trajectory_points) > 1:
                # Convert hex color to RGB with alpha
                rgb_color = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
                rgba_color = rgb_color + (180,)  # More opaque for predictions
                
                for i in range(len(trajectory_points) - 1):
                    draw.line([trajectory_points[i], trajectory_points[i+1]], 
                             fill=rgba_color, width=3)
            
            # Draw points along trajectory
            point_radius = h / 120
            for step_idx, (x, y) in enumerate(trajectory_points):
                # Use color gradient for time (lighter = later)
                alpha = int(100 + (step_idx / max(1, len(trajectory_points) - 1)) * 155)
                rgb_color = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
                point_color = rgb_color + (alpha,)
                
                draw.ellipse((x - point_radius, y - point_radius, 
                            x + point_radius, y + point_radius),
                           fill=point_color, outline='white', width=2)
    
    # Draw legend
    draw_legend(draw, finger_colors, w, h, show_ground_truth=(gt_trajectory is not None))
    
    # Draw prompt text at the bottom of the image
    if prompt:
        draw_prompt_text(draw, prompt, w, h)
    
    return image_pil


def draw_prompt_text(draw: ImageDraw.Draw, prompt: str, 
                     image_width: int, image_height: int) -> None:
    """Draw the language command/prompt at the bottom of the image."""
    # Text settings
    text_margin = int(image_height * 0.02)
    text_padding = int(image_height * 0.015)
    
    # Try to load a font
    # Scale font size based on image size - use smaller divisor for smaller images
    # For 256x256 images: ~6-8, for 1080p images: ~27
    min_font_size = 8
    max_font_size = 24
    font_size = max(min_font_size, min(max_font_size, int(image_height/50)))
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except OSError:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except OSError:
            font = ImageFont.load_default()
    
    # Wrap text if too long
    max_width = image_width - 2 * text_margin
    words = prompt.split()
    lines = []
    current_line = []
    
    for word in words:
        test_line = ' '.join(current_line + [word])
        try:
            bbox = draw.textbbox((0, 0), test_line, font=font)
            text_width = bbox[2] - bbox[0]
        except:
            text_width = len(test_line) * font_size * 0.6
        
        if text_width <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
                current_line = [word]
            else:
                lines.append(word)
    
    if current_line:
        lines.append(' '.join(current_line))
    
    # Calculate text box dimensions
    line_height = font_size + int(font_size * 0.3)
    total_text_height = len(lines) * line_height
    box_height = total_text_height + 2 * text_padding
    box_width = image_width - 2 * text_margin
    
    # Position at bottom of image
    box_y = image_height - box_height - text_margin
    box_x = text_margin
    
    # Draw semi-transparent background
    background = Image.new('RGBA', (box_width, box_height), (0, 0, 0, 180))
    image = draw._image
    image.paste(background, (box_x, box_y), background)
    
    # Draw text lines
    text_y = box_y + text_padding
    for line in lines:
        draw.text((box_x + text_padding, text_y), line, fill='white', font=font)
        text_y += line_height


def draw_legend(draw: ImageDraw.Draw, finger_colors: Dict[str, str], 
                image_width: int, image_height: int,
                show_ground_truth: bool = False) -> None:
    """Draw a legend showing finger colors and optionally a ground truth indicator.
    
    Note: All colored trajectories shown are model predictions, so no separate prediction label is needed.
    """
    legend_x = int(image_width * 0.02)
    legend_y = int(image_height * 0.02)
    legend_spacing = int(image_height * 0.03)
    
    # Calculate legend height based on content
    legend_items = len(finger_colors)
    if show_ground_truth:
        legend_items += 1  # Add space for GT indicator (all colored trajectories are predictions)
    
    # Create a semi-transparent background for legend
    legend_width = int(image_width * 0.15)
    legend_height = legend_items * legend_spacing + int(image_height * 0.02)
    legend_bg = Image.new('RGBA', (legend_width, legend_height), (0, 0, 0, 128))
    
    # Get the image from the draw object to paste the background
    image = draw._image
    image.paste(legend_bg, (legend_x - int(image_width * 0.01), 
                            legend_y - int(image_height * 0.01)), legend_bg)
    
    # Try to load a font
    # Scale font size based on image size - use smaller divisor for smaller images
    # For 256x256 images: ~5, for 1080p images: ~21
    min_legend_font_size = 6
    max_legend_font_size = 18
    legend_font_size = max(min_legend_font_size, min(max_legend_font_size, int(image_height/60)))
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", legend_font_size)
    except OSError:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", legend_font_size)
        except OSError:
            font = ImageFont.load_default()
    
    # Draw finger color legend items
    for i, (finger_type, color) in enumerate(finger_colors.items()):
        y_pos = legend_y + i * legend_spacing
        legend_r = int(image_height * 0.01)
        
        # Draw colored circle
        draw.ellipse((legend_x, y_pos - legend_r, legend_x + 2*legend_r, y_pos + legend_r), 
                    fill=color, outline='white', width=1)
        
        # Draw text
        text = finger_type.capitalize()
        draw.text((legend_x + int(image_width * 0.02), y_pos - int(image_height * 0.01)), 
                 text, fill='white', font=font)
    
    # Add ground truth vs prediction legend if needed
    if show_ground_truth:
        gt_y = legend_y + len(finger_colors) * legend_spacing
        legend_r = int(image_height * 0.01)
        
        # Ground truth indicator (red circle with thinner line)
        draw.ellipse((legend_x, gt_y - legend_r, legend_x + 2*legend_r, gt_y + legend_r), 
                    fill='red', outline='black', width=1)
        draw.text((legend_x + int(image_width * 0.02), gt_y - int(image_height * 0.01)), 
                 "Ground Truth", fill='white', font=font)
        
        # Note: All colored trajectories shown are model predictions, so no separate "Prediction" label needed


def convert_delta_to_absolute(delta_trajectory: np.ndarray, initial_state: np.ndarray) -> np.ndarray:
    """
    Convert delta positions (velocities) to absolute positions.
    
    For a delta trajectory where v_t = x_{t+1} - x_t, we reconstruct absolute positions as:
    x_0 = initial_state
    x_t = x_0 + sum(v_0 to v_{t-1}) for t > 0
    
    Args:
        delta_trajectory: Delta trajectory of shape (num_steps, num_joints, coords)
                          Each entry is v_t representing the velocity/delta
        initial_state: Initial absolute position of shape (num_joints, coords)
                      This is the absolute position at timestep 0
    
    Returns:
        Absolute trajectory of shape (num_steps, num_joints, coords)
    """
    num_steps = delta_trajectory.shape[0]
    absolute_trajectory = np.zeros_like(delta_trajectory)
    
    # First timestep is the initial state
    absolute_trajectory[0] = initial_state
    
    # For subsequent timesteps, cumulatively add deltas
    # x_t = x_{t-1} + v_{t-1}
    for t in range(1, num_steps):
        absolute_trajectory[t] = absolute_trajectory[t-1] + delta_trajectory[t-1]
    
    return absolute_trajectory


def load_test_examples(num_examples: int = 10, 
                       action_chunking_horizon: int = 10,
                       trajectory_representation: str = "absolute",
                       split: str = "test",
                       dataset_type: str = "egodex") -> List[Dict]:
    """
    Load examples from the specified split for 3D trajectory prediction.
    
    Args:
        num_examples: Number of examples to load
        action_chunking_horizon: Number of timesteps in trajectory
        trajectory_representation: Either 'absolute' or 'delta' for trajectory representation
        split: Dataset split to load (e.g., train, test)
        dataset_type: Either 'egodex' or 'robocasa' to specify which dataset to load
    
    Returns:
        List of example dictionaries
    """
    # Check if stats file is available for normalization
    if dataset_type == "egodex":
        stats_file = os.environ.get("TRAJECTORY_STATS_FILE")
        normalize_coords = bool(stats_file)
        
        # Load EgoDex dataset (always 3D)
        dataset = TrajectoryDataset(
            split=split,
            action_chunking_horizon=action_chunking_horizon,
            output_2d_trajectory=False,
            normalize_coordinates=False, # Set to be False as in inference time, the trajectory itself is only for visualization, but this only works for delta representation.
            output_format="flow_matching",  # Use flow matching format
            trajectory_representation=trajectory_representation,
            frame_downsampling_ratio=10,
        )
    elif dataset_type == "robocasa":
        stats_file = os.environ.get("ROBOCASA_STATS_FILE")
        normalize_coords = bool(stats_file)
        
        # Load RoboCasa dataset (always 3D)
        dataset = RoboCasaTrajectoryDataset(
            split=split,
            action_chunking_horizon=action_chunking_horizon,
            output_2d_trajectory=False,
            normalize_coordinates=False, # Set to be False as in inference time, the trajectory itself is only for visualization, but this only works for delta representation.
            trajectory_representation=trajectory_representation,
            frame_downsampling_ratio=10,
        )
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}. Must be 'egodex' or 'robocasa'")
    
    print(f"Loaded '{split}' dataset with {len(dataset)} examples")
    
    # Sample examples (using evenly spaced indices for diversity)
    indices = np.linspace(0, len(dataset) - 1, min(num_examples, len(dataset)), dtype=int)
    
    examples = []
    for idx in indices:
        example_data = dataset.get(idx, rng=np.random.RandomState(42))
        
        # Extract data
        image = example_data["image"]
        metadata = example_data.get("metadata", {})
        
        # Handle both message_list (text-based) and top-level fields (flow matching)
        if "message_list" in example_data and example_data["message_list"] and len(example_data["message_list"]) > 0:
            # Text-based mode: read from message_list
            message_list = example_data["message_list"]
            instruction = message_list[0].get("label", "")
            gt_trajectory = message_list[0].get("points", None)
            point_scale = message_list[0].get("point_scale", None)
            style = message_list[0].get("style", "")
            state = message_list[0].get("state", None)
        else:
            # Flow matching mode: read from top-level fields
            instruction = example_data.get("label", "")
            gt_trajectory = example_data.get("trajectory_target", None)  # Use trajectory_target for flow matching
            point_scale = None  # Not used for flow matching
            style = example_data.get("style", "")
            state = example_data.get("state", None)
            
            # Save image temporarily for processing
            temp_image_path = f"temp_trajectory_image_{idx}.jpg"
            
            if hasattr(image, 'save'):
                image.save(temp_image_path)
            else:
                # Convert numpy array to PIL Image
                if isinstance(image, np.ndarray):
                    Image.fromarray(image).save(temp_image_path)

            examples.append({
                "image_path": temp_image_path,
                "prompt": instruction,
                "ground_truth_trajectory": gt_trajectory,
                "point_scale": point_scale,
                "style": style,
                "task_name": metadata.get("task_name", "unknown"),
                "frame_idx": metadata.get("frame_idx", 0),
                "is_test_data": True,
                "test_idx": idx,
                "state": state,
                "trajectory_representation": metadata.get("trajectory_representation", "absolute"),
            })
            
            print(f"Loaded test example {idx}: {instruction[:50]}...")
            if gt_trajectory is not None:
                print(f"  Ground truth trajectory shape: {gt_trajectory.shape}")
    
    return examples


def load_video_examples(num_videos: int = 5,
                        action_chunking_horizon: int = 10,
                        trajectory_representation: str = "absolute",
                        split: str = "test",
                        frame_downsampling_ratio: int = 10,
                        dataset_type: str = "egodex") -> List[Dict]:
    """
    Load all frames from multiple videos for video evaluation.
    
    Args:
        num_videos: Number of videos to sample
        action_chunking_horizon: Number of timesteps in trajectory
        trajectory_representation: Either 'absolute' or 'delta' for trajectory representation
        split: Dataset split to load (e.g., train, test)
        frame_downsampling_ratio: Frame downsampling ratio (e.g., 10 = every 10th frame)
        dataset_type: Either 'egodex' or 'robocasa' to specify which dataset to load
    
    Returns:
        List of video dictionaries, each containing frames and metadata
    """
    # Load dataset with specified downsampling ratio
    if dataset_type == "egodex":
        dataset = TrajectoryDataset(
            split=split,
            action_chunking_horizon=action_chunking_horizon,
            output_2d_trajectory=False,
            normalize_coordinates=False,
            output_format="flow_matching",
            trajectory_representation=trajectory_representation,
            frame_downsampling_ratio=frame_downsampling_ratio,
        )
    elif dataset_type == "robocasa":
        dataset = RoboCasaTrajectoryDataset(
            split=split,
            action_chunking_horizon=action_chunking_horizon,
            output_2d_trajectory=False,
            normalize_coordinates=False,
            trajectory_representation=trajectory_representation,
            frame_downsampling_ratio=frame_downsampling_ratio,
        )
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}. Must be 'egodex' or 'robocasa'")
    
    print(f"Loaded '{split}' dataset with {len(dataset)} examples (downsampling ratio: {frame_downsampling_ratio})")
    
    # Use dataset's built-in video sampling methods
    sampled_videos = dataset.sample_videos(num_videos=num_videos)
    
    print(f"Found {len(dataset.get_videos_info())} unique videos")
    print(f"Sampled {len(sampled_videos)} videos")
    
    if not sampled_videos:
        print(f"WARNING: No videos found in dataset.")
        return []
    
    # Load all frames for each sampled video
    video_examples = []
    for vid_idx, video_info in enumerate(sampled_videos):
        frames = dataset.iter_video_frames(video_info=video_info)
        
        video_data = {
            'video_path': video_info['video_path'],
            'task_name': video_info['task_name'],
            'frames': frames,
        }
        
        print(f"Loaded video {vid_idx + 1}/{len(sampled_videos)}: {Path(video_info['video_path']).name} with {len(frames)} frames")
        video_examples.append(video_data)
    
    return video_examples


def create_video_from_frames(frames: List[Image.Image], 
                             output_path: str,
                             fps: int = 10,
                             codec: str = 'mp4v') -> bool:
    """
    Create a video from a list of PIL Image frames.
    
    Args:
        frames: List of PIL Image frames
        output_path: Path to save the output video
        fps: Frames per second for the output video
        codec: Video codec to use (default: 'mp4v' for .mp4 files)
    
    Returns:
        True if video was created successfully, False otherwise
    """
    if not frames:
        print("ERROR: No frames to create video")
        return False
    
    # Get frame dimensions from the first frame
    first_frame = frames[0]
    if isinstance(first_frame, Image.Image):
        width, height = first_frame.size
    else:
        height, width = first_frame.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"ERROR: Could not create video writer for {output_path}")
        return False
    
    try:
        for i, frame in enumerate(frames):
            # Convert PIL Image to numpy array
            if isinstance(frame, Image.Image):
                frame_np = np.array(frame)
            else:
                frame_np = frame
            
            # Convert RGB to BGR for OpenCV
            if len(frame_np.shape) == 3 and frame_np.shape[2] == 3:
                frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame_np
            
            # Ensure frame is the correct size
            if frame_bgr.shape[1] != width or frame_bgr.shape[0] != height:
                frame_bgr = cv2.resize(frame_bgr, (width, height))
            
            out.write(frame_bgr)
        
        print(f"Created video with {len(frames)} frames at {fps} FPS: {output_path}")
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to write video: {e}")
        return False
        
    finally:
        out.release()


def evaluate_video(model, preprocessor, tokenizer, video_data: Dict,
                   args, intrinsic: np.ndarray,
                   stats_mean: Optional[np.ndarray] = None,
                   stats_std: Optional[np.ndarray] = None) -> Tuple[List[Image.Image], List[Dict]]:
    """
    Evaluate all frames in a video and return visualized frames.
    
    Args:
        model: The trajectory prediction model
        preprocessor: Data preprocessor
        tokenizer: Tokenizer for decoding
        video_data: Dictionary containing video frames and metadata
        args: Command-line arguments
        intrinsic: Camera intrinsic matrix
        stats_mean: Mean for denormalization (optional)
        stats_std: Std for denormalization (optional)
    
    Returns:
        Tuple of (list of visualized PIL Images, list of metrics dictionaries)
    """
    device = torch.device(args.device)
    visualized_frames = []
    frame_metrics = []
    
    is_direct_prediction = getattr(model.config, "use_direct_trajectory_prediction", False)
    
    for frame_idx, frame_data in enumerate(video_data['frames']):
        image_np = frame_data["image"]
        prompt = frame_data["prompt"]
        gt_trajectory_raw = frame_data.get("ground_truth_trajectory", None)
        style = frame_data.get("style", "trajectory_3d_fm")
        
        h, w = image_np.shape[:2]
        
        # Prepare example for model
        # Save image temporarily for preprocessing
        temp_image_path = f"temp_video_frame_{frame_idx}.jpg"
        Image.fromarray(image_np).save(temp_image_path)
        
        example = {
            "image": image_np,
            "prompt": prompt,
            "style": style,
            "state": frame_data.get("state"),
            "point_scale": frame_data.get("point_scale"),
        }
        batch = preprocessor(example)
        
        # Move tensors to device
        input_ids = torch.tensor(batch["input_tokens"], dtype=torch.long).unsqueeze(0).to(device)
        attention_mask = (input_ids != -1).to(device)
        images = torch.tensor(batch["images"], dtype=torch.float32).unsqueeze(0).to(device)
        image_input_idx = torch.tensor(batch["image_input_idx"], dtype=torch.long).unsqueeze(0).to(device)
        image_masks = None
        if "image_masks" in batch:
            image_masks = torch.tensor(batch["image_masks"]).unsqueeze(0).to(device)
        
        position_ids = None
        if "position_ids" in batch:
            position_ids = torch.tensor(batch["position_ids"], dtype=torch.long).unsqueeze(0).to(device)
        
        proprio_state = None
        if "proprio_state" in batch:
            proprio_state = torch.tensor(batch["proprio_state"], dtype=torch.float32).unsqueeze(0).to(device)
        
        expert_type = None
        if "expert_type" in batch:
            expert_type = torch.tensor([batch["expert_type"]], dtype=torch.long).to(device)
        elif hasattr(model.config, "num_action_experts") and model.config.num_action_experts > 1:
            print("WARNING: Multi-expert mode but no expert_type in batch. Defaulting to expert 0 (human).")
            expert_type = torch.tensor([0], dtype=torch.long).to(device)
        
        # Generate trajectory
        with torch.no_grad():
            if is_direct_prediction:
                pred_trajectory = model.predict_actions_direct(
                    input_ids=input_ids,
                    attention_mask=None,
                    images=images,
                    image_masks=image_masks,
                    image_input_idx=image_input_idx,
                    position_ids=position_ids,
                    proprio_state=proprio_state,
                    expert_type=expert_type,
                )
            else:
                initial_noise = torch.randn(
                    1,
                    model.config.action_horizon,
                    model.config.action_dim,
                    device=device
                )
                pred_trajectory = model.sample_actions_flow_matching(
                    input_ids=input_ids,
                    attention_mask=None,
                    images=images,
                    image_masks=image_masks,
                    image_input_idx=image_input_idx,
                    num_steps=args.num_ode_steps,
                    initial_noise=initial_noise,
                    position_ids=position_ids,
                    proprio_state=proprio_state,
                    expert_type=expert_type,
                )
        
        # Convert to numpy and process
        if isinstance(pred_trajectory, torch.Tensor):
            pred_trajectory = pred_trajectory.cpu().numpy()
        
        # Denormalize if stats available
        if stats_mean is not None and stats_std is not None:
            pred_trajectory = pred_trajectory * stats_std + stats_mean
        
        # Reshape from (batch_size, action_horizon, action_dim) to (action_horizon, num_joints, 3)
        batch_size, action_horizon, action_dim = pred_trajectory.shape
        pred_trajectory = pred_trajectory.squeeze(0)
        num_joints = action_dim // 3
        pred_trajectory = pred_trajectory.reshape(action_horizon, num_joints, 3)
        
        # Convert delta to absolute if needed
        trajectory_rep = frame_data.get("trajectory_representation", "absolute")
        if trajectory_rep == "delta":
            initial_state = frame_data.get("state")
            if initial_state is not None:
                if len(initial_state.shape) == 1:
                    num_joints_state = initial_state.shape[0] // 3
                    initial_state = initial_state.reshape(num_joints_state, 3)
                pred_trajectory = convert_delta_to_absolute(pred_trajectory, initial_state)
        
        # Process ground truth trajectory
        gt_trajectory_2d = None
        if gt_trajectory_raw is not None:
            if isinstance(gt_trajectory_raw, torch.Tensor):
                gt_trajectory_raw = gt_trajectory_raw.numpy()
            
            if len(gt_trajectory_raw.shape) == 3 and gt_trajectory_raw.shape[0] == 1:
                gt_trajectory_raw = gt_trajectory_raw.squeeze(0)
            
            if len(gt_trajectory_raw.shape) == 2:
                gt_action_horizon, gt_action_dim = gt_trajectory_raw.shape
                gt_num_joints = gt_action_dim // 3
                gt_trajectory_raw = gt_trajectory_raw.reshape(gt_action_horizon, gt_num_joints, 3)
            
            if trajectory_rep == "delta":
                initial_state = frame_data.get("state")
                if initial_state is not None:
                    if len(initial_state.shape) == 1:
                        num_joints_state = initial_state.shape[0] // 3
                        initial_state = initial_state.reshape(num_joints_state, 3)
                    gt_trajectory_raw = convert_delta_to_absolute(gt_trajectory_raw, initial_state)
            
            gt_trajectory_2d = project_3d_trajectory_to_2d(
                gt_trajectory_raw, intrinsic, w, h, normalize=True
            )
        
        # Project prediction to 2D
        pred_trajectory_2d = project_3d_trajectory_to_2d(
            pred_trajectory, intrinsic, w, h, normalize=True
        )
        
        # Compute metrics
        metrics = {
            "frame_idx": frame_data['frame_idx'],
            "video_frame_idx": frame_idx,
        }
        
        if gt_trajectory_2d is not None:
            mse = np.mean((pred_trajectory_2d - gt_trajectory_2d) ** 2)
            ade = np.mean(np.linalg.norm(pred_trajectory_2d - gt_trajectory_2d, axis=-1))
            fde = np.linalg.norm(pred_trajectory_2d[-1] - gt_trajectory_2d[-1], axis=-1).mean()
            metrics["mse"] = float(mse)
            metrics["ade"] = float(ade)
            metrics["fde"] = float(fde)
        
        # Visualize
        visualized_image = visualize_trajectory_on_image(
            image_np,
            pred_trajectory=pred_trajectory_2d,
            gt_trajectory=gt_trajectory_2d,
            is_normalized=True,
            prompt=prompt
        )
        
        visualized_frames.append(visualized_image)
        frame_metrics.append(metrics)
        
        # Clean up temp file
        if os.path.exists(temp_image_path):
            try:
                os.remove(temp_image_path)
            except:
                pass
    
    return visualized_frames, frame_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate 3D trajectory predictions (flow matching or direct regression)"
    )
    parser.add_argument("checkpoint", type=str, help="Path to model checkpoint directory")
    parser.add_argument("--num_examples", type=int, default=10,
                       help="Number of test examples to visualize (image mode)")
    parser.add_argument("--action_chunking_horizon", type=int, default=30,
                       help="Number of timesteps in trajectory")
    parser.add_argument("--num_ode_steps", type=int, default=10,
                       help="Number of ODE integration steps for flow matching (ignored for direct prediction)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run the model on")
    parser.add_argument("--output_dir", type=str, default="trajectory_flow_matching_output",
                       help="Directory to save visualizations")
    parser.add_argument("--camera_intrinsic", type=str, default=None,
                       help="Path to numpy file containing camera intrinsic matrix")
    parser.add_argument("--save_metrics", action="store_true",
                       help="Save trajectory metrics (MSE, ADE, FDE) to JSON")
    parser.add_argument("--trajectory_representation", type=str, default="delta",
                       choices=["absolute", "delta"],
                       help="Trajectory representation mode: 'absolute' positions or 'delta' (velocity)")
    parser.add_argument("--save_3d_trajectory", action="store_true",
                       help="Save 3D trajectory data to .npz file")
    parser.add_argument("--normalize_coordinates", action="store_true", default=True,
                       help="Whether the model was trained with normalized coordinates (requires TRAJECTORY_STATS_FILE env var)")
    parser.add_argument("--num_inference_samples", type=int, default=1,
                       help="Number of inference samples to generate and choose the best one. When > 1, selects the sample with minimum MSE to ground truth (useful for multi-modal tasks)")
    parser.add_argument("--inference_seed", type=int, default=None,
                       help="Random seed for reproducible inference sampling (optional)")
    parser.add_argument("--split", type=str, default="test",
                       choices=["train", "test", "test_pick_and_place", "overfit"],
                       help="Dataset split to load examples from")
    
    # Video evaluation mode arguments
    parser.add_argument("--eval_mode", type=str, default="image",
                       choices=["image", "video"],
                       help="Evaluation mode: 'image' generates individual images, 'video' generates evaluation videos")
    parser.add_argument("--num_videos", type=int, default=5,
                       help="Number of videos to sample (video mode only)")
    parser.add_argument("--frame_downsampling_ratio", type=int, default=10,
                       help="Frame downsampling ratio for video mode (e.g., 10 = every 10th frame)")
    parser.add_argument("--video_fps", type=int, default=3,
                       help="Frames per second for output videos (video mode only)")
    parser.add_argument("--dataset", type=str, default="egodex",
                       choices=["egodex", "robocasa"],
                       help="Dataset to load: 'egodex' for EgoDex (human) dataset or 'robocasa' for RoboCasa dataset")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\n{'='*80}")
    print(f"Evaluation mode: {args.eval_mode.upper()}")
    print(f"Dataset: {args.dataset}")
    print(f"Trajectory representation: {args.trajectory_representation}")
    print(f"Split: {args.split}")
    print(f"{'='*80}\n")
    
    # Load examples only for image mode (video mode loads its own examples)
    examples = None
    if args.eval_mode == "image":
        print(f"Loading {args.num_examples} examples from split '{args.split}' for 3D trajectory task...")
        examples = load_test_examples(args.num_examples, args.action_chunking_horizon, 
                                      args.trajectory_representation, args.split, args.dataset)
        
        if not examples:
            print("No examples loaded. Exiting.")
            return
    
    # Load model and preprocessor
    model = None
    preprocessor = None
    print("Loading model...")
    model = Molmo.from_checkpoint(args.checkpoint, device=args.device)
    model.eval()
    model.config.action_horizon = args.action_chunking_horizon
    
    print("Building preprocessor...")
    preprocessor = build_mm_preprocessor(model.config, for_inference=True, is_training=False)
    
    # Get tokenizer for decoding input tokens to text
    tokenizer = model.config.get_tokenizer()
    
    # Load camera intrinsic and stats
    intrinsic = None
    stats_mean = None
    stats_std = None
    
    # Load stats if normalization was used
    if args.normalize_coordinates:
        if args.dataset == "egodex":
            stats_file = os.environ.get("TRAJECTORY_STATS_FILE")
        elif args.dataset == "robocasa":
            stats_file = os.environ.get("ROBOCASA_STATS_FILE")
        else:
            stats_file = None
        
        if stats_file and os.path.exists(stats_file):
            print(f"Loading trajectory stats from {stats_file}")
            stats = torch.load(stats_file, map_location="cpu")
            stats_mean = stats["mean"].numpy()
            stats_std = stats["std"].numpy()
            print("Loaded mean/std for denormalization")
        else:
            env_var_name = "TRAJECTORY_STATS_FILE" if args.dataset == "egodex" else "ROBOCASA_STATS_FILE"
            print(f"WARNING: normalize_coordinates is True but {env_var_name} not set or not found!")
            print("Model predictions will NOT be denormalized (results may be incorrect)")
    
    if args.camera_intrinsic and os.path.exists(args.camera_intrinsic):
        intrinsic = np.load(args.camera_intrinsic)
        print(f"Loaded camera intrinsic from {args.camera_intrinsic}")
    else:
        if args.dataset == "egodex":
            # Use default intrinsic from EgoDex
            intrinsic = np.array([[736.6339, 0., 960.], 
                                [0., 736.6339, 540.], 
                                [0., 0., 1.]])
        elif args.dataset == "robocasa":
            intrinsic = np.array([[160.91805426, 0., 128.],
                                 [  0., 160.91805426, 128.],
                                 [  0., 0., 1.]])
        print("Using default EgoDex camera intrinsic")
    
    # Branch based on evaluation mode
    if args.eval_mode == "video":
        # VIDEO EVALUATION MODE
        print(f"\n{'='*80}")
        print(f"VIDEO EVALUATION MODE")
        print(f"Number of videos: {args.num_videos}")
        print(f"Frame downsampling ratio: {args.frame_downsampling_ratio}")
        print(f"Output FPS: {args.video_fps}")
        print(f"{'='*80}\n")
        
        # Load video examples
        video_examples = load_video_examples(
            num_videos=args.num_videos,
            action_chunking_horizon=args.action_chunking_horizon,
            trajectory_representation=args.trajectory_representation,
            split=args.split,
            frame_downsampling_ratio=args.frame_downsampling_ratio,
            dataset_type=args.dataset,
        )
        
        if not video_examples:
            print("No video examples loaded. Exiting.")
            return
        
        all_video_metrics = []
        
        # Process each video
        for vid_idx, video_data in enumerate(video_examples):
            video_path = video_data['video_path']
            task_name = video_data['task_name']
            
            print(f"\n{'='*80}")
            print(f"Processing video {vid_idx+1}/{len(video_examples)}")
            print(f"Video: {Path(video_path).name}")
            print(f"Task: {task_name}")
            print(f"Frames: {len(video_data['frames'])}")
            print(f"{'='*80}")
            
            # Evaluate all frames in the video
            start_time = time.time()
            visualized_frames, frame_metrics = evaluate_video(
                model=model,
                preprocessor=preprocessor,
                tokenizer=tokenizer,
                video_data=video_data,
                args=args,
                intrinsic=intrinsic,
                stats_mean=stats_mean,
                stats_std=stats_std,
            )
            end_time = time.time()
            
            print(f"Evaluated {len(visualized_frames)} frames in {end_time - start_time:.2f} seconds "
                  f"({(end_time - start_time) / len(visualized_frames):.2f} s/frame)")
            
            # Create output video
            sanitized_task = sanitize_filename(task_name)
            video_filename = f"eval_video_{sanitized_task}_{vid_idx+1}.mp4"
            video_output_path = str(output_dir / video_filename)
            
            success = create_video_from_frames(
                frames=visualized_frames,
                output_path=video_output_path,
                fps=args.video_fps,
            )
            
            if success:
                print(f"Saved evaluation video to {video_output_path}")
            else:
                print(f"WARNING: Failed to create video {video_output_path}")
            
            # Aggregate metrics for this video
            video_metrics = {
                "video_idx": vid_idx,
                "video_path": video_path,
                "task_name": task_name,
                "num_frames": len(visualized_frames),
                "video_output_file": video_filename if success else None,
                "frame_metrics": frame_metrics,
            }
            
            # Compute video-level aggregate metrics
            if frame_metrics and all(m.get("mse") is not None for m in frame_metrics):
                video_metrics["mean_mse"] = float(np.mean([m["mse"] for m in frame_metrics]))
                video_metrics["mean_ade"] = float(np.mean([m["ade"] for m in frame_metrics]))
                video_metrics["mean_fde"] = float(np.mean([m["fde"] for m in frame_metrics]))
                
                print(f"Video metrics - MSE: {video_metrics['mean_mse']:.6f}, "
                      f"ADE: {video_metrics['mean_ade']:.6f}, "
                      f"FDE: {video_metrics['mean_fde']:.6f}")
            
            all_video_metrics.append(video_metrics)
        
        # Save video metrics to JSON
        if args.save_metrics and all_video_metrics:
            metrics_path = output_dir / "video_evaluation_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(all_video_metrics, f, indent=2)
            print(f"\nSaved video evaluation metrics to {metrics_path}")
            
            # Print summary statistics
            videos_with_metrics = [v for v in all_video_metrics if v.get("mean_mse") is not None]
            if videos_with_metrics:
                mses = [v["mean_mse"] for v in videos_with_metrics]
                ades = [v["mean_ade"] for v in videos_with_metrics]
                fdes = [v["mean_fde"] for v in videos_with_metrics]
                
                print(f"\nSummary Statistics (across {len(videos_with_metrics)} videos):")
                print(f"  MSE - Mean: {np.mean(mses):.6f}, Std: {np.std(mses):.6f}")
                print(f"  ADE - Mean: {np.mean(ades):.6f}, Std: {np.std(ades):.6f}")
                print(f"  FDE - Mean: {np.mean(fdes):.6f}, Std: {np.std(fdes):.6f}")
        
        print(f"\n{'='*80}")
        print(f"Video evaluation complete! {len(video_examples)} videos processed.")
        print(f"Output saved to {output_dir}")
        print(f"{'='*80}")
        
        return
    
    # IMAGE EVALUATION MODE (original behavior)
    # Store evaluation metrics
    all_metrics = []
    
    # Process each example
    for idx, example_row in enumerate(examples):
        image_path = example_row["image_path"]
        prompt = example_row["prompt"]
        gt_trajectory_raw = example_row.get("ground_truth_trajectory", None)
        style = example_row.get("style", "trajectory_3d_fm")
        task_name = example_row.get("task_name", "unknown")
        
        print(f"\n{'='*80}")
        print(f"Processing example {idx+1}/{len(examples)}")
        print(f"Task: {task_name}")
        print(f"Prompt: {prompt}")
        print(f"{'='*80}")
        
        # Load and preprocess image
        image_np = load_image(image_path)
        h, w = image_np.shape[:2]
        
        pred_trajectory = None
        pred_trajectory_2d = None
        # Prepare example for model
        example = {
            "image": image_np,
            "prompt": prompt,
            "style": style,
            # Include current state and scale so DataFormatter can inject it into the prompt
            "state": example_row.get("state"),
            "point_scale": example_row.get("point_scale"),
        }
        batch = preprocessor(example)
        
        # Decode and print the text prompt (VLM input)
        input_tokens = batch["input_tokens"]
        # Get image token positions to filter them out
        image_positions = set(batch["image_input_idx"].flatten().tolist()) if "image_input_idx" in batch else set()
        # Filter out image tokens and negative/padding tokens
        text_tokens = [t for i, t in enumerate(input_tokens) if t >= 0 and i not in image_positions]
        decoded_prompt = tokenizer.decode(text_tokens)
        # print(f"\n--- VLM Text Prompt ---")
        # print(decoded_prompt)
        # print(f"--- End of Prompt ---\n")
        
        # Move tensors to device
        device = torch.device(args.device)
        input_ids = torch.tensor(batch["input_tokens"], dtype=torch.long).unsqueeze(0).to(device)
        attention_mask = (input_ids != -1).to(device)
        images = torch.tensor(batch["images"], dtype=torch.float32).unsqueeze(0).to(device)
        image_input_idx = torch.tensor(batch["image_input_idx"], dtype=torch.long).unsqueeze(0).to(device)
        image_masks = None
        if "image_masks" in batch:
            image_masks = torch.tensor(batch["image_masks"]).unsqueeze(0).to(device)
        
        # Extract position_ids from batch if available
        position_ids = None
        if "position_ids" in batch:
            position_ids = torch.tensor(batch["position_ids"], dtype=torch.long).unsqueeze(0).to(device)
        
        # Extract proprio_state from batch if available
        proprio_state = None
        if "proprio_state" in batch:
            print("Using proprioceptive state for trajectory prediction")
            proprio_state = torch.tensor(batch["proprio_state"], dtype=torch.float32).unsqueeze(0).to(device)
        
        # Extract expert_type from batch if available (for multi-expert routing)
        # Default to 0 (human expert) if not present
        expert_type = None
        if "expert_type" in batch:
            expert_type = torch.tensor([batch["expert_type"]], dtype=torch.long).to(device)
        elif hasattr(model.config, "num_action_experts") and model.config.num_action_experts > 1:
            # Multi-expert mode but no expert_type in batch - default to 0
            expert_type = torch.tensor([0], dtype=torch.long).to(device)
            print("WARNING: Multi-expert mode but no expert_type in batch. Defaulting to expert 0 (human).")
        
        # Generate trajectory
        is_direct_prediction = getattr(model.config, "use_direct_trajectory_prediction", False)
        if is_direct_prediction:
            print("Predicting trajectory directly (regression)...")
        else:
            print(f"Sampling trajectory using flow matching ODE (steps={args.num_ode_steps})...")

        # Multi-sample inference: generate multiple samples and select the best one
        num_samples = args.num_inference_samples
        if num_samples > 1 and is_direct_prediction:
            print("WARNING: Direct prediction mode does not support multi-sample inference. Using single sample.")
            num_samples = 1
        
        start_time = time.time()
        all_pred_trajectories = []
        
        # Set base seed for reproducibility if provided
        base_seed = args.inference_seed if args.inference_seed is not None else int(time.time()) + idx
        
        for sample_idx in range(num_samples):
            # Use different seed for each sample to get different initial noise
            sample_seed = base_seed + sample_idx * 12345
            torch.manual_seed(sample_seed)
            
            # Generate initial noise with the new seed
            initial_noise = torch.randn(
                1,  # batch_size
                model.config.action_horizon,
                model.config.action_dim,
                device=device
            )
            
            with torch.no_grad():
                pred_traj_sample = model.sample_actions_flow_matching(
                    input_ids=input_ids,
                    attention_mask=None,
                    images=images,
                    image_masks=image_masks,
                    image_input_idx=image_input_idx,
                    num_steps=args.num_ode_steps,
                    initial_noise=initial_noise,
                    position_ids=position_ids,
                    proprio_state=proprio_state,
                    expert_type=expert_type,
                )
            all_pred_trajectories.append(pred_traj_sample)
        
        end_time = time.time()
        
        if num_samples > 1:
            print(f"Generated {num_samples} samples in {end_time - start_time:.2f} seconds ({(end_time - start_time)/num_samples:.2f} s/sample)")
        else:
            print(f"Time taken for prediction: {end_time - start_time:.2f} seconds")
        
        # Select the best sample based on MSE with ground truth (if available and multiple samples)
        pred_trajectory = all_pred_trajectories[0]  # Default to first sample
        selected_sample_idx = 0
        sample_mses = None  # Track MSEs for all samples
        
        if num_samples > 1 and gt_trajectory_raw is not None:
            # Get ground truth in the same format for comparison
            # GT is already in absolute coordinates (not normalized, not delta)
            gt_for_comparison = gt_trajectory_raw
            if isinstance(gt_for_comparison, torch.Tensor):
                gt_for_comparison = gt_for_comparison.numpy()
            
            # Reshape GT if needed to match prediction format
            if len(gt_for_comparison.shape) == 3 and gt_for_comparison.shape[0] == 1:
                gt_for_comparison = gt_for_comparison.squeeze(0)
            if len(gt_for_comparison.shape) == 2:
                gt_action_horizon, gt_action_dim = gt_for_comparison.shape
                gt_num_joints = gt_action_dim // 3
                gt_for_comparison = gt_for_comparison.reshape(gt_action_horizon, gt_num_joints, 3)
            
            # For delta representation, convert GT to absolute for fair comparison
            # (GT from dataset is already in absolute form after delta-to-absolute conversion in the dataset)
            trajectory_rep = example_row.get("trajectory_representation", "delta")
            initial_state = example_row.get("state")
            if trajectory_rep == "delta" and initial_state is not None:
                # GT is in delta format, convert to absolute for comparison
                gt_initial_state = initial_state
                if len(gt_initial_state.shape) == 1:
                    num_joints_state = gt_initial_state.shape[0] // 3
                    gt_initial_state = gt_initial_state.reshape(num_joints_state, 3)
                gt_for_comparison = convert_delta_to_absolute(gt_for_comparison, gt_initial_state)
            
            # Compute MSE for each sample and select the best
            # We need to apply the same transformations (denormalize + delta-to-absolute) to each sample
            best_mse = float('inf')
            sample_mses = []
            
            for s_idx, pred_traj_sample in enumerate(all_pred_trajectories):
                # Convert to numpy
                if isinstance(pred_traj_sample, torch.Tensor):
                    pred_sample_np = pred_traj_sample.cpu().numpy()
                else:
                    pred_sample_np = pred_traj_sample.copy()
                
                # Denormalize if stats available (same as main code path)
                if stats_mean is not None and stats_std is not None:
                    pred_sample_np = pred_sample_np * stats_std + stats_mean
                
                # Remove batch dimension and reshape
                pred_sample_np = pred_sample_np.squeeze(0)
                sample_action_dim = pred_sample_np.shape[-1]
                sample_num_joints = sample_action_dim // 3
                pred_sample_np = pred_sample_np.reshape(-1, sample_num_joints, 3)
                
                # Convert delta to absolute if needed (same as main code path)
                if trajectory_rep == "delta" and initial_state is not None:
                    sample_initial_state = initial_state
                    if len(sample_initial_state.shape) == 1:
                        num_joints_state = sample_initial_state.shape[0] // 3
                        sample_initial_state = sample_initial_state.reshape(num_joints_state, 3)
                    pred_sample_np = convert_delta_to_absolute(pred_sample_np, sample_initial_state)
                
                # Compute MSE in the same coordinate space as GT (absolute positions)
                mse = np.mean((pred_sample_np - gt_for_comparison) ** 2)
                sample_mses.append(mse)
                
                if mse < best_mse:
                    best_mse = mse
                    best_sample_idx = s_idx
            
            selected_sample_idx = best_sample_idx
            pred_trajectory = all_pred_trajectories[selected_sample_idx]
            
            print(f"Multi-sample selection: Sample MSEs = {[f'{m:.6f}' for m in sample_mses]}")
            print(f"Selected sample {selected_sample_idx + 1}/{num_samples} with MSE = {best_mse:.6f}")
        elif num_samples > 1:
            print(f"WARNING: Multi-sample inference requested but no ground truth available. Using first sample.")
        
        # Convert to numpy
        if isinstance(pred_trajectory, torch.Tensor):
            pred_trajectory = pred_trajectory.cpu().numpy()
            
        # Denormalize if stats available (before reshaping, as stats are flattened)
        if stats_mean is not None and stats_std is not None:
            # pred_trajectory is (1, action_horizon, action_dim) or (batch_size, action_horizon, action_dim)
            # stats are (action_dim,)
            # Broadcast: (..., action_dim) * (action_dim,) + (action_dim,)
            pred_trajectory = pred_trajectory * stats_std + stats_mean
            print("Denormalized predicted trajectory")
                
        # Reshape from (batch_size, action_horizon, action_dim) to (action_horizon, num_joints, 3)
        # Remove batch dimension and reshape flattened coordinates
        batch_size, action_horizon, action_dim = pred_trajectory.shape
        pred_trajectory = pred_trajectory.squeeze(0)  # Remove batch dimension: (action_horizon, action_dim)
        
        # For 3D: action_dim = num_joints * 3
        num_joints = action_dim // 3
        if action_dim % 3 != 0:
            raise ValueError(f"action_dim ({action_dim}) must be divisible by 3 for 3D trajectories")
        pred_trajectory = pred_trajectory.reshape(action_horizon, num_joints, 3)
                
        # Convert delta predictions to absolute positions if needed
        trajectory_rep = example_row.get("trajectory_representation", "absolute")
        if trajectory_rep == "delta":
            initial_state = example_row.get("state")
            if initial_state is not None:
                print(f"Converting delta predictions to absolute positions using initial state...")
                print(f"Initial state shape: {initial_state.shape}")
                
                # Ensure initial_state has the right shape: (num_joints, 3)
                if len(initial_state.shape) == 1:
                    # Flatten format, reshape to (num_joints, 3)
                    num_joints_state = initial_state.shape[0] // 3
                    initial_state = initial_state.reshape(num_joints_state, 3)
                
                # Convert delta to absolute
                pred_trajectory = convert_delta_to_absolute(pred_trajectory, initial_state)
            else:
                print("WARNING: Delta mode but no initial state found! Treating as absolute positions.")
                
        # Process ground truth trajectory
        gt_trajectory_2d = None
        metrics = {
            "example_idx": idx,
            "task_name": task_name,
            "prompt": prompt,
            "trajectory_representation": trajectory_rep,
            "num_inference_samples": num_samples,
            "selected_sample_idx": selected_sample_idx,
        }
        
        # Add all sample MSEs if multi-sample was used
        if sample_mses is not None:
            metrics["all_sample_mses"] = sample_mses
        
        if gt_trajectory_raw is not None:
            if isinstance(gt_trajectory_raw, torch.Tensor):
                gt_trajectory_raw = gt_trajectory_raw.numpy()
                        
            # Reshape GT trajectory if needed (it might also be flattened)
            if len(gt_trajectory_raw.shape) == 3 and gt_trajectory_raw.shape[0] == 1:
                # Remove batch dimension
                gt_trajectory_raw = gt_trajectory_raw.squeeze(0)
            
            # Check if GT needs reshaping
            if len(gt_trajectory_raw.shape) == 2:
                # Flattened format: (action_horizon, action_dim)
                gt_action_horizon, gt_action_dim = gt_trajectory_raw.shape
                gt_num_joints = gt_action_dim // 3
                if gt_action_dim % 3 != 0:
                    raise ValueError(f"GT action_dim ({gt_action_dim}) must be divisible by 3 for 3D trajectories")
                gt_trajectory_raw = gt_trajectory_raw.reshape(gt_action_horizon, gt_num_joints, 3)
            
            # Convert ground truth from delta to absolute if needed
            if trajectory_rep == "delta":
                initial_state = example_row.get("state")
                if initial_state is not None:
                    print(f"Converting GT delta trajectory to absolute positions...")
                    
                    # Ensure initial_state has the right shape: (num_joints, 3)
                    if len(initial_state.shape) == 1:
                        # Flatten format, reshape to (num_joints, 3)
                        num_joints_state = initial_state.shape[0] // 3
                        initial_state = initial_state.reshape(num_joints_state, 3)
                    
                    # Convert delta to absolute
                    gt_trajectory_raw = convert_delta_to_absolute(gt_trajectory_raw, initial_state)
                    print(f"Converted GT to absolute trajectory. Shape: {gt_trajectory_raw.shape}")
            
            # Project 3D ground truth to 2D for visualization
            gt_trajectory_2d = project_3d_trajectory_to_2d(
                gt_trajectory_raw, intrinsic, w, h, normalize=True
            )
            print(f"Projected GT trajectory to 2D: {gt_trajectory_2d.shape}")
        
        # Project 3D prediction to 2D for visualization
        pred_trajectory_2d = project_3d_trajectory_to_2d(
            pred_trajectory, intrinsic, w, h, normalize=True
        )
        print(f"Projected prediction trajectory to 2D: {pred_trajectory_2d.shape}")
        
        # Compute metrics if requested (after both trajectories are in 2D)
        if gt_trajectory_2d is not None and args.save_metrics:
            # MSE
            mse = np.mean((pred_trajectory_2d - gt_trajectory_2d) ** 2)
            metrics["mse"] = float(mse)
            
            # ADE (Average Displacement Error)
            ade = np.mean(np.linalg.norm(pred_trajectory_2d - gt_trajectory_2d, axis=-1))
            metrics["ade"] = float(ade)
            
            # FDE (Final Displacement Error)
            fde = np.linalg.norm(pred_trajectory_2d[-1] - gt_trajectory_2d[-1], axis=-1).mean()
            metrics["fde"] = float(fde)
            
            print(f"MSE: {mse:.6f}")
            print(f"ADE: {ade:.6f}")
            print(f"FDE: {fde:.6f}")
        
        # Visualize trajectories
        print("Creating visualization...")
        visualized_image = visualize_trajectory_on_image(
            image_np, 
            pred_trajectory=pred_trajectory_2d,
            gt_trajectory=gt_trajectory_2d,
            is_normalized=True,
            prompt=prompt
        )
        
        # Save visualization
        sanitized_task = sanitize_filename(task_name)
        sanitized_prompt = sanitize_filename(prompt[:50])
        suffix = "_with_gt" if gt_trajectory_2d is not None else ""
        mode_tag = "direct" if is_direct_prediction else "fm"
        out_filename = f"3d_traj_{mode_tag}_{sanitized_task}_{idx+1}{suffix}.jpg"
        out_path = output_dir / out_filename
        
        visualized_image.save(out_path)
        print(f"Saved visualization to {out_path}")
        
        # Save 3D trajectory data if requested
        if args.save_3d_trajectory:
            # Save both predicted and ground truth 3D trajectories
            traj_filename = f"3d_traj_{mode_tag}_{sanitized_task}_{idx+1}.npz"
            traj_path = output_dir / traj_filename
            
            save_data = {
                'predicted_trajectory_3d': pred_trajectory,  # Shape: (num_steps, num_joints, 3)
                'task_name': task_name,
                'prompt': prompt,
                'trajectory_representation': trajectory_rep,
                'num_steps': pred_trajectory.shape[0],
                'num_joints': pred_trajectory.shape[1],
            }
            
            # Add ground truth if available
            if gt_trajectory_raw is not None:
                save_data['ground_truth_trajectory_3d'] = gt_trajectory_raw
            
            # Add initial state if available (for delta representation)
            if example_row.get("state") is not None:
                initial_state = example_row.get("state")
                if len(initial_state.shape) == 1:
                    # Reshape to (num_joints, 3)
                    num_joints_state = initial_state.shape[0] // 3
                    initial_state = initial_state.reshape(num_joints_state, 3)
                save_data['initial_state'] = initial_state
            
            # Add camera intrinsic if available
            if intrinsic is not None:
                save_data['camera_intrinsic'] = intrinsic
            
            np.savez(traj_path, **save_data)
            print(f"Saved 3D trajectory data to {traj_path}")
            metrics["trajectory_3d_file"] = traj_filename
        
        # Store metrics
        metrics["output_filename"] = out_filename
        all_metrics.append(metrics)
        
        # Clean up temporary image
        if os.path.exists(image_path) and example_row.get("is_test_data", False):
            try:
                os.remove(image_path)
            except:
                pass
    
    # Save metrics to JSON if requested
    if args.save_metrics and all_metrics:
        metrics_path = output_dir / "evaluation_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\nSaved evaluation metrics to {metrics_path}")
        
        # Print summary statistics
        if all([m.get("mse") is not None for m in all_metrics]):
            mses = [m["mse"] for m in all_metrics]
            ades = [m["ade"] for m in all_metrics]
            fdes = [m["fde"] for m in all_metrics]
            
            print(f"\nSummary Statistics:")
            print(f"  MSE - Mean: {np.mean(mses):.6f}, Std: {np.std(mses):.6f}")
            print(f"  ADE - Mean: {np.mean(ades):.6f}, Std: {np.std(ades):.6f}")
            print(f"  FDE - Mean: {np.mean(fdes):.6f}, Std: {np.std(fdes):.6f}")
    
    print(f"\n{'='*80}")
    print(f"Evaluation complete! Visualizations saved to {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
