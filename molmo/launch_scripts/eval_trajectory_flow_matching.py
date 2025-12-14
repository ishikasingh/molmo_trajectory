#!/usr/bin/env python3
"""
Evaluation script for 3D trajectory predictions.
Supports both flow matching (ODE sampling) and direct regression modes.
Trajectories are projected to 2D for visualization on images.
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

from olmo.model import Molmo
from olmo.data.model_preprocessor import load_image
from olmo.data import build_mm_preprocessor
from olmo.config import ModelConfig


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
    font_size = max(16, int(image_height/40))
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
    legend_font_size = max(12, int(image_height/50))
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
                       split: str = "test") -> List[Dict]:
    """
    Load examples from the specified split for 3D trajectory prediction.
    
    Args:
        num_examples: Number of examples to load
        action_chunking_horizon: Number of timesteps in trajectory
        trajectory_representation: Either 'absolute' or 'delta' for trajectory representation
        split: Dataset split to load (e.g., train, test)
    
    Returns:
        List of example dictionaries
    """
    from olmo.data.trajectory_datasets import TrajectoryDataset
    
    # Check if stats file is available for normalization
    stats_file = os.environ.get("TRAJECTORY_STATS_FILE")
    normalize_coords = bool(stats_file)
    
    # Load dataset (always 3D)
    dataset = TrajectoryDataset(
        split=split,
        action_chunking_horizon=action_chunking_horizon,
        output_2d_trajectory=False,
        normalize_coordinates=False, # Set to be False as in inference time, the trajectory itself is only for visualization, but this only works for delta representation.
        output_format="flow_matching",  # Use flow matching format
        trajectory_representation=trajectory_representation,
        frame_downsampling_ratio=10,
    )
    
    print(f"Loaded '{split}' dataset with {len(dataset)} examples")
    
    # Sample examples (using evenly spaced indices for diversity)
    indices = np.linspace(0, len(dataset) - 1, min(num_examples, len(dataset)), dtype=int)
    
    examples = []
    for idx in indices:
        example_data = dataset.get(idx, rng=np.random.RandomState(42))
        
        # Extract data
        image = example_data["image"]
        message_list = example_data["message_list"]
        metadata = example_data.get("metadata", {})
        
        if message_list and len(message_list) > 0:
            instruction = message_list[0].get("label", "")
            gt_trajectory = message_list[0].get("points", None)  # For flow matching format
            point_scale = message_list[0].get("point_scale", None)
            style = message_list[0].get("style", "")
            state = message_list[0].get("state", None)
            
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


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate 3D trajectory predictions (flow matching or direct regression)"
    )
    parser.add_argument("checkpoint", type=str, help="Path to model checkpoint directory")
    parser.add_argument("--num_examples", type=int, default=10,
                       help="Number of test examples to visualize")
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
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load examples
    print(f"Loading {args.num_examples} examples from split '{args.split}' for 3D trajectory task...")
    print(f"Trajectory representation: {args.trajectory_representation}")
    examples = load_test_examples(args.num_examples, args.action_chunking_horizon, 
                                  args.trajectory_representation, args.split)
    
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
        stats_file = os.environ.get("TRAJECTORY_STATS_FILE")
        if stats_file and os.path.exists(stats_file):
            print(f"Loading trajectory stats from {stats_file}")
            stats = torch.load(stats_file, map_location="cpu")
            stats_mean = stats["mean"].numpy()
            stats_std = stats["std"].numpy()
            print("Loaded mean/std for denormalization")
        else:
            print("WARNING: normalize_coordinates is True but TRAJECTORY_STATS_FILE not set or not found!")
            print("Model predictions will NOT be denormalized (results may be incorrect)")
    
    if args.camera_intrinsic and os.path.exists(args.camera_intrinsic):
        intrinsic = np.load(args.camera_intrinsic)
        print(f"Loaded camera intrinsic from {args.camera_intrinsic}")
    else:
        # Use default intrinsic from EgoDex
        intrinsic = np.array([[736.6339, 0., 960.], 
                             [0., 736.6339, 540.], 
                             [0., 0., 1.]])
        print("Using default EgoDex camera intrinsic")
    
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
