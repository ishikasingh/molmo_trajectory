#!/usr/bin/env python3
"""
Closed-Loop Evaluation Script for Robot Action Prediction in RoboCasa Simulation.

This script performs closed-loop evaluation of a VLA (Vision-Language-Action) model
in the RoboCasa simulation environment. It:
1. Loads episodes from the original RoboCasa dataset (HDF5 files)
2. Creates environments from exact dataset configuration (env_args from data.attrs)
3. Resets to the exact initial state from each dataset episode (model_file XML and initial state)
4. Uses language commands from the dataset episodes
5. Queries the policy for actions based on rendered observations
6. Executes the first step of the action chunking output
7. Repeats until done or max steps reached
8. Records videos and computes success metrics

The environment is created from the exact configuration stored in the original dataset
(f["data"].attrs["env_args"]), and reset to the exact initial state (model_file XML
and initial simulator state). This ensures evaluation matches the original dataset conditions.

Dataset Structure (original RoboCasa dataset):
    dataset_path/
        TaskName1.hdf5
        TaskName2.hdf5
        ...

Each HDF5 file contains:
    - data.attrs["env_args"]: Environment configuration JSON
    - data/{episode_name}/:
        - states: (num_frames, state_dim) - Robot states at each frame
        - actions: (num_frames, action_dim) - Robot actions at each frame
        - attrs["model_file"]: MuJoCo XML string for the scene
        - attrs["ep_meta"]: Episode metadata JSON (contains language command)

Usage:
    python eval_closed_loop.py <checkpoint_path> --dataset_path <dataset_path> --output_dir <output_dir> [options]

Example:
    python eval_closed_loop.py /path/to/checkpoint --dataset_path /path/to/datasets --output_dir closed_loop_eval_results --num_tasks 10
"""

import argparse
import cv2
import json
import os
import sys
import time
import random
import numpy as np
import torch
import h5py
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image
from tqdm import tqdm
from dataclasses import dataclass, field
# RoboCasa/Robosuite imports (required for closed-loop evaluation)
import robocasa
import robosuite

# Model imports
from olmo.model import Molmo
from olmo.data import build_mm_preprocessor


@dataclass
class EvalConfig:
    """Configuration for closed-loop evaluation."""
    checkpoint: str
    dataset_path: str  # Path to original RoboCasa dataset (HDF5 file or directory)
    output_dir: str = "closed_loop_eval_results"
    num_tasks: int = 10
    max_steps: int = 300  # Maximum steps per episode
    camera_name: str = "egoview"
    render_height: int = 256
    render_width: int = 256
    device: str = "cuda"
    seed: int = 42
    action_horizon: int = 30
    num_ode_steps: int = 10
    steps_per_chunk: int = 1  # Number of steps to execute from each predicted action chunk before predicting next chunk
    video_fps: int = 20
    normalize_actions: bool = True
    trajectory_representation: str = "delta"
    use_robot_actions: bool = True  # If True, output robot joint actions; if False, output fingertip trajectories
    max_episodes: Optional[int] = None  # Maximum episodes per dataset file
    filter_key: Optional[str] = None  # Filter key to select subset of episodes
    debug: bool = False  # If True, skip model loading and use random actions


@dataclass
class EpisodeResult:
    """Results from a single episode evaluation."""
    task_name: str
    episode_idx: int
    success: bool
    num_steps: int
    total_reward: float
    video_path: str
    language_command: str
    errors: List[str] = field(default_factory=list)


def get_keypoint_site_names() -> List[str]:
    """Get the list of keypoint site names for the GR1 robot (10 fingertips only)."""
    return [
        # Right fingertips
        "gripper0_right_site_R_thumb_distal_link",
        "gripper0_right_site_R_index_intermediate_link",
        "gripper0_right_site_R_middle_intermediate_link",
        "gripper0_right_site_R_ring_intermediate_link",
        "gripper0_right_site_R_pinky_intermediate_link",
        # Left fingertips
        "gripper0_left_site_L_thumb_distal_link",
        "gripper0_left_site_L_index_intermediate_link",
        "gripper0_left_site_L_middle_intermediate_link",
        "gripper0_left_site_L_ring_intermediate_link",
        "gripper0_left_site_L_pinky_intermediate_link",
        # Wrists (commented out to match training)
        # "robot0_r_wrist_site",
        # "robot0_l_wrist_site",
    ]


# Fingertip offset distances (in meters)
FINGERTIP_OFFSETS = {
    "gripper0_right_site_R_thumb_distal_link": 0.035,
    "gripper0_left_site_L_thumb_distal_link": 0.035,
    "gripper0_right_site_R_index_intermediate_link": 0.032,
    "gripper0_right_site_R_middle_intermediate_link": 0.032,
    "gripper0_right_site_R_ring_intermediate_link": 0.032,
    "gripper0_right_site_R_pinky_intermediate_link": 0.031,
    "gripper0_left_site_L_index_intermediate_link": 0.032,
    "gripper0_left_site_L_middle_intermediate_link": 0.032,
    "gripper0_left_site_L_ring_intermediate_link": 0.032,
    "gripper0_left_site_L_pinky_intermediate_link": 0.031,
    "robot0_r_wrist_site": 0.0,
    "robot0_l_wrist_site": 0.0,
}


def get_keypoint_positions_flat(sim, apply_fingertip_offset: bool = True) -> np.ndarray:
    """
    Get flattened array of keypoint positions (3D coordinates only).
    
    Args:
        sim: MuJoCo simulation object
        apply_fingertip_offset: Whether to apply fingertip offset
        
    Returns:
        np.ndarray: Flattened array of shape (num_keypoints * 3,)
    """
    positions = []
    for site_name in get_keypoint_site_names():
        try:
            site_id = sim.model.site_name2id(site_name)
            site_pos = sim.data.site_xpos[site_id].copy()
            
            if apply_fingertip_offset and site_name in FINGERTIP_OFFSETS:
                offset_distance = FINGERTIP_OFFSETS[site_name]
                if offset_distance > 0:
                    site_rot = sim.data.site_xmat[site_id].reshape(3, 3)
                    local_x_axis = site_rot[:, 0]
                    site_pos = site_pos + offset_distance * local_x_axis
            
            positions.append(site_pos)
        except Exception as e:
            print(f"Warning: Could not get position for site '{site_name}': {e}")
            positions.append(np.zeros(3))
    
    return np.concatenate(positions)


class VideoWriter:
    """Wrapper for cv2.VideoWriter with automatic file management."""
    
    def __init__(self, filepath: str, fps: int = 20, width: int = 256, height: int = 256):
        self.filepath = filepath
        self.fps = fps
        self.width = width
        self.height = height
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
        self.frame_count = 0
        
    def append_frame(self, frame: np.ndarray):
        """Append a frame to the video."""
        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            frame = cv2.resize(frame, (self.width, self.height))
        
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.writer.write(frame_bgr)
        self.frame_count += 1
        
    def close(self):
        """Release the video writer."""
        self.writer.release()


def list_available_task_names() -> List[str]:
    """
    List all available RoboCasa task names that can be used with robocasa.make().
    
    Returns:
        List of available task/environment names
    """
    from robosuite.environments import ALL_ENVIRONMENTS
    
    # Filter to only RoboCasa tabletop environments
    robocasa_tasks = [
        name for name in ALL_ENVIRONMENTS.keys()
        if name.startswith(('PnP', 'Tabletop', 'PutAll'))
    ]
    
    return sorted(robocasa_tasks)


def make_ik_indicator_invisible(str_xml: str) -> str:
    """Make IK indicators invisible in the XML."""
    import xml.etree.ElementTree as ET
    
    raw_xml = ET.fromstring(str_xml)
    for site in raw_xml.findall(".//site"):
        name = site.get("name", "")
        if "pinch_spheres" in name:
            site.set("rgba", "0 0 0 0")
    return ET.tostring(raw_xml, encoding="unicode")


def make_env_from_metadata(env_meta: Dict[str, Any], use_abs_actions: bool = False) -> Any:
    """Create environment from metadata."""
    if use_abs_actions:
        env_meta["env_kwargs"]["controller_configs"]["control_delta"] = False
    
    env_kwargs = env_meta["env_kwargs"].copy()
    env_kwargs["env_name"] = env_meta["env_name"]
    env_kwargs["has_renderer"] = False
    env_kwargs["renderer"] = "mjviewer"
    env_kwargs["has_offscreen_renderer"] = True
    env_kwargs["use_camera_obs"] = False
    
    if "env_lang" in env_kwargs:
        env_kwargs.pop("env_lang")
    
    env = robosuite.make(**env_kwargs)
    return env


def reset_to(env, state: Dict[str, Any]) -> None:
    """
    Reset environment to a specific simulator state.
    
    Args:
        env: Environment instance
        state: Dictionary containing state information with keys:
            - "model": MuJoCo XML string (optional)
            - "states": Flattened simulator state (required)
            - "ep_meta": Episode metadata JSON string (optional)
    """
    if "model" in state and state["model"] is not None:
        if state.get("ep_meta", None) is not None:
            if isinstance(state["ep_meta"], str):
                ep_meta = json.loads(state["ep_meta"])
            else:
                ep_meta = state["ep_meta"]
        else:
            ep_meta = {}
        
        if hasattr(env, "set_attrs_from_ep_meta"):
            env.set_attrs_from_ep_meta(ep_meta)
        elif hasattr(env, "set_ep_meta"):
            env.set_ep_meta(ep_meta)
        
        env.reset()
        robosuite_version_id = int(robosuite.__version__.split(".")[1])
        if robosuite_version_id <= 3:
            from robosuite.utils.mjcf_utils import postprocess_model_xml
            xml = postprocess_model_xml(state["model"])
        else:
            xml = env.edit_model_xml(state["model"])
        
        env.reset_from_xml_string(xml)
        env.sim.reset()
    
    if "states" in state:
        env.sim.set_state_from_flattened(state["states"])
        env.sim.forward()
    
    if hasattr(env, "update_sites"):
        env.update_sites()
    if hasattr(env, "update_state"):
        env.update_state()


def find_datasets_and_episodes(
    dataset_path: str,
    max_episodes: Optional[int] = None,
    filter_key: Optional[str] = None,
) -> List[Tuple[str, str, str]]:
    """
    Find all dataset files and their episodes from the original RoboCasa dataset.
    
    Args:
        dataset_path: Path to dataset file or directory containing dataset files
        max_episodes: Maximum episodes per dataset file
        filter_key: Filter key to select subset of episodes
        
    Returns:
        List of (dataset_path, task_name, episode_name) tuples
    """
    dataset_path = os.path.expanduser(dataset_path)
    
    # Collect all dataset files
    dataset_files = []
    if os.path.isfile(dataset_path):
        dataset_files.append(dataset_path)
    elif os.path.isdir(dataset_path):
        hdf5_files = sorted([
            os.path.join(dataset_path, f) 
            for f in os.listdir(dataset_path)
            if f.endswith('.hdf5') or f.endswith('.h5')
        ])
        dataset_files.extend(hdf5_files)
    else:
        raise ValueError(f"Dataset path does not exist: {dataset_path}")
    
    if not dataset_files:
        raise ValueError(f"No HDF5 files found at {dataset_path}")
    
    # Collect episodes from all dataset files
    all_episodes = []
    for dataset_file in dataset_files:
        task_name = Path(dataset_file).stem
        
        with h5py.File(dataset_file, 'r') as f:
            # Get episode list (same as update_dataset_with_keypoints.py)
            if filter_key is not None:
                demos = [
                    elem.decode("utf-8") if isinstance(elem, bytes) else elem
                    for elem in np.array(f[f"mask/{filter_key}"])
                ]
            else:
                demos = list(f["data"].keys())
            
            # Sort demos
            inds = np.argsort([int(elem.split("_")[-1]) for elem in demos])
            demos = [demos[i] for i in inds]
            
            if max_episodes is not None:
                demos = demos[:max_episodes]
            
            # Add all episodes from this dataset
            for ep_name in demos:
                all_episodes.append((dataset_file, task_name, ep_name))
    
    return all_episodes


def sample_episodes(
    episodes: List[Tuple[str, str, str]],
    num_tasks: int,
    seed: int = 42,
) -> List[Tuple[str, str, str]]:
    """
    Randomly sample episodes, ensuring one episode per task.
    
    Args:
        episodes: List of (dataset_path, task_name, episode_name) tuples
        num_tasks: Number of tasks to sample
        seed: Random seed
        
    Returns:
        List of sampled (dataset_path, task_name, episode_name) tuples
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Group episodes by task
    task_episodes = {}
    for dataset_path, task_name, ep_name in episodes:
        if task_name not in task_episodes:
            task_episodes[task_name] = []
        task_episodes[task_name].append((dataset_path, task_name, ep_name))
    
    task_names = list(task_episodes.keys())
    
    if num_tasks > len(task_names):
        print(f"Warning: Requested {num_tasks} tasks but only {len(task_names)} available. Using all.")
        num_tasks = len(task_names)
    
    sampled_tasks = random.sample(task_names, num_tasks)
    
    samples = []
    for task_name in sampled_tasks:
        # Randomly pick one episode from this task
        episode = random.choice(task_episodes[task_name])
        samples.append(episode)
    
    return samples


def get_env_metadata_from_dataset(dataset_path: str) -> Dict[str, Any]:
    """
    Retrieves env metadata from original dataset.
    
    Args:
        dataset_path: Path to original HDF5 dataset
        
    Returns:
        Dict containing environment metadata
    """
    dataset_path = os.path.expanduser(dataset_path)
    with h5py.File(dataset_path, "r") as f:
        env_meta = json.loads(f["data"].attrs["env_args"])
    return env_meta


def load_episode_data_from_dataset(
    dataset_path: str,
    episode_name: str
) -> Dict[str, Any]:
    """
    Load episode data from the original RoboCasa dataset.
    
    Args:
        dataset_path: Path to the original dataset HDF5 file
        episode_name: Episode name in the dataset (e.g., 'demo_1')
        
    Returns:
        Dictionary containing episode data for environment reset
    """
    dataset_path = os.path.expanduser(dataset_path)
    
    with h5py.File(dataset_path, 'r') as f:
        # Get episode data (same as process_episode in update_dataset_with_keypoints.py)
        demo_grp = f[f"data/{episode_name}"]
        
        # Get episode metadata
        ep_meta_str = demo_grp.attrs.get("ep_meta", None)
        if ep_meta_str:
            ep_meta = json.loads(ep_meta_str)
            language_command = ep_meta.get("lang", "")
        else:
            ep_meta = {}
            language_command = ""
        
        # Get model XML
        model_file = demo_grp.attrs.get("model_file", None)
        if model_file:
            model_file = make_ik_indicator_invisible(model_file)
        
        # Get initial state
        states = demo_grp["states"][()]
        initial_state = states[0]
    
    return {
        'model_file': model_file,
        'ep_meta': ep_meta_str,
        'initial_state': initial_state,
        'language_command': language_command,
    }


class ClosedLoopEvaluator:
    """Closed-loop evaluator for VLA models in RoboCasa."""
    
    def __init__(self, config: EvalConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Set random seeds
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        
        # Load model (skip in debug mode)
        if config.debug:
            print("DEBUG MODE: Skipping model loading, will use random actions")
            self.model = None
            self.preprocessor = None
        else:
            print(f"Loading model from {config.checkpoint}...")
            self.model = Molmo.from_checkpoint(config.checkpoint, device=config.device)
            self.model.eval()
            self.model.config.action_horizon = config.action_horizon
            
            # Build preprocessor
            print("Building preprocessor...")
            self.preprocessor = build_mm_preprocessor(
                self.model.config, for_inference=True, is_training=False
            )
        
        # Load action normalization stats if needed
        self.action_stats_mean = None
        self.action_stats_std = None
        if config.normalize_actions:
            stats_file = os.environ.get("ROBOCASA_STATS_FILE")
            if stats_file and os.path.exists(stats_file):
                print(f"Loading action normalization stats from {stats_file}")
                stats = torch.load(stats_file, map_location="cpu")
                
                # Check which stats to use based on config
                if config.use_robot_actions:
                    # Look for robot action stats
                    self.action_stats_mean = stats['robot_action_stats']['mean'].numpy()
                    self.action_stats_std = stats['robot_action_stats']['std'].numpy()
                    print(f"Using robot action stats: mean shape {self.action_stats_mean.shape}")
                else:
                    self.action_stats_mean = stats['mean'].numpy()
                    self.action_stats_std = stats['std'].numpy()
                    print(f"Using trajectory stats: mean shape {self.action_stats_mean.shape}")
                
                if self.action_stats_mean is None:
                    print("Warning: Could not find appropriate stats in stats file")
            else:
                raise ValueError("ROBOCASA_STATS_FILE not set, actions will not be denormalized")
        
        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Store dataset path
        self.dataset_path = config.dataset_path
        self.max_episodes = config.max_episodes
        self.filter_key = config.filter_key
        
        print(f"Using dataset: {self.dataset_path}")
    
    def preprocess_observation(
        self, 
        image: np.ndarray, 
        instruction: str,
        proprio_state: np.ndarray,
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocess observation for model input.
        
        Args:
            image: RGB image array (H, W, 3)
            instruction: Language instruction
            proprio_state: Current proprioceptive state (flattened keypoint positions)
            
        Returns:
            Dictionary of tensors ready for model forward pass
        """
        # Convert to PIL Image
        pil_image = Image.fromarray(image)
        
        # Prepare example for preprocessor
        example = {
            "image": pil_image,
            "prompt": instruction,
            "style": "trajectory_3d_fm",  # Use flow matching style
            "state": proprio_state,
        }
        
        # Run preprocessor
        batch = self.preprocessor(example)
        
        # Convert to tensors and move to device
        input_ids = torch.tensor(batch["input_tokens"], dtype=torch.long).unsqueeze(0).to(self.device)
        images = torch.tensor(batch["images"], dtype=torch.float32).unsqueeze(0).to(self.device)
        image_input_idx = torch.tensor(batch["image_input_idx"], dtype=torch.long).unsqueeze(0).to(self.device)
        
        image_masks = None
        if "image_masks" in batch:
            image_masks = torch.tensor(batch["image_masks"]).unsqueeze(0).to(self.device)
        
        position_ids = None
        if "position_ids" in batch:
            position_ids = torch.tensor(batch["position_ids"], dtype=torch.long).unsqueeze(0).to(self.device)
        
        proprio_state_tensor = None
        if "proprio_state" in batch:
            proprio_state_tensor = torch.tensor(batch["proprio_state"], dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Expert type = 1 for robot
        expert_type = torch.tensor([1], dtype=torch.long).to(self.device)
        
        return {
            "input_ids": input_ids,
            "images": images,
            "image_input_idx": image_input_idx,
            "image_masks": image_masks,
            "position_ids": position_ids,
            "proprio_state": proprio_state_tensor,
            "expert_type": expert_type,
        }
    
    def predict_actions(self, obs_dict: Dict[str, torch.Tensor]) -> np.ndarray:
        """
        Query the model for action predictions.
        
        Args:
            obs_dict: Preprocessed observation dictionary
            
        Returns:
            Action array of shape (action_horizon, action_dim)
        """
        # Debug mode: return random actions
        if self.config.debug:
            # Use a reasonable default action dimension (e.g., 14 for GR1 robot)
            # This can be adjusted based on your robot configuration
            action_dim = 24  # Default for GR1 robot actions
            actions_np = np.random.randn(self.config.action_horizon, action_dim).astype(np.float32)
            return actions_np
        
        with torch.no_grad():
            # Use the model's configured action_dim, as in training
            action_dim = self.model.config.action_dim
            
            # Generate initial noise
            initial_noise = torch.randn(
                1,
                self.model.config.action_horizon,
                action_dim,
                device=self.device
            )
            
            # Sample actions using flow matching
            actions = self.model.sample_actions_flow_matching(
                input_ids=obs_dict["input_ids"],
                attention_mask=None,
                images=obs_dict["images"],
                image_masks=obs_dict.get("image_masks"),
                image_input_idx=obs_dict["image_input_idx"],
                num_steps=self.config.num_ode_steps,
                initial_noise=initial_noise,
                position_ids=obs_dict.get("position_ids"),
                proprio_state=obs_dict.get("proprio_state"),
                expert_type=obs_dict.get("expert_type"),
            )
            
            # Check if sequential expert mode returns a tuple (Expert A output, Expert B robot actions)
            # In sequential mode, we want Expert B's robot actions (the second element)
            action_expert_mode = getattr(self.model.config, 'action_expert_mode', 'shared')
            if action_expert_mode == 'sequential' and isinstance(actions, tuple):
                # Sequential mode: (fingertip_trajectory, robot_actions)
                # We want the robot actions from Expert B (second element)
                actions = actions[1]
        
        # Convert to numpy
        actions_np = actions.cpu().numpy()[0]  # (action_horizon, action_dim)

        # Match training-time robot expert setup: slice to the effective robot dim
        if self.config.use_robot_actions:
            # Decide whether robot expert was trained on joint actions or trajectories
            robot_use_trajectory = getattr(self.model.config, "robot_use_trajectory_as_action", True)
            robot_action_dim = getattr(self.model.config, "robot_action_dim", None)
            robot_trajectory_dim = getattr(self.model.config, "robot_trajectory_dim", None)

            if not robot_use_trajectory and robot_action_dim is not None:
                # Joint actions: keep the supervised robot_action_dim subset
                actions_np = actions_np[..., :robot_action_dim]
            elif robot_use_trajectory and robot_trajectory_dim is not None:
                # Fingertip trajectories: keep the supervised robot_trajectory_dim subset
                actions_np = actions_np[..., :robot_trajectory_dim]
        
        # Denormalize if stats available
        if self.action_stats_mean is not None and self.action_stats_std is not None:
            # Reshape stats if needed
            if actions_np.shape[-1] == self.action_stats_mean.size:
                mean = self.action_stats_mean.reshape(1, -1)
                std = self.action_stats_std.reshape(1, -1)
                actions_np = actions_np * std + mean
            else:
                print(
                    f"Warning: Action dim mismatch. Model output: {actions_np.shape[-1]}, "
                    f"stats: {self.action_stats_mean.size}"
                )
        
        return actions_np
    
    def run_episode(
        self, 
        dataset_path: str,
        task_name: str,
        episode_name: str,
        episode_idx: int,
        env: Any,  # Environment instance (reused across episodes from same dataset)
    ) -> EpisodeResult:
        """
        Run a single closed-loop evaluation episode.
        
        Uses the environment created from the dataset and resets to the exact
        initial state from the original dataset episode.
        
        Args:
            dataset_path: Path to the original dataset HDF5 file
            task_name: Name of the task
            episode_name: Episode name in the dataset (e.g., 'demo_1')
            episode_idx: Index of this episode in the evaluation
            env: Environment instance (reused for episodes from same dataset)
            
        Returns:
            EpisodeResult with evaluation metrics
        """
        errors = []
        
        # Load episode data from original dataset
        try:
            episode_data = load_episode_data_from_dataset(dataset_path, episode_name)
        except Exception as e:
            return EpisodeResult(
                task_name=task_name,
                episode_idx=episode_idx,
                success=False,
                num_steps=0,
                total_reward=0.0,
                video_path="",
                language_command="",
                errors=[f"Failed to load episode data: {e}"],
            )
        
        language_command = episode_data['language_command']
        print(f"Language command: {language_command}")
        
        # Reset to exact initial state from dataset (same as process_episode)
        try:
            initial_state = {
                "states": episode_data['initial_state'],
                "model": episode_data['model_file'],
                "ep_meta": episode_data['ep_meta'],
            }
            reset_to(env, initial_state)
        except Exception as e:
            return EpisodeResult(
                task_name=task_name,
                episode_idx=episode_idx,
                success=False,
                num_steps=0,
                total_reward=0.0,
                video_path="",
                language_command=language_command,
                errors=[f"Failed to reset environment to exact state: {e}"],
            )
        
        # Create video writer
        video_filename = f"{task_name}_ep{episode_idx}.mp4"
        video_path = self.output_dir / video_filename
        video_writer = VideoWriter(
            str(video_path), 
            fps=self.config.video_fps,
            width=self.config.render_width,
            height=self.config.render_height,
        )
        
        # Run closed-loop evaluation
        total_reward = 0.0
        done = False
        step = 0
        success = False
        
        # Track current action chunk and position within it
        current_action_chunk = None
        chunk_step_idx = 0

        # Progress bar for this episode
        pbar = tqdm(
            total=self.config.max_steps,
            desc=f"Episode {episode_idx} ({task_name})",
            leave=False,
        )
        
        try:
            while not done and step < self.config.max_steps:
                # Check if we need to predict a new action chunk
                if current_action_chunk is None or chunk_step_idx >= self.config.steps_per_chunk:
                    # Render current observation
                    frame = env.sim.render(
                        height=self.config.render_height,
                        width=self.config.render_width,
                        camera_name=self.config.camera_name,
                    )[::-1]  # Flip vertically
                    
                    video_writer.append_frame(frame)
                    
                    # Get proprioceptive state (fingertip positions)
                    proprio_state = get_keypoint_positions_flat(env.sim)
                    
                    # Preprocess observation (skip in debug mode)
                    if self.config.debug:
                        # Create dummy obs_dict for debug mode
                        obs_dict = {}
                    else:
                        obs_dict = self.preprocess_observation(
                            image=frame,
                            instruction=language_command,
                            proprio_state=proprio_state,
                        )
                    
                    # Predict new action chunk
                    current_action_chunk = self.predict_actions(obs_dict)
                    chunk_step_idx = 0
                
                # Execute action from current chunk
                # Use min to ensure we don't go beyond the chunk size
                action_idx = min(chunk_step_idx, len(current_action_chunk) - 1)
                action = current_action_chunk[action_idx]
                
                # Step environment
                try:
                    obs, reward, done, info = env.step(action)
                    total_reward += reward
                    pbar.update(1)
                    
                    # Check success
                    if hasattr(env, 'is_success') and callable(env.is_success):
                        success = env.is_success()
                    elif 'success' in info:
                        success = info['success']
                    
                    if success:
                        done = True
                except Exception as e:
                    errors.append(f"Step {step} error: {e}")
                    break
                
                step += 1
                chunk_step_idx += 1
                
                # Render frame after step for video (if not done)
                if not done:
                    try:
                        frame = env.sim.render(
                            height=self.config.render_height,
                            width=self.config.render_width,
                            camera_name=self.config.camera_name,
                        )[::-1]  # Flip vertically
                        video_writer.append_frame(frame)
                    except:
                        pass
                
                # Update sites if needed
                if hasattr(env, "update_sites"):
                    env.update_sites()
                if hasattr(env, "update_state"):
                    env.update_state()
        
        except Exception as e:
            errors.append(f"Episode execution error: {e}")
        
        finally:
            # Close progress bar
            pbar.close()
            # Record final frame
            try:
                frame = env.sim.render(
                    height=self.config.render_height,
                    width=self.config.render_width,
                    camera_name=self.config.camera_name,
                )[::-1]
                video_writer.append_frame(frame)
            except:
                pass
            
            video_writer.close()
            env.close()
        
        return EpisodeResult(
            task_name=task_name,
            episode_idx=episode_idx,
            success=success,
            num_steps=step,
            total_reward=total_reward,
            video_path=str(video_path),
            language_command=language_command,
            errors=errors,
        )
    
    def run_evaluation(self) -> Dict[str, Any]:
        """
        Run full closed-loop evaluation across sampled tasks.
        
        Returns:
            Dictionary with evaluation results and statistics
        """
        print("\n" + "="*80)
        print("CLOSED-LOOP EVALUATION")
        print("="*80)
        print(f"Output directory: {self.output_dir}")
        print(f"Number of tasks: {self.config.num_tasks}")
        print(f"Max steps per episode: {self.config.max_steps}")
        print(f"Action horizon: {self.config.action_horizon}")
        print(f"Steps per chunk: {self.config.steps_per_chunk}")
        print("="*80 + "\n")
        
        # Find all episodes from dataset(s)
        print("Scanning for episodes in dataset...")
        all_episodes = find_datasets_and_episodes(
            self.dataset_path,
            max_episodes=self.max_episodes,
            filter_key=self.filter_key,
        )
        print(f"Found {len(all_episodes)} episodes across all datasets")
        
        # Sample episodes (one per task)
        samples = sample_episodes(
            all_episodes,
            self.config.num_tasks,
            self.config.seed,
        )
        print(f"Sampled {len(samples)} episodes for evaluation\n")
        
        # Group episodes by dataset (to reuse environments)
        episodes_by_dataset = {}
        for dataset_path, task_name, episode_name in samples:
            if dataset_path not in episodes_by_dataset:
                episodes_by_dataset[dataset_path] = []
            episodes_by_dataset[dataset_path].append((task_name, episode_name))
        
        # Run evaluation
        results = []
        start_time = time.time()
        
        episode_counter = 0
        for dataset_path, episodes in episodes_by_dataset.items():
            # Create environment for this dataset (reused for all episodes)
            print(f"\nCreating environment for dataset: {Path(dataset_path).name}")
            try:
                env_meta = get_env_metadata_from_dataset(dataset_path)
                env = make_env_from_metadata(env_meta)
                print(f"Environment: {env_meta.get('env_name', 'Unknown')}")
            except Exception as e:
                print(f"Failed to create environment for {dataset_path}: {e}")
                # Mark all episodes from this dataset as failed
                for task_name, episode_name in episodes:
                    results.append(EpisodeResult(
                        task_name=task_name,
                        episode_idx=episode_counter,
                        success=False,
                        num_steps=0,
                        total_reward=0.0,
                        video_path="",
                        language_command="",
                        errors=[f"Failed to create environment: {e}"],
                    ))
                    episode_counter += 1
                continue
            
            # Evaluate all episodes from this dataset
            for task_name, episode_name in episodes:
                print(f"\n--- Task {episode_counter+1}/{len(samples)}: {task_name} ---")
                print(f"Episode: {episode_name}")
                
                result = self.run_episode(
                    dataset_path=dataset_path,
                    task_name=task_name,
                    episode_name=episode_name,
                    episode_idx=episode_counter,
                    env=env,
                )
                results.append(result)
                episode_counter += 1
                
                status = "SUCCESS" if result.success else "FAILED"
                print(f"Result: {status}, Steps: {result.num_steps}, Reward: {result.total_reward:.3f}")
                if result.errors:
                    print(f"Errors: {result.errors}")
            
            # Close environment after processing all episodes from this dataset
            env.close()
        
        total_time = time.time() - start_time
        
        # Compute statistics
        num_success = sum(1 for r in results if r.success)
        success_rate = num_success / len(results) if results else 0.0
        avg_steps = np.mean([r.num_steps for r in results]) if results else 0.0
        avg_reward = np.mean([r.total_reward for r in results]) if results else 0.0
        
        # Per-task success rates
        task_success = {}
        for r in results:
            if r.task_name not in task_success:
                task_success[r.task_name] = {'total': 0, 'success': 0}
            task_success[r.task_name]['total'] += 1
            if r.success:
                task_success[r.task_name]['success'] += 1
        
        # Summary statistics
        summary = {
            'num_tasks': len(results),
            'num_success': num_success,
            'success_rate': success_rate,
            'avg_steps': avg_steps,
            'avg_reward': avg_reward,
            'total_time_seconds': total_time,
            'avg_time_per_episode': total_time / len(results) if results else 0.0,
            'task_success_rates': {
                task: data['success'] / data['total'] 
                for task, data in task_success.items()
            },
            'config': {
                'checkpoint': self.config.checkpoint,
                'num_tasks': self.config.num_tasks,
                'max_steps': self.config.max_steps,
                'action_horizon': self.config.action_horizon,
                'num_ode_steps': self.config.num_ode_steps,
                'steps_per_chunk': self.config.steps_per_chunk,
                'seed': self.config.seed,
            },
            'results': [
                {
                    'task_name': r.task_name,
                    'episode_idx': r.episode_idx,
                    'success': r.success,
                    'num_steps': r.num_steps,
                    'total_reward': r.total_reward,
                    'video_path': r.video_path,
                    'language_command': r.language_command,
                    'errors': r.errors,
                }
                for r in results
            ],
        }
        
        # Print summary
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        print(f"Total episodes:      {len(results)}")
        print(f"Successful:          {num_success}")
        print(f"Success rate:        {success_rate*100:.1f}%")
        print(f"Average steps:       {avg_steps:.1f}")
        print(f"Average reward:      {avg_reward:.3f}")
        print(f"Total time:          {total_time:.1f}s")
        print(f"Time per episode:    {total_time/len(results):.1f}s" if results else "N/A")
        print("\nPer-task success rates:")
        for task, rate in summary['task_success_rates'].items():
            print(f"  {task}: {rate*100:.1f}%")
        print("="*80 + "\n")
        
        # Save results to JSON
        results_path = self.output_dir / "evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Results saved to: {results_path}")
        
        return summary


def main():
    parser = argparse.ArgumentParser(
        description="Closed-loop evaluation of VLA model in RoboCasa simulation"
    )
    parser.add_argument(
        "checkpoint", 
        type=str, 
        nargs='?',
        help="Path to model checkpoint directory (required unless --list_tasks is used)"
    )
    parser.add_argument(
        "--list_tasks",
        action="store_true",
        help="List all available RoboCasa task names and exit"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="closed_loop_eval_results",
        help="Directory to save evaluation results and videos"
    )
    parser.add_argument(
        "--num_tasks", 
        type=int, 
        default=10,
        help="Number of tasks to randomly sample for evaluation"
    )
    parser.add_argument(
        "--max_steps", 
        type=int, 
        default=300,
        help="Maximum steps per episode"
    )
    parser.add_argument(
        "--camera_name", 
        type=str, 
        default="egoview",
        help="Camera name for rendering observations"
    )
    parser.add_argument(
        "--render_height", 
        type=int, 
        default=256,
        help="Height of rendered images"
    )
    parser.add_argument(
        "--render_width", 
        type=int, 
        default=256,
        help="Width of rendered images"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda",
        help="Device to run the model on"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--action_horizon", 
        type=int, 
        default=30,
        help="Number of action steps to predict"
    )
    parser.add_argument(
        "--num_ode_steps", 
        type=int, 
        default=10,
        help="Number of ODE integration steps for flow matching"
    )
    parser.add_argument(
        "--steps_per_chunk", 
        type=int, 
        default=1,
        help="Number of steps to execute from each predicted action chunk before predicting next chunk"
    )
    parser.add_argument(
        "--video_fps", 
        type=int, 
        default=20,
        help="Frames per second for output videos"
    )
    parser.add_argument(
        "--no_normalize_actions", 
        action="store_true",
        help="Disable action normalization"
    )
    parser.add_argument(
        "--dataset_path", 
        type=str, 
        required=True,
        help="Path to original RoboCasa dataset (HDF5 file or directory containing HDF5 files)"
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Maximum number of episodes to process per dataset file"
    )
    parser.add_argument(
        "--filter_key",
        type=str,
        default=None,
        help="Filter key to select subset of episodes"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: skip model loading and use random actions instead"
    )
    
    args = parser.parse_args()
    
    # Handle --list_tasks option
    if args.list_tasks:
        print("\n" + "="*80)
        print("Available RoboCasa Task Names")
        print("="*80)
        try:
            # Find all datasets and extract task names
            all_episodes = find_datasets_and_episodes(
                args.dataset_path,
                max_episodes=None,
                filter_key=args.filter_key,
            )
            # Extract unique task names
            task_names = sorted(set(task_name for _, task_name, _ in all_episodes))
            print(f"\nFound {len(task_names)} RoboCasa tasks:\n")
            for i, task in enumerate(task_names, 1):
                print(f"  {i:3d}. {task}")
            print("\n" + "="*80)
            print(f"\nTotal episodes found: {len(all_episodes)}")
            print("="*80 + "\n")
        except Exception as e:
            print(f"Error listing tasks: {e}")
            import traceback
            traceback.print_exc()
        return
    
    if not args.checkpoint and not args.debug:
        parser.error("checkpoint is required unless --list_tasks or --debug is used")
    
    # Create config
    # In debug mode, checkpoint can be None
    checkpoint = args.checkpoint if args.checkpoint else "dummy_checkpoint" if args.debug else None
    config = EvalConfig(
        checkpoint=checkpoint,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        num_tasks=args.num_tasks,
        max_steps=args.max_steps,
        camera_name=args.camera_name,
        render_height=args.render_height,
        render_width=args.render_width,
        device=args.device,
        seed=args.seed,
        action_horizon=args.action_horizon,
        num_ode_steps=args.num_ode_steps,
        steps_per_chunk=args.steps_per_chunk,
        video_fps=args.video_fps,
        normalize_actions=not args.no_normalize_actions,
        max_episodes=args.max_episodes,
        filter_key=args.filter_key,
        use_robot_actions=True,
        debug=args.debug,
    )
    
    # Run evaluation
    evaluator = ClosedLoopEvaluator(config)
    results = evaluator.run_evaluation()
    
    if 'error' not in results:
        print(f"\nEvaluation complete. Success rate: {results['success_rate']*100:.1f}%")
    else:
        print(f"\nEvaluation failed: {results['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
