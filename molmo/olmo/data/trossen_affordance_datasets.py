#!/usr/bin/env python3
"""
Trossen (LeRobot) Affordance Dataset for EgoDex-style training.

Loads Trossen/LeRobot data with precomputed end-effector positions (from add_trossen_ee_to_dataset.py).
At each sample frame we provide:
  - Current image (head camera)
  - Current robot state
  - Current end-effector positions in the current frame's camera frame (one point per arm)
  - Future action sequence
  - Future end-effector position sequence (in current frame's camera frame)

Normalization: Unlike EgoDex or RoboCasa, this is a small dataset; normalization stats (mean, variance/std,
min, max) are computed online in __init__ when normalize_coordinates=True and no stats_file is provided.
You can still pass a precomputed stats file (e.g. from compute_trajectory_stats.py) if desired.

Frame alignment: The EE HDF5 must be produced by data/add_trossen_ee_to_dataset.py from the same
LeRobot dataset (same repo_id, data_root, and episodes). HDF5 row index i = LeRobot global frame index i.

Requires: lerobot, and an HDF5 of EE positions produced by data/add_trossen_ee_to_dataset.py.
"""

import contextlib
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from olmo.data.dataset import Dataset

# try:
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
# except ImportError:
#     LeRobotDataset = None

# Default camera key for Trossen (head camera)
DEFAULT_CAMERA_KEY = "observation.images.cam_high"

# Same default dataset as data/add_trossen_ee_to_dataset.py (and compute_fk_from_dataset.py)
# DEFAULT_REPO_ID = "ykorkmaz/aloha_play_dataset_part_3"
DEFAULT_REPO_ID = "ishika/aloha_play_dataset_part_3_with_fk_full_split"
STATE_KEY = "observation.state"

# Camera intrinsics for head camera (same as lerobot/script/compute_fk_from_dataset.py)
DEFAULT_FX = 381.092
DEFAULT_FY = 381.092
DEFAULT_CX = 310.085
DEFAULT_CY = 245.318
ACTION_KEY = "action"


def _world_to_camera(
    p_world: np.ndarray,
    cam_position: np.ndarray,
    cam_quat_xyzw: np.ndarray,
) -> np.ndarray:
    """Transform world-frame point to camera frame. p_world (3,), returns (3,)."""
    from scipy.spatial.transform import Rotation
    R_world_to_cam = Rotation.from_quat(cam_quat_xyzw).as_matrix().T
    p_cam = R_world_to_cam @ (np.asarray(p_world).reshape(3) - np.asarray(cam_position).reshape(3))
    return p_cam


@contextlib.contextmanager
def _hdf5_with_retry(path: str, max_retries: int = 3):
    """Open HDF5 with retry for NFS/IO errors."""
    last_e = None
    for attempt in range(max_retries):
        try:
            f = h5py.File(path, "r")
            try:
                yield f
                return
            finally:
                f.close()
        except (OSError, IOError) as e:
            last_e = e
            if attempt < max_retries - 1:
                time.sleep((2 ** attempt) * 0.1 + random.uniform(0, 0.1))
                continue
            raise
    if last_e is not None:
        raise last_e


class TrossenAffordanceDataset(Dataset):
    """
    Dataset for Trossen (LeRobot) data with EE positions in camera frame.
    One end-effector point per arm (left and right gripper center).
    not "state" means the keypoint positions, not the robot state
    "robot_states" means joint angles
    """

    def __init__(
        self,
        repo_id: Optional[str] = None,
        data_root: Optional[str] = None,
        ee_hdf5_path: Optional[str] = None,
        split: str = "train",
        action_chunking_horizon: int = 30,
        camera_key: Optional[str] = None,
        normalize_coordinates: bool = False,
        stats_file: Optional[str] = None,
        trajectory_representation: str = "absolute",
        load_images: bool = True,
        frame_downsampling_ratio: int = 1,
        train_ratio: float = 0.9,
        pad_action_chunk: bool = False,
        episodes: Optional[List[int]] = None,
    ):
        """
        Args:
            repo_id: LeRobot dataset repo id (HuggingFace). Use with data_root for local.
            data_root: Local path to dataset (if loading from disk).
            ee_hdf5_path: Path to HDF5 from add_trossen_ee_to_dataset.py. Default: <data_root>/trossen_ee_world.hdf5.
            split: 'train', 'test', or 'overfit'.
            action_chunking_horizon: Number of future steps.
            camera_key: Observation key for images (default: observation.images.cam_high).
            normalize_coordinates: Whether to normalize EE coordinates (e.g. with stats).
            stats_file: Path to stats for normalization (optional). If not provided and
                normalize_coordinates=True, stats (mean, std, min, max) are computed online in __init__.
            trajectory_representation: 'absolute' or 'delta'.
            load_images: Whether to load images.
            frame_downsampling_ratio: Sample every n frames.
            train_ratio: Train/test split ratio per repo.
            pad_action_chunk: Pad action chunks with last step when near end.
            episodes: If set, only use these episode indices (for add_trossen_ee_to_dataset --episodes).
        """
        if LeRobotDataset is None:
            raise ImportError("lerobot is required for TrossenAffordanceDataset. Install with: pip install lerobot")

        self.repo_id = repo_id or ""
        self.data_root = Path(data_root) if data_root else None
        self.ee_hdf5_path = Path(ee_hdf5_path) if ee_hdf5_path else None
        self.split = split
        self.action_chunking_horizon = action_chunking_horizon
        self.camera_key = camera_key or DEFAULT_CAMERA_KEY
        self.normalize_coordinates = normalize_coordinates
        self.stats_file = stats_file
        self.trajectory_representation = trajectory_representation
        self.load_images = load_images
        self.frame_downsampling_ratio = max(1, frame_downsampling_ratio)
        self.train_ratio = train_ratio
        self.pad_action_chunk = pad_action_chunk
        self.episodes = episodes

        # Resolve data root from env if not set
        if self.data_root is None:
            env_root = os.environ.get("TROSSEN_DATA_DIR")
            self.data_root = Path(env_root) if env_root else Path(".")
        if not self.repo_id and not self.data_root.exists():
            raise ValueError(
                "TrossenAffordanceDataset needs either repo_id or an existing data_root. Set TROSSEN_DATA_DIR for local data."
            )

        # Load dataset (meta + index only; we use it for __getitem__)
        self.lerobot_dataset = LeRobotDataset(
            self.repo_id,
            root=str(self.data_root) if self.data_root else None,
            episodes=self.episodes,
            # video_backend="decord",
        )


        # Resolve EE HDF5 path
        if self.ee_hdf5_path is None:
            self.ee_hdf5_path = self.data_root / "trossen_ee_world.hdf5"
        if not self.ee_hdf5_path.exists():
            raise FileNotFoundError(
                f"EE HDF5 not found: {self.ee_hdf5_path}. "
                "Run data/add_trossen_ee_to_dataset.py first."
            )

        # Normalization stats (optional). For small Trossen dataset we compute online in __init__
        # when normalize_coordinates=True and no stats_file is provided.
        self.trajectory_stats_mean = None
        self.trajectory_stats_std = None
        self.trajectory_stats_min = None
        self.trajectory_stats_max = None
        self.robot_action_stats_mean = None
        self.robot_action_stats_std = None
        self.robot_action_stats_min = None
        self.robot_action_stats_max = None

        # Build flat index: list of {global_frame_idx, episode_idx, num_frames}
        self.index_mapping = self._build_index_mapping()

        if self.frame_downsampling_ratio > 1:
            self.index_mapping = [
                e for e in self.index_mapping
                if (e["global_frame_idx"] - self._episode_start(e["episode_idx"])) % self.frame_downsampling_ratio == 0
            ]

        # Normalization stats (optional). For small Trossen dataset we compute online when
        # normalize_coordinates=True and no stats_file is provided.
        if self.normalize_coordinates:
            if self.stats_file and os.path.exists(self.stats_file):
                stats = torch.load(self.stats_file)
                if "mean" in stats and "std" in stats:
                    self.trajectory_stats_mean = torch.tensor(stats["mean"]).float()
                    self.trajectory_stats_std = torch.tensor(stats["std"]).float()
                if "min" in stats and "max" in stats:
                    self.trajectory_stats_min = torch.tensor(stats["min"]).float()
                    self.trajectory_stats_max = torch.tensor(stats["max"]).float()
                if "robot_action_stats" in stats:
                    ra = stats["robot_action_stats"]
                    self.robot_action_stats_mean = torch.tensor(ra["mean"]).float()
                    self.robot_action_stats_std = torch.tensor(ra["std"]).float()
                    if "min" in ra and "max" in ra:
                        self.robot_action_stats_min = torch.tensor(ra["min"]).float()
                        self.robot_action_stats_max = torch.tensor(ra["max"]).float()
            else:
                # Small dataset: compute mean, variance (std), min, max online
                self._compute_normalization_stats_online()

        print(
            f"[TrossenAffordanceDataset] {self.split}: {len(self.index_mapping)} samples, "
            f"horizon={action_chunking_horizon}, camera={self.camera_key}"
        )

    def _episode_start(self, ep_idx: int) -> int:
        return int(self.lerobot_dataset.episode_data_index["from"][ep_idx].item())

    def _episode_end(self, ep_idx: int) -> int:
        return int(self.lerobot_dataset.episode_data_index["to"][ep_idx].item())

    def _build_index_mapping(self) -> List[Dict]:
        """Build list of (global_frame_idx, episode_idx, num_frames) with enough future steps."""
        mapping = []
        # import ipdb; ipdb.set_trace()
        ep_from = self.lerobot_dataset.episode_data_index["from"].numpy()
        ep_to = self.lerobot_dataset.episode_data_index["to"].numpy()
        num_episodes = len(ep_from)

        # Train/test split by episode
        n_train = int(num_episodes * self.train_ratio)
        if self.split == "train":
            ep_range = range(0, n_train)
        elif self.split == "test":
            ep_range = range(n_train, num_episodes)
        elif self.split == "overfit":
            ep_range = range(0, min(3, num_episodes))
        else:
            raise ValueError(f"Unknown split: {self.split}")

        for ep_idx in ep_range:
            start, end = int(ep_from[ep_idx]), int(ep_to[ep_idx])
            n_frames = end - start
            if self.pad_action_chunk:
                if n_frames < self.action_chunking_horizon:
                    continue
                for frame_in_ep in range(n_frames):
                    mapping.append({
                        "global_frame_idx": start + frame_in_ep,
                        "episode_idx": ep_idx,
                        "num_frames": n_frames,
                        "task_name": self.lerobot_dataset.meta.episodes[ep_idx]['tasks'][0],
                    })
            else:
                if n_frames < self.action_chunking_horizon:
                    continue
                for frame_in_ep in range(n_frames - self.action_chunking_horizon):
                    mapping.append({
                        "global_frame_idx": start + frame_in_ep,
                        "episode_idx": ep_idx,
                        "num_frames": n_frames,
                    })
        return mapping

    def _compute_normalization_stats_online(self) -> None:
        """
        Compute normalization stats (mean, std, min, max) over all samples in index_mapping.
        Used for the small Trossen dataset so we don't need a precomputed stats file.
        """
        all_trajectories: List[np.ndarray] = []
        all_robot_actions: List[np.ndarray] = []
        _to_float32 = lambda x: np.asarray(x, dtype=np.float32).ravel()

        with _hdf5_with_retry(str(self.ee_hdf5_path)) as f:
            for entry in tqdm(self.index_mapping, desc="Computing trajectory/action stats"):
                global_idx = entry["global_frame_idx"]
                ep_idx = entry["episode_idx"]
                num_frames = entry["num_frames"]
                ep_start = self._episode_start(ep_idx)

                head_pos = np.asarray(f["head_camera_position"][global_idx])
                head_quat = np.asarray(f["head_camera_quat_xyzw"][global_idx])
                n_future = min(
                    self.action_chunking_horizon,
                    num_frames - (global_idx - ep_start) - 1,
                )
                left_ee_fut = f["left_ee_position"][global_idx + 1:global_idx + 1 + n_future]
                right_ee_fut = f["right_ee_position"][global_idx + 1:global_idx + 1 + n_future]

                trajectory_list = []
                for i in range(left_ee_fut.shape[0]):
                    left_cam = _world_to_camera(
                        np.asarray(left_ee_fut[i]), head_pos, head_quat
                    )
                    right_cam = _world_to_camera(
                        np.asarray(right_ee_fut[i]), head_pos, head_quat
                    )
                    trajectory_list.append(np.concatenate([left_cam, right_cam]))
                trajectory = np.stack(trajectory_list, axis=0).astype(np.float32)

                if self.pad_action_chunk and trajectory.shape[0] < self.action_chunking_horizon:
                    pad = np.repeat(
                        trajectory[-1:],
                        self.action_chunking_horizon - trajectory.shape[0],
                        axis=0,
                    )
                    trajectory = np.concatenate([trajectory, pad], axis=0)

                if self.trajectory_representation == "delta":
                    deltas = trajectory[1:] - trajectory[:-1]
                    trajectory = np.concatenate([deltas, deltas[-1:]], axis=0)

                all_trajectories.append(trajectory.reshape(-1, trajectory.shape[-1]))

                start_fut = global_idx + 1
                end_fut = min(
                    global_idx + self.action_chunking_horizon + 1,
                    len(self.lerobot_dataset),
                )
                if start_fut < end_fut:
                    indices = list(range(start_fut, end_fut))
                    subset = self.lerobot_dataset.hf_dataset.select(indices)
                    robot_actions = np.stack(
                        [_to_float32(x) for x in subset[ACTION_KEY]]
                    )
                    if self.pad_action_chunk and robot_actions.shape[0] < self.action_chunking_horizon:
                        robot_actions = np.concatenate([
                            robot_actions,
                            np.repeat(
                                robot_actions[-1:],
                                self.action_chunking_horizon - robot_actions.shape[0],
                                axis=0,
                            ),
                        ], axis=0)
                else:
                    state_dim = len(_to_float32(self.lerobot_dataset[global_idx][STATE_KEY]))
                    robot_actions = np.zeros(
                        (self.action_chunking_horizon, state_dim), dtype=np.float32
                    )
                all_robot_actions.append(robot_actions)

        if all_trajectories:
            traj_data = np.concatenate(all_trajectories, axis=0)
            self.trajectory_stats_mean = torch.from_numpy(
                np.mean(traj_data, axis=0)
            ).float()
            traj_std = np.std(traj_data, axis=0)
            traj_std[traj_std < 1e-8] = 1.0
            self.trajectory_stats_std = torch.from_numpy(traj_std).float()
            self.trajectory_stats_min = torch.from_numpy(np.min(traj_data, axis=0)).float()
            self.trajectory_stats_max = torch.from_numpy(np.max(traj_data, axis=0)).float()
            print(
                f"[TrossenAffordanceDataset] Online trajectory stats: mean/std/min/max from {len(all_trajectories)} samples"
            )

        if all_robot_actions:
            ra_data = np.concatenate(all_robot_actions, axis=0)
            self.robot_action_stats_mean = torch.from_numpy(
                np.mean(ra_data, axis=0)
            ).float()
            ra_std = np.std(ra_data, axis=0)
            ra_std[ra_std < 1e-8] = 1.0
            self.robot_action_stats_std = torch.from_numpy(ra_std).float()
            self.robot_action_stats_min = torch.from_numpy(np.min(ra_data, axis=0)).float()
            self.robot_action_stats_max = torch.from_numpy(np.max(ra_data, axis=0)).float()
            print(
                f"[TrossenAffordanceDataset] Online robot action stats: mean/std/min/max from {len(all_robot_actions)} samples"
            )

    def __len__(self) -> int:
        return len(self.index_mapping)

    def get(self, idx: int, rng) -> dict:
        entry = self.index_mapping[idx]
        global_idx = entry["global_frame_idx"]
        ep_idx = entry["episode_idx"]
        num_frames = entry["num_frames"]

        # Current frame from LeRobot
        item = self.lerobot_dataset[global_idx]
        state = item[STATE_KEY]
        if hasattr(state, "numpy"):
            state = state.numpy()
        state = np.asarray(state, dtype=np.float32).ravel()

        if self.load_images:
            img = item[self.camera_key]
            if hasattr(img, "numpy"):
                img = img.numpy()
            if isinstance(img, np.ndarray):
                if img.ndim == 3 and img.shape[0] == 3:
                    img = np.transpose(img, (1, 2, 0))
                image = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8) if img.dtype in (np.float32, np.float64) else img)
            else:
                image = img  # PIL already
        else:
            image = Image.fromarray(np.zeros((1, 1, 3), dtype=np.uint8))

        # Load EE and camera pose from HDF5 (current frame's camera pose for all transforms)
        # HDF5 index = LeRobot global frame index (same as in add_trossen_ee_to_dataset.py).
        with _hdf5_with_retry(str(self.ee_hdf5_path)) as f:
            head_pos = np.asarray(f["head_camera_position"][global_idx])
            head_quat = np.asarray(f["head_camera_quat_xyzw"][global_idx])
            # Current EE (for state) at global_idx
            left_ee_now = np.asarray(f["left_ee_position"][global_idx])
            right_ee_now = np.asarray(f["right_ee_position"][global_idx])
            # Future EE sequence: global_idx+1 to global_idx+action_chunking_horizon
            n_future = min(
                self.action_chunking_horizon,
                num_frames - (global_idx - self._episode_start(ep_idx)) - 1,
            )
            left_ee_fut = f["left_ee_position"][global_idx + 1:global_idx + 1 + n_future]
            right_ee_fut = f["right_ee_position"][global_idx + 1:global_idx + 1 + n_future]

        # Current EE in camera frame (state / initial)
        left_cam_now = _world_to_camera(left_ee_now, head_pos, head_quat)
        right_cam_now = _world_to_camera(right_ee_now, head_pos, head_quat)
        initial_ee = np.concatenate([left_cam_now, right_cam_now]).astype(np.float32)

        # Future trajectory: transform each future step to current frame's camera
        trajectory_list = []
        for i in range(left_ee_fut.shape[0]):
            left_cam = _world_to_camera(left_ee_fut[i], head_pos, head_quat)
            right_cam = _world_to_camera(right_ee_fut[i], head_pos, head_quat)
            trajectory_list.append(np.concatenate([left_cam, right_cam]))
        trajectory = np.stack(trajectory_list, axis=0).astype(np.float32)

        if self.pad_action_chunk and trajectory.shape[0] < self.action_chunking_horizon:
            pad = np.repeat(trajectory[-1:], self.action_chunking_horizon - trajectory.shape[0], axis=0)
            trajectory = np.concatenate([trajectory, pad], axis=0)

        # Delta representation if requested
        if self.trajectory_representation == "delta":
            deltas = trajectory[1:] - trajectory[:-1]
            trajectory = np.concatenate([deltas, deltas[-1:]], axis=0)

        # Normalize trajectory if stats available (trajectory: T, 6 -> T, 2, 3 for 2 arms)
        if self.normalize_coordinates and self.trajectory_stats_mean is not None:
            t = torch.from_numpy(trajectory)
            num_steps, dim = t.shape
            mean = self.trajectory_stats_mean.view(1, -1).expand(1, dim)
            std = self.trajectory_stats_std.view(1, -1).expand(1, dim)
            trajectory = (t - mean) / std
            trajectory = trajectory.numpy()

        trajectory_flat = trajectory.reshape(trajectory.shape[0], -1).astype(np.float32)
        trajectory_shape = trajectory_flat.shape

        # Future actions and robot states: read from parquet in one batch to avoid
        # decoding the same episode video multiple times (LeRobot __getitem__ decodes video per call).
        start_fut = global_idx + 1
        end_fut = min(global_idx + self.action_chunking_horizon + 1, len(self.lerobot_dataset))
        if start_fut < end_fut:
            indices = list(range(start_fut, end_fut))
            subset = self.lerobot_dataset.hf_dataset.select(indices)
            _to_float32 = lambda x: np.asarray(x, dtype=np.float32).ravel()
            robot_actions = np.stack([_to_float32(x) for x in subset[ACTION_KEY]])
            robot_states = np.stack([_to_float32(x) for x in subset[STATE_KEY]])
            if self.pad_action_chunk and robot_actions.shape[0] < self.action_chunking_horizon:
                robot_actions = np.concatenate([
                    robot_actions,
                    np.repeat(robot_actions[-1:], self.action_chunking_horizon - robot_actions.shape[0], axis=0),
                ], axis=0)
                robot_states = np.concatenate([
                    robot_states,
                    np.repeat(robot_states[-1:], self.action_chunking_horizon - robot_states.shape[0], axis=0),
                ], axis=0)
        else:
            robot_actions = np.zeros((self.action_chunking_horizon, state.size), dtype=np.float32)
            robot_states = np.zeros((self.action_chunking_horizon, state.size), dtype=np.float32)

        if self.normalize_coordinates and self.robot_action_stats_mean is not None:
            robot_actions = (torch.from_numpy(robot_actions) - self.robot_action_stats_mean) / self.robot_action_stats_std
            robot_actions = robot_actions.numpy()

        result = {
            "image": image,
            "state": initial_ee,
            "trajectory_target": trajectory_flat,
            "trajectory_shape": trajectory_shape,
            "expert_type": 1,
            "robot_actions": robot_actions,
            "robot_states": robot_states,
            "label": self.lerobot_dataset.meta.episodes[ep_idx]['tasks'][0], #entry['task_name'], 
            "style": "trajectory_3d_fm",
            "metadata": {
                "image": image,
                "frame_idx": global_idx,
                "episode_idx": ep_idx,
                "task_name": entry['task_name'],
                "output_2d_trajectory": False,
                "trajectory_representation": self.trajectory_representation,
                "trajectory_dim": trajectory_flat.shape[-1],
                "robot_action_dim": robot_actions.shape[-1],
                "robot_state_dim": robot_states.shape[-1],
            },
        }
        return result


# Alias
TrossenTrajectoryDataset = TrossenAffordanceDataset


def _default_data_root():
    """Default data root: same place add_trossen_ee_to_dataset.py writes when using default repo (LeRobot cache)."""
    try:
        from lerobot.common.constants import HF_LEROBOT_HOME
        return str(HF_LEROBOT_HOME / DEFAULT_REPO_ID)
    except ImportError:
        import os
        from pathlib import Path
        from huggingface_hub.constants import HF_HOME
        cache = Path(os.getenv("HF_LEROBOT_HOME", Path(HF_HOME).expanduser() / "lerobot")).expanduser()
        return str(cache / DEFAULT_REPO_ID)


def _project_camera_frame_to_image(
    p_cam: np.ndarray,
    fx: float = DEFAULT_FX,
    fy: float = DEFAULT_FY,
    cx: float = DEFAULT_CX,
    cy: float = DEFAULT_CY,
) -> tuple[int, int] | None:
    """Project 3D point in camera frame (x right, y down, z forward) to pixel (u, v). Returns None if behind camera."""
    p_cam = np.asarray(p_cam).reshape(3)
    if p_cam[2] <= 1e-6:
        return None
    u = int(round(fx * p_cam[0] / p_cam[2] + cx))
    v = int(round(fy * p_cam[1] / p_cam[2] + cy))
    return (u, v)


def _image_to_numpy_uint8(img) -> np.ndarray:
    """Convert PIL or tensor image to (H, W, 3) uint8 numpy."""
    if hasattr(img, "numpy"):
        img = img.numpy()
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    if img.dtype in (np.float32, np.float64):
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    return np.ascontiguousarray(img)


def visualize_ee_trajectory_on_frame(
    dataset: "TrossenAffordanceDataset",
    sample_idx: int = 0,
    fx: float = DEFAULT_FX,
    fy: float = DEFAULT_FY,
    cx: float = DEFAULT_CX,
    cy: float = DEFAULT_CY,
    output_path: str | Path = "trossen_ee_vis.png",
    horizon: int | None = None,
) -> Path:
    """
    Load one sample from the dataset, denormalize if needed, project current + future EE positions
    (in camera frame) onto the image, and save.
    Left EE: blue, right EE: red. Current position = filled circle, future trajectory = line + points.
    """
    import cv2

    rng = np.random.default_rng(42)
    sample = dataset.get(sample_idx, rng)
    image = sample["image"]
    img = _image_to_numpy_uint8(image)
    h, w = img.shape[:2]

    # EE positions in camera frame (6D = left_xyz + right_xyz).
    # state is always raw (current frame); trajectory may be normalized when dataset uses normalize_coordinates.
    state = np.asarray(sample["state"], dtype=np.float64).ravel()  # (6,)
    trajectory = np.asarray(sample["trajectory_target"], dtype=np.float64)  # (T, 6)
    if trajectory.ndim == 1:
        trajectory = trajectory.reshape(1, -1)

    # Denormalize only the trajectory for display (state is never normalized)
    if dataset.trajectory_stats_mean is not None and dataset.trajectory_stats_std is not None:
        mean = dataset.trajectory_stats_mean.numpy()
        std = dataset.trajectory_stats_std.numpy()
        trajectory = trajectory * std + mean

    rep = sample["metadata"].get("trajectory_representation", "absolute")
    if rep == "delta":
        # trajectory is (T, 6) deltas: trajectory[t] = pos[t+1] - pos[t]. Integrate from current state.
        trajectory = state.reshape(1, 6) + np.cumsum(trajectory, axis=0)  # (T, 6) absolute

    # Current: state is (6,) = left_xyz (3) + right_xyz (3)
    left_cam_current = state[0:3]
    right_cam_current = state[3:6]
    # Future: trajectory is now (T, 6) absolute in both modes
    left_cam_future = trajectory[:, 0:3]  # (T, 3)
    right_cam_future = trajectory[:, 3:6]  # (T, 3)

    # Build lists of 3D points: current + future for each arm
    left_points = np.concatenate([left_cam_current.reshape(1, 3), left_cam_future], axis=0)
    right_points = np.concatenate([right_cam_current.reshape(1, 3), right_cam_future], axis=0)

    # Project to image
    left_uvs = []
    right_uvs = []
    for i in range(left_points.shape[0]):
        left_uv = _project_camera_frame_to_image(left_points[i], fx, fy, cx, cy)
        right_uv = _project_camera_frame_to_image(right_points[i], fx, fy, cx, cy)
        if left_uv and 0 <= left_uv[0] < w and 0 <= left_uv[1] < h:
            left_uvs.append(left_uv)
        if right_uv and 0 <= right_uv[0] < w and 0 <= right_uv[1] < h:
            right_uvs.append(right_uv)

    # Draw: left = blue, right = red (cv2 uses BGR)
    vis = img.copy()
    if vis.shape[2] == 3:
        vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
    left_bgr = (255, 0, 0)   # blue
    right_bgr = (0, 0, 255)  # red
    for i, uv in enumerate(left_uvs):
        if i == 0:
            cv2.circle(vis, uv, 10, left_bgr, -1)
            cv2.putText(vis, "L", (uv[0] + 12, uv[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, left_bgr, 1)
        else:
            cv2.circle(vis, uv, 4, left_bgr, -1)
        if i > 0:
            cv2.line(vis, left_uvs[i - 1], uv, left_bgr, 2)
    for i, uv in enumerate(right_uvs):
        if i == 0:
            cv2.circle(vis, uv, 10, right_bgr, -1)
            cv2.putText(vis, "R", (uv[0] + 12, uv[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, right_bgr, 1)
        else:
            cv2.circle(vis, uv, 4, right_bgr, -1)
        if i > 0:
            cv2.line(vis, right_uvs[i - 1], uv, right_bgr, 2)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    print(f"Saved EE trajectory visualization to {output_path} (sample_idx={sample_idx}, frame={sample['metadata']['frame_idx']})")
    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Visualize end-effector trajectories projected onto the current camera frame.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help=f"Dataset root; default: LeRobot cache for {DEFAULT_REPO_ID}",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=DEFAULT_REPO_ID,
        help=f"LeRobot repo id (default: {DEFAULT_REPO_ID})",
    )
    parser.add_argument(
        "--ee_hdf5",
        type=str,
        default="/home/ishikasi/.cache/huggingface/lerobot/ishika/aloha_play_dataset_part_3_with_fk_full_split/trossen_ee_world_test.hdf5",
        help="Path to trossen_ee_world.hdf5; default: <data_root>/trossen_ee_world.hdf5",
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument(
        "--trajectory_representation",
        type=str,
        choices=("absolute", "delta"),
        default="delta",
        help="Trajectory representation (default: absolute); delta integrates to absolute for viz",
    )
    parser.add_argument("--sample_idx", type=int, default=30, help="Dataset sample index to visualize")
    parser.add_argument("--output", type=str, default="trossen_ee_vis.png", help="Output image path")
    parser.add_argument("--fx", type=float, default=DEFAULT_FX, help="Camera focal length x")
    parser.add_argument("--fy", type=float, default=DEFAULT_FY, help="Camera focal length y")
    parser.add_argument("--cx", type=float, default=DEFAULT_CX, help="Camera principal point x")
    parser.add_argument("--cy", type=float, default=DEFAULT_CY, help="Camera principal point y")
    args = parser.parse_args()

    data_root = args.data_root if args.data_root is not None else _default_data_root()

    ds = TrossenAffordanceDataset(
        repo_id=args.repo_id,
        data_root=data_root,
        ee_hdf5_path=args.ee_hdf5,
        split=args.split,
        action_chunking_horizon=args.horizon,
        trajectory_representation=args.trajectory_representation,
        normalize_coordinates=True,
    )
    print(f"Dataset length: {len(ds)}")
    if len(ds) == 0:
        print("No samples; run add_trossen_ee_to_dataset.py first.")
    else:
        visualize_ee_trajectory_on_frame(
            ds,
            sample_idx=args.sample_idx,
            fx=args.fx,
            fy=args.fy,
            cx=args.cx,
            cy=args.cy,
            output_path=args.output,
            horizon=args.horizon,
        )
