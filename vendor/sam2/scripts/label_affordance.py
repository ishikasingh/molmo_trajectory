import os
import tempfile
import shutil
import cv2
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from sam2.hand_contact.utils import filter_contact_frames, get_contact_statistics
from PIL import Image
import json
import h5py
import multiprocessing as mp
from multiprocessing import Pool
import time

# Add these constants at the top of the file after imports
LEFT_HAND_ID = 0
RIGHT_HAND_ID = 1
LEFT_OBJECT_ID = 2
RIGHT_OBJECT_ID = 3

HAND_IDS = {LEFT_HAND_ID, RIGHT_HAND_ID}
OBJECT_IDS = {LEFT_OBJECT_ID, RIGHT_OBJECT_ID}

# Contact detection threshold (minimum overlap area to consider contact)
CONTACT_THRESHOLD = 0  # pixels
DEFAULT_INTRINSIC = np.array([
    [736.6339, 0., 960.], 
    [0., 736.6339, 540.], 
    [0., 0., 1.]
])

JOINT_NAMES_OF_INTEREST = ['leftHand', 'leftThumbTip', 'leftIndexFingerTip', 'leftMiddleFingerTip', 'leftRingFingerTip', 'leftLittleFingerTip',
                           'rightHand', 'rightThumbTip', 'rightIndexFingerTip', 'rightMiddleFingerTip', 'rightRingFingerTip', 'rightLittleFingerTip',
                           'camera']

# Configuration options
SAVE_KEYPOINT_VIDEO = False  # Set to False to skip video generation and only save keypoint data

def detect_hand_object_contact_dilation(hand_masks, object_masks, dilation_pixels=1):
    """
    Detect contact using morphological dilation to handle boundary issues.
    
    Args:
        hand_masks (dict): Dictionary of hand_id -> mask
        object_masks (dict): Dictionary of object_id -> mask
        dilation_pixels (int): Number of pixels to dilate (creates buffer zone)
    
    Returns:
        dict: Contact information with intersection areas
    """
    contact_info = {}
    
    # Create dilation kernel
    kernel = np.ones((dilation_pixels*2+1, dilation_pixels*2+1), np.uint8)
    
    for hand_id, hand_mask in hand_masks.items():
        for object_id, object_mask in object_masks.items():
            # Convert to binary if needed
            hand_binary = (hand_mask.squeeze() > 0).astype(np.uint8)
            object_binary = (object_mask.squeeze() > 0).astype(np.uint8)
            
            # Dilate hand mask to create buffer zone
            dilated_hand = cv2.dilate(hand_binary, kernel, iterations=1)
            
            # Check intersection with dilated hand
            intersection = np.logical_and(dilated_hand, object_binary)
            intersection_area = np.sum(intersection)
            
            if intersection_area > 30:
                contact_key = f"hand_{hand_id}_object_{object_id}"
                contact_info[contact_key] = {
                    'intersection_area': intersection_area,
                    'intersection_mask': intersection,
                    'hand_id': hand_id,
                    'object_id': object_id,
                    'method': 'dilation'
                }
    
    return contact_info


# select the device for computation
# if torch.cuda.is_available():
#     device = torch.device("cuda")
# elif torch.backends.mps.is_available():
#     device = torch.device("mps")
# else:
#     device = torch.device("cpu")
# print(f"using device: {device}")

# if device.type == "cuda":
#     # use bfloat16 for the entire notebook
#     torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
#     # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
#     if torch.cuda.get_device_properties(0).major >= 8:
#         torch.backends.cuda.matmul.allow_tf32 = True
#         torch.backends.cudnn.allow_tf32 = True
# elif device.type == "mps":
#     print(
#         "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
#         "give numerically different outputs and sometimes degraded performance on MPS. "
#         "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
#     )

from sam2.build_sam import build_sam2_video_predictor

sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

# predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

def extract_frames_from_video(video_path, output_dir):
    """
    Extract frames from a video file and save them as JPEG images.
    
    Args:
        video_path (str): Path to the input video file
        output_dir (str): Directory to save the extracted frames
    
    Returns:
        list: List of frame filenames
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")
    
    frame_names = []
    frame_idx = 0
    
    print(f"Extracting frames from {video_path}...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB (OpenCV uses BGR by default)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Save frame as JPEG
        frame_filename = f"{frame_idx}.jpg"
        frame_path = os.path.join(output_dir, frame_filename)
        
        # Convert to PIL Image and save
        pil_image = Image.fromarray(frame_rgb)
        pil_image.save(frame_path, "JPEG", quality=95)
        
        frame_names.append(frame_filename)
        frame_idx += 1
    
    cap.release()
    print(f"Extracted {len(frame_names)} frames")
    
    return frame_names

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def filter_contact_temporal(contact_log, min_contact_duration=5, max_gap_to_fill=10):
    """
    Apply temporal filtering to contact sequences to remove noise and fill gaps.
    
    Args:
        contact_log (list): List of contact data for each frame
        min_contact_duration (int): Minimum frames for a contact to be considered valid
        max_gap_to_fill (int): Maximum gap size (in frames) to fill within contact sequences
    
    Returns:
        list: Filtered contact log with the same structure
    """
    if not contact_log:
        return contact_log
    
    # Extract binary sequences for each hand
    left_contacts = [frame['left_hand_contact'] for frame in contact_log]
    right_contacts = [frame['right_hand_contact'] for frame in contact_log]
    
    # Apply temporal filtering to each hand's contact sequence
    filtered_left = _filter_binary_sequence(left_contacts, min_contact_duration, max_gap_to_fill)
    filtered_right = _filter_binary_sequence(right_contacts, min_contact_duration, max_gap_to_fill)
    
    # Reconstruct the contact log with filtered results
    filtered_contact_log = []
    for i, frame_data in enumerate(contact_log):
        filtered_frame = frame_data.copy()
        filtered_frame['left_hand_contact'] = filtered_left[i]
        filtered_frame['right_hand_contact'] = filtered_right[i]
        filtered_contact_log.append(filtered_frame)
    
    return filtered_contact_log

def _filter_binary_sequence(binary_seq, min_duration, max_gap):
    """
    Filter a binary sequence to remove short bursts and fill short gaps.
    
    Args:
        binary_seq (list): List of boolean values
        min_duration (int): Minimum duration for a True sequence to be kept
        max_gap (int): Maximum gap size to fill within True sequences
    
    Returns:
        list: Filtered binary sequence
    """
    if not binary_seq:
        return binary_seq
    
    # Convert to numpy array for easier processing
    seq = np.array(binary_seq, dtype=bool)
    filtered = seq.copy()
    
    # Step 1: Remove short contact bursts (erosion-like operation)
    # Find all True segments and remove those shorter than min_duration
    diff = np.diff(np.concatenate(([False], seq, [False])).astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    for start, end in zip(starts, ends):
        if end - start < min_duration:
            filtered[start:end] = False
    
    # Step 2: Fill short gaps (dilation-like operation)
    # Find all False segments within True regions and fill short ones
    seq = filtered  # Use the result from step 1
    diff = np.diff(np.concatenate(([False], seq, [False])).astype(int))
    gap_starts = np.where(diff == -1)[0]
    gap_ends = np.where(diff == 1)[0]
    
    for gap_start, gap_end in zip(gap_starts, gap_ends):
        gap_length = gap_end - gap_start
        if gap_length <= max_gap:
            # Check if this gap is surrounded by True values
            has_contact_before = gap_start > 0 and seq[gap_start - 1]
            has_contact_after = gap_end < len(seq) and seq[gap_end]
            
            if has_contact_before and has_contact_after:
                filtered[gap_start:gap_end] = True
    
    return filtered.tolist()

def detect_transitions(contact_log):
    """
    Detect transitions in contact states for each frame.
    
    Args:
        contact_log (list): List of contact data for each frame
    
    Returns:
        dict: Dictionary with transition lists for left and right hands
    """
    left_hand_transitions = []
    right_hand_transitions = []
    
    for i in range(len(contact_log)):
        if i == 0:
            # First frame has no transition
            left_hand_transitions.append("no_change")
            right_hand_transitions.append("no_change")
        else:
            prev_left = contact_log[i-1]['left_hand_contact']
            curr_left = contact_log[i]['left_hand_contact']
            prev_right = contact_log[i-1]['right_hand_contact']
            curr_right = contact_log[i]['right_hand_contact']
            
            # Left hand
            if prev_left == False and curr_left == True:
                left_hand_transitions.append("0_to_1")
            elif prev_left == True and curr_left == False:
                left_hand_transitions.append("1_to_0")
            else:
                left_hand_transitions.append("no_change")
            
            # Right hand
            if prev_right == False and curr_right == True:
                right_hand_transitions.append("0_to_1")
            elif prev_right == True and curr_right == False:
                right_hand_transitions.append("1_to_0")
            else:
                right_hand_transitions.append("no_change")
    
    return {
        'left_hand_transitions': left_hand_transitions,
        'right_hand_transitions': right_hand_transitions
    }

def analyze_hand_object_contact(video_segments, frame_names, output_contact_path=None, apply_temporal_filtering=True):
    """
    Analyze hand-object contact throughout video segments - simplified version.
    
    Args:
        video_segments (dict): Segmentation results for each frame
        frame_names (list): List of frame filenames
        output_contact_path (str, optional): Path to save contact analysis results
        apply_temporal_filtering (bool): Whether to apply temporal filtering to reduce noise
    
    Returns:
        list: Simple contact log with binary flags for each frame
    """
    if not video_segments:
        print("No segmentation results to analyze")
        return []
    
    print(f"Analyzing hand-object contact across {len(frame_names)} frames...")
    
    contact_log = []
    left_hand_contact_count = 0
    right_hand_contact_count = 0

    for frame_idx in range(len(frame_names)):
        # Initialize contact flags for this frame
        left_hand_contact = False
        right_hand_contact = False
        
        if frame_idx in video_segments:
            frame_segments = video_segments[frame_idx]
            
            # Check for hand-object contact
            present_hand_ids = HAND_IDS.intersection(frame_segments.keys())
            present_object_ids = OBJECT_IDS.intersection(frame_segments.keys())
            
            if present_hand_ids and present_object_ids:
                # Extract hand and object masks
                hand_masks = {obj_id: frame_segments[obj_id] for obj_id in present_hand_ids}
                object_masks = {obj_id: frame_segments[obj_id] for obj_id in present_object_ids}
                
                # Detect contact
                contact_info = detect_hand_object_contact_dilation(hand_masks, object_masks)
                
                if contact_info:
                    # Check which hands are in contact
                    for contact_key, contact_data in contact_info.items():
                        hand_id = contact_data['hand_id']
                        if hand_id == LEFT_HAND_ID:
                            left_hand_contact = True
                        elif hand_id == RIGHT_HAND_ID:
                            right_hand_contact = True
        
        # Store simple contact info for this frame
        contact_log.append({
            'frame_idx': frame_idx,
            'left_hand_contact': left_hand_contact,
            'right_hand_contact': right_hand_contact
        })
    
    # Apply temporal filtering if requested
    if apply_temporal_filtering:
        print("Applying temporal filtering to reduce noise...")
        original_contact_log = contact_log.copy()
        contact_log = filter_contact_temporal(contact_log)
        
        # Compare before and after filtering
        original_left_count = sum(1 for frame in original_contact_log if frame['left_hand_contact'])
        original_right_count = sum(1 for frame in original_contact_log if frame['right_hand_contact'])
        
        print(f"Before filtering - Left: {original_left_count}, Right: {original_right_count}")
    
    # Count contacts for summary (after filtering)
    left_hand_contact_count = sum(1 for frame in contact_log if frame['left_hand_contact'])
    right_hand_contact_count = sum(1 for frame in contact_log if frame['right_hand_contact'])
    
    # Print summary statistics
    print(f"\n=== Contact Analysis Summary ===")
    print(f"Total frames processed: {len(frame_names)}")
    print(f"Frames with left hand contact: {left_hand_contact_count}")
    print(f"Frames with right hand contact: {right_hand_contact_count}")
    print(f"Left hand contact rate: {left_hand_contact_count/len(frame_names)*100:.1f}%")
    print(f"Right hand contact rate: {right_hand_contact_count/len(frame_names)*100:.1f}%")
    
    # Save contact information if output path is provided
    if output_contact_path:
        contact_summary = {
            'total_frames': len(frame_names),
            'left_hand_contact_frames': left_hand_contact_count,
            'right_hand_contact_frames': right_hand_contact_count,
            'left_hand_contact_rate': left_hand_contact_count/len(frame_names),
            'right_hand_contact_rate': right_hand_contact_count/len(frame_names),
            'frame_contacts': contact_log,
            'temporal_filtering_applied': apply_temporal_filtering
        }
        
        with open(output_contact_path, 'w') as f:
            json.dump(contact_summary, f, indent=2, default=str)
        print(f"Contact analysis saved to: {output_contact_path}")
    
    return contact_log

def find_closest_future_transitions(transitions, current_frame_idx, total_frames):
    """
    Find the closest future transitions for each hand from the current frame.
    
    Args:
        transitions (dict): Dictionary with left/right hand transitions
        current_frame_idx (int): Current frame index
        total_frames (int): Total number of frames
    
    Returns:
        dict: Dictionary with closest future transition info for each hand
    """
    left_transitions = transitions['left_hand_transitions']
    right_transitions = transitions['right_hand_transitions']
    
    result = {
        'left_hand': None,
        'right_hand': None
    }
    
    # Find closest future transition for left hand
    for i in range(current_frame_idx + 1, total_frames):
        if left_transitions[i] in ["0_to_1", "1_to_0"]:
            result['left_hand'] = {
                'frame_idx': i,
                'transition_type': left_transitions[i],
                'distance': i - current_frame_idx
            }
            break
    
    # Find closest future transition for right hand
    for i in range(current_frame_idx + 1, total_frames):
        if right_transitions[i] in ["0_to_1", "1_to_0"]:
            result['right_hand'] = {
                'frame_idx': i,
                'transition_type': right_transitions[i],
                'distance': i - current_frame_idx
            }
            break
    
    return result

def load_keypoints_from_hdf5(hdf5_path):
    """
    Load 3D keypoint trajectories from HDF5 file.
    
    Args:
        hdf5_path (str): Path to HDF5 file
    
    Returns:
        dict: Dictionary containing keypoint trajectories and camera poses
    """
    with h5py.File(hdf5_path, 'r') as f:
        keypoints_traj = f['transforms']
        
        # Load camera trajectory
        camera_traj = keypoints_traj['camera'][:]  # [T, 4, 4]
        
        # Load keypoint trajectories for joints of interest
        joint_trajectories = {}
        for joint_name in JOINT_NAMES_OF_INTEREST:
            if joint_name in keypoints_traj:
                joint_trajectories[joint_name] = keypoints_traj[joint_name][:]  # [T, 4, 4]
        
        return {
            'camera_poses': camera_traj,
            'joint_trajectories': joint_trajectories
        }

def project_3d_to_2d(points_3d, camera_pose, intrinsic_matrix):
    """
    Project 3D points to 2D using camera pose and intrinsic matrix.
    
    Args:
        points_3d (np.ndarray): 3D points in world coordinates [N, 3]
        camera_pose (np.ndarray): Camera pose transformation matrix [4, 4]
        intrinsic_matrix (np.ndarray): Camera intrinsic matrix [3, 3]
    
    Returns:
        np.ndarray: 2D projected points [N, 2]
    """
    # Convert 3D points to homogeneous coordinates
    points_3d_homo = np.concatenate([points_3d, np.ones((points_3d.shape[0], 1))], axis=1)
    
    # Transform points to camera coordinate system
    # Note: camera_pose is world-to-camera transform, so we use it directly
    world_to_camera = np.linalg.inv(camera_pose)
    points_cam = (world_to_camera @ points_3d_homo.T).T
    
    # Extract 3D coordinates in camera frame
    points_cam_3d = points_cam[:, :3]
    
    # Project to 2D using intrinsic matrix
    points_2d_homo = (intrinsic_matrix @ points_cam_3d.T).T
    
    # Convert from homogeneous to 2D coordinates
    points_2d = points_2d_homo[:, :2] / points_2d_homo[:, 2:3]
    return points_2d

def get_hand_keypoints(joint_name):
    """
    Get keypoint names associated with each hand.
    
    Args:
        joint_name (str): Joint name
    
    Returns:
        str: Hand identifier ('left' or 'right') or None if not a hand keypoint
    """
    if joint_name.startswith('left'):
        return 'left'
    elif joint_name.startswith('right'):
        return 'right'
    else:
        return None

def render_keypoints_on_image(image_path, keypoints_2d, keypoints_info, output_path):
    """
    Render 2D keypoints as dots on an image.
    
    Args:
        image_path (str): Path to input image
        keypoints_2d (dict): Dictionary of joint_name -> 2D coordinates
        keypoints_info (dict): Information about which keypoints to render for each hand
        output_path (str): Path to save output image
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return
    
    # Define colors for different keypoints
    colors = {
        'left': (0, 255, 0),    # Green for left hand
        'right': (0, 0, 255),   # Red for right hand
        'other': (255, 255, 0)  # Yellow for other joints
    }
    
    # Define keypoint sizes
    keypoint_sizes = {
        'Hand': 8,
        'Tip': 6,
        'camera': 10
    }
    
    # Render keypoints
    for joint_name, point_2d in keypoints_2d.items():
        if point_2d is None:
            continue
            
        # Get hand association
        hand = get_hand_keypoints(joint_name)
        
        # Only render if this hand has future transitions
        if hand == 'left' and not keypoints_info['left_hand']:
            continue
        if hand == 'right' and not keypoints_info['right_hand']:
            continue
        
        # Determine color and size
        if hand == 'left':
            color = colors['left']
        elif hand == 'right':
            color = colors['right']
        else:
            color = colors['other']
        
        # Determine size based on keypoint type
        size = 6  # default
        if 'Hand' in joint_name:
            size = keypoint_sizes['Hand']
        elif 'Tip' in joint_name:
            size = keypoint_sizes['Tip']
        elif 'camera' in joint_name:
            size = keypoint_sizes['camera']
            color = (255, 255, 255)  # White for camera
        
        # Draw keypoint
        x, y = int(point_2d[0]), int(point_2d[1])
        
        # Check if point is within image bounds
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            cv2.circle(image, (x, y), size, color, -1)
            cv2.circle(image, (x, y), size + 1, (0, 0, 0), 2)  # Black border
    
    # Save image
    cv2.imwrite(output_path, image)

def generate_keypoint_projection_video(video_dir, frame_names, keypoint_data, transitions, output_video_path, intrinsic_matrix=DEFAULT_INTRINSIC, save_video=True):
    """
    Generate video with projected keypoints for frames with future transitions and save keypoints to JSON.
    
    Args:
        video_dir (str): Directory containing video frames
        frame_names (list): List of frame filenames
        keypoint_data (dict): Keypoint and camera data from HDF5
        transitions (dict): Transition information
        output_video_path (str): Path to save output video
        intrinsic_matrix (np.ndarray): Camera intrinsic matrix
        save_video (bool): Whether to save the video or just the keypoint data
    """
    camera_poses = keypoint_data['camera_poses']
    joint_trajectories = keypoint_data['joint_trajectories']
    
    # Create temporary directory for rendered frames
    temp_render_dir = tempfile.mkdtemp(prefix="keypoint_render_")
    print(f"Creating temporary render directory: {temp_render_dir}")
    
    rendered_frames = []
    
    # Initialize data structure for saving keypoints to JSON
    transition_keypoints_data = {
        'total_frames': len(frame_names),
        'frame_keypoints': []
    }
    
    # Get the final frame index for fallback
    final_frame_idx = len(frame_names) - 1
    
    try:
        print(f"Generating keypoint projection video for {len(frame_names)} frames...")
        
        for frame_idx in range(len(frame_names)):
            # Initialize frame data for JSON
            frame_data = {
                'frame_idx': frame_idx,
                'frame_name': frame_names[frame_idx],
                'has_future_transitions': False,
                'future_transitions': None,
                'keypoints_2d': None,
                'keypoints_3d_camera': None,
                'keypoints_3d_world': None
            }
            
            # Find closest future transitions for this frame
            future_transitions = find_closest_future_transitions(transitions, frame_idx, len(frame_names))
            
            # Check if this frame has future transitions
            has_future_transitions = future_transitions['left_hand'] or future_transitions['right_hand']
            
            if has_future_transitions:
                frame_data['has_future_transitions'] = True
                frame_data['future_transitions'] = future_transitions
                # print(f"Processing frame {frame_idx}: Left={future_transitions['left_hand']}, Right={future_transitions['right_hand']}")
            
            # Get current camera pose
            if frame_idx >= len(camera_poses):
                print(f"Warning: Frame {frame_idx} exceeds camera pose data length")
                transition_keypoints_data['frame_keypoints'].append(frame_data)
                continue
            
            current_camera_pose = camera_poses[frame_idx]
            
            # Project keypoints from future transition frames OR final frame
            keypoints_2d = {}
            keypoints_3d_camera = {}
            keypoints_3d_world = {}
            
            # Determine which hands to process
            hands_to_process = []
            hands_to_process.append(('left_hand', future_transitions['left_hand']['frame_idx'] if future_transitions['left_hand'] else final_frame_idx))
            hands_to_process.append(('right_hand', future_transitions['right_hand']['frame_idx'] if future_transitions['right_hand'] else final_frame_idx))
            
            # Process each hand
            for hand, source_frame_idx in hands_to_process:
                # Get keypoints for relevant joints at the source frame
                for joint_name in JOINT_NAMES_OF_INTEREST:
                    if joint_name in joint_trajectories:
                        # Check if this joint belongs to the current hand
                        joint_hand = get_hand_keypoints(joint_name)
                        if (hand == 'left_hand' and joint_hand == 'left') or \
                           (hand == 'right_hand' and joint_hand == 'right') or \
                           (joint_name == 'camera'):  # Always include camera
                            
                            if source_frame_idx < len(joint_trajectories[joint_name]):
                                # Get 3D position at source frame (world coordinates)
                                joint_transform = joint_trajectories[joint_name][source_frame_idx]
                                joint_3d_world = joint_transform[:3, 3].reshape(1, 3)  # Extract position
                                
                                # Transform to camera coordinates
                                world_to_camera = np.linalg.inv(current_camera_pose)
                                joint_3d_world_homo = np.concatenate([joint_3d_world, np.ones((1, 1))], axis=1)
                                joint_3d_camera_homo = (world_to_camera @ joint_3d_world_homo.T).T
                                joint_3d_camera = joint_3d_camera_homo[:, :3]  # Remove homogeneous coordinate
                                
                                # Project to 2D using current camera pose
                                joint_2d = project_3d_to_2d(joint_3d_world, current_camera_pose, intrinsic_matrix)
                                
                                # Store all three coordinate systems
                                keypoints_2d[joint_name] = joint_2d[0]
                                keypoints_3d_camera[joint_name] = joint_3d_camera[0]
                                keypoints_3d_world[joint_name] = joint_3d_world[0]
            
            # Save keypoints data for JSON (convert numpy arrays to lists)
            if keypoints_2d:
                frame_data['keypoints_2d'] = {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in keypoints_2d.items()}
                frame_data['keypoints_3d_camera'] = {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in keypoints_3d_camera.items()}
                frame_data['keypoints_3d_world'] = {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in keypoints_3d_world.items()}
            
            # Add frame data to the collection
            transition_keypoints_data['frame_keypoints'].append(frame_data)
            
            # Render keypoints on image
            input_image_path = os.path.join(video_dir, frame_names[frame_idx])
            output_image_path = os.path.join(temp_render_dir, f"frame_{frame_idx:06d}.jpg")
            
            # Create render info based on which hands have keypoint data available
            render_keypoints_info = {
                'left_hand': any(get_hand_keypoints(joint_name) == 'left' for joint_name in keypoints_2d.keys()),
                'right_hand': any(get_hand_keypoints(joint_name) == 'right' for joint_name in keypoints_2d.keys())
            }
            
            # If there are future transitions, use that information for better context
            if has_future_transitions:
                if future_transitions['left_hand']:
                    render_keypoints_info['left_hand'] = future_transitions['left_hand']
                if future_transitions['right_hand']:
                    render_keypoints_info['right_hand'] = future_transitions['right_hand']            # For rendering, only show keypoints if there are actual future transitions
            
            
            # render_keypoints_info = future_transitions if has_future_transitions else {'left_hand': None, 'right_hand': None}
            render_keypoints_on_image(input_image_path, keypoints_2d, render_keypoints_info, output_image_path)
            rendered_frames.append(output_image_path)
        
        # Save keypoints data to JSON file
        json_output_path = output_video_path.replace('.mp4', '_keypoints.json')
        with open(json_output_path, 'w') as f:
            json.dump(transition_keypoints_data, f, indent=2, default=str)
        
        print(f"Transition keypoints saved to: {json_output_path}")
        
        # Print summary statistics
        frames_with_transitions = sum(1 for frame in transition_keypoints_data['frame_keypoints'] 
                                    if frame['has_future_transitions'])
        frames_with_keypoints = sum(1 for frame in transition_keypoints_data['frame_keypoints'] 
                                  if frame['keypoints_2d'] is not None)
        print(f"Frames with future transitions: {frames_with_transitions}/{len(frame_names)}")
        print(f"Frames with keypoint data: {frames_with_keypoints}/{len(frame_names)}")
        
        # Create video from rendered frames
        if save_video and rendered_frames:
            print(f"Creating video from {len(rendered_frames)} rendered frames...")
            
            # Get frame dimensions from first frame
            first_frame = cv2.imread(rendered_frames[0])
            height, width = first_frame.shape[:2]
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width, height))
            
            # Write frames to video
            for frame_path in rendered_frames:
                frame = cv2.imread(frame_path)
                if frame is not None:
                    video_writer.write(frame)
            
            video_writer.release()
            print(f"Keypoint projection video saved to: {output_video_path}")
        elif save_video and not rendered_frames:
            print("No frames found - no video generated")
    
    finally:
        # Clean up temporary render directory
        print(f"Cleaning up temporary render directory: {temp_render_dir}")
        shutil.rmtree(temp_render_dir)

def find_video_files(folder_path):
    """
    Find all video files and their corresponding .json and .hdf5 files in a folder.
    
    Args:
        folder_path (str): Path to folder containing video files
    
    Returns:
        list: List of tuples (video_path, json_path, hdf5_path, base_name)
    """
    video_files = []
    
    # Get all files in the folder
    if not os.path.exists(folder_path):
        print(f"Error: Folder {folder_path} does not exist")
        return video_files
    
    files = os.listdir(folder_path)
    
    # Find all .mp4 files
    mp4_files = [f for f in files if f.endswith('.mp4')]
    
    for mp4_file in mp4_files:
        base_name = os.path.splitext(mp4_file)[0]  # Remove .mp4 extension
        
        # Construct paths for corresponding files
        video_path = os.path.join(folder_path, mp4_file)
        json_path = os.path.join(folder_path, f"{base_name}.json")
        hdf5_path = os.path.join(folder_path, f"{base_name}.hdf5")
        
        # Check if corresponding files exist
        if os.path.exists(json_path) and os.path.exists(hdf5_path):
            video_files.append((video_path, json_path, hdf5_path, base_name))
            print(f"Found complete set: {base_name}")
        else:
            missing = []
            if not os.path.exists(json_path):
                missing.append("json")
            if not os.path.exists(hdf5_path):
                missing.append("hdf5")
            print(f"Warning: Missing {', '.join(missing)} file(s) for {base_name}")
    
    print(f"Found {len(video_files)} complete video sets to process")
    return video_files

def get_available_gpus():
    """
    Get list of available GPU devices.
    
    Returns:
        list: List of GPU device IDs
    """
    if not torch.cuda.is_available():
        print("CUDA is not available. Using CPU.")
        return [torch.device("cpu")]
    
    gpu_count = torch.cuda.device_count()
    gpu_devices = [torch.device(f"cuda:{i}") for i in range(gpu_count)]
    print(f"Found {gpu_count} GPU(s): {gpu_devices}")
    return gpu_devices

def create_sam2_predictor(device):
    """
    Create a SAM2 predictor for a specific device.
    
    Args:
        device (torch.device): Device to load the model on
    
    Returns:
        SAM2VideoPredictor: Initialized predictor
    """
    from sam2.build_sam import build_sam2_video_predictor
    
    sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    
    # Configure device-specific settings
    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs
        if torch.cuda.get_device_properties(device.index).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    return predictor

def gpu_worker(args):
    """
    Worker function for GPU processing.
    
    Args:
        args (tuple): (gpu_device, video_files_batch)
    
    Returns:
        dict: Processing results
    """
    gpu_device, video_files_batch = args
    
    print(f"Worker on {gpu_device} processing {len(video_files_batch)} videos")
    
    # Initialize SAM2 predictor for this GPU
    predictor = create_sam2_predictor(gpu_device)
    
    results = {
        'gpu_device': str(gpu_device),
        'total_videos': len(video_files_batch),
        'successful': 0,
        'failed': 0,
        'video_results': []
    }
    
    for i, (video_path, json_path, hdf5_path, base_name) in enumerate(video_files_batch, 1):
        print(f"[{gpu_device}] Processing {i}/{len(video_files_batch)}: {base_name}")
        
        start_time = time.time()
        success = process_single_video_with_predictor(video_path, json_path, hdf5_path, base_name, predictor)
        processing_time = time.time() - start_time
        
        result = {
            'video_name': base_name,
            'success': success,
            'processing_time': processing_time
        }
        
        results['video_results'].append(result)
        
        if success:
            results['successful'] += 1
            print(f"[{gpu_device}] ✓ {base_name} completed in {processing_time:.2f}s")
        else:
            results['failed'] += 1
            print(f"[{gpu_device}] ✗ {base_name} failed after {processing_time:.2f}s")
    
    return results

def process_single_video_with_predictor(video_file_path, detection_result_path, hdf5_path, base_name, predictor):
    """
    Process a single video file with a provided predictor.
    
    Args:
        video_file_path (str): Path to video file
        detection_result_path (str): Path to detection JSON file
        hdf5_path (str): Path to HDF5 keypoint file
        base_name (str): Base name for output files
        predictor: SAM2 predictor instance
    
    Returns:
        bool: True if processing succeeded, False otherwise
    """
    try:
        # Create output path for contact analysis
        output_contact_path = video_file_path.replace('.mp4', '_contact_analysis.json')

        # Create a temporary directory for extracted frames
        temp_dir = tempfile.mkdtemp(prefix=f"sam2_frames_{base_name}_")

        try:
            # Extract frames from video
            frame_names = extract_frames_from_video(video_file_path, temp_dir)
            
            # Set video_dir to our temporary directory
            video_dir = temp_dir

            # Sort frame names (they should already be in order, but just to be safe)
            frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

            inference_state = predictor.init_state(video_path=video_dir)
            predictor.reset_state(inference_state)

            contact_statistics = get_contact_statistics(detection_result_path)
            left_hand_frame_idx = None
            right_hand_frame_idx = None
            
            for contact in contact_statistics["contact_details"]:
                if contact[1] == "left_hand" and left_hand_frame_idx is None:
                    left_hand_frame_idx = contact[0]
                    left_hand_bbox = contact[2]
                    left_hand_object_bbox = contact[3]

                    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=left_hand_frame_idx,
                        obj_id=LEFT_HAND_ID,
                        box=left_hand_bbox,
                    )

                    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=left_hand_frame_idx,
                        obj_id=LEFT_OBJECT_ID,
                        box=left_hand_object_bbox,
                    )

                if contact[1] == "right_hand" and right_hand_frame_idx is None:
                    right_hand_frame_idx = contact[0]
                    right_hand_bbox = contact[2]
                    right_hand_object_bbox = contact[3]
                    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=right_hand_frame_idx,
                        obj_id=RIGHT_HAND_ID,
                        box=right_hand_bbox,
                    )

                    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=right_hand_frame_idx,
                        obj_id=RIGHT_OBJECT_ID,
                        box=right_hand_object_bbox,
                    )
                if left_hand_frame_idx is not None and right_hand_frame_idx is not None:
                    break

            # run propagation throughout the video and collect the results in a dict
            video_segments = {}  # video_segments contains the per-frame segmentation results
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, reverse=True):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }

            # Analyze hand-object contact without creating video
            contact_log = analyze_hand_object_contact(video_segments, frame_names, output_contact_path, apply_temporal_filtering=True)

            # Detect transitions
            transitions = detect_transitions(contact_log)
            
            # Save transitions
            transitions_output_path = video_file_path.replace('.mp4', '_transitions.json')
            with open(transitions_output_path, 'w') as f:
                json.dump(transitions, f, indent=2)

            # Load keypoint data from HDF5
            if hdf5_path is not None:
                keypoint_data = load_keypoints_from_hdf5(hdf5_path)
                
                # Generate keypoint projection video (or just keypoint data)
                output_video_path = video_file_path.replace('.mp4', '_keypoint_projections.mp4')
                generate_keypoint_projection_video(
                    video_dir, 
                    frame_names, 
                    keypoint_data, 
                    transitions, 
                    output_video_path, 
                    DEFAULT_INTRINSIC,
                    save_video=SAVE_KEYPOINT_VIDEO
                )

            return True

        finally:
            # Clean up: remove the temporary directory and all its contents
            shutil.rmtree(temp_dir)
            
    except Exception as e:
        print(f"Error processing {base_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def distribute_videos_across_gpus(video_files, gpu_devices):
    """
    Distribute video files across available GPUs.
    
    Args:
        video_files (list): List of video file tuples
        gpu_devices (list): List of GPU devices
    
    Returns:
        list: List of (gpu_device, video_batch) tuples
    """
    gpu_assignments = []
    videos_per_gpu = len(video_files) // len(gpu_devices)
    remainder = len(video_files) % len(gpu_devices)
    
    start_idx = 0
    for i, gpu_device in enumerate(gpu_devices):
        # Give one extra video to the first 'remainder' GPUs
        batch_size = videos_per_gpu + (1 if i < remainder else 0)
        end_idx = start_idx + batch_size
        
        video_batch = video_files[start_idx:end_idx]
        gpu_assignments.append((gpu_device, video_batch))
        
        print(f"GPU {gpu_device}: {len(video_batch)} videos (indices {start_idx}-{end_idx-1})")
        start_idx = end_idx
    
    return gpu_assignments

def process_folder_multi_gpu(folder_path, use_all_gpus=True, max_gpus=None):
    """
    Process all videos in a folder using multiple GPUs in parallel.
    
    Args:
        folder_path (str): Path to folder containing video files
        use_all_gpus (bool): Whether to use all available GPUs
        max_gpus (int): Maximum number of GPUs to use (None for no limit)
    """
    # Set multiprocessing start method for CUDA compatibility
    mp.set_start_method('spawn', force=True)
    
    # Find all video files and their corresponding files
    video_files = find_video_files(folder_path)
    
    if not video_files:
        print("No complete video sets found to process")
        return
    
    # Get available GPUs
    gpu_devices = get_available_gpus()
    
    if max_gpus is not None:
        gpu_devices = gpu_devices[:max_gpus]
    
    if not use_all_gpus:
        gpu_devices = gpu_devices[:1]  # Use only first GPU
    
    print(f"\nStarting multi-GPU batch processing:")
    print(f"- Total videos: {len(video_files)}")
    print(f"- GPUs to use: {len(gpu_devices)} {gpu_devices}")
    print(f"- Videos per GPU: ~{len(video_files) // len(gpu_devices)}")
    
    # Distribute videos across GPUs
    gpu_assignments = distribute_videos_across_gpus(video_files, gpu_devices)
    
    # Start processing
    start_time = time.time()
    
    # Use multiprocessing to run on multiple GPUs
    with Pool(processes=len(gpu_devices)) as pool:
        all_results = pool.map(gpu_worker, gpu_assignments)
    
    total_time = time.time() - start_time
    
    # Aggregate results
    total_successful = sum(result['successful'] for result in all_results)
    total_failed = sum(result['failed'] for result in all_results)
    
    print(f"\n{'='*70}")
    print(f"MULTI-GPU BATCH PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"Total processing time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    print(f"Total videos: {len(video_files)}")
    print(f"Successful: {total_successful}")
    print(f"Failed: {total_failed}")
    print(f"Success rate: {total_successful/len(video_files)*100:.1f}%")
    print(f"Average time per video: {total_time/len(video_files):.2f} seconds")
    
    # Print per-GPU statistics
    print(f"\nPer-GPU Results:")
    for result in all_results:
        print(f"  {result['gpu_device']}: {result['successful']}/{result['total_videos']} successful")
        avg_time = np.mean([v['processing_time'] for v in result['video_results']])
        print(f"    Average time per video: {avg_time:.2f}s")

# Main execution
if __name__ == "__main__":
    # Configuration
    input_folder = "/home/ANT.AMAZON.COM/fanyangr/Downloads/small_test/basic_pick_place"
    
    # Multi-GPU processing options
    MAX_GPUS = None  # Set to a number to limit GPU usage, None for all available
    
    process_folder_multi_gpu(input_folder, use_all_gpus=True, max_gpus=MAX_GPUS)
