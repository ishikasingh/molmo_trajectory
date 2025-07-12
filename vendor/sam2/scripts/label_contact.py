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
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

from sam2.build_sam import build_sam2_video_predictor

sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

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

def filter_contact_temporal(contact_log, min_contact_duration=3, max_gap_to_fill=2):
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

# Specify your video file path here
video_file_path = "/home/ANT.AMAZON.COM/fanyangr/Downloads/small_test/basic_pick_place/0.mp4"  # Change this to your video file path
detection_result_path = "/home/ANT.AMAZON.COM/fanyangr/Downloads/small_test/basic_pick_place/0.json"
hdf5_path = "/home/ANT.AMAZON.COM/fanyangr/Downloads/small_test/basic_pick_place/0.hdf5"

# Create output path for contact analysis
output_contact_path = video_file_path.replace('.mp4', '_contact_analysis.json')

# Create a temporary directory for extracted frames
temp_dir = tempfile.mkdtemp(prefix="sam2_frames_")
print(f"Creating temporary directory: {temp_dir}")

try:
    # Extract frames from video
    frame_names = extract_frames_from_video(video_file_path, temp_dir)
    
    # Set video_dir to our temporary directory
    video_dir = temp_dir

    # Sort frame names (they should already be in order, but just to be safe)
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    # take a look the first video frame
    frame_idx = 0
    # plt.figure(figsize=(9, 6))
    # plt.title(f"frame {frame_idx}")
    # plt.imshow(Image.open(os.path.join(video_dir, frame_names[frame_idx])))

    inference_state = predictor.init_state(video_path=video_dir)
    # inference_state.reset_state()

    predictor.reset_state(inference_state)


    contact_statistics = get_contact_statistics(detection_result_path)
    left_hand_frame_idx = None
    right_hand_frame_idx = None
    for contact in contact_statistics["contact_details"]:
        if contact[1] == "left_hand" and left_hand_frame_idx is None:
            left_hand_frame_idx = contact[0]
            left_hand_bbox = contact[2]
            left_hand_object_bbox = contact[3]

            # Let's add a box at (x_min, y_min, x_max, y_max) = (300, 0, 500, 400) to get started
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
    
    print(f"Transitions saved to: {transitions_output_path}")

    if hdf5_path is not None:
        with h5py.File(hdf5_path, 'r') as f:
            keypoints_traj = f['transforms']
            camera_traj = keypoints_traj['camera'] # [T, 4, 4]
            print(f.keys())
        

    # Print simple contact information
    print(f"\n=== Simple Contact Information ===")
    # simple_contact_info = []
    
    # for frame_data in contact_log:
    #     frame_idx = frame_data['frame_idx']
    #     left_contact = frame_data['left_hand_contact']
    #     right_contact = frame_data['right_hand_contact']
        
    #     if left_contact or right_contact:
    #         contact_status = []
            
    #         simple_contact_info.append({
    #             'frame_idx': frame_idx,
    #             'left_hand_contact': left_contact,
    #             'right_hand_contact': right_contact
    #         })
    
    # Save to JSON file
    # simple_contact_output_path = video_file_path.replace('.mp4', '_simple_contact_info.json')
    # with open(simple_contact_output_path, 'w') as f:
    #     json.dump(simple_contact_info, f, indent=2)
    

finally:
    # Clean up: remove the temporary directory and all its contents
    print(f"Cleaning up temporary directory: {temp_dir}")
    shutil.rmtree(temp_dir)
    print("Cleanup completed")