"""this script outputs a video with contact info and segmentation masks"""
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

# Add these constants at the top of the file after imports
LEFT_HAND_ID = 0
RIGHT_HAND_ID = 1
LEFT_OBJECT_ID = 2
RIGHT_OBJECT_ID = 3

HAND_IDS = {LEFT_HAND_ID, RIGHT_HAND_ID}
OBJECT_IDS = {LEFT_OBJECT_ID, RIGHT_OBJECT_ID}

# Contact detection threshold (minimum overlap area to consider contact)
CONTACT_THRESHOLD = 0  # pixels

def detect_hand_object_contact(hand_masks, object_masks):
    """
    Detect contact between hand and object masks by calculating intersection areas.
    
    Args:
        hand_masks (dict): Dictionary of hand_id -> mask
        object_masks (dict): Dictionary of object_id -> mask
    
    Returns:
        dict: Contact information with intersection areas
    """
    contact_info = {}
    
    for hand_id, hand_mask in hand_masks.items():
        for object_id, object_mask in object_masks.items():
            # Calculate intersection
            intersection = np.logical_and(hand_mask.squeeze(), object_mask.squeeze())
            intersection_area = np.sum(intersection)
            
            if intersection_area > CONTACT_THRESHOLD:
                contact_key = f"hand_{hand_id}_object_{object_id}"
                contact_info[contact_key] = {
                    'intersection_area': intersection_area,
                    'intersection_mask': intersection,
                    'hand_id': hand_id,
                    'object_id': object_id
                }
    
    return contact_info

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


def create_tracking_video(video_segments, frame_names, video_dir, output_path, fps=30):
    """
    Create a video showing the tracking results with masks overlaid on original frames.
    
    Args:
        video_segments (dict): Segmentation results for each frame
        frame_names (list): List of frame filenames
        video_dir (str): Directory containing the frames
        output_path (str): Path to save the output video
        fps (int): Frames per second for the output video
    """
    if not video_segments:
        print("No segmentation results to create video")
        return
    
    # Read the first frame to get dimensions
    first_frame = cv2.imread(os.path.join(video_dir, frame_names[0]))
    height, width = first_frame.shape[:2]
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Creating tracking video with {len(frame_names)} frames...")
    
    contact_log = []

    for frame_idx in range(len(frame_names)):
        # Read the original frame
        frame_path = os.path.join(video_dir, frame_names[frame_idx])
        frame = cv2.imread(frame_path)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create overlay with masks
        overlay = frame_rgb.copy()
        
        if frame_idx in video_segments:
            frame_segments = video_segments[frame_idx]
            
            # Apply mask overlays
            for obj_id, mask in frame_segments.items():
                # Convert boolean mask to float and ensure it's 2D
                mask_float = mask.squeeze().astype(np.float32)
                
                # Create 3D mask
                mask_3d = np.stack([mask_float, mask_float, mask_float], axis=-1)
                
                # Generate a color for this object (using the same logic as show_mask)
                cmap = plt.get_cmap("tab10")
                color = np.array([*cmap(obj_id)[:3]]) # Keep color in [0,1] range
                
                # Apply colored mask with transparency
                alpha = 0.6
                # Ensure frame_rgb is float for calculations
                frame_float = frame_rgb.astype(np.float32)
                colored_mask = mask_3d * color.reshape(1, 1, 3) * 255.0
                
                # Proper alpha blending
                overlay = overlay.astype(np.float32)
                overlay = np.where(mask_3d > 0, 
                                 frame_float * (1 - alpha) + colored_mask * alpha, 
                                 overlay)
            
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
                    # Log contact for this frame
                    contact_log.append({
                        'frame_idx': frame_idx,
                        'contacts': contact_info
                    })
                    
                    # Instead of highlighting with overlay, add text labels
                    for contact_key, contact_data in contact_info.items():
                        intersection_mask = contact_data['intersection_mask']
                        
                        # Find the center of the contact area
                        contact_coords = np.where(intersection_mask)
                        if len(contact_coords[0]) > 0:
                            center_y = int(np.mean(contact_coords[0]))
                            center_x = int(np.mean(contact_coords[1]))
                            
                            # Create contact text
                            hand_id = contact_data['hand_id']
                            object_id = contact_data['object_id']
                            area = contact_data['intersection_area']
                            
                            # Determine hand type for better readability
                            hand_type = "L" if hand_id == LEFT_HAND_ID else "R"
                            contact_text = f"CONTACT: {hand_type}H-OBJ ({area}px)"
                            
                            # Convert overlay to BGR for OpenCV text rendering
                            overlay_bgr = cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR)
                            
                            # Add text with background rectangle for better visibility
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 0.6
                            font_thickness = 2
                            text_color = (0, 0, 255)  # Red text
                            bg_color = (255, 255, 255)  # White background
                            
                            # Get text size to create background rectangle
                            (text_width, text_height), baseline = cv2.getTextSize(
                                contact_text, font, font_scale, font_thickness
                            )
                            
                            # Position text near contact but avoid going off-screen
                            text_x = max(10, min(center_x - text_width//2, width - text_width - 10))
                            text_y = max(text_height + 10, min(center_y - 20, height - 10))
                            
                            # Draw background rectangle
                            cv2.rectangle(overlay_bgr, 
                                        (text_x - 5, text_y - text_height - 5),
                                        (text_x + text_width + 5, text_y + baseline + 5),
                                        bg_color, -1)
                            
                            # Draw border around rectangle
                            cv2.rectangle(overlay_bgr, 
                                        (text_x - 5, text_y - text_height - 5),
                                        (text_x + text_width + 5, text_y + baseline + 5),
                                        (0, 0, 0), 1)
                            
                            # Draw text
                            cv2.putText(overlay_bgr, contact_text, (text_x, text_y),
                                      font, font_scale, text_color, font_thickness)
                            
                            # Convert back to RGB
                            overlay = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
                        
                        # print(f"Frame {frame_idx}: {contact_key} - Contact area: {contact_data['intersection_area']} pixels")
        
        # Convert back to BGR for OpenCV and ensure uint8
        overlay_bgr = cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR)
        out.write(overlay_bgr)
    
    # Release everything
    out.release()
    print(f"Tracking video saved to {output_path}")


# Specify your video file path here
video_file_path = "/home/ANT.AMAZON.COM/fanyangr/code/hand_object_detector/videos/0.mp4"  # Change this to your video file path
detection_result_path = "/home/ANT.AMAZON.COM/fanyangr/code/hand_object_detector/videos/0_video_data.json"

# Get original video FPS
original_fps = 30
print(f"Original video FPS: {original_fps}")

# Create output path for tracking video
output_video_path = video_file_path.replace('.mp4', '_tracking.mp4')

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

    # predictor.reset_state(inference_state)


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

    # ann_frame_idx = 0  # the frame index we interact with
    # left_hand_id = 0  # give a unique id to each object we interact with (it can be any integers)

    # # Let's add a box at (x_min, y_min, x_max, y_max) = (300, 0, 500, 400) to get started
    # box = np.array([300, 0, 500, 400], dtype=np.float32)
    # _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    #     inference_state=inference_state,
    #     frame_idx=ann_frame_idx,
    #     obj_id=left_hand_id,
    #     box=box,
    # )

    # show the results on the current (interacted) frame
    # plt.figure(figsize=(9, 6))
    # plt.title(f"frame {right_hand_frame_idx}")
    # plt.imshow(Image.open(os.path.join(video_dir, frame_names[right_hand_frame_idx])))
    # show_box(right_hand_bbox, plt.gca())
    # show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])


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

    # Generate tracking video
    create_tracking_video(video_segments, frame_names, video_dir, output_video_path, fps=original_fps)

    # render the segmentation results every few frames
    vis_frame_stride = 30
    plt.close("all")
    # for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
    #     plt.figure(figsize=(6, 4))
    #     plt.title(f"frame {out_frame_idx}")
    #     plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
    #     for out_obj_id, out_mask in video_segments[out_frame_idx].items():
    #         show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
    # plt.show()

finally:
    # Clean up: remove the temporary directory and all its contents
    print(f"Cleaning up temporary directory: {temp_dir}")
    shutil.rmtree(temp_dir)
    print("Cleanup completed")