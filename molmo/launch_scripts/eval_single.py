import argparse
import torch
from pathlib import Path
import re
import os
import numpy as np
from olmo.model import Molmo
from olmo.data.model_preprocessor import load_image
from olmo.data import build_mm_preprocessor
from olmo.config import ModelConfig
from olmo.util import extract_points, extract_points_no_filter
from olmo.data.model_preprocessor import load_image
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Dict, Optional

def sanitize_filename(s: str) -> str:
    """Replace any character that is not alphanumeric or underscore with underscore."""
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', s)

def calculate_point_wise_errors(pred_points: List[Tuple[int, int]], gt_points: List[Tuple[int, int]]) -> List[float]:
    """
    Calculate individual point errors (Euclidean distance) between predicted and ground truth points.
    
    Args:
        pred_points: List of predicted (x, y) coordinate tuples
        gt_points: List of ground truth (x, y) coordinate tuples
    
    Returns:
        List of Euclidean distances for each point pair
    """
    if len(pred_points) != len(gt_points):
        raise ValueError(f"Point count mismatch: predicted {len(pred_points)}, ground truth {len(gt_points)}")
    
    errors = []
    for (px, py), (gx, gy) in zip(pred_points, gt_points):
        error = np.sqrt((px - gx) ** 2 + (py - gy) ** 2)
        errors.append(error)
    
    return errors

def get_finger_colors() -> Dict[str, str]:
    """Get color mapping for different finger types."""
    return {
        'thumb': '#FF6B6B',      # Red
        'index': '#4ECDC4',      # Teal
        'middle': '#45B7D1',     # Blue
        'ring': '#96CEB4',       # Green
        'pinky': '#FFEAA7',      # Yellow
        'wrist': '#DDA0DD'       # Plum
    }

def get_finger_names(is_new_format: bool = False) -> List[str]:
    """Get finger names in the expected order."""
    if is_new_format:
        # New format: 5 points per hand (no wrists)
        return ['leftThumb', 'leftIndex', 'leftMiddle', 'leftRing', 'leftPinky',
                'rightThumb', 'rightIndex', 'rightMiddle', 'rightRing', 'rightPinky']
    else:
        # Original format: 6 points per hand (with wrists)
        return ['leftHand', 'leftThumb', 'leftIndex', 'leftMiddle', 'leftRing', 'leftPinky',
                'rightHand', 'rightThumb', 'rightIndex', 'rightMiddle', 'rightRing', 'rightPinky']

def draw_legend(draw: ImageDraw.Draw, finger_colors: Dict[str, str], 
                image_width: int, image_height: int, font: ImageFont.FreeTypeFont,
                show_ground_truth: bool = False) -> None:
    """Draw a legend showing finger colors and optionally ground truth vs prediction indicators."""
    legend_x = int(image_width * 0.02)  # 2% of image width
    legend_y = int(image_height * 0.02)  # 2% of image height
    legend_spacing = int(image_height * 0.03)  # 3% of image height
    
    # Calculate legend height based on content
    legend_items = len(finger_colors)
    if show_ground_truth:
        legend_items += 2  # Add space for GT/Pred indicators
    
    # Create a semi-transparent background for legend
    legend_width = int(image_width * 0.18)  # Slightly wider for GT/Pred info
    legend_height = legend_items * legend_spacing + int(image_height * 0.02)
    legend_bg = Image.new('RGBA', (legend_width, legend_height), (0, 0, 0, 128))
    
    # Get the image from the draw object to paste the background
    image = draw._image
    image.paste(legend_bg, (legend_x - int(image_width * 0.01), legend_y - int(image_height * 0.01)), legend_bg)
    
    # Draw finger color legend items
    for i, (finger_type, color) in enumerate(finger_colors.items()):
        y_pos = legend_y + i * legend_spacing
        # Draw colored circle - scaled based on image height
        legend_r = int(image_height * 0.01)  # 1% of image height
        draw.ellipse((legend_x, y_pos - legend_r, legend_x + 2*legend_r, y_pos + legend_r), 
                    fill=color, outline='white', width=1)
        # Draw text
        text = finger_type.capitalize()
        draw.text((legend_x + int(image_width * 0.02), y_pos - int(image_height * 0.01)), 
                 text, fill='white', font=font)
    
    # Add ground truth vs prediction legend if needed
    if show_ground_truth:
        gt_y = legend_y + len(finger_colors) * legend_spacing
        pred_y = gt_y + legend_spacing
        
        # Ground truth indicator (solid circle)
        draw.ellipse((legend_x, gt_y - legend_r, legend_x + 2*legend_r, gt_y + legend_r), 
                    fill='white', outline='black', width=2)
        draw.text((legend_x + int(image_width * 0.02), gt_y - int(image_height * 0.01)), 
                 "Ground Truth", fill='white', font=font)
        
        # Prediction indicator (dashed circle - approximated with dotted outline)
        draw.ellipse((legend_x, pred_y - legend_r, legend_x + 2*legend_r, pred_y + legend_r), 
                    fill='white', outline='gray', width=3)
        draw.text((legend_x + int(image_width * 0.02), pred_y - int(image_height * 0.01)), 
                 "Prediction", fill='white', font=font)

def visualize_new_format(draw: ImageDraw.Draw, points: List[Tuple[int, int]], 
                        finger_colors: Dict[str, str], finger_names: List[str], 
                        point_radius: float, image_width: int, image_height: int,
                        is_ground_truth: bool = False) -> None:
    """Visualize points for the new affordance format (5 points per hand, no wrists)."""
    if len(points) != 10:
        # Fallback for non-standard point counts
        for x, y in points:
            outline_color = 'black' if is_ground_truth else 'gray'
            outline_width = 2 if is_ground_truth else 3
            draw.ellipse((x - point_radius, y - point_radius, x + point_radius, y + point_radius), 
                        fill='red', outline=outline_color, width=outline_width)
        return
    
    # Draw lines connecting fingers in order for each hand
    line_width = 2 if is_ground_truth else 3
    
    # Left hand: points[0:5] - thumb, index, middle, ring, pinky
    for i in range(0, 4):  # Connect 0-1, 1-2, 2-3, 3-4
        finger1_x, finger1_y = points[i]
        finger2_x, finger2_y = points[i + 1]
        finger_type = finger_names[i].replace('left', '').replace('right', '').lower()
        color = finger_colors[finger_type]
        draw.line([(finger1_x, finger1_y), (finger2_x, finger2_y)], fill=color, width=line_width)
    
    # Right hand: points[5:10] - thumb, index, middle, ring, pinky
    for i in range(5, 9):  # Connect 5-6, 6-7, 7-8, 8-9
        finger1_x, finger1_y = points[i]
        finger2_x, finger2_y = points[i + 1]
        finger_type = finger_names[i].replace('left', '').replace('right', '').lower()
        color = finger_colors[finger_type]
        draw.line([(finger1_x, finger1_y), (finger2_x, finger2_y)], fill=color, width=line_width)
    
    # Draw points with appropriate colors and styles
    outline_color = 'black' if is_ground_truth else 'gray'
    outline_width = 2 if is_ground_truth else 3
    
    for i, (x, y) in enumerate(points):
        # Use red for ground truth points, finger-specific colors for predictions
        if is_ground_truth:
            fill_color = 'red'
        else:
            finger_type = finger_names[i].replace('left', '').replace('right', '').lower()
            fill_color = finger_colors[finger_type]
        
        draw.ellipse((x - point_radius, y - point_radius, x + point_radius, y + point_radius), 
                    fill=fill_color, outline=outline_color, width=outline_width)

def visualize_original_format(draw: ImageDraw.Draw, points: List[Tuple[int, int]], 
                             finger_colors: Dict[str, str], finger_names: List[str], 
                             point_radius: float, image_width: int, image_height: int,
                             is_ground_truth: bool = False) -> None:
    """Visualize points for the original affordance format (6 points per hand, with wrists)."""
    if len(points) != 12:
        # Fallback for non-standard point counts
        outline_color = 'black' if is_ground_truth else 'gray'
        outline_width = 2 if is_ground_truth else 3
        for x, y in points:
            draw.ellipse((x - point_radius, y - point_radius, x + point_radius, y + point_radius), 
                        fill='red', outline=outline_color, width=outline_width)
        return
    
    # Draw lines connecting fingers to wrists first
    left_wrist = points[0]  # leftHand
    right_wrist = points[6]  # rightHand
    line_width = 2 if is_ground_truth else 3
    
    # Connect left hand fingers to left wrist
    for i in range(1, 6):  # leftThumb to leftPinky
        finger_x, finger_y = points[i]
        wrist_x, wrist_y = left_wrist
        finger_type = finger_names[i].replace('left', '').lower()
        color = finger_colors[finger_type]
        draw.line([(wrist_x, wrist_y), (finger_x, finger_y)], fill=color, width=line_width)
    
    # Connect right hand fingers to right wrist
    for i in range(7, 12):  # rightThumb to rightPinky
        finger_x, finger_y = points[i]
        wrist_x, wrist_y = right_wrist
        finger_type = finger_names[i].replace('right', '').lower()
        color = finger_colors[finger_type]
        draw.line([(wrist_x, wrist_y), (finger_x, finger_y)], fill=color, width=line_width)
    
    # Draw points with appropriate colors and styles
    outline_color = 'black' if is_ground_truth else 'gray'
    outline_width = 2 if is_ground_truth else 3
    
    for i, (x, y) in enumerate(points):
        # Use red for ground truth points, finger-specific colors for predictions
        if is_ground_truth:
            fill_color = 'red'
        else:
            if i == 0:  # leftHand
                fill_color = finger_colors['wrist']
            elif i == 6:  # rightHand
                fill_color = finger_colors['wrist']
            else:
                finger_type = finger_names[i].replace('left', '').replace('right', '').lower()
                fill_color = finger_colors[finger_type]
        
        draw.ellipse((x - point_radius, y - point_radius, x + point_radius, y + point_radius), 
                    fill=fill_color, outline=outline_color, width=outline_width)

def convert_ground_truth_points(gt_points: np.ndarray, image_width: int, image_height: int, 
                               point_scale: float = 100.0) -> List[Tuple[int, int]]:
    """
    Convert ground truth points from dataset format to pixel coordinates.
    
    Args:
        gt_points: Ground truth points as numpy array from dataset
        image_width: Image width in pixels
        image_height: Image height in pixels  
        point_scale: Scale factor for points (100 means percentage coordinates)
    
    Returns:
        List of (x, y) coordinate tuples in pixel coordinates
    """
    if gt_points is None or len(gt_points) == 0:
        return []
    
    pixel_points = []
    for point in gt_points:
        if len(point) >= 2:
            # Convert from percentage (0-100) to pixel coordinates
            x_pixel = int((point[0] / point_scale) * image_width)
            y_pixel = int((point[1] / point_scale) * image_height)
            pixel_points.append((x_pixel, y_pixel))
    
    return pixel_points

def visualize_affordance_points(image_np: np.ndarray, pred_points: List[Tuple[int, int]], 
                              gt_points: Optional[List[Tuple[int, int]]] = None,
                              is_new_format: bool = False) -> Image.Image:
    """
    Visualize affordance points on an image, optionally showing both ground truth and predictions.
    
    Args:
        image_np: Input image as numpy array
        pred_points: List of predicted (x, y) coordinate tuples
        gt_points: Optional list of ground truth (x, y) coordinate tuples
        is_new_format: Whether to use new format (5 points per hand) or original (6 points per hand)
    
    Returns:
        PIL Image with visualized points
    """
    image_pil = Image.fromarray(image_np)
    draw = ImageDraw.Draw(image_pil)
    h, w = image_np.shape[:2]
    point_radius = h / 80
    
    # Get color and name mappings
    finger_colors = get_finger_colors()
    finger_names = get_finger_names(is_new_format)
    
    # Remove wrist color if using new format
    if is_new_format:
        finger_colors.pop('wrist', None)
    
    # Visualize ground truth points first (so predictions appear on top)
    if gt_points:
        if is_new_format:
            visualize_new_format(draw, gt_points, finger_colors, finger_names, 
                                point_radius, w, h, is_ground_truth=True)
        else:
            visualize_original_format(draw, gt_points, finger_colors, finger_names, 
                                    point_radius, w, h, is_ground_truth=True)
    
    # Visualize prediction points
    if pred_points:
        if is_new_format:
            visualize_new_format(draw, pred_points, finger_colors, finger_names, 
                                point_radius, w, h, is_ground_truth=False)
        else:
            visualize_original_format(draw, pred_points, finger_colors, finger_names, 
                                    point_radius, w, h, is_ground_truth=False)
    
    # Draw legend
    legend_font_size = max(12, int(h/50))
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", legend_font_size)
    except OSError:
        # Fallback to default font if DejaVuSans-Bold.ttf is not available
        font = ImageFont.load_default()
    
    show_ground_truth = gt_points is not None and len(gt_points) > 0
    draw_legend(draw, finger_colors, w, h, font, show_ground_truth)
    
    return image_pil

def load_training_examples(dataset_name: str, split: str, num_examples: int = 5, seed: int = 42) -> List[Dict]:
    """Load examples from training data."""
    import random
    from olmo.data import get_dataset_by_name
    import numpy as np
    from pathlib import Path
    
    # Set random seed for reproducible sampling
    random.seed(seed)
    
    try:
        dataset = get_dataset_by_name(dataset_name, split)
        print(f"Loaded {dataset_name} dataset with {len(dataset)} examples")
        
        # Sample random examples
        indices = random.sample(range(len(dataset)), min(num_examples, len(dataset)))
        
        examples = []
        for idx in indices:
            # Get example from dataset
            example_data = dataset.get(idx, rng=random.Random(seed + idx))
            
            # Extract image and prompt from the dataset format
            image = example_data["image"]
            message_list = example_data["message_list"]
            
            if message_list and len(message_list) > 0:
                # Get the instruction/prompt
                instruction = message_list[0].get("label", "")
                
                # Get ground truth points
                gt_points = message_list[0].get("points", None)
                point_scale = message_list[0].get("point_scale", 100.0)
                
                # Save image temporarily for processing
                temp_image_path = f"temp_training_image_{idx}.jpg"
                
                # Handle different image formats
                if hasattr(image, 'save'):
                    # PIL Image object
                    image.save(temp_image_path)
                    print(f"Saved PIL image to {temp_image_path}")
                elif isinstance(image, (str, Path)):
                    # Image path - copy the file
                    import shutil
                    image_path = str(image)
                    if os.path.exists(image_path):
                        shutil.copy2(image_path, temp_image_path)
                        print(f"Copied image from {image_path} to {temp_image_path}")
                    else:
                        print(f"Warning: Image path {image_path} does not exist")
                        continue
                
                # Verify the file was created
                if not os.path.exists(temp_image_path):
                    print(f"Error: Failed to create temporary image file {temp_image_path}")
                    continue
                
                examples.append({
                    "image_path": temp_image_path,
                    "prompt": instruction,
                    "is_training_data": True,
                    "dataset_idx": idx,
                    "ground_truth_points": gt_points,
                    "point_scale": point_scale
                })
                
                print(f"Loaded training example {idx}: {instruction[:50]}...")
                if gt_points is not None:
                    print(f"  Ground truth points: {len(gt_points)} keypoints")
                    
        return examples
        
    except Exception as e:
        print(f"Error loading training dataset {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return []

def main():
    parser = argparse.ArgumentParser(description="Batch VLM inference: image + prompt → output")
    parser.add_argument("checkpoint", type=str, help="Path to model checkpoint directory")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max tokens to generate")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on")
    parser.add_argument("--affordance_new", action="store_true", help="Use the new affordance output format")
    parser.add_argument("--use_training_data", action="store_true", help="Load examples from training data")
    parser.add_argument("--use_eval_data", action="store_true", help="Load examples from evaluation dataset")
    parser.add_argument("--training_split", type=str, default="train", help="Training dataset split (train/validation/test)")
    parser.add_argument("--num_training_examples", type=int, default=15, help="Number of training examples to load")
    parser.add_argument("--training_seed", type=int, default=42, help="Random seed for sampling training examples")
    parser.add_argument("--max_error_threshold", type=float, default=2000.0, help="Maximum error threshold in pixels - errors above this are considered invalid and excluded from statistics")
    args = parser.parse_args()

    # Initialize error tracking variables
    point_errors = []
    valid_examples = 0
    skipped_examples = 0
    expected_points = 10 if args.affordance_new else 12

    examples = []
    
    if args.use_training_data or args.use_eval_data:
        print("Loading examples from dataset...")
        # Set environment variable if not already set
        if "AFFORDANCE_DATA_PATH" not in os.environ:
            print("Warning: AFFORDANCE_DATA_PATH environment variable not set.")
            print("Please set it to your affordance dataset path:")
            print("export AFFORDANCE_DATA_PATH=/path/to/your/affordance/dataset")
            return
        
        # Determine which dataset to use
        if args.use_eval_data:
            if args.affordance_new:
                dataset_name = "affordance_eval_new"
            else:
                dataset_name = "affordance_eval"
            split_name = "train"  # Evaluation dataset uses "train" split
            print(f"Using evaluation dataset: {dataset_name}")
        else:
            if args.affordance_new:
                dataset_name = "affordance_new"
            else:
                dataset_name = "affordance"
            split_name = args.training_split
            print(f"Using training dataset: {dataset_name}")
        
        training_examples = load_training_examples(
            dataset_name, 
            split_name, 
            args.num_training_examples, 
            args.training_seed
        )
        examples.extend(training_examples)
    
    # Add hardcoded examples if no dataset data or as additional examples
    if not (args.use_training_data or args.use_eval_data):
        examples = [
            {"image_path": "./temp_test_image/test.jpg", "prompt": "pick up the toy"},
            {"image_path": "./temp_test_image/test.jpg", "prompt": "pick up the cup"},
            {"image_path": "./temp_test_image/test.jpg", "prompt": "pick up the marker"},
            {"image_path": "./temp_test_image/test.jpg", "prompt": "clean up the table"},
            {"image_path": "./temp_test_image/test.jpg", "prompt": "clean up the table, please start with small objects"},
            {"image_path": "./temp_test_image/test.jpg", "prompt": "Give me something for writing"},
            {"image_path": "./temp_test_image/test_1.jpeg", "prompt": "pick up the yellow dice"},
            {"image_path": "./temp_test_image/test_1.jpeg", "prompt": "pick up the coffee lid from the green basket"},
            # Add more examples as needed
            {"image_path": "./temp_test_image/casa_test_1.png", "prompt": "pick up the water bottle and place it in the fridge"},
            {"image_path": "./temp_test_image/casa_test_2.png", "prompt": "pick up the water bottle and place it in the cabinet"},
            {"image_path": "./temp_test_image/casa_test_3.png", "prompt": "pick up the water bottle and place it in the fridge"},
            {"image_path": "./temp_test_image/casa_test_4.png", "prompt": "pick up the water bottle and place it in the cabinet"},
            {"image_path": "./temp_test_image/casa_test_5.png", "prompt": "pick up the lemon and place it in the basket"},
            {"image_path": "./temp_test_image/casa_test_6.png", "prompt": "pick up the croissant and place it in the basket"},
        ]

    # 1. Load model from checkpoint
    print("Loading model...")
    model = Molmo.from_checkpoint(args.checkpoint, device=args.device)
    model.eval()

    # 2. Build preprocessor
    print("Building preprocessor...")
    preprocessor = build_mm_preprocessor(model.config, for_inference=True, is_training=False)

    for idx, example_row in enumerate(examples):
        image_path = example_row["image_path"]
        prompt = example_row["prompt"]
        is_training = example_row.get("is_training_data", False)
        dataset_idx = example_row.get("dataset_idx", "")
        gt_points_raw = example_row.get("ground_truth_points", None)
        point_scale = example_row.get("point_scale", 100.0)
        
        print(f"\nProcessing example {idx+1}/{len(examples)}: {image_path} | {prompt}")
        if is_training:
            print(f"  (Training data example, dataset index: {dataset_idx})")

        # 3. Load and preprocess image
        print("Loading and preprocessing image...")
        image = load_image(image_path)

        # 4. Prepare example for preprocessor
        example = {
            "image": image,
            "prompt": prompt,
            "style": "affordance" if not args.affordance_new else "affordance_new",
        }
        batch = preprocessor(example)

        # 5. Move tensors to device
        device = torch.device(args.device)
        input_ids = torch.tensor(batch["input_tokens"]).unsqueeze(0).to(device)
        images = torch.tensor(batch["images"]).unsqueeze(0).to(device)
        image_input_idx = torch.tensor(batch["image_input_idx"]).unsqueeze(0).to(device)
        image_masks = None
        if "image_masks" in batch:
            image_masks = torch.tensor(batch["image_masks"]).unsqueeze(0).to(device)

        # 6. Generate output
        print("Generating output...")
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                images=images,
                image_masks=image_masks,
                image_input_idx=image_input_idx,
                max_steps=args.max_new_tokens,
                beam_size=1,
                is_distributed=False
            )
        
        # 7. Decode output tokens
        tokenizer = preprocessor.tokenizer
        generated_ids = output.token_ids[0, 0]  # [batch, beam, seq]
        generated_ids = generated_ids[generated_ids != -1]
        generated_text = tokenizer.decode(generated_ids.tolist())
        print("Model output:")
        print(generated_text)

        # 8. Load the original image (as numpy array)
        image_np = load_image(image_path)
        h, w = image_np.shape[:2]

        # 9. Extract prediction points from the generated text
        pred_points = extract_points_no_filter(generated_text, w, h)

        # 10. Convert ground truth points to pixel coordinates if available
        gt_points = None
        if gt_points_raw is not None and len(gt_points_raw) > 0:
            gt_points = convert_ground_truth_points(gt_points_raw, w, h, point_scale)
            print(f"Ground truth: {len(gt_points)} points")

        # 10.5. Calculate pointwise errors if both predictions and ground truth are available with correct point counts
        if pred_points and gt_points and len(pred_points) == expected_points and len(gt_points) == expected_points:
            try:
                errors = calculate_point_wise_errors(pred_points, gt_points)
                mean_error = np.mean(errors)
                
                # Check if the mean error is below the threshold
                if mean_error <= args.max_error_threshold:
                    point_errors.extend(errors)
                    valid_examples += 1
                    
                    print(f"Point errors (pixels): {[f'{e:.1f}' for e in errors]}")
                    print(f"Mean point error: {mean_error:.2f} pixels")
                    print(f"Max point error: {np.max(errors):.2f} pixels")
                    print(f"Min point error: {np.min(errors):.2f} pixels")
                    
                else:
                    print(f"Example REJECTED: Mean error {mean_error:.2f} pixels exceeds threshold {args.max_error_threshold:.1f}")
                    print(f"Point errors (pixels): {[f'{e:.1f}' for e in errors]}")
                    skipped_examples += 1
                
            except Exception as e:
                print(f"Error calculating pointwise errors: {e}")
                skipped_examples += 1
        elif pred_points and gt_points:
            print(f"Skipping error calculation: pred_points={len(pred_points)}, gt_points={len(gt_points)}, expected={expected_points}")
            skipped_examples += 1
        else:
            if not pred_points:
                print("No predicted points found")
            if not gt_points:
                print("No ground truth points available")
            skipped_examples += 1

        if pred_points or gt_points:
            # 11. Visualize points on the image
            print(f"Found {len(pred_points) if pred_points else 0} prediction points to visualize")
            visualized_image = visualize_affordance_points(image_np, pred_points, gt_points, args.affordance_new)
            
            # 12. Save the image with prompt as filename
            sanitized_prompt = sanitize_filename(prompt)
            if not os.path.exists("temp_output"):
                os.makedirs("temp_output")
            
            # Add training data prefix to filename if it's from training data
            prefix = "training_" if is_training else ""
            suffix = "_with_gt" if gt_points else ""
            out_path = Path(f"temp_output/{prefix}{sanitized_prompt}_{idx+1}{suffix}.jpg")
            visualized_image.save(out_path)
            print(f"Saved image with points to {out_path}")
        else:
            print("No points found in model output or ground truth.")
    
    # Print statistical summary
    print("\n" + "="*60)
    print("POINTWISE ERROR STATISTICAL SUMMARY")
    print("="*60)
    print(f"Total examples processed: {len(examples)}")
    print(f"Valid examples for error calculation: {valid_examples}")
    print(f"Skipped examples: {skipped_examples}")
    print(f"Expected points per example: {expected_points}")
    print(f"Mean error threshold: {args.max_error_threshold:.1f} pixels")
    print(f"Total points evaluated: {len(point_errors)}")
    
    if point_errors:
        point_errors_array = np.array(point_errors)
        
        print(f"\nPointwise Error Statistics (pixels, valid examples only):")
        print(f"  Mean error: {np.mean(point_errors_array):.2f}")
        print(f"  Median error: {np.median(point_errors_array):.2f}")
        print(f"  Standard deviation: {np.std(point_errors_array):.2f}")
        print(f"  Min error: {np.min(point_errors_array):.2f}")
        print(f"  Max error: {np.max(point_errors_array):.2f}")
    else:
        print(f"\nNo valid examples found (all mean errors exceeded {args.max_error_threshold:.1f} pixel threshold)")

    # Clean up temporary training images
    if args.use_training_data:
        for example in examples:
            if example.get("is_training_data", False) and os.path.exists(example["image_path"]):
                try:
                    os.remove(example["image_path"])
                except:
                    pass

if __name__ == "__main__":
    main()