import argparse
import torch
from pathlib import Path
import re

from olmo.model import Molmo
from olmo.data.model_preprocessor import load_image
from olmo.data import build_mm_preprocessor
from olmo.config import ModelConfig
from olmo.util import extract_points, extract_points_no_filter
from olmo.data.model_preprocessor import load_image
from PIL import Image, ImageDraw, ImageFont

def sanitize_filename(s):
    # Replace any character that is not alphanumeric or underscore with underscore
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', s)

def main():
    parser = argparse.ArgumentParser(description="Batch VLM inference: image + prompt → output")
    parser.add_argument("checkpoint", type=str, help="Path to model checkpoint directory")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max tokens to generate")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on")
    args = parser.parse_args()

    # Hardcoded list of examples: replace with your own image paths and prompts
    # examples = [
    #     {"image_path": "./temp_test_image/casa_test_1.png", "prompt": "pick up the water bottle and place it in the fridge"},
    #     {"image_path": "./temp_test_image/casa_test_2.png", "prompt": "pick up the water bottle and place it in the cabinet"},
    #     {"image_path": "./temp_test_image/casa_test_3.png", "prompt": "pick up the water bottle and place it in the fridge"},
    #     {"image_path": "./temp_test_image/casa_test_4.png", "prompt": "pick up the water bottle and place it in the cabinet"},
    #     {"image_path": "./temp_test_image/casa_test_5.png", "prompt": "pick up the lemon and place it in the basket"},
    #     {"image_path": "./temp_test_image/casa_test_6.png", "prompt": "pick up the lemon and place it in the basket"},
    #     # Add more examples as needed
    # ]

    examples = [
        {"image_path": "./temp_test_image/test.jpg", "prompt": "pick up the toy"},
        {"image_path": "./temp_test_image/test.jpg", "prompt": "pick up the cup"},
        {"image_path": "./temp_test_image/test.jpg", "prompt": "pick up the marker"},
        {"image_path": "./temp_test_image/test_1.jpeg", "prompt": "pick up the yellow dice"},
        {"image_path": "./temp_test_image/test_1.jpeg", "prompt": "pick up the coffee lid from the green basket"},
        # Add more examples as needed
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
        print(f"\nProcessing example {idx+1}/{len(examples)}: {image_path} | {prompt}")

        # 3. Load and preprocess image
        print("Loading and preprocessing image...")
        image = load_image(image_path)

        # 4. Prepare example for preprocessor
        example = {
            "image": image,
            "prompt": prompt,
            "style": "affordance",
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

        # 9. Extract points from the generated text
        points = extract_points_no_filter(generated_text, w, h)

        if points:
            # 10. Draw points on the image
            image_pil = Image.fromarray(image_np)
            draw = ImageDraw.Draw(image_pil)
            r = h/80
            
            # Define colors for each finger type (same color for left and right)
            finger_colors = {
                'thumb': '#FF6B6B',      # Red
                'index': '#4ECDC4',      # Teal
                'middle': '#45B7D1',     # Blue
                'ring': '#96CEB4',       # Green
                'pinky': '#FFEAA7',      # Yellow
                'wrist': '#DDA0DD'       # Plum
            }
            
            # Define finger names in order
            finger_names = ['leftHand', 'leftThumb', 'leftIndex', 'leftMiddle', 'leftRing', 'leftPinky',
                           'rightHand', 'rightThumb', 'rightIndex', 'rightMiddle', 'rightRing', 'rightPinky']
            
            # the key points are ordered as
            # [leftHand, leftThumb, leftIndex, leftMiddle, leftRing, leftPinky, 
            # rightHand, rightThumb, rightIndex, rightMiddle, rightRing, rightPinky]
            print("len(points)", len(points))
            if len(points) == 12:
                # Draw lines connecting fingers to wrists first
                left_wrist = points[0]  # leftHand
                right_wrist = points[6]  # rightHand
                
                # Connect left hand fingers to left wrist
                for i in range(1, 6):  # leftThumb to leftPinky
                    finger_x, finger_y = points[i]
                    wrist_x, wrist_y = left_wrist
                    # Determine finger type for color
                    finger_type = finger_names[i].replace('left', '').lower()
                    color = finger_colors[finger_type]
                    draw.line([(wrist_x, wrist_y), (finger_x, finger_y)], fill=color, width=3)
                
                # Connect right hand fingers to right wrist
                for i in range(7, 12):  # rightThumb to rightPinky
                    finger_x, finger_y = points[i]
                    wrist_x, wrist_y = right_wrist
                    # Determine finger type for color
                    finger_type = finger_names[i].replace('right', '').lower()
                    color = finger_colors[finger_type]
                    draw.line([(wrist_x, wrist_y), (finger_x, finger_y)], fill=color, width=3)
                
                # Draw points with appropriate colors
                for i, (x, y) in enumerate(points):
                    if i == 0:  # leftHand
                        color = finger_colors['wrist']
                    elif i == 6:  # rightHand
                        color = finger_colors['wrist']
                    else:
                        # Determine finger type for color
                        finger_type = finger_names[i].replace('left', '').replace('right', '').lower()
                        color = finger_colors[finger_type]
                    
                    draw.ellipse((x - r, y - r, x + r, y + r), fill=color, outline='white', width=2)
                
                # Draw legend - scaled based on image resolution
                legend_x = int(w * 0.02)  # 2% of image width
                legend_y = int(h * 0.02)  # 2% of image height
                legend_spacing = int(h * 0.03)  # 3% of image height
                legend_font_size = max(12, int(h/50))
                font = ImageFont.truetype("DejaVuSans-Bold.ttf", legend_font_size)
                
                # Create a semi-transparent background for legend
                legend_width = int(w * 0.15)  # 15% of image width
                legend_height = len(finger_colors) * legend_spacing + int(h * 0.02)
                legend_bg = Image.new('RGBA', (legend_width, legend_height), (0, 0, 0, 128))
                image_pil.paste(legend_bg, (legend_x - int(w * 0.01), legend_y - int(h * 0.01)), legend_bg)
                
                # Draw legend items
                for i, (finger_type, color) in enumerate(finger_colors.items()):
                    y_pos = legend_y + i * legend_spacing
                    # Draw colored circle - scaled based on image height
                    legend_r = int(h * 0.01)  # 1% of image height
                    draw.ellipse((legend_x, y_pos - legend_r, legend_x + 2*legend_r, y_pos + legend_r), 
                               fill=color, outline='white', width=1)
                    # Draw text
                    text = finger_type.capitalize()
                    draw.text((legend_x + int(w * 0.02), y_pos - int(h * 0.01)), text, fill='white', font=font)
                
            else:
                # Fallback for non-standard point counts
                for i, (x, y) in enumerate(points):
                    draw.ellipse((x - r, y - r, x + r, y + r), fill='red', outline='white', width=2)
            
            # 11. Save the image with prompt as filename
            sanitized_prompt = sanitize_filename(prompt)
            out_path = Path(f"{sanitized_prompt}_{idx+1}.jpg")
            image_pil.save(out_path)
            print(f"Saved image with points to {out_path}")
        else:
            print("No points found in model output.")

if __name__ == "__main__":
    main()