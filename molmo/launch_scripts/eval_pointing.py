import argparse
import torch
from pathlib import Path
import re
import os
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
    parser = argparse.ArgumentParser(description="Batch VLM inference for pointing task")
    parser.add_argument("checkpoint", type=str, help="Path to model checkpoint directory")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max tokens to generate")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on")
    args = parser.parse_args()

    # Examples for pointing task
    examples = [
        {"image_path": "./temp_test_image/test.jpg", "prompt": "point to the cup"},
        {"image_path": "./temp_test_image/test.jpg", "prompt": "point to the toy"},
        {"image_path": "./temp_test_image/test.jpg", "prompt": "point to the marker"},
        {"image_path": "./temp_test_image/test_1.jpeg", "prompt": "point to the yellow die"},
        {"image_path": "./temp_test_image/casa_test_5.png", "prompt": "point to the lemon"},
        {"image_path": "./temp_test_image/casa_test_1.png", "prompt": "point to the water bottle"},
        # Add more examples as needed
    ]

    # Load model from checkpoint
    print("Loading model...")
    model = Molmo.from_checkpoint(args.checkpoint, device=args.device)
    model.eval()

    # Build preprocessor
    print("Building preprocessor...")
    preprocessor = build_mm_preprocessor(model.config, for_inference=True, is_training=False)

    for idx, example_row in enumerate(examples):
        image_path = example_row["image_path"]
        prompt = example_row["prompt"]
        print(f"\nProcessing example {idx+1}/{len(examples)}: {image_path} | {prompt}")

        # Load and preprocess image
        print("Loading and preprocessing image...")
        image = load_image(image_path)

        # Prepare example for preprocessor
        example = {
            "image": image,
            "prompt": prompt,
            "style": "pointing",  # Changed from "affordance" to "pointing"
        }
        batch = preprocessor(example)

        # Move tensors to device
        device = torch.device(args.device)
        input_ids = torch.tensor(batch["input_tokens"]).unsqueeze(0).to(device)
        images = torch.tensor(batch["images"]).unsqueeze(0).to(device)
        image_input_idx = torch.tensor(batch["image_input_idx"]).unsqueeze(0).to(device)
        image_masks = None
        if "image_masks" in batch:
            image_masks = torch.tensor(batch["image_masks"]).unsqueeze(0).to(device)

        # Generate output
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

        # Decode output tokens
        tokenizer = preprocessor.tokenizer
        generated_ids = output.token_ids[0, 0]
        generated_ids = generated_ids[generated_ids != -1]
        generated_text = tokenizer.decode(generated_ids.tolist())
        print("Model output:")
        print(generated_text)

        # Load the original image
        image_np = load_image(image_path)
        h, w = image_np.shape[:2]

        # Extract points from the generated text
        points = extract_points_no_filter(generated_text, w, h)

        if points:
            # Draw points on the image
            image_pil = Image.fromarray(image_np)
            draw = ImageDraw.Draw(image_pil)
            r = h/40  # Slightly larger points for better visibility

            # Draw points with a distinctive style
            for x, y in points:
                # Draw outer circle
                draw.ellipse((x - r, y - r, x + r, y + r), fill='yellow', outline='red', width=3)
                # Draw center point
                draw.ellipse((x - r/4, y - r/4, x + r/4, y + r/4), fill='red')

                # Add coordinates as text
                # font_size = max(12, int(h/50))
                # try:
                #     font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
                # except:
                #     font = ImageFont.load_default()
                # text = f"({int(x)}, {int(y)})"
                # draw.text((x + r + 5, y + r + 5), text, fill='white', stroke_width=2, stroke_fill='black', font=font)

            # Save the image
            sanitized_prompt = sanitize_filename(prompt)
            if not os.path.exists("temp_output_pointing"):
                os.makedirs("temp_output_pointing")
            out_path = Path(f"temp_output_pointing/{sanitized_prompt}_{idx+1}.jpg")
            image_pil.save(out_path)
            print(f"Saved image with points to {out_path}")
        else:
            print("No points found in model output.")

if __name__ == "__main__":
    main() 