import argparse
import torch
from pathlib import Path
import re

from olmo.model import Molmo
from olmo.data.model_preprocessor import load_image
from olmo.data import build_mm_preprocessor
from olmo.config import ModelConfig
from olmo.util import extract_points
from PIL import Image, ImageDraw

def sanitize_filename(s):
    # Replace any character that is not alphanumeric or underscore with underscore
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', s)

def main():
    parser = argparse.ArgumentParser(description="Interactive VLM inference: image + prompt → output")
    parser.add_argument("checkpoint", type=str, help="Path to model checkpoint directory")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max tokens to generate")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on")
    args = parser.parse_args()

    # 1. Load model from checkpoint
    print("Loading model...")
    model = Molmo.from_checkpoint(args.checkpoint, device=args.device)
    model.eval()

    # 2. Build preprocessor
    print("Building preprocessor...")
    preprocessor = build_mm_preprocessor(model.config, for_inference=True, is_training=False)

    print("Ready for interactive inference. Type 'exit' or press Enter with no input to quit.")

    while True:
        image_path = input("\nEnter image path (or 'exit' to quit): ").strip()
        if not image_path or image_path.lower() == "exit":
            print("Exiting.")
            break
        prompt = input("Enter language prompt: ").strip()
        if not prompt:
            print("Prompt cannot be empty. Try again.")
            continue

        # 3. Load and preprocess image
        try:
            image = load_image(image_path)
        except Exception as e:
            print(f"Error loading image: {e}")
            continue

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

        # 1. Load the original image (as numpy array)
        image_np = load_image(image_path)
        h, w = image_np.shape[:2]

        # 2. Extract points from the generated text
        points = extract_points(generated_text, w, h)

        if points:
            # 3. Draw points on the image
            image_pil = Image.fromarray(image_np)
            draw = ImageDraw.Draw(image_pil)
            for x, y in points:
                r = 8  # radius of the point
                draw.ellipse((x - r, y - r, x + r, y + r), fill='red', outline='white', width=2)
            # 4. Save the image with prompt as filename
            sanitized_prompt = sanitize_filename(prompt)
            out_path = Path(f"{sanitized_prompt}.jpg")
            image_pil.save(out_path)
            print(f"Saved image with points to {out_path}")
        else:
            print("No points found in model output.")

if __name__ == "__main__":
    main()