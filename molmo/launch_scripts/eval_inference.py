import argparse
import torch
from pathlib import Path
import re
import os
from olmo.model import Molmo
from olmo.data.model_preprocessor import load_image
from olmo.data import build_mm_preprocessor
from olmo.util import extract_points_no_filter

def sanitize_filename(s: str) -> str:
    """Replace any character that is not alphanumeric or underscore with underscore."""
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', s)

def main():
    parser = argparse.ArgumentParser(description="Simple VLM inference: image + prompt → output")
    parser.add_argument("checkpoint", type=str, help="Path to model checkpoint directory")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max tokens to generate")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on")
    parser.add_argument("--affordance_new", action="store_true", help="Use the new affordance output format")
    # affordance new: predict 5 points for each hand, affordance: predict 6 points for each hand
    args = parser.parse_args()

    # define your inference input here
    examples = [
        {"image_path": "./temp_test_image/test.jpg", "prompt": "pick up the toy"},
        {"image_path": "./temp_test_image/test.jpg", "prompt": "pick up the cup"},
    ]

    # 1. Load model from checkpoint
    print("Loading model...")
    model = Molmo.from_checkpoint(args.checkpoint, device="cuda")
    model.eval()

    # 2. Build preprocessor
    print("Building preprocessor...")
    preprocessor = build_mm_preprocessor(model.config, for_inference=True, is_training=False)

    for idx, example_row in enumerate(examples):
        image_path = example_row["image_path"]
        prompt = example_row["prompt"]
        
        print(f"\nProcessing example {idx+1}/{len(examples)}: {image_path} | {prompt}")

        # 3. Load and preprocess image, if it is already an image, you can skip this step. Image should be in PIL format
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
        device = torch.device("cuda")
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

        # 8. Extract prediction points from the generated text
        image_np = load_image(image_path)
        h, w = image_np.shape[:2]
        # the points are normalized to 0-100, you need to convert them to pixel coordinates
        pred_points = extract_points_no_filter(generated_text, w, h)
        """
        if 12 points (affordance): the output order is left wrist, left thumb, left index, left middle, left ring, left pinky, right wrist, right thumb, right index, right middle, right ring, right pinky
        if 10 points (affordance_new): the output order is left thumb, left index, left middle, left ring, left pinky, right thumb, right index, right middle, right ring, right pinky
        """

if __name__ == "__main__":
    main()