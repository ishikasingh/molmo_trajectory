import argparse
import torch
from pathlib import Path

from olmo.model import Molmo
from olmo.data.model_preprocessor import load_image
from olmo.data import build_mm_preprocessor
from olmo.config import ModelConfig

def main():
    parser = argparse.ArgumentParser(description="Casual VLM inference: image + prompt → output")
    parser.add_argument("checkpoint", type=str, help="Path to model checkpoint directory")
    parser.add_argument("image_path", type=str, help="Path to input image")
    parser.add_argument("prompt", type=str, help="Text prompt for the model")
    parser.add_argument("--max_new_tokens", type=int, default=32, help="Max tokens to generate")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on")
    args = parser.parse_args()

    # 1. Load model from checkpoint
    print("Loading model...")
    model = Molmo.from_checkpoint(args.checkpoint, device=args.device)
    model.eval()

    # 2. Build preprocessor
    print("Building preprocessor...")
    preprocessor = build_mm_preprocessor(model.config, for_inference=True, is_training=False)

    # 3. Load and preprocess image
    print("Loading and preprocessing image...")
    image = load_image(args.image_path)

    # 4. Prepare example for preprocessor
    example = {
        "image": image,
        "messages": [args.prompt],  # You can adjust this if your model expects a system/user format
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
    # The output token_ids do not include the prompt, so you may want to concatenate input_ids and output.token_ids
    generated_ids = output.token_ids[0, 0]  # [batch, beam, seq]
    # Remove padding (-1) and decode
    generated_ids = generated_ids[generated_ids != -1]
    generated_text = tokenizer.decode(generated_ids.tolist())
    print("Model output:")
    print(generated_text)

if __name__ == "__main__":
    main()