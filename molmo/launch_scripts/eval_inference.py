import argparse
import torch
from pathlib import Path
import re
import os
from olmo.model import Molmo
from olmo.data.model_preprocessor import load_image
from olmo.data import build_mm_preprocessor
from olmo.data.collator import MMCollator
from olmo.util import extract_points_no_filter

def sanitize_filename(s: str) -> str:
    """Replace any character that is not alphanumeric or underscore with underscore."""
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', s)

def process_batch(examples_batch, preprocessor, model, args):
    """Process a batch of examples through the model."""
    device = torch.device(args.device)
    
    # Preprocess each example individually
    processed_examples = []
    for example_row in examples_batch:
        image_path = example_row["image_path"]
        prompt = example_row["prompt"]
        
        # Load and preprocess image
        image = load_image(image_path)
        
        # Prepare example for preprocessor
        example = {
            "image": image,
            "prompt": prompt,
            "style": "affordance" if not args.affordance_new else "affordance_new",
        }
        # Preprocess individual example
        processed_example = preprocessor(example)
        processed_examples.append(processed_example)
    
    # Use MMCollator to batch the preprocessed examples
    collator = MMCollator(include_metadata=False)
    batch = collator(processed_examples)
    
    # Move tensors to device
    input_ids = batch["input_ids"].to(device)
    images = batch["images"].to(device)
    image_input_idx = batch["image_input_idx"].to(device)
    image_masks = None
    if "image_masks" in batch:
        image_masks = batch["image_masks"].to(device)
    
    # Generate output for the batch
    print(f"Generating output for batch of {len(examples_batch)} examples...")
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
    
    # Process outputs for each example in the batch
    tokenizer = preprocessor.tokenizer
    results = []
    
    for i, example_row in enumerate(examples_batch):
        generated_ids = output.token_ids[i, 0]  # [batch, beam, seq]
        generated_ids = generated_ids[generated_ids != -1]
        generated_text = tokenizer.decode(generated_ids.tolist())
        
        # Extract prediction points
        image_np = load_image(example_row["image_path"])
        h, w = image_np.shape[:2]
        pred_points = extract_points_no_filter(generated_text, w, h)
        
        results.append({
            "image_path": example_row["image_path"],
            "prompt": example_row["prompt"],
            "generated_text": generated_text,
            "pred_points": pred_points
        })
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Simple VLM inference: image + prompt → output")
    parser.add_argument("checkpoint", type=str, help="Path to model checkpoint directory")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max tokens to generate")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on")
    parser.add_argument("--affordance_new", action="store_true", help="Use the new affordance output format")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    # affordance new: predict 5 points for each hand, affordance: predict 6 points for each hand
    args = parser.parse_args()

    # define your inference input here
    examples = [
        {"image_path": "./temp_test_image/test.jpg", "prompt": "pick up the toy"},
        {"image_path": "./temp_test_image/test.jpg", "prompt": "pick up the cup"},
    ]

    # 1. Load model from checkpoint
    print("Loading model...")
    model = Molmo.from_checkpoint(args.checkpoint, device=args.device)
    model.eval()

    # 2. Build preprocessor
    print("Building preprocessor...")
    preprocessor = build_mm_preprocessor(model.config, for_inference=True, is_training=False)

    # 3. Process examples in batches
    total_examples = len(examples)
    print(f"Processing {total_examples} examples with batch size {args.batch_size}")
    
    all_results = []
    for i in range(0, total_examples, args.batch_size):
        batch_end = min(i + args.batch_size, total_examples)
        examples_batch = examples[i:batch_end]
        
        print(f"\nProcessing batch {i//args.batch_size + 1}/{(total_examples + args.batch_size - 1)//args.batch_size}")
        print(f"Examples {i+1}-{batch_end} of {total_examples}")
        
        batch_results = process_batch(examples_batch, preprocessor, model, args)
        all_results.extend(batch_results)
        
        # Print results for this batch
        for j, result in enumerate(batch_results):
            print(f"\nResult for example {i+j+1}:")
            print(f"Image: {result['image_path']}")
            print(f"Prompt: {result['prompt']}")
            print(f"Model output: {result['generated_text']}")
            print(f"Extracted points: {len(result['pred_points']) if result['pred_points'] else 0} points")

    print(f"\nCompleted processing all {total_examples} examples in {(total_examples + args.batch_size - 1)//args.batch_size} batches")
    
    """
    if 12 points (affordance): the output order is left wrist, left thumb, left index, left middle, left ring, left pinky, right wrist, right thumb, right index, right middle, right ring, right pinky
    if 10 points (affordance_new): the output order is left thumb, left index, left middle, left ring, left pinky, right thumb, right index, right middle, right ring, right pinky
    """

if __name__ == "__main__":
    main()