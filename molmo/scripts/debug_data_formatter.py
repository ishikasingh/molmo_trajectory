#!/usr/bin/env python3
"""
Simple debug script for the affordance data formatter
"""

import argparse
import os
import sys
import numpy as np
from pathlib import Path

# Add the molmo directory to the path
sys.path.insert(0, str(Path(__file__).parent / "molmo"))

from olmo.data import get_dataset_by_name
from olmo.data.data_formatter import DataFormatter

def debug_example(dataset, formatter, index, rng):
    """Debug a single example"""
    print(f"\n{'='*60}")
    print(f"EXAMPLE {index}")
    print(f"{'='*60}")
    
    # Get raw example
    raw_example = dataset.get(index, rng)
    
    print("Raw example keys:", list(raw_example.keys()))
    
    if "message_list" in raw_example:
        for i, msg in enumerate(raw_example["message_list"]):
            print(f"Message {i}:")
            print(f"  Style: {msg.get('style', 'None')}")
            print(f"  Label: {msg.get('label', 'None')}")
            if "points" in msg:
                points = msg["points"]
                print(f"  Points shape: {points.shape}")
                print(f"  Points: {points}")
    
    # Format with data formatter
    print("\n--- FORMATTED OUTPUT ---")
    messages, metadata = formatter(raw_example, is_training=True, for_inference=False, rng=rng)
    
    print(f"Input:  {messages[0][0]}")
    print(f"Output: {messages[0][1]}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="affordance_new", choices=["affordance", "affordance_new", "affordance_with_transitions"])
    parser.add_argument("--split", default="train")
    parser.add_argument("--num_examples", type=int, default=3)
    args = parser.parse_args()
    
    # Load dataset
    print(f"Loading {args.dataset} dataset...")
    dataset = get_dataset_by_name(args.dataset, args.split)
    print(f"Dataset length: {len(dataset)}")
    
    # Create formatter
    formatter = DataFormatter(
        prompt_templates="uber_model",
        message_format="none",
        system_prompt="demo_or_style",
        debug=True
    )
    
    # Debug examples
    rng = np.random.RandomState(42)
    for i in range(args.num_examples):
        debug_example(dataset, formatter, i, rng)

if __name__ == "__main__":
    main() 