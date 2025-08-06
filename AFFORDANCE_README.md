# Affordance Dataset for Molmo

This guide explains how to use the new affordance prediction task in Molmo, which predicts hand keypoints (finger positions and wrist locations) for given instructions.

## Overview

The affordance dataset allows you to train Molmo to predict where hands should be positioned for specific tasks or instructions. This is useful for:
- Robotics applications
- Human-computer interaction
- Gesture recognition
- Action understanding

## Quick Start

1. **Prepare your data** in the expected format
2. **Convert to HuggingFace dataset** using the provided utilities
3. **Train** with the affordance dataset
4. **Evaluate** using affordance-specific metrics

## Data Format

Your raw data should include:

```python
{
    "image_path": "path/to/image.jpg",
    "instruction": "open the door",
    "hand_keypoints": {
        "left_hand": {
            "fingertips": [
                {"x": 25.0, "y": 30.0},  # thumb
                {"x": 22.0, "y": 28.0},  # index finger
                {"x": 20.0, "y": 26.0},  # middle finger
                {"x": 18.0, "y": 28.0},  # ring finger
                {"x": 16.0, "y": 30.0},  # pinky
            ],
            "wrist": {"x": 20.0, "y": 35.0}
        },
        "right_hand": {
            "fingertips": [
                {"x": 75.0, "y": 30.0},  # thumb
                {"x": 78.0, "y": 28.0},  # index finger
                {"x": 80.0, "y": 26.0},  # middle finger
                {"x": 82.0, "y": 28.0},  # ring finger
                {"x": 84.0, "y": 30.0},  # pinky
            ],
            "wrist": {"x": 80.0, "y": 35.0}
        }
    }
}
```

**Important**: 
- Coordinates are in percentage (0-100) relative to image dimensions
- You can have left hand only, right hand only, or both hands
- Each hand has 5 fingertips + 1 wrist = 6 keypoints maximum
- Total maximum keypoints = 12 (6 per hand × 2 hands)

## Converting Your Data

Use the provided utility to convert your raw data:

```python
from molmo.olmo.data.affordance_datsets import HandPositioningDataset

# Your raw data
raw_data = [
    {
        "image_path": "path/to/image1.jpg",
        "instruction": "open the door",
        "hand_keypoints": { ... }  # Your hand keypoint data
    },
    # ... more data
]

# Convert to HuggingFace dataset
dataset = HandPositioningDataset.create_hf_dataset(
    raw_data=raw_data,
    output_path="data/affordance_dataset",
    splits={"train": 0.8, "validation": 0.1, "test": 0.1}
)
```

## Training Configuration

Add to your training YAML config:

```yaml
data:
  datasets:
    - affordance:0.1      # 10% affordance data
    - pointing:0.3        # 30% pointing data  
    - user_qa:0.6         # 60% user QA data
  
  # Other data config...

evaluators:
  - data:
      dataset_name: affordance
      split: validation
    label: affordance_val
    affordance_eval: true
```

Set the environment variable:
```bash
export AFFORDANCE_DATA_DIR="path/to/your/affordance/dataset"
```

## Evaluation

Run evaluation with:

```bash
python -m molmo.launch_scripts.eval_model \
    --model_path path/to/your/model \
    --dataset_name affordance \
    --split test \
    --evaluator affordance_eval \
    --output_dir eval_results/
```

## Evaluation Metrics

The affordance evaluator provides:

- **Precision**: Fraction of predicted keypoints that match ground truth
- **Recall**: Fraction of ground truth keypoints that are predicted
- **F1 Score**: Harmonic mean of precision and recall
- **Mean Keypoint Distance**: Average pixel distance between predicted and ground truth keypoints
- **Accuracy at Thresholds**: Percentage of keypoints within 10px, 20px, 30px, 50px of ground truth

## Customizing for Your Data

If your data format is different, modify the `_process_hand_positions` method in `HandPositioningDataset`:

```python
def _process_hand_positions(self, hand_data):
    """Adapt this method to your data format"""
    points = []
    
    # Example: If your data has different structure
    if "hands" in hand_data:
        for hand in hand_data["hands"]:
            for keypoint in hand["keypoints"]:
                points.append([keypoint["x"], keypoint["y"]])
    
    return np.array(points, dtype=np.float32)
```

## Example Usage

See `affordance_example.py` for a complete example:

```bash
python affordance_example.py
```

This will show you:
1. How to create sample data
2. How to convert to HuggingFace format
3. How to load and use the dataset
4. Training and evaluation examples

## Advanced Usage

### Multiple Hands per Image

The system supports multiple hands per image. Each hand contributes up to 6 keypoints (5 fingertips + 1 wrist).

### Custom Prompts

The system uses various prompts for affordance prediction:
- "Predict the hand keypoints for {instruction}"
- "Show me where the hands should be positioned for {instruction}"
- "Indicate the hand positions for {instruction}"

### Integration with Other Tasks

You can mix affordance training with other tasks:

```yaml
data:
  datasets:
    - affordance:0.1
    - pointing:0.2
    - user_qa:0.4
    - long_caption:0.3
```

## Troubleshooting

1. **Dataset not found**: Make sure `AFFORDANCE_DATA_DIR` is set correctly
2. **Wrong data format**: Check that your `_process_hand_positions` method matches your data structure
3. **Poor performance**: Ensure your keypoint coordinates are in percentage (0-100) format
4. **Memory issues**: Use `keep_in_memory=False` when loading large datasets

## Files Added/Modified

- `molmo/olmo/data/affordance_datsets.py` - Main dataset class
- `molmo/olmo/data/data_formatter.py` - Added affordance prompts and formatting
- `molmo/olmo/data/__init__.py` - Registered the dataset
- `molmo/olmo/eval/evaluators.py` - Added AffordanceEval evaluator
- `molmo/olmo/eval/inf_evaluator.py` - Added affordance evaluation support
- `molmo/olmo/config.py` - Added affordance_eval config option
- `molmo/launch_scripts/utils.py` - Added affordance evaluation utilities

## Next Steps

1. Prepare your affordance data in the expected format
2. Run the example script to test the integration
3. Adapt the `_process_hand_positions` method for your data
4. Create your HuggingFace dataset
5. Train your model with affordance data
6. Evaluate using the affordance metrics

For questions or issues, please check the example script and ensure your data format matches the expected structure. 