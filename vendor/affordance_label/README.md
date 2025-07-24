# Key Frame Detection for Affordance Dataset

This script automatically detects key frames in video sequences based on hand keypoint velocities. Key frames are defined as frames where at least one hand has keypoints with velocities below a specified threshold.

## Features

- **Hand-specific detection**: Separately analyzes left and right hand keypoints
- **Flexible input**: Supports both JSON and HDF5 keypoint files
- **Binary output**: Generates binary strings (e.g., "00011001") indicating key frames
- **Video annotation**: Creates annotated videos with key frame indicators
- **Detailed logging**: Provides comprehensive analysis results

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Structure

The script expects your dataset to follow this structure:

```
dataset/
├── 0.mp4                                    # Video file
├── 0_keypoint_projections_keypoints.json   # Keypoints (preferred)
├── 0.hdf5                                   # Alternative keypoints source
├── 1.mp4
├── 1_keypoint_projections_keypoints.json
├── 1.hdf5
└── ...
```

## Usage

### Command Line Interface

```bash
python key_frame_extract.py \
    --data_folder /path/to/your/dataset \
    --output_folder ./results \
    --velocity_threshold 5.0 \
    --fps 30 \
    --create_videos \
    --font_size 50
```

### Parameters

- `--data_folder`: Path to folder containing episode data (required)
- `--output_folder`: Path to save output files (required)
- `--velocity_threshold`: Maximum velocity (pixels/frame) for keypoint to be considered stationary (default: 5.0)
- `--fps`: Video frame rate (default: 30)
- `--create_videos`: Flag to create annotated videos
- `--font_size`: Font size for video annotations (default: 50)

### Python API

```python
from key_frame_extract import KeyFrameDetector

# Initialize detector
detector = KeyFrameDetector(velocity_threshold=3.0, fps=30)

# Process a single episode
keyframe_flags, total_frames = detector.process_episode(
    mp4_path="path/to/video.mp4",
    keypoints_source="path/to/keypoints.json"
)

# Save results
detector.save_keyframe_json(
    keyframe_flags, 
    "output/keyframes.json", 
    "episode_1"
)

# Create annotated video
detector.create_annotated_video(
    "path/to/video.mp4",
    keyframe_flags,
    "output/annotated_video.mp4"
)
```

## Output Format

### JSON Output

Each episode generates a JSON file with the following structure:

```json
{
  "episode_id": "0",
  "total_frames": 150,
  "keyframe_count": 12,
  "keyframe_binary": "00011001000110010001100100011001...",
  "keyframe_indices": [3, 4, 8, 9, 13, 14, 18, 19, 23, 24, 28, 29],
  "parameters": {
    "velocity_threshold": 5.0,
    "fps": 30
  }
}
```

### Binary String Format

The `keyframe_binary` field contains a string where:
- `1` indicates a key frame
- `0` indicates a regular frame
- Each character corresponds to one frame in sequence

Example: `"00011001"` means frames 2, 3, and 6 are key frames.

## Key Frame Detection Logic

1. **Velocity Calculation**: For each keypoint, calculate the Euclidean distance between consecutive frames
2. **Hand Classification**: Identify which keypoints belong to left/right hands using pattern matching
3. **Threshold Comparison**: Compare velocities against the specified threshold
4. **Frame Classification**: A frame is marked as a key frame if at least one hand has all keypoints below the velocity threshold

### Hand Keypoint Patterns

The script recognizes these patterns as hand keypoints:
- `left_hand`, `lefthand`, `left_wrist`, `left_thumb`, etc.
- `right_hand`, `righthand`, `right_wrist`, `right_thumb`, etc.
- Generic patterns: `hand`, `wrist`, `thumb`, `index`, `middle`, `ring`, `pinky`

## Example Workflow

1. **Prepare your dataset** with MP4 videos and corresponding keypoint files
2. **Run the detection script**:
   ```bash
   python key_frame_extract.py --data_folder ./dataset --output_folder ./results --create_videos
   ```
3. **Review the results**:
   - Check JSON files for binary key frame sequences
   - Watch annotated videos to verify detection quality
   - Adjust `velocity_threshold` if needed

## Troubleshooting

### Common Issues

1. **Keypoints length mismatch**: The script automatically handles cases where keypoint data doesn't match video frame count
2. **Missing files**: Ensure each MP4 has a corresponding JSON or HDF5 file
3. **Invalid keypoint format**: The script supports multiple keypoint data structures

### Adjusting Sensitivity

- **Lower threshold** (e.g., 1.0): More sensitive, detects more key frames
- **Higher threshold** (e.g., 10.0): Less sensitive, detects fewer key frames

### Performance Tips

- Process episodes in parallel by modifying the main loop
- Use smaller videos for testing threshold values
- Consider subsampling frames if processing is slow

## Testing

Use the provided test script to verify the setup:

```bash
python test_keyframe_detection.py
```

Update the `data_folder` path in the test script to point to your dataset.
