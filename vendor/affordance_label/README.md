# Key Frame Detection for Affordance Dataset

This script automatically detects key frames in video sequences based on hand velocity transitions. Key frames are defined as frames where at least one hand transitions from moving (velocity > threshold) to stationary (velocity <= threshold). This captures the moment when an action stops or pauses, rather than detecting frames where hands are already stationary.

## Features

- **Transition-based detection**: Detects when hands change from moving to stationary states
- **Hand-specific analysis**: Separately analyzes left and right hand keypoints based on exact keypoint names from the affordance dataset
- **HDF5 support**: Reads 3D keypoint trajectories from HDF5 files and projects them to 2D coordinates
- **JSON support**: Also supports 2D keypoint data from JSON files
- **Binary output**: Generates binary strings (e.g., "00011001") indicating key frames
- **Video annotation**: Creates annotated videos with key frame indicators
- **Detailed logging**: Provides comprehensive analysis results

## Keypoint Structure

The script is specifically designed for the affordance dataset with these hand keypoints:

**Left Hand:**
- `leftHand`, `leftThumbTip`, `leftIndexFingerTip`, `leftMiddleFingerTip`, `leftRingFingerTip`, `leftLittleFingerTip`

**Right Hand:**
- `rightHand`, `rightThumbTip`, `rightIndexFingerTip`, `rightMiddleFingerTip`, `rightRingFingerTip`, `rightLittleFingerTip`

## HDF5 Data Structure

The script expects HDF5 files with the following structure:
```
file.hdf5/
├── transforms/
│   ├── camera          # [T, 4, 4] camera poses
│   ├── leftHand        # [T, 4, 4] transformation matrices
│   ├── leftThumbTip    # [T, 4, 4] transformation matrices
│   ├── rightHand       # [T, 4, 4] transformation matrices
│   └── ...             # Other keypoint trajectories
```

Each keypoint has a 4x4 transformation matrix per frame, where the 3D position is extracted from the translation component `[:3, 3]` and then projected to 2D coordinates using camera intrinsics.

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

1. **3D to 2D Projection**: For HDF5 files, extract 3D positions from transformation matrices and project to 2D using camera intrinsics
2. **Velocity Calculation**: For each keypoint, calculate the Euclidean distance between consecutive frames
3. **Hand State Analysis**: For each hand, determine if it's "moving" or "stationary" based on velocity thresholds:
   - **Moving**: Majority (≥50%) of hand keypoints have velocity > threshold
   - **Stationary**: Majority (≥50%) of hand keypoints have velocity ≤ threshold
4. **Transition Detection**: A frame is marked as a key frame if at least one hand transitions from moving to stationary
5. **Frame Classification**: Key frames represent moments when hands stop moving (action endpoints)

### Key Frame Definition

**Key Frame = Transition from Moving → Stationary**

- ✅ Hand was moving in previous frame AND is stationary in current frame
- ❌ Hand is stationary in both frames (not a transition)
- ❌ Hand is moving in both frames (no transition)
- ❌ Hand transitions from stationary to moving (start of movement, not endpoint)

This approach captures meaningful action boundaries where hands come to rest, indicating completion of manipulation tasks.

### Hand Keypoint Groupings

**Left Hand Keypoints:**
- `leftHand`, `leftThumbTip`, `leftIndexFingerTip`, `leftMiddleFingerTip`, `leftRingFingerTip`, `leftLittleFingerTip`

**Right Hand Keypoints:**  
- `rightHand`, `rightThumbTip`, `rightIndexFingerTip`, `rightMiddleFingerTip`, `rightRingFingerTip`, `rightLittleFingerTip`

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

- **Lower threshold** (e.g., 1.0): More sensitive, detects smaller movements as "moving" state
- **Higher threshold** (e.g., 10.0): Less sensitive, requires larger movements to be considered "moving"

**Typical values:**
- `--velocity_threshold 3.0`: Good for precise manipulation tasks
- `--velocity_threshold 5.0`: Good balance for general actions  
- `--velocity_threshold 10.0`: For detecting only major movements

### Expected Results

Key frames should correspond to:
- End of reaching motions
- Completion of grasping actions
- Moments when objects are placed down
- Pauses between manipulation steps

Key frames should NOT be detected during:
- Continuous movement phases
- Static periods where hands don't move at all

### Performance Tips

- Process episodes in parallel by modifying the main loop
- Use smaller videos for testing threshold values
- Consider subsampling frames if processing is slow

## Testing

Use the provided test script to verify the setup and see detection results:

```bash
# Test with a specific dataset folder
python test_keyframe_detection.py /path/to/your/dataset

# Or run without arguments and edit the script to set the path
python test_keyframe_detection.py
```

The test script will:
1. Find all episodes in your dataset
2. Test key frame detection on the first few episodes
3. Show detailed results including keypoint velocities and detection statistics
4. Verify that the HDF5 structure is correctly read

## Full Pipeline

Once testing is successful, run the full pipeline:

```bash
python key_frame_extract.py --data_folder /path/to/dataset --output_folder ./results --create_videos --velocity_threshold 5.0
```
