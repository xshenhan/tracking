# PointTracker

A real-time point tracking tool based on SpaTrackerV2, designed for RGB-D cameras.

## Author

- **Xiaoshen Han** - hxs001@sjtu.edu.cn

## Features

- Real-time point tracking with RGB-D input
- 2D pixel coordinate tracking
- 3D world coordinate estimation (camera frame)
- Interactive testing interface
- On-demand inference with frame downsampling

## Requirements

### Hardware

- NVIDIA GPU with CUDA support (tested on RTX 5090)
- Intel RealSense D435 or compatible RGB-D camera

### Software Dependencies

- Python 3.10+
- PyTorch 2.9+ with CUDA
- Intel RealSense SDK (`pyrealsense2`)
- OpenCV (`opencv-python`)
- NumPy

### Required Subpackage

This project depends on **SpaTrackerV2**:

```
SpaTrackerV2/
├── models/
│   └── SpaTrackV2/
│       └── models/
│           └── predictor.py    # Predictor class
```

The model is loaded from HuggingFace: `Yuxihenry/SpatialTrackerV2-Online`

## Installation

1. Create conda environment:
```bash
conda create -n SpaTrack python=3.10
conda activate SpaTrack
```

2. Install PyTorch with CUDA:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

3. Install dependencies:
```bash
pip install pyrealsense2 opencv-python numpy
```

4. Clone SpaTrackerV2 (required subpackage):
```bash
git clone https://github.com/henry123-boy/SpaTrackerV2.git
```

## Project Structure

```
tracking/
├── tracker_tool.py           # Core PointTracker class
├── interactive_tracker.py    # Interactive testing script
├── SpaTrackerV2/             # Required subpackage (SpaTrackerV2 repo)
├── tests/                    # Test scripts and data
│   ├── test_realsense.py
│   ├── test_model_load.py
│   ├── test_static_track.py
│   ├── test_2d_to_3d.py
│   ├── record_video.py
│   ├── annotate_frames.py
│   ├── compare_tracking.py
│   ├── test_downsample_track.py
│   └── generate_tracking_video.py
└── README.md
```

## Usage

### Basic API

```python
from tracker_tool import PointTracker

# Create tracker (8-frame downsampling)
tracker = PointTracker(sample_frames=8)

# Set the point to track (pixel coordinates on first frame)
tracker.set_query_point(x, y)

# Add frames continuously
tracker.add_frame(rgb_image, depth_image, intrinsic_matrix)

# Get current position (triggers inference)
result = tracker.get_position()
# result = {
#     'position_2d': (u, v),           # Pixel coordinates
#     'position_3d': (x, y, z),        # 3D position in camera frame (meters)
#     'visibility': 0.95,              # Visibility score
#     'frame_count': 100,              # Frames in buffer
#     'inference_time': 0.105          # Inference time (seconds)
# }

# Reset tracker
tracker.reset()
```

### Interactive Testing

```bash
conda activate SpaTrack
python interactive_tracker.py
```

**Controls:**
- **Mouse click**: Set tracking point
- **Space**: Trigger tracking, show result
- **r**: Reset
- **q**: Quit

### API Reference

#### `PointTracker(sample_frames=8, buffer_size=1000, model_input_size=384, device="cuda")`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sample_frames` | int | 8 | Number of frames to downsample for inference |
| `buffer_size` | int | 1000 | Maximum frames to keep in buffer (~33s at 30fps) |
| `model_input_size` | int | 384 | Model input resolution |
| `device` | str | "cuda" | Compute device |

#### Methods

| Method | Description |
|--------|-------------|
| `set_query_point(x, y)` | Set pixel coordinates to track |
| `add_frame(rgb, depth, intrinsic)` | Add a frame to the buffer |
| `get_position()` | Get current tracked position (triggers inference) |
| `reset()` | Clear buffer and reset state |
| `get_status()` | Get tracker status |

## Performance

- **Inference time**: ~0.1s (with model loaded)
- **Tracking accuracy**: <5 pixels error
- **Minimum frames**: 4 frames required for tracking

## Notes

- The first frame (when `set_query_point` is called) is used as the reference
- Model is lazy-loaded on first `get_position()` call
- Requires at least 4 frames before tracking can be performed
- Fixed camera assumption (stage 1)

## License

MIT License
