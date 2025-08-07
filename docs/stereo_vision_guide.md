# Stereo Vision Setup and Usage Guide

## Overview
This guide covers the complete setup and usage of the stereo vision system using dual IMX219 CSI cameras for accurate 3D depth perception and obstacle detection.

## Table of Contents
1. [Hardware Setup](#hardware-setup)
2. [Software Installation](#software-installation)
3. [Camera Calibration](#camera-calibration)
4. [Testing and Validation](#testing-and-validation)
5. [Troubleshooting](#troubleshooting)
6. [Advanced Usage](#advanced-usage)

## Hardware Setup

### Required Components
- **Dual IMX219 CSI Camera Module** (ribbon cable connection)
- **NVIDIA Jetson Orin Nano**
- **Chessboard pattern** (9x6 internal corners, 2.5cm squares)
- **Camera mounting bracket** (for proper alignment)

### Physical Connection
1. **Connect CSI cables**:
   - Left camera → CSI0 port
   - Right camera → CSI1 port
   - Ensure cables are fully seated and locked

2. **Camera alignment**:
   - Cameras should be parallel and level
   - Typical separation: 12cm (adjustable in config)
   - Ensure no obstructions between cameras

3. **Power connection**:
   - Cameras are powered through CSI interface
   - No additional power required

### Verification
```bash
# Check camera detection
v4l2-ctl --list-devices

# Expected output should show:
# /dev/video0 - Left IMX219 camera
# /dev/video1 - Right IMX219 camera
```

## Software Installation

### 1. System Dependencies
```bash
# Install Jetson-specific camera packages
sudo apt update
sudo apt install -y \
    nvidia-l4t-camera \
    nvidia-l4t-multimedia \
    v4l-utils \
    libv4l-dev

# Install OpenCV dependencies
sudo apt install -y \
    libopencv-dev \
    python3-opencv \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev
```

### 2. Python Environment
```bash
# Create virtual environment
python3 -m venv cursorPathlight_env
source cursorPathlight_env/bin/activate

# Install Python dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 3. OpenCV Verification
```python
import cv2
import numpy as np

# Check OpenCV version
print(f"OpenCV version: {cv2.__version__}")

# Check CUDA support
print(f"CUDA devices: {cv2.cuda.getCudaEnabledDeviceCount()}")

# Test camera access
cap = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L2)
if cap.isOpened():
    print("✓ Camera access working")
    cap.release()
else:
    print("✗ Camera access failed")
```

## Camera Calibration

### Why Calibration is Important
- **Accuracy**: Calibrated cameras provide much more accurate depth measurements
- **Rectification**: Proper stereo rectification improves depth computation
- **Error Reduction**: Reduces depth estimation errors by 50-80%

### Calibration Process

#### 1. Prepare Calibration Pattern
- Print a chessboard pattern (9x6 internal corners)
- Mount on a rigid, flat surface
- Ensure squares are exactly 2.5cm (adjustable in config)

#### 2. Run Calibration Tool
```bash
# Activate environment
source cursorPathlight_env/bin/activate

# Run calibration
python scripts/stereo_calibration.py
```

#### 3. Calibration Instructions
1. **Hold chessboard** at different angles and distances
2. **Ensure both cameras** can see the pattern clearly
3. **Press 'c'** when pattern is detected (green overlay appears)
4. **Capture 15-20 images** from various positions
5. **Press 's'** to start calibration when enough images are captured

#### 4. Calibration Positions
- **Frontal**: Pattern facing cameras directly
- **Angled**: Pattern rotated 15-45 degrees
- **Distances**: 0.5m to 3m from cameras
- **Heights**: Different vertical positions
- **Orientations**: Pattern tilted in different directions

### Calibration Quality Assessment
```bash
# Check calibration file
ls -la data/stereo_calibration.pkl

# Expected calibration error: < 0.5 pixels
# Higher errors indicate poor calibration
```

## Testing and Validation

### 1. Basic Camera Test
```bash
# Test camera detection
python scripts/test_stereo_vision.py --test camera

# Expected output:
# ✓ At least 2 cameras detected
# ✓ Both cameras captured frames
# ✓ Frame sizes match
```

### 2. Depth Computation Test
```bash
# Test depth computation
python scripts/test_stereo_vision.py --test depth

# Check output images in test_output/:
# - left_frame.jpg
# - right_frame.jpg
# - depth_map.jpg (color-coded depth)
# - disparity_map.jpg
```

### 3. Full System Test
```bash
# Run all tests
python scripts/test_stereo_vision.py

# This tests:
# - Camera detection and synchronization
# - Stereo calibration status
# - Depth computation
# - 3D obstacle detection
# - Safe path planning
# - Performance metrics
```

### 4. Performance Benchmarks
**Expected Performance on Jetson Orin Nano:**
- **Depth computation**: 15-30ms per frame
- **Obstacle detection**: 5-10ms per frame
- **Path planning**: 2-5ms per frame
- **Total pipeline**: 25-50ms per frame
- **Overall FPS**: 10-20 FPS

## Configuration

### Stereo Vision Settings
```yaml
# config/config.yaml
stereo_vision:
  calibration_file: "data/stereo_calibration.pkl"
  camera_separation: 0.12  # meters (adjust based on your setup)
  focal_length: 1000.0     # pixels (will be calibrated)
  min_depth: 0.1           # meters
  max_depth: 10.0          # meters
  
  # Stereo matching parameters
  block_size: 11           # Larger = smoother but slower
  num_disparities: 128     # Higher = better depth range
  uniqueness_ratio: 15     # Higher = more strict matching
  speckle_window_size: 100 # Noise reduction
  speckle_range: 32        # Disparity change threshold
```

### Camera Settings
```yaml
camera:
  device_id: 0             # 0 for left, 1 for right
  width: 1280              # 720p for good performance
  height: 720
  fps: 30                  # IMX219 can handle 30fps
  format: "MJPEG"          # Better performance than YUYV
```

## Usage Examples

### 1. Basic Depth Measurement
```python
from hardware.camera.camera_manager import CameraManager
from core.vision.stereo_vision import StereoVision
import yaml

# Load configuration
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize components
camera = CameraManager(config['camera'])
stereo = StereoVision(config['stereo_vision'])

# Capture stereo frames
left_frame, right_frame = camera.get_stereo_frames()

# Compute depth
result = stereo.compute_depth(left_frame, right_frame)
depth_map = result['depth_map']

# Get distance to center point
center_depth = depth_map[depth_map.shape[0]//2, depth_map.shape[1]//2]
print(f"Distance to center: {center_depth:.2f}m")
```

### 2. 3D Obstacle Detection
```python
# Detect obstacles
obstacles = stereo.detect_obstacles_3d(depth_map)

for obstacle in obstacles:
    print(f"Obstacle at {obstacle['distance']:.2f}m")
    print(f"Height: {obstacle['height']:.2f}m")
    print(f"3D position: {obstacle['position_3d']}")
    print(f"Dangerous: {obstacle['is_dangerous']}")
```

### 3. Safe Path Planning
```python
# Calculate safe path
path = stereo.calculate_safe_path_3d(obstacles)

print(f"Safe direction: {np.degrees(path['safe_direction']):.1f}°")
print(f"Confidence: {path['confidence']:.2f}")
print(f"Available directions: {len(path['safe_directions'])}")
```

## Troubleshooting

### Common Issues

#### 1. Camera Not Detected
**Symptoms:** `No cameras found` or `Could not open camera`
**Solutions:**
```bash
# Check physical connections
ls -la /dev/video*

# Restart camera service
sudo systemctl restart nvidia-l4t-camera

# Check kernel modules
lsmod | grep tegra
```

#### 2. Poor Depth Quality
**Symptoms:** Noisy depth maps, incorrect distances
**Solutions:**
- **Recalibrate cameras** with more images
- **Check camera alignment** (parallel and level)
- **Adjust stereo parameters** in config
- **Ensure good lighting** conditions

#### 3. Low Performance
**Symptoms:** <5 FPS, high computation times
**Solutions:**
- **Reduce resolution** to 640x480
- **Use MJPEG format** instead of YUYV
- **Adjust block_size** parameter
- **Check CPU/GPU usage** with `htop` and `nvidia-smi`

#### 4. Calibration Errors
**Symptoms:** High calibration error (>1.0 pixels)
**Solutions:**
- **Use more calibration images** (20-30)
- **Ensure pattern is flat** and rigid
- **Check square size** measurement
- **Use different angles** and distances

### Performance Optimization

#### 1. Resolution Optimization
```yaml
# For maximum performance
camera:
  width: 640
  height: 480
  fps: 15

# For maximum accuracy
camera:
  width: 1280
  height: 720
  fps: 30
```

#### 2. Stereo Parameter Tuning
```yaml
# Fast processing
stereo_vision:
  block_size: 15
  num_disparities: 64
  speckle_window_size: 50

# High accuracy
stereo_vision:
  block_size: 11
  num_disparities: 128
  speckle_window_size: 100
```

## Advanced Usage

### 1. Custom Depth Processing
```python
# Apply custom filters to depth map
def filter_depth_map(depth_map, min_depth=0.1, max_depth=10.0):
    # Remove invalid depths
    filtered = depth_map.copy()
    filtered[filtered < min_depth] = 0
    filtered[filtered > max_depth] = 0
    
    # Apply median filter for noise reduction
    filtered = cv2.medianBlur(filtered.astype(np.float32), 5)
    
    return filtered
```

### 2. Point Cloud Generation
```python
# Generate 3D point cloud
points_3d = stereo.get_3d_points(depth_map, left_frame)

# Save as PLY file
def save_point_cloud(points, filename):
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        for point in points:
            f.write(f"{point[0]} {point[1]} {point[2]} {point[3]} {point[4]} {point[5]}\n")

save_point_cloud(points_3d, "point_cloud.ply")
```

### 3. Real-time Visualization
```python
import cv2

def visualize_depth_realtime():
    camera = CameraManager(config['camera'])
    stereo = StereoVision(config['stereo_vision'])
    
    while True:
        left_frame, right_frame = camera.get_stereo_frames()
        result = stereo.compute_depth(left_frame, right_frame)
        
        # Create visualization
        depth_vis = cv2.normalize(result['depth_map'], None, 0, 255, cv2.NORM_MINMAX)
        depth_vis = cv2.applyColorMap(depth_vis.astype(np.uint8), cv2.COLORMAP_JET)
        
        # Display
        cv2.imshow('Left Camera', left_frame)
        cv2.imshow('Depth Map', depth_vis)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
```

## Maintenance

### Regular Tasks
1. **Clean camera lenses** monthly
2. **Check camera alignment** quarterly
3. **Recalibrate** if accuracy degrades
4. **Update software** when new versions are available

### Performance Monitoring
```bash
# Monitor system resources
htop
nvidia-smi

# Check camera status
v4l2-ctl --device=/dev/video0 --all
v4l2-ctl --device=/dev/video1 --all
```

## Support

### Useful Commands
```bash
# Check OpenCV installation
python3 -c "import cv2; print(cv2.__version__)"

# Test camera access
v4l2-ctl --list-devices

# Check calibration
ls -la data/stereo_calibration.pkl

# Run diagnostics
python scripts/test_stereo_vision.py
```

### Logs and Debugging
- **Camera logs**: `dmesg | grep -i camera`
- **OpenCV errors**: Check Python exception messages
- **Performance issues**: Monitor with `htop` and `nvidia-smi`

### Getting Help
1. **Check this guide** for common solutions
2. **Run test scripts** to identify issues
3. **Review logs** for error messages
4. **Verify hardware connections** and alignment 