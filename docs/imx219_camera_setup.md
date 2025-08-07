# IMX219 CSI Camera Setup Guide

## Overview
The Pathlight system uses a **dual IMX219 CSI camera module** that connects directly to the NVIDIA Jetson Orin Nano via CSI (Camera Serial Interface) ribbon cables. This provides superior performance compared to USB cameras.

## Hardware Specifications

### IMX219 Camera Module
- **Sensor**: Sony IMX219 8MP CMOS sensor
- **Resolution**: Up to 3280x2464 (8MP)
- **Frame Rate**: Up to 30fps at 720p
- **Interface**: CSI-2 (MIPI)
- **Focus**: Fixed focus (no autofocus)
- **Dual Camera**: Left and right cameras for stereo vision

### Supported Resolutions
- 3280x2464 (8MP) - 15fps
- 1920x1080 (1080p) - 30fps
- 1640x1232 (2MP) - 30fps
- 1280x720 (720p) - 30fps
- 640x480 (480p) - 30fps

## Physical Connection

### CSI Cable Connection
1. **Locate CSI ports** on Jetson Orin Nano:
   - CSI0 (left camera)
   - CSI1 (right camera)

2. **Connect ribbon cables**:
   - Connect left camera to CSI0
   - Connect right camera to CSI1
   - Ensure cables are seated properly and locked

3. **Power connection**:
   - Camera modules are powered through CSI interface
   - No additional power connection required

### Cable Orientation
- **Blue side** of ribbon cable faces the **blue side** of the connector
- **Red stripe** indicates pin 1
- Ensure cables are fully inserted and locked

## Software Configuration

### Device Paths
- **Left camera**: `/dev/video0`
- **Right camera**: `/dev/video1`

### Configuration Settings
```yaml
# config/config.yaml
camera:
  device_id: 0  # 0 for left camera, 1 for right camera
  width: 1280   # 720p resolution
  height: 720
  fps: 30
  format: "MJPEG"  # Better performance than YUYV
  auto_exposure: true
  auto_focus: true  # IMX219 has fixed focus
```

## Testing the Camera

### Check Camera Detection
```bash
# List all video devices
v4l2-ctl --list-devices

# Check IMX219 camera properties
v4l2-ctl --device=/dev/video0 --list-formats-ext
v4l2-ctl --device=/dev/video1 --list-formats-ext
```

### Test Camera Capture
```bash
# Test left camera
python3 -c "
import cv2
cap = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
ret, frame = cap.read()
if ret:
    print('Left camera working!')
    cv2.imwrite('left_camera_test.jpg', frame)
cap.release()
"

# Test right camera
python3 -c "
import cv2
cap = cv2.VideoCapture('/dev/video1', cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
ret, frame = cap.read()
if ret:
    print('Right camera working!')
    cv2.imwrite('right_camera_test.jpg', frame)
cap.release()
"
```

### Test with Pathlight
```bash
# Run the test script
python scripts/test_installation.py

# Or test camera module directly
python -c "
from hardware.camera.camera_manager import CameraManager
import yaml

with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

camera = CameraManager(config['camera'])
camera.start()
frame = camera.get_frame()
if frame is not None:
    print(f'Camera working! Frame shape: {frame.shape}')
    import cv2
    cv2.imwrite('pathlight_camera_test.jpg', frame)
camera.stop()
"
```

## Troubleshooting

### Camera Not Detected
1. **Check physical connection**:
   - Ensure ribbon cables are properly seated
   - Check cable orientation (blue to blue)
   - Verify cables are locked in place

2. **Check device paths**:
   ```bash
   ls -la /dev/video*
   ```

3. **Check kernel modules**:
   ```bash
   lsmod | grep tegra
   ```

4. **Restart camera service**:
   ```bash
   sudo systemctl restart nvidia-l4t-camera
   ```

### Poor Image Quality
1. **Check resolution settings**:
   - Ensure resolution is supported by IMX219
   - Try different resolutions for optimal performance

2. **Check format settings**:
   - MJPEG for better performance
   - YUYV for better quality

3. **Check exposure settings**:
   ```bash
   v4l2-ctl --device=/dev/video0 --set-ctrl=exposure_auto=1
   ```

### Performance Issues
1. **Reduce resolution**:
   - Use 720p instead of 1080p
   - Use 480p for maximum performance

2. **Use MJPEG format**:
   - Better compression and performance
   - Less CPU usage

3. **Adjust frame rate**:
   - Lower FPS for better performance
   - Higher FPS for smoother video

## Advanced Configuration

### Stereo Vision Setup
For stereo vision applications, configure both cameras:

```python
# Initialize both cameras
left_camera = CameraManager(config['camera'])
right_camera = CameraManager(config['camera'])
right_camera.device_id = 1

# Synchronize captures
left_frame = left_camera.get_frame()
right_frame = right_camera.get_frame()
```

### Custom Camera Settings
```python
# Set custom camera properties
camera.camera.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)
camera.camera.set(cv2.CAP_PROP_CONTRAST, 0.5)
camera.camera.set(cv2.CAP_PROP_SATURATION, 0.5)
camera.camera.set(cv2.CAP_PROP_HUE, 0.0)
```

## Performance Optimization

### Recommended Settings for Pathlight
```yaml
camera:
  device_id: 0
  width: 1280    # 720p - good balance of quality and performance
  height: 720
  fps: 30        # Real-time performance
  format: "MJPEG" # Better performance
  auto_exposure: true
  auto_focus: true
```

### Memory Optimization
- Use smaller resolution for object detection
- Process frames at lower resolution for AI tasks
- Use frame skipping if needed

### GPU Acceleration
- IMX219 works well with Jetson's hardware acceleration
- Use CUDA for AI processing
- Optimize YOLOv8 for Jetson

## Support

### Useful Commands
```bash
# Check camera status
v4l2-ctl --device=/dev/video0 --all

# Set camera properties
v4l2-ctl --device=/dev/video0 --set-ctrl=brightness=128

# Capture test image
v4l2-ctl --device=/dev/video0 --stream-mmap --stream-count=1 --stream-to=test.raw
```

### Logs
- Camera logs: `dmesg | grep -i camera`
- System logs: `journalctl -u nvidia-l4t-camera`

### Documentation
- NVIDIA Jetson Camera documentation
- IMX219 datasheet
- CSI interface specifications 