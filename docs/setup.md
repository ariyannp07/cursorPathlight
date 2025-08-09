# Pathlight Setup Guide for NVIDIA Jetson Orin Nano

## Prerequisites
- NVIDIA Jetson Orin Nano Developer Kit
- 16GB+ microSD card or NVMe SSD
- USB keyboard and mouse
- HDMI monitor (for initial setup)
- Internet connection
- Camera module (compatible with Jetson)

## Step 1: Initial Jetson Setup

### 1.1 Flash JetPack SDK
1. Download JetPack SDK 5.1.2 from NVIDIA Developer website
2. Install JetPack SDK on your host computer
3. Connect Jetson to host computer via USB-C
4. Put Jetson in recovery mode:
   - Power off Jetson
   - Hold RECOVERY button
   - Press POWER button while holding RECOVERY
   - Release RECOVERY after 2 seconds
5. Run JetPack SDK and follow flashing instructions
6. Set up username and password when prompted

### 1.2 Initial Configuration
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install essential tools
sudo apt install -y git curl wget vim nano htop

# Set up Python environment
sudo apt install -y python3-pip python3-venv python3-dev

# Note: CUDA support disabled for compatibility
# We will use CPU-only operation for reliable performance
```

## Step 2: Python Environment Setup

### 2.1 Create Virtual Environment
```bash
# Create project directory
mkdir -p ~/cursorPathlight
cd ~/cursorPathlight

# Create virtual environment
python3 -m venv cursorPathlight_env
source cursorPathlight_env/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### 2.2 Install PyTorch for Jetson (CPU-only)
```bash
# Install PyTorch CPU-only version for better compatibility on Jetson
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 2.3 Install Core Dependencies
```bash
# Install OpenCV (CPU version for compatibility)
pip install opencv-python-headless
pip install opencv-contrib-python-headless

# Install other core dependencies
pip install numpy pillow matplotlib scipy
pip install ultralytics  # YOLOv8
pip install face-recognition
pip install scikit-learn
```

## Step 3: Hardware Setup

### 3.1 Camera Setup
```bash
# Install camera utilities
sudo apt install -y v4l-utils

# Check available cameras
v4l2-ctl --list-devices

# Test camera (replace /dev/video0 with your camera device)
v4l2-ctl --device=/dev/video0 --list-formats-ext
```

### 3.2 LED Array Setup
```bash
# Install GPIO libraries
sudo apt install -y python3-gpiozero

# For I2C communication (if using I2C LED drivers)
sudo apt install -y i2c-tools
sudo i2cdetect -y 1  # Check I2C devices
```

### 3.3 Audio Setup
```bash
# Install audio dependencies
sudo apt install -y portaudio19-dev python3-pyaudio
sudo apt install -y espeak espeak-data  # Text-to-speech engine

# Install Python audio libraries
pip install pyaudio pyttsx3 speechrecognition
```

## Step 4: Project Installation

### 4.1 Clone/Transfer Project
```bash
# If using git
git clone <repository-url> ~/cursorPathlight

# Or transfer files from your Mac
# Copy the pathlight_project folder to ~/cursorPathlight on Jetson
```

### 4.2 Install Project Dependencies
```bash
cd ~/cursorPathlight
pip install -r requirements.txt
```

### 4.3 Environment Configuration
```bash
# Create environment file
cp config/config.example.yaml config/config.yaml

# Edit configuration
nano config/config.yaml
```

## Step 5: System Configuration

### 5.1 Enable Required Services
```bash
# Enable I2C
sudo raspi-config  # If using Raspberry Pi GPIO
# Navigate to Interface Options > I2C > Enable

# Set up auto-start service
sudo nano /etc/systemd/system/cursorPathlight.service
```

### 5.2 Performance Optimization
```bash
# Set GPU memory allocation
sudo nano /etc/systemd/system/nvzramconfig.service

# Optimize for real-time processing
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
echo 'vm.vfs_cache_pressure=50' | sudo tee -a /etc/sysctl.conf
```

## Step 6: Testing and Validation

### 6.1 Test Camera
```python
import cv2
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if ret:
    print("Camera working!")
    cv2.imwrite("test_image.jpg", frame)
cap.release()
```

### 6.2 Test YOLOv8
```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
results = model('test_image.jpg')
print("YOLOv8 working!")
```

### 6.3 Test Audio
```python
import pyttsx3
engine = pyttsx3.init()
engine.say("Pathlight audio system test")
engine.runAndWait()
```

## Step 7: Troubleshooting

### Common Issues:
1. **Camera not detected**: Check USB connection and permissions
2. **CUDA errors**: Verify JetPack installation and GPU memory
3. **Audio issues**: Check ALSA configuration and device permissions
4. **Performance issues**: Monitor GPU usage with `tegrastats`

### Useful Commands:
```bash
# Monitor system resources
tegrastats
htop
nvidia-smi

# Check camera
v4l2-ctl --list-devices
v4l2-ctl --device=/dev/video0 --list-formats-ext

# Test audio
speaker-test -t wav -c 2
arecord -l  # List recording devices
```

## Next Steps
After completing this setup:
1. Configure the main application settings
2. Test individual components
3. Run integration tests
4. Begin development of specific features

See `docs/development.md` for development guidelines. 