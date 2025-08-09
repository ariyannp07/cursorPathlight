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
1. Download JetPack SDK 6.2.1 from NVIDIA Developer website
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

# Verify CUDA installation (should be pre-installed with JetPack 6.2.1)
nvidia-smi  # Should show CUDA 12.6
nvcc --version  # Should show CUDA 12.6
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

### 2.2 Install PyTorch for Jetson with CUDA Support
```bash
# Install NVIDIA's official PyTorch wheel with CUDA 12.6 support
pip install --no-cache https://developer.download.nvidia.com/compute/redist/jp/v62/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl

# Install compatible torchvision and torchaudio
pip install torchvision==0.20.0 --no-deps
pip install torchaudio==2.5.0 --no-deps
```

### 2.3 Install Core Dependencies
```bash
# Install OpenCV with CUDA support
pip install opencv-python
pip install opencv-contrib-python

# Install other core dependencies
pip install numpy pillow matplotlib scipy
pip install ultralytics  # YOLOv8 with CUDA support
pip install face-recognition
pip install scikit-learn
```

## Step 3: Hardware Setup

### 3.1 Camera Setup
```bash
# Install camera utilities
sudo apt install -y v4l-utils

# Test IMX219 CSI camera
v4l2-ctl --list-devices

# You should see something like:
# vi-output, imx219 10-0010 (platform:tegra-capt):
#         /dev/video0
```

### 3.2 Audio Setup
```bash
# Install audio dependencies
sudo apt install -y portaudio19-dev python3-pyaudio espeak pulseaudio

# Test audio
speaker-test -t sine -f 1000 -c 2 -s 1
```

### 3.3 I2C and GPIO Setup
```bash
# Install I2C tools
sudo apt install -y i2c-tools python3-smbus2 python3-gpiozero

# Enable I2C in boot config
echo "i2c_arm=on" | sudo tee -a /boot/firmware/config.txt

# Test I2C
sudo i2cdetect -y 1
```

## Step 4: Pathlight Installation

### 4.1 Clone Repository
```bash
cd ~/cursorPathlight
git clone https://github.com/ariyannp07/cursorPathlight.git .
```

### 4.2 Run Setup Script
```bash
chmod +x scripts/setup_jetson.sh
./scripts/setup_jetson.sh
```

### 4.3 Configure Settings
```bash
# Copy example configuration
cp config/config.example.yaml config/config.yaml

# Edit configuration file
nano config/config.yaml

# Add your API keys:
# - OpenAI API key for AI assistant
# - Google Maps API key for navigation
```

## Step 5: Performance Optimization

### 5.1 Set Performance Mode
```bash
# Set maximum performance mode
sudo nvpmodel -m 0

# Enable GPU persistence mode
sudo nvidia-smi -pm 1

# Check current power mode
sudo nvpmodel -q
```

### 5.2 Memory Optimization
```bash
# Add swap space if needed (for 4GB models)
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

## Step 6: Testing and Verification

### 6.1 Run Installation Test
```bash
cd ~/cursorPathlight
source cursorPathlight_env/bin/activate
python scripts/test_installation.py
```

Expected results:
- ✅ Module Imports: PASS
- ✅ CUDA PyTorch Support: PASS
- ✅ Project Modules: PASS
- ✅ Configuration: PASS
- ✅ Directories: PASS
- ✅ YOLOv8 Model: PASS
- ✅ Audio System: PASS
- ✅ Camera Test: PASS

### 6.2 Test CUDA Performance
```bash
# Test PyTorch CUDA
python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Device: {torch.cuda.get_device_name(0)}')

# Performance test
import time
x = torch.randn(1000, 1000).cuda()
start = time.time()
y = torch.matmul(x, x.T)
end = time.time()
print(f'CUDA matrix multiply time: {(end-start)*1000:.2f}ms')
"

# Test YOLOv8 performance
python3 -c "
from ultralytics import YOLO
import torch
model = YOLO('yolov8n.pt')
if torch.cuda.is_available():
    model.to('cuda')
    print('YOLOv8 running on CUDA')
else:
    print('YOLOv8 running on CPU')
"
```

## Step 7: Service Setup

### 7.1 Enable Auto-Start Service
```bash
# Service should be automatically configured by setup script
sudo systemctl enable cursorPathlight.service
sudo systemctl status cursorPathlight.service
```

### 7.2 Manual Start
```bash
# Start manually for testing
cd ~/cursorPathlight
source cursorPathlight_env/bin/activate
python main.py
```

## Expected Performance with CUDA

| Component | CPU-only | CUDA Accelerated | Improvement |
|-----------|----------|------------------|-------------|
| Object Detection (YOLOv8) | 2-5 FPS | 15-30 FPS | 6-10x faster |
| Face Recognition | 1-3 FPS | 5-15 FPS | 3-5x faster |
| Stereo Vision | 5-10 FPS | 10-20 FPS | 2x faster |

## Troubleshooting

### CUDA Issues
```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Check PyTorch CUDA support
python -c "import torch; print(torch.cuda.is_available())"

# Check GPU memory usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

### Performance Issues
```bash
# Check current power mode
sudo nvpmodel -q

# Monitor GPU utilization
nvidia-smi -l 1

# Check temperature
tegrastats
```

### Memory Issues
```bash
# Check memory usage
free -h

# Monitor with htop
htop

# Check swap usage
swapon --show
```

### Camera Issues
```bash
# List video devices
v4l2-ctl --list-devices

# Test camera capture
gst-launch-1.0 nvarguscamerasrc ! 'video/x-raw(memory:NVMM),width=1920,height=1080' ! nvvidconv flip-method=0 ! 'video/x-raw,width=960,height=540' ! nvvidconv ! nvegltransform ! nveglglessink -e
```

## Additional Resources

- [NVIDIA Jetson Developer Guide](https://docs.nvidia.com/jetson/)
- [JetPack 6.2.1 Release Notes](https://docs.nvidia.com/jetson/jetpack/release-notes/)
- [PyTorch for Jetson](https://forums.developer.nvidia.com/t/pytorch-for-jetson)
- [Jetson Zoo](https://elinux.org/Jetson_Zoo)

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review system logs: `journalctl -u cursorPathlight.service`
3. Test individual components with the test script
4. Monitor system resources with `htop` and `nvidia-smi`