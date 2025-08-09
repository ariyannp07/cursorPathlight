# 🎯 COMPLETE JETSON SETUP GUIDE - FINAL VERSION

**This is the definitive guide to set up Pathlight on Jetson Orin Nano with CUDA acceleration.**

---

## 📋 **OVERVIEW**

This guide provides **exhaustive, step-by-step instructions** to:
1. **Clean house** - Remove all old, conflicting installations
2. **Install the EXACT compatible software stack** 
3. **Ensure 8/8 tests pass** with full CUDA acceleration
4. **Get Pathlight running at 5+ FPS**

---

## 🧹 **STEP 1: COMPLETE CLEANUP ON JETSON**

### **1.1 Stop All Running Services**
```bash
# Stop any running Pathlight services
sudo systemctl stop pathlight.service 2>/dev/null || true
sudo systemctl stop cursorPathlight.service 2>/dev/null || true
sudo systemctl disable pathlight.service 2>/dev/null || true
sudo systemctl disable cursorPathlight.service 2>/dev/null || true

# Kill any Python processes
sudo pkill -f "python.*pathlight" 2>/dev/null || true
sudo pkill -f "python.*cursor" 2>/dev/null || true
```

### **1.2 Remove Old Project Directories**
```bash
# Remove ALL old project directories
cd ~
rm -rf cursorPathlight/ 2>/dev/null || true
rm -rf pathlight/ 2>/dev/null || true
rm -rf pathlight_project/ 2>/dev/null || true
rm -rf Downloads/pathlight* 2>/dev/null || true

# Clear cache directories
rm -rf ~/.cache/pip/
rm -rf ~/.cache/torch/
rm -rf ~/.local/share/ultralytics/
```

### **1.3 Remove All Python Virtual Environments**
```bash
# Remove all virtual environments
rm -rf ~/cursorPathlight_env/ 2>/dev/null || true
rm -rf ~/pathlight_env/ 2>/dev/null || true
rm -rf ~/.local/lib/python3.*/site-packages/torch* 2>/dev/null || true
rm -rf ~/.local/lib/python3.*/site-packages/ultralytics* 2>/dev/null || true
```

### **1.4 Clean System Python Packages**
```bash
# Uninstall potentially conflicting packages
pip3 uninstall -y torch torchvision torchaudio ultralytics opencv-python opencv-contrib-python face-recognition dlib 2>/dev/null || true

# Clear pip cache
pip3 cache purge
```

### **1.5 Verify Clean State**
```bash
# Verify no PyTorch remains
python3 -c "import torch" 2>&1 && echo "⚠️  PyTorch still installed!" || echo "✅ PyTorch cleaned"

# Verify no ultralytics remains  
python3 -c "import ultralytics" 2>&1 && echo "⚠️  ultralytics still installed!" || echo "✅ ultralytics cleaned"

# Check CUDA is available
nvcc --version || echo "⚠️  CUDA not available"
nvidia-smi || echo "⚠️  nvidia-smi not working"
```

---

## 📦 **STEP 2: TRANSFER FILES TO JETSON**

### **2.1 Option A: USB Transfer (Jetson Offline)**
```bash
# On your laptop (already done):
# - pathlight_final_cuda.tar.gz created
# - Copy to USB drive
# - Plug USB into Jetson

# On Jetson:
cd ~
# Find USB drive
ls /media/*/

# Copy and extract
cp /media/*/pathlight_final_cuda.tar.gz ~/
tar -xzf pathlight_final_cuda.tar.gz
mv pathlight_final cursorPathlight
cd cursorPathlight
```

### **2.2 Option B: GitHub Direct (Jetson Online)**
```bash
# On Jetson (if online):
cd ~
git clone https://github.com/ariyannp07/cursorPathlight.git
cd cursorPathlight
```

---

## ⚙️ **STEP 3: FRESH INSTALLATION**

### **3.1 Create New Virtual Environment**
```bash
cd ~/cursorPathlight

# Create fresh virtual environment
python3 -m venv cursorPathlight_env
source cursorPathlight_env/bin/activate

# Verify clean environment
pip list | grep -E "(torch|ultralytics|opencv)" || echo "✅ Clean environment"
```

### **3.2 Update System Packages**
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install essential build dependencies
sudo apt install -y \
    build-essential cmake git wget curl \
    pkg-config libssl-dev libffi-dev \
    python3-dev python3-pip python3-venv \
    libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libopenblas-dev liblapack-dev libatlas-base-dev \
    libgtk-3-dev libboost-all-dev
```

### **3.3 Run Setup Script**
```bash
# Make script executable
chmod +x scripts/setup_jetson.sh

# Run setup (this will take 30-45 minutes)
sudo ./scripts/setup_jetson.sh

# Follow prompts and wait for completion
```

---

## 🧪 **STEP 4: VERIFICATION**

### **4.1 Test Installation**
```bash
# Activate environment
source ~/cursorPathlight/cursorPathlight_env/bin/activate

# Run test script
python3 scripts/test_installation.py

# Expected output:
# ✅ 8/8 tests passed
# ✅ CUDA available: True
# ✅ PyTorch version: 2.4.0a0+07cecf4168
# ✅ TorchVision version: 0.19.0
# ✅ CUDA PyTorch operations working
# ✅ YOLOv8 model loaded successfully
# ✅ Face recognition working
# ✅ OpenCV CUDA: X devices
# ✅ All modules imported successfully
```

### **4.2 Test Pathlight Components**
```bash
# Test object detection
python3 -c "
from core.vision.object_detector import ObjectDetector
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
detector = ObjectDetector()
print(f'Detector device: {detector.device}')
print('✅ Object detection working')
"

# Test face recognition
python3 -c "
from core.vision.face_recognizer import FaceRecognizer
recognizer = FaceRecognizer()
print(f'Face recognition model: {recognizer.model}')
print('✅ Face recognition working')
"
```

---

## 🚀 **STEP 5: RUN PATHLIGHT**

### **5.1 Start Pathlight**
```bash
# Activate environment
source ~/cursorPathlight/cursorPathlight_env/bin/activate

# Run main application
python3 main.py

# Expected output:
# 🎯 Pathlight starting...
# ✅ CUDA acceleration enabled
# ✅ All systems operational
# 🚀 Ready for navigation!
```

### **5.2 Monitor Performance**
```bash
# In another terminal, monitor GPU usage
watch -n 1 nvidia-smi

# You should see:
# - GPU utilization > 0%
# - GPU memory usage
# - Python processes using GPU
```

---

## 📊 **FINAL VERIFIED SOFTWARE STACK**

| **Component** | **Version** | **Status** |
|:-------------:|:-----------:|:----------:|
| **PyTorch** | `2.4.0a0+07cecf4168` | ✅ **Stable Alpha** |
| **TorchVision** | `0.19.0` | ✅ **Source Built** |
| **TorchAudio** | `2.4.0` | ✅ **Compatible** |
| **CUDA** | `12.6` (JetPack) | ✅ **Native** |
| **cuDNN** | `9.3.0` | ✅ **JetPack Native** |
| **OpenCV** | `4.10.0.84` | ✅ **CUDA Enabled** |
| **YOLOv8** | `8.3.0` | ✅ **GPU Optimized** |
| **face_recognition** | `1.3.0` | ✅ **CNN Model** |
| **dlib** | `19.24.6` | ✅ **CUDA Support** |

---

## 🔧 **TROUBLESHOOTING**

### **Issue: "torch not found"**
```bash
# Solution: Activate virtual environment
source ~/cursorPathlight/cursorPathlight_env/bin/activate
```

### **Issue: "CUDA not available"**
```bash
# Check CUDA installation
nvcc --version
nvidia-smi

# Verify PyTorch CUDA
python3 -c "import torch; print(torch.cuda.is_available())"
```

### **Issue: "operator torchvision::nms does not exist"**
```bash
# This means mismatched PyTorch/TorchVision
# Solution: Re-run setup script (it builds compatible versions)
```

### **Issue: "Low FPS performance"**
```bash
# Check GPU utilization
nvidia-smi

# Verify CUDA acceleration
python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Device count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'Current device: {torch.cuda.current_device()}')
    print(f'Device name: {torch.cuda.get_device_name(0)}')
"
```

---

## ✅ **SUCCESS CRITERIA**

After following this guide, you should achieve:

- ✅ **8/8 tests passing** in test_installation.py
- ✅ **CUDA acceleration working** for PyTorch operations
- ✅ **GPU utilization > 0%** during inference
- ✅ **5+ FPS performance** for real-time detection
- ✅ **No compatibility errors** between components
- ✅ **All Pathlight modules** importing successfully

---

## 🎯 **FINAL NOTES**

1. **This setup uses PyTorch 2.4.0a0** - it's the most stable "alpha" for Jetson
2. **TorchVision 0.19.0** is built from source for guaranteed compatibility  
3. **All versions are specifically chosen** based on extensive compatibility research
4. **This eliminates the need for source building PyTorch** (only TorchVision)
5. **Performance should be 5+ FPS** with full CUDA acceleration

**THIS IS THE FINAL, TESTED, COMPATIBLE STACK. No more version conflicts!** 🎉