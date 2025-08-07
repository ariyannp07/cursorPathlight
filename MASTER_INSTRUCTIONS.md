# 🎯 PATHLIGHT MASTER INSTRUCTIONS
## Complete Setup Guide for Jetson Orin Nano with Dual IMX219 Stereo Vision

---

## 📋 **QUICK START CHECKLIST**

### **Phase 1: Preparation** ⏳ 30 minutes
- [ ] **1.1** Download and organize project files
- [ ] **1.2** Prepare Jetson for connection
- [ ] **1.3** Connect to Jetson and assess current state
- [ ] **1.4** Clean up existing OpenCV installations

### **Phase 2: Installation** ⏳ 60 minutes
- [ ] **2.1** Install system dependencies
- [ ] **2.2** Set up Python environment
- [ ] **2.3** Install OpenCV and AI libraries
- [ ] **2.4** Configure project settings

### **Phase 3: Hardware Setup** ⏳ 30 minutes
- [ ] **3.1** Connect dual IMX219 cameras
- [ ] **3.2** Test camera detection
- [ ] **3.3** Verify stereo camera operation

### **Phase 4: Calibration** ⏳ 45 minutes
- [ ] **4.1** Prepare calibration pattern
- [ ] **4.2** Run stereo calibration
- [ ] **4.3** Validate calibration quality

### **Phase 5: Testing & Validation** ⏳ 30 minutes
- [ ] **5.1** Run comprehensive tests
- [ ] **5.2** Verify 3D depth perception
- [ ] **5.3** Test obstacle detection
- [ ] **5.4** Validate safety systems

### **Phase 6: Launch** ⏳ 15 minutes
- [ ] **6.1** Start Pathlight system
- [ ] **6.2** Verify all components working
- [ ] **6.3** Test real-world navigation

---

## 🔗 **HOW TO CONNECT ME TO YOUR JETSON**

### **Option 1: Direct SSH Connection (Recommended)**
```bash
# On your Jetson, enable SSH if not already enabled:
sudo systemctl enable ssh
sudo systemctl start ssh

# Get your Jetson's IP address:
hostname -I

# From your Mac, connect via SSH:
ssh nvidia@[JETSON_IP_ADDRESS]
# Password: [your Jetson password]
```

### **Option 2: USB Connection with ADB**
```bash
# Install ADB on your Mac:
brew install android-platform-tools

# Connect Jetson via USB and enable ADB:
adb connect [JETSON_IP]:5555
```

### **Option 3: Direct Terminal Access**
- Connect monitor, keyboard, and mouse to Jetson
- Work directly on the Jetson terminal

### **What I Need to See:**
1. **Current system state**: `uname -a`, `nvidia-smi`, `python3 --version`
2. **OpenCV installations**: `pip list | grep opencv`, `dpkg -l | grep opencv`
3. **Camera status**: `v4l2-ctl --list-devices`
4. **File system**: `ls -la ~/`, `pwd`
5. **Error messages**: Any OpenCV import errors or system issues

---

## 📁 **PROJECT FILE ORGANIZATION**

### **Current Project Structure:**
```
pathlight_project/
├── 📄 README.md                           # Project overview
├── 📄 requirements.txt                    # Python dependencies
├── 📄 main.py                            # Main application
├── 📄 MASTER_INSTRUCTIONS.md             # This file
├── 📄 TRANSFER_INSTRUCTIONS.md           # Transfer guide
├── 📁 config/
│   ├── 📄 config.example.yaml            # Configuration template
│   └── 📄 config.yaml                    # Your settings (create this)
├── 📁 core/
│   ├── 📁 vision/
│   │   ├── 📄 object_detector.py         # YOLOv8 object detection
│   │   ├── 📄 face_recognizer.py         # Face recognition
│   │   └── 📄 stereo_vision.py           # 3D stereo vision
│   ├── 📁 navigation/
│   │   └── 📄 path_planner.py            # Path planning
│   ├── 📁 audio/
│   │   └── 📄 audio_manager.py           # Text-to-speech
│   └── 📁 memory/
│       └── 📄 memory_manager.py          # Face memory
├── 📁 hardware/
│   ├── 📁 camera/
│   │   └── 📄 camera_manager.py          # IMX219 camera control
│   ├── 📁 leds/
│   │   └── 📄 led_controller.py          # LED array control
│   └── 📁 audio/
│       └── 📄 audio_io.py                # Audio I/O
├── 📁 ai/
│   ├── 📁 assistant/
│   │   └── 📄 ai_assistant.py            # AI assistant
│   └── 📁 voice/
│       └── 📄 voice_processor.py         # Voice processing
├── 📁 scripts/
│   ├── 📄 setup_jetson.sh                # Automated setup
│   ├── 📄 test_installation.py           # System tests
│   ├── 📄 stereo_calibration.py          # Camera calibration
│   └── 📄 test_stereo_vision.py          # Stereo vision tests
└── 📁 docs/
    ├── 📄 setup.md                       # Detailed setup guide
    ├── 📄 development.md                 # Development guide
    ├── 📄 imx219_camera_setup.md         # Camera setup
    └── 📄 stereo_vision_guide.md         # Stereo vision guide
```

---

## 🚀 **STEP-BY-STEP EXECUTION GUIDE**

### **STEP 1: PREPARATION** ⏳ 30 minutes

#### **1.1 Download and Organize Files**
```bash
# On your Mac, ensure all project files are ready:
ls -la pathlight_project/
# Should show all files listed above

# Create a backup of your current Jetson setup (if any):
# [We'll do this when connected to Jetson]
```

#### **1.2 Prepare Jetson for Connection**
- [ ] Power on Jetson Orin Nano
- [ ] Connect to network (WiFi or Ethernet)
- [ ] Note the IP address: `hostname -I`
- [ ] Ensure SSH is enabled: `sudo systemctl status ssh`

#### **1.3 Connect and Assess Current State**
```bash
# Connect to Jetson:
ssh nvidia@[JETSON_IP]

# Run these commands and share output with me:
echo "=== SYSTEM INFO ==="
uname -a
nvidia-smi
python3 --version
pip3 --version

echo "=== OPENCV STATUS ==="
pip3 list | grep opencv
dpkg -l | grep opencv
python3 -c "import cv2; print('OpenCV version:', cv2.__version__)" 2>/dev/null || echo "OpenCV import failed"

echo "=== CAMERA STATUS ==="
v4l2-ctl --list-devices
ls -la /dev/video*

echo "=== CURRENT DIRECTORY ==="
pwd
ls -la ~/
```

#### **1.4 Clean Up Existing Installations**
```bash
# Remove conflicting OpenCV installations:
sudo apt remove --purge python3-opencv opencv-* libopencv*
sudo apt autoremove
sudo apt autoclean

# Clean Python environments:
pip3 uninstall opencv-python opencv-contrib-python opencv-python-headless -y
conda remove opencv -y 2>/dev/null || echo "No conda environment found"

# Verify clean state:
pip3 list | grep opencv
dpkg -l | grep opencv
```

**✅ CHECKPOINT 1:** System assessed and cleaned

---

### **STEP 2: INSTALLATION** ⏳ 60 minutes

#### **2.1 Transfer Project Files**
```bash
# From your Mac, transfer files to Jetson:
scp -r pathlight_project/ nvidia@[JETSON_IP]:~/pathlight/

# On Jetson, verify transfer:
cd ~/pathlight
ls -la
```

#### **2.2 Install System Dependencies**
```bash
# Run the automated setup script:
cd ~/pathlight
chmod +x scripts/setup_jetson.sh
./scripts/setup_jetson.sh

# This script will:
# - Update system packages
# - Install Jetson-specific camera packages
# - Install OpenCV and AI dependencies
# - Set up Python virtual environment
# - Configure system services
```

#### **2.3 Verify Installation**
```bash
# Activate virtual environment:
source ~/pathlight/pathlight_env/bin/activate

# Test OpenCV installation:
python3 -c "
import cv2
import numpy as np
print('✓ OpenCV version:', cv2.__version__)
print('✓ NumPy version:', np.__version__)
print('✓ CUDA devices:', cv2.cuda.getCudaEnabledDeviceCount())
"

# Test camera access:
python3 -c "
import cv2
cap = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L2)
if cap.isOpened():
    print('✓ Camera access working')
    cap.release()
else:
    print('✗ Camera access failed')
"
```

#### **2.4 Configure Project Settings**
```bash
# Copy configuration template:
cp config/config.example.yaml config/config.yaml

# Edit configuration with your settings:
nano config/config.yaml

# Key settings to configure:
# - API keys (OpenAI, Google Maps)
# - Camera settings (resolution, FPS)
# - Stereo vision parameters
# - LED array configuration
```

**✅ CHECKPOINT 2:** Software installed and configured

---

### **STEP 3: HARDWARE SETUP** ⏳ 30 minutes

#### **3.1 Connect Dual IMX219 Cameras**
```bash
# Physical connection:
# - Left camera → CSI0 port
# - Right camera → CSI1 port
# - Ensure cables are fully seated and locked

# Verify camera detection:
v4l2-ctl --list-devices
# Should show:
# /dev/video0 - Left IMX219 camera
# /dev/video1 - Right IMX219 camera
```

#### **3.2 Test Camera Operation**
```bash
# Test individual cameras:
python3 -c "
import cv2
import numpy as np

# Test left camera
cap = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
ret, frame = cap.read()
if ret:
    print('✓ Left camera working')
    cv2.imwrite('left_test.jpg', frame)
cap.release()

# Test right camera
cap = cv2.VideoCapture('/dev/video1', cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
ret, frame = cap.read()
if ret:
    print('✓ Right camera working')
    cv2.imwrite('right_test.jpg', frame)
cap.release()
"
```

#### **3.3 Run Camera Tests**
```bash
# Run comprehensive camera tests:
python3 scripts/test_installation.py

# Expected output:
# ✓ Module Imports: PASS
# ✓ CUDA Support: PASS
# ✓ Project Modules: PASS
# ✓ Configuration: PASS
# ✓ Directories: PASS
# ✓ YOLOv8 Model: PASS
# ✓ Audio System: PASS
# ✓ IMX219 Camera: PASS
```

**✅ CHECKPOINT 3:** Hardware connected and tested

---

### **STEP 4: CALIBRATION** ⏳ 45 minutes

#### **4.1 Prepare Calibration Pattern**
- [ ] Print chessboard pattern (9x6 internal corners)
- [ ] Mount on rigid, flat surface
- [ ] Measure square size (should be 2.5cm)

#### **4.2 Run Stereo Calibration**
```bash
# Start calibration tool:
python3 scripts/stereo_calibration.py

# Follow on-screen instructions:
# 1. Hold chessboard at different angles
# 2. Press 'c' when pattern detected (green overlay)
# 3. Capture 15-20 images from various positions
# 4. Press 's' to start calibration
# 5. Wait for calibration to complete

# Expected calibration error: < 0.5 pixels
```

#### **4.3 Validate Calibration**
```bash
# Check calibration file:
ls -la data/stereo_calibration.pkl

# Test calibration quality:
python3 -c "
import pickle
with open('data/stereo_calibration.pkl', 'rb') as f:
    calib = pickle.load(f)
print('Calibration error:', calib['calibration_error'])
print('Number of images used:', calib['num_images'])
"
```

**✅ CHECKPOINT 4:** Cameras calibrated

---

### **STEP 5: TESTING & VALIDATION** ⏳ 30 minutes

#### **5.1 Run Comprehensive Tests**
```bash
# Run all stereo vision tests:
python3 scripts/test_stereo_vision.py

# This will test:
# ✓ Camera Detection
# ✓ Camera Synchronization
# ✓ Stereo Calibration
# ✓ Depth Computation
# ✓ 3D Obstacle Detection
# ✓ Safe Path Planning
# ✓ Performance

# Check output images in test_output/:
ls -la test_output/
# Should show:
# - left_frame.jpg
# - right_frame.jpg
# - depth_map.jpg
# - disparity_map.jpg
# - obstacles_detected.jpg
# - safe_path.jpg
```

#### **5.2 Verify 3D Depth Perception**
```bash
# Test depth computation:
python3 scripts/test_stereo_vision.py --test depth

# Check depth map quality:
# - Should show color-coded depth
# - Closer objects = red/orange
# - Farther objects = blue/purple
# - No depth = black
```

#### **5.3 Test Obstacle Detection**
```bash
# Test 3D obstacle detection:
python3 scripts/test_stereo_vision.py --test obstacles

# Place objects at different distances:
# - Close object (< 1m) should be marked as dangerous
# - Farther objects should be detected but not dangerous
```

#### **5.4 Validate Safety Systems**
```bash
# Test safe path planning:
python3 scripts/test_stereo_vision.py --test path

# Verify:
# - Safe direction arrow points away from obstacles
# - Confidence score > 0.5 for clear paths
# - Emergency warnings for close obstacles
```

**✅ CHECKPOINT 5:** All systems tested and validated

---

### **STEP 6: LAUNCH** ⏳ 15 minutes

#### **6.1 Start Pathlight System**
```bash
# Launch main application:
python3 main.py

# Expected startup sequence:
# 1. Loading configuration...
# 2. Initializing components...
# 3. Starting cameras...
# 4. Loading AI models...
# 5. Pathlight system started successfully
```

#### **6.2 Verify All Components**
```bash
# Check system status:
# - Camera feeds should be active
# - LED array should show direction
# - Audio system should respond to voice
# - AI assistant should answer questions
# - 3D depth perception should be working
```

#### **6.3 Test Real-World Navigation**
- [ ] Walk around with the system
- [ ] Test obstacle detection with real objects
- [ ] Verify audio guidance is accurate
- [ ] Test LED direction indicators
- [ ] Confirm emergency stops work

**✅ CHECKPOINT 6:** Pathlight fully operational

---

## 🆘 **TROUBLESHOOTING GUIDE**

### **Common Issues and Solutions**

#### **OpenCV Import Errors**
```bash
# Symptom: ImportError: libopencv_core.so.4.5: cannot open shared object file
# Solution: Complete cleanup and reinstall
sudo apt remove --purge python3-opencv opencv-*
pip3 uninstall opencv-python opencv-contrib-python
sudo apt install python3-opencv libopencv-dev
```

#### **Camera Not Detected**
```bash
# Symptom: No cameras found in v4l2-ctl --list-devices
# Solution: Check physical connections and restart services
sudo systemctl restart nvidia-l4t-camera
ls -la /dev/video*
```

#### **Poor Depth Quality**
```bash
# Symptom: Noisy or incorrect depth maps
# Solution: Recalibrate with more images
python3 scripts/stereo_calibration.py
# Use 20-30 calibration images from various positions
```

#### **Low Performance**
```bash
# Symptom: <5 FPS, high computation times
# Solution: Reduce resolution and optimize parameters
# Edit config/config.yaml:
# camera:
#   width: 640
#   height: 480
#   fps: 15
```

---

## 📞 **GETTING HELP**

### **When to Contact Me:**
1. **During any step** if you encounter errors
2. **Before proceeding** if unsure about any step
3. **After each checkpoint** to confirm success
4. **If performance** is below expected levels

### **What to Share with Me:**
1. **Error messages** (copy exact text)
2. **Command outputs** (copy terminal output)
3. **Current step** you're on
4. **What you've tried** so far

### **Emergency Recovery:**
```bash
# If system becomes unstable:
sudo reboot

# If OpenCV completely broken:
sudo apt remove --purge python3-opencv opencv-*
sudo apt autoremove
# Then restart from Step 2.1
```

---

## 🎯 **SUCCESS CRITERIA**

### **System is Working When:**
- [ ] **Cameras**: Both IMX219 cameras detected and capturing
- [ ] **Stereo Vision**: Depth maps generated with reasonable quality
- [ ] **Obstacle Detection**: 3D obstacles detected with accurate distances
- [ ] **Path Planning**: Safe directions calculated with confidence > 0.5
- [ ] **Performance**: >5 FPS overall system performance
- [ ] **Safety**: Emergency stops triggered for obstacles <50cm away
- [ ] **Audio**: Clear voice guidance provided
- [ ] **LEDs**: Direction indicators working correctly

### **Performance Benchmarks:**
- **Depth Computation**: <30ms per frame
- **Obstacle Detection**: <10ms per frame
- **Path Planning**: <5ms per frame
- **Total Pipeline**: <50ms per frame
- **Overall FPS**: >10 FPS

---

## 📝 **PROGRESS TRACKING**

### **Your Progress Checklist:**
- [ ] **Phase 1 Complete**: Preparation and assessment
- [ ] **Phase 2 Complete**: Software installation
- [ ] **Phase 3 Complete**: Hardware setup
- [ ] **Phase 4 Complete**: Camera calibration
- [ ] **Phase 5 Complete**: Testing and validation
- [ ] **Phase 6 Complete**: System launch

### **Current Status:**
**Phase:** [ ] 1 [ ] 2 [ ] 3 [ ] 4 [ ] 5 [ ] 6
**Step:** [Current step number]
**Status:** [In Progress / Complete / Blocked]

---

## 🚀 **READY TO START?**

**Next Action:** Connect me to your Jetson and let's begin with Step 1.3 (Connect and Assess Current State).

**What I Need:**
1. Your Jetson's IP address
2. SSH connection established
3. Output from the assessment commands

**Let's make your Pathlight prototype a reality!** 🎉 