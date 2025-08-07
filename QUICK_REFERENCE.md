# 🚀 PATHLIGHT QUICK REFERENCE CARD

## 🔗 **CONNECT TO JETSON**
```bash
# Get Jetson IP
hostname -I

# Connect via SSH
ssh nvidia@[JETSON_IP]

# Enable SSH (if needed)
sudo systemctl enable ssh
sudo systemctl start ssh
```

## 📋 **ESSENTIAL COMMANDS**

### **System Assessment**
```bash
# Check system info
uname -a
nvidia-smi
python3 --version

# Check OpenCV
pip3 list | grep opencv
dpkg -l | grep opencv
python3 -c "import cv2; print(cv2.__version__)"

# Check cameras
v4l2-ctl --list-devices
ls -la /dev/video*
```

### **Installation**
```bash
# Transfer files
scp -r pathlight_project/ nvidia@[JETSON_IP]:~/cursorPathlight/

# Run setup
cd ~/cursorPathlight
chmod +x scripts/setup_jetson.sh
./scripts/setup_jetson.sh

# Activate environment
source ~/cursorPathlight/cursorPathlight_env/bin/activate
```

### **Testing**
```bash
# Test installation
python3 scripts/test_installation.py

# Test stereo vision
python3 scripts/test_stereo_vision.py

# Test specific component
python3 scripts/test_stereo_vision.py --test depth
python3 scripts/test_stereo_vision.py --test obstacles
python3 scripts/test_stereo_vision.py --test path
```

### **Calibration**
```bash
# Run calibration
python3 scripts/stereo_calibration.py

# Check calibration
ls -la data/stereo_calibration.pkl
```

### **Launch**
```bash
# Start Pathlight
python3 main.py

# Check configuration
nano config/config.yaml
```

## 🆘 **TROUBLESHOOTING**

### **OpenCV Issues**
```bash
# Clean OpenCV
sudo apt remove --purge python3-opencv opencv-*
pip3 uninstall opencv-python opencv-contrib-python
sudo apt autoremove

# Reinstall
sudo apt install python3-opencv libopencv-dev
```

### **Camera Issues**
```bash
# Restart camera service
sudo systemctl restart nvidia-l4t-camera

# Check camera status
v4l2-ctl --device=/dev/video0 --all
v4l2-ctl --device=/dev/video1 --all
```

### **Performance Issues**
```bash
# Monitor resources
htop
nvidia-smi

# Reduce resolution in config/config.yaml
camera:
  width: 640
  height: 480
  fps: 15
```

## 📁 **KEY FILES**

### **Configuration**
- `config/config.yaml` - Main settings
- `config/config.example.yaml` - Template

### **Scripts**
- `scripts/setup_jetson.sh` - Automated setup
- `scripts/test_installation.py` - System tests
- `scripts/stereo_calibration.py` - Camera calibration
- `scripts/test_stereo_vision.py` - Stereo vision tests

### **Documentation**
- `MASTER_INSTRUCTIONS.md` - Complete guide
- `docs/stereo_vision_guide.md` - Stereo vision guide
- `docs/imx219_camera_setup.md` - Camera setup

## 🎯 **SUCCESS INDICATORS**

### **System Working When:**
- [ ] `python3 -c "import cv2; print(cv2.__version__)"` works
- [ ] `v4l2-ctl --list-devices` shows 2 cameras
- [ ] `python3 scripts/test_installation.py` all PASS
- [ ] `python3 scripts/test_stereo_vision.py` all PASS
- [ ] `python3 main.py` starts without errors

### **Performance Targets:**
- [ ] >10 FPS overall
- [ ] <50ms total pipeline time
- [ ] <30ms depth computation
- [ ] <10ms obstacle detection

## 📞 **GET HELP**

### **When to Contact:**
1. Any error during setup
2. Performance below targets
3. Cameras not detected
4. OpenCV import failures

### **What to Share:**
1. Exact error messages
2. Command outputs
3. Current step number
4. What you've tried

---

## 🚀 **START HERE**

1. **Connect to Jetson**: `ssh nvidia@[JETSON_IP]`
2. **Run assessment**: Copy commands from MASTER_INSTRUCTIONS.md Step 1.3
3. **Follow MASTER_INSTRUCTIONS.md** step by step
4. **Contact me** at any checkpoint or error

**Good luck! Your Pathlight prototype is within reach!** 🎉 