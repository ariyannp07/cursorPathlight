# Pathlight Project Transfer Instructions

## Overview
This document provides step-by-step instructions for transferring the Pathlight project from your Mac to the NVIDIA Jetson Orin Nano.

## What We've Created
We've built a comprehensive AI wearable navigation system with the following components:

### Core Features Implemented
1. **Real-time Object Detection** - YOLOv8-based obstacle identification
2. **Face Recognition** - Familiar face detection and memory system
3. **Safe Path Planning** - AI-powered navigation with obstacle avoidance
4. **LED Direction Display** - Visual course heading through LED array
5. **Audio Guidance** - Text-to-speech navigation instructions
6. **AI Assistant** - Voice interaction and query processing
7. **Memory System** - Interaction history and face database

### Project Structure
```
pathlight_project/
├── main.py                    # Main application entry point
├── requirements.txt           # Python dependencies
├── README.md                  # Project overview
├── config/
│   └── config.example.yaml    # Configuration template
├── core/                      # Core AI modules
│   ├── vision/
│   │   ├── object_detector.py     # YOLOv8 object detection
│   │   └── face_recognizer.py     # Face recognition
│   ├── navigation/
│   │   └── path_planner.py        # Path planning
│   ├── audio/
│   │   └── audio_manager.py       # Text-to-speech
│   └── memory/
│       └── memory_manager.py      # Database management
├── hardware/                  # Hardware interfaces
│   ├── camera/
│   │   └── camera_manager.py     # Camera control
│   ├── leds/
│   │   └── led_controller.py     # LED array control
│   └── audio/
│       └── audio_io.py           # Audio I/O
├── ai/                        # AI components
│   ├── assistant/
│   │   └── ai_assistant.py       # AI assistant
│   └── voice/
│       └── voice_processor.py    # Speech recognition
├── scripts/
│   ├── setup_jetson.sh           # Automated setup script
│   └── test_installation.py      # Installation test
└── docs/
    ├── setup.md                  # Detailed setup guide
    └── development.md            # Development guide
```

## Transfer Process

### Step 1: Prepare the Jetson
1. **Connect Jetson SSD to Mac**:
   - Safely shut down the Jetson
   - Remove the SSD and connect it to your Mac using an adapter
   - Mount the SSD on your Mac

2. **Navigate to Jetson Home Directory**:
   ```bash
   cd /Volumes/[JETSON_SSD_NAME]/home/nvidia/
   ```

### Step 2: Transfer Project Files
1. **Copy the entire project**:
   ```bash
   cp -r /Users/ariyanp/pathlight_project ~/cursorPathlight
   ```

2. **Set proper permissions**:
   ```bash
   chmod +x ~/cursorPathlight/scripts/*.sh
chmod +x ~/cursorPathlight/scripts/*.py
chmod +x ~/cursorPathlight/main.py
   ```

### Step 3: Safely Disconnect
1. **Unmount the SSD**:
   ```bash
   sudo umount /Volumes/[JETSON_SSD_NAME]
   ```

2. **Reinstall SSD in Jetson**:
   - Carefully reinstall the SSD in the Jetson
   - Power on the Jetson

### Step 4: Setup on Jetson
1. **SSH into Jetson** (or use direct connection):
   ```bash
   ssh nvidia@[JETSON_IP_ADDRESS]
   ```

2. **Navigate to project directory**:
   ```bash
   cd ~/cursorPathlight
   ```

3. **Run the automated setup script**:
   ```bash
   ./scripts/setup_jetson.sh
   ```

   This script will:
   - Update system packages
   - Install all dependencies
   - Set up Python environment
   - Configure hardware interfaces
   - Create necessary directories
   - Set up system services

4. **Test the installation**:
   ```bash
   python scripts/test_installation.py
   ```

### Step 5: Configure the System
1. **Create configuration file**:
   ```bash
   cp config/config.example.yaml config/config.yaml
   ```

2. **Edit configuration**:
   ```bash
   nano config/config.yaml
   ```

   Key settings to configure:
   - Camera device ID
   - LED array configuration
   - Audio device settings
   - API keys (OpenAI, Google Maps)

3. **Add API keys** (if you have them):
   ```yaml
   ai_assistant:
     api_key: "your_openai_api_key"
   
   google_maps:
     api_key: "your_google_maps_api_key"
   ```

### Step 6: Test Individual Components
1. **Test IMX219 camera**:
   ```bash
   python -c "
   import cv2
   # Test left camera (device 0)
   cap = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L2)
   ret, frame = cap.read()
   if ret:
       print('Left IMX219 camera working!')
       cv2.imwrite('test_image_left.jpg', frame)
   cap.release()
   
   # Test right camera (device 1) if available
   cap = cv2.VideoCapture('/dev/video1', cv2.CAP_V4L2)
   ret, frame = cap.read()
   if ret:
       print('Right IMX219 camera working!')
       cv2.imwrite('test_image_right.jpg', frame)
   cap.release()
   "
   ```

2. **Test YOLOv8**:
   ```bash
   python -c "
   from ultralytics import YOLO
   model = YOLO('yolov8n.pt')
   print('YOLOv8 working!')
   "
   ```

3. **Test audio**:
   ```bash
   python -c "
   import pyttsx3
   engine = pyttsx3.init()
   engine.say('Pathlight audio test')
   engine.runAndWait()
   "
   ```

### Step 7: Run the System
1. **Start Pathlight**:
   ```bash
   python main.py
   ```

2. **Or start as a service**:
   ```bash
   sudo systemctl start pathlight
   sudo systemctl enable pathlight
   ```

## Hardware Requirements

### Required Hardware
- **Camera**: Dual IMX219 CSI camera module (ribbon cable connection)
- **LED Array**: 16-LED array (I2C or GPIO controlled)
- **Audio**: Microphone and speakers/headphones
- **Power**: Adequate power supply for Jetson

### Optional Hardware
- **Depth Sensor**: For more accurate distance measurement
- **GPS Module**: For location-based navigation
- **IMU Sensor**: For orientation tracking

## Troubleshooting

### Common Issues
1. **IMX219 camera not detected**: Check CSI ribbon cable connection and camera module power
2. **CUDA errors**: Verify JetPack installation
3. **Audio issues**: Check ALSA configuration
4. **LED array not working**: Verify I2C/GPIO connections

### Useful Commands
```bash
# Monitor system resources
tegrastats
htop

# Check IMX219 camera
v4l2-ctl --list-devices
v4l2-ctl --device=/dev/video0 --list-formats-ext

# Check I2C devices
sudo i2cdetect -y 1

# Check audio devices
arecord -l
aplay -l

# View logs
tail -f logs/cursorPathlight.log
```

## Next Steps

### Immediate Actions
1. Transfer the project files to Jetson
2. Run the setup script
3. Configure the system
4. Test all components
5. Start the system

### Future Development
1. **Add known faces**: Use the face recognition system to add familiar people
2. **Calibrate distance estimation**: Adjust distance calculation for your camera
3. **Customize LED patterns**: Modify LED patterns for your specific array
4. **Add voice commands**: Extend the voice processing system
5. **Integrate GPS**: Add location-based navigation features

### Performance Optimization
1. **Model optimization**: Use TensorRT for faster inference
2. **Resolution adjustment**: Optimize camera resolution for performance
3. **FPS tuning**: Adjust frame rate based on hardware capabilities
4. **Memory management**: Monitor and optimize memory usage

## Support

### Documentation
- `docs/setup.md` - Detailed setup instructions
- `docs/development.md` - Development guide
- `README.md` - Project overview

### Testing
- `scripts/test_installation.py` - Comprehensive installation test
- Individual component tests in each module

### Logs
- System logs: `logs/cursorPathlight.log`
- Service logs: `sudo journalctl -u cursorPathlight.service`

## Success Criteria
The system is successfully set up when:
1. ✅ All tests pass in `test_installation.py`
2. ✅ Camera captures images
3. ✅ YOLOv8 detects objects
4. ✅ LED array responds to commands
5. ✅ Audio system works
6. ✅ Main application runs without errors

## Notes
- The system is designed to be robust and handle hardware failures gracefully
- All components have fallback mechanisms
- The configuration system allows easy customization
- The modular design makes it easy to add new features

Good luck with your Pathlight implementation! 🚀 