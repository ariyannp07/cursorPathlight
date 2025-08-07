# Pathlight Development Guide

## Overview
This guide provides detailed information for developing and extending the Pathlight AI wearable navigation assistant.

## Project Structure
```
pathlight/
├── main.py                 # Main application entry point
├── requirements.txt        # Python dependencies
├── config/                 # Configuration files
│   ├── config.example.yaml # Example configuration
│   └── config.yaml         # Active configuration (create from example)
├── core/                   # Core AI and processing modules
│   ├── vision/            # Computer vision components
│   │   ├── object_detector.py    # YOLOv8 object detection
│   │   └── face_recognizer.py    # Face recognition system
│   ├── navigation/        # Path planning and navigation
│   │   └── path_planner.py       # Safe path calculation
│   ├── audio/             # Audio processing
│   │   └── audio_manager.py      # Text-to-speech system
│   └── memory/            # Memory and storage
│       └── memory_manager.py     # Interaction history
├── hardware/              # Hardware interface modules
│   ├── camera/            # Camera control
│   │   └── camera_manager.py     # Camera input management
│   ├── leds/              # LED array control
│   │   └── led_controller.py     # Navigation display
│   └── audio/             # Audio I/O
│       └── audio_io.py           # Microphone and speaker control
├── ai/                    # AI assistant components
│   ├── assistant/         # AI assistant
│   │   └── ai_assistant.py       # Query processing
│   └── voice/             # Voice processing
│       └── voice_processor.py    # Speech recognition
├── tests/                 # Unit and integration tests
├── scripts/               # Setup and utility scripts
│   └── setup_jetson.sh    # Jetson setup script
└── docs/                  # Documentation
    ├── setup.md           # Setup instructions
    └── development.md     # This file
```

## Development Environment Setup

### Prerequisites
- NVIDIA Jetson Orin Nano with JetPack SDK 5.1.2+
- Python 3.8+
- Git
- USB camera (compatible with Jetson)
- LED array (I2C or GPIO controlled)
- Microphone and speakers

### Local Development Setup
1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd pathlight
   ```

2. **Create virtual environment**:
   ```bash
   python3 -m venv cursorPathlight_env
source cursorPathlight_env/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure the system**:
   ```bash
   cp config/config.example.yaml config/config.yaml
   # Edit config/config.yaml with your settings
   ```

### Jetson Setup
Use the provided setup script for automatic Jetson configuration:
```bash
chmod +x scripts/setup_jetson.sh
./scripts/setup_jetson.sh
```

## Configuration

### Main Configuration File
The main configuration is in `config/config.yaml`. Key sections:

- **System**: Basic system settings
- **Camera**: Camera device and format settings
- **YOLO**: Object detection parameters
- **Face Recognition**: Face detection settings
- **Navigation**: Path planning parameters
- **LEDs**: LED array configuration
- **Audio**: Text-to-speech and audio settings
- **AI Assistant**: OpenAI and AI settings
- **Database**: Storage configuration

### Environment Variables
Create a `.env` file for sensitive information:
```bash
OPENAI_API_KEY=your_openai_api_key
GOOGLE_MAPS_API_KEY=your_google_maps_api_key
```

## Core Components

### Object Detection (YOLOv8)
The `ObjectDetector` class provides real-time object detection:

```python
from core.vision.object_detector import ObjectDetector

detector = ObjectDetector(config['yolo'])
detections = detector.detect(frame)
```

**Key Features**:
- Real-time object detection
- Distance estimation
- Obstacle classification
- Performance optimization for Jetson

### Face Recognition
The `FaceRecognizer` class handles face detection and recognition:

```python
from core.vision.face_recognizer import FaceRecognizer

recognizer = FaceRecognizer(config['face_recognition'])
faces = recognizer.recognize(frame)
```

**Key Features**:
- Face detection and encoding
- Known face recognition
- Interaction history tracking
- Database persistence

### Path Planning
The `PathPlanner` class calculates safe navigation paths:

```python
from core.navigation.path_planner import PathPlanner

planner = PathPlanner(config['navigation'])
path = planner.plan_path(detections)
```

**Key Features**:
- Obstacle avoidance
- Safety zone analysis
- Direction recommendation
- Confidence scoring

### LED Control
The `LEDController` class manages the LED array display:

```python
from hardware.leds.led_controller import LEDController

leds = LEDController(config['leds'])
leds.display_direction('forward')
```

**Key Features**:
- Direction pattern display
- I2C and GPIO support
- Animation patterns
- Brightness control

## Adding New Features

### Creating a New Module
1. **Create the module file**:
   ```python
   # core/new_feature/new_module.py
   
   import logging
   from typing import Dict, Any
   
   class NewModule:
       def __init__(self, config: Dict[str, Any]):
           self.logger = logging.getLogger(__name__)
           self.config = config
           self._initialize()
       
       def _initialize(self):
           # Initialize your module
           pass
       
       def process(self, data):
           # Process data
           return result
   ```

2. **Add configuration**:
   ```yaml
   # config/config.yaml
   new_feature:
     enabled: true
     parameter1: value1
     parameter2: value2
   ```

3. **Integrate into main system**:
   ```python
   # main.py
   from core.new_feature.new_module import NewModule
   
   # In _initialize_components method:
   self.components['new_feature'] = NewModule(self.config['new_feature'])
   ```

### Adding Hardware Support
1. **Create hardware interface**:
   ```python
   # hardware/new_hardware/hardware_interface.py
   
   class HardwareInterface:
       def __init__(self, config):
           self.config = config
           self._initialize_hardware()
       
       def _initialize_hardware(self):
           # Initialize hardware
           pass
       
       def start(self):
           # Start hardware
           pass
       
       def stop(self):
           # Stop hardware
           pass
   ```

2. **Add error handling**:
   ```python
   def _initialize_hardware(self):
       try:
           # Hardware initialization
           pass
       except Exception as e:
           self.logger.error(f"Hardware initialization failed: {e}")
           # Fallback or graceful degradation
   ```

## Testing

### Unit Tests
Create tests in the `tests/` directory:

```python
# tests/test_object_detector.py
import pytest
from core.vision.object_detector import ObjectDetector

def test_object_detector_initialization():
    config = {'model': 'yolov8n.pt', 'confidence': 0.5}
    detector = ObjectDetector(config)
    assert detector is not None

def test_object_detection():
    # Test object detection with sample image
    pass
```

### Integration Tests
Test component interactions:

```python
# tests/test_integration.py
def test_camera_to_detection_pipeline():
    # Test full pipeline from camera to object detection
    pass

def test_voice_to_ai_pipeline():
    # Test voice input to AI response pipeline
    pass
```

### Performance Testing
Monitor system performance:

```python
# tests/test_performance.py
def test_fps_requirements():
    # Ensure system meets real-time requirements
    pass

def test_memory_usage():
    # Monitor memory consumption
    pass
```

## Debugging

### Logging
The system uses Python's logging module. Configure log levels in config:

```yaml
system:
  log_level: DEBUG  # DEBUG, INFO, WARNING, ERROR
```

### Debug Mode
Enable debug mode for detailed output:

```bash
python main.py --debug
```

### Performance Monitoring
Use system monitoring tools:

```bash
# Monitor GPU usage
tegrastats

# Monitor system resources
htop

# Monitor camera
v4l2-ctl --list-devices
```

## Deployment

### Production Setup
1. **Configure system service**:
   ```bash
   sudo systemctl enable cursorPathlight.service
sudo systemctl start cursorPathlight.service
   ```

2. **Monitor logs**:
   ```bash
   sudo journalctl -u cursorPathlight.service -f
   ```

3. **Auto-restart on failure**:
   The service is configured to restart automatically on failure.

### Backup and Recovery
1. **Database backup**:
   ```bash
   cp ~/cursorPathlight/data/cursorPathlight.db ~/cursorPathlight/data/cursorPathlight.db.backup
   ```

2. **Configuration backup**:
   ```bash
   cp ~/cursorPathlight/config/config.yaml ~/cursorPathlight/config/config.yaml.backup
   ```

## Troubleshooting

### Common Issues

#### Camera Not Detected
```bash
# Check camera devices
v4l2-ctl --list-devices

# Test camera
v4l2-ctl --device=/dev/video0 --list-formats-ext
```

#### CUDA Errors
```bash
# Check CUDA installation
nvidia-smi

# Check PyTorch CUDA support
python -c "import torch; print(torch.cuda.is_available())"
```

#### Audio Issues
```bash
# Check audio devices
arecord -l
aplay -l

# Test audio
speaker-test -t wav -c 2
```

#### LED Array Not Working
```bash
# Check I2C devices
sudo i2cdetect -y 1

# Check GPIO
gpio readall
```

### Performance Issues
1. **Reduce resolution**: Lower camera resolution in config
2. **Optimize model**: Use smaller YOLOv8 model (yolov8n instead of yolov8x)
3. **Adjust FPS**: Lower frame rate in camera config
4. **Memory optimization**: Reduce batch sizes and model complexity

## Contributing

### Code Style
- Follow PEP 8 style guidelines
- Use type hints
- Add docstrings to all functions
- Keep functions small and focused

### Git Workflow
1. Create feature branch
2. Make changes
3. Add tests
4. Update documentation
5. Submit pull request

### Documentation
- Update README.md for user-facing changes
- Update this guide for developer-facing changes
- Add inline comments for complex logic

## Future Enhancements

### Planned Features
- [ ] Depth sensor integration
- [ ] GPS navigation
- [ ] Voice command system
- [ ] Mobile app companion
- [ ] Cloud synchronization
- [ ] Advanced path planning algorithms

### Performance Improvements
- [ ] TensorRT optimization
- [ ] Multi-threading improvements
- [ ] Memory optimization
- [ ] Battery life optimization

### Hardware Support
- [ ] Additional camera types
- [ ] Different LED array configurations
- [ ] Alternative audio systems
- [ ] Sensor fusion

## Support

### Getting Help
- Check the troubleshooting section
- Review logs in `logs/cursorPathlight.log`
- Search existing issues
- Create new issue with detailed information

### Reporting Bugs
Include the following information:
- System configuration
- Error logs
- Steps to reproduce
- Expected vs actual behavior

### Feature Requests
- Describe the feature clearly
- Explain the use case
- Consider implementation complexity
- Provide mockups if applicable 