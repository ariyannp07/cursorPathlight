# Pathlight - AI Wearable Navigation Assistant

## Project Overview
Pathlight is an AI-powered wearable headset designed to help blind individuals navigate safely and socialize effectively. The system operates like Tesla's self-driving technology, providing real-time obstacle detection, path planning, and social interaction assistance.

## Core Features
1. **Real-time Object & Face Detection** - Using YOLOv8 for obstacle identification
2. **Safe Path Calculation** - AI-powered navigation with obstacle avoidance
3. **Face Recognition & Memory** - Recall familiar faces and interaction history
4. **LED Direction Display** - Visual course heading through LED array
5. **Audio Guidance** - Voice directions and environmental descriptions
6. **AI Assistant** - General Q&A, navigation, and task completion

## Hardware Requirements
- NVIDIA Jetson Orin Nano
- Dual IMX219 CSI camera module (ribbon cable connection)
- LED array for direction display
- Audio output system
- Microphone for voice input

## Software Architecture
```
pathlight/
├── core/                    # Core AI and processing modules
│   ├── vision/             # Computer vision (YOLOv8, face detection)
│   ├── navigation/         # Path planning and obstacle avoidance
│   ├── audio/              # Text-to-speech and speech recognition
│   └── memory/             # Face recognition and interaction history
├── hardware/               # Hardware interface modules
│   ├── camera/             # Camera control and image processing
│   ├── leds/               # LED array control
│   └── audio/              # Audio I/O management
├── ai/                     # AI assistant and plugins
│   ├── assistant/          # Main AI assistant
│   ├── plugins/            # External service integrations
│   └── voice/              # Voice interaction system
├── config/                 # Configuration files
├── tests/                  # Unit and integration tests
├── scripts/                # Setup and utility scripts
└── docs/                   # Documentation
```

## Setup Instructions
See `docs/setup.md` for detailed installation and configuration instructions.

## Development Status
- [ ] Basic Jetson setup and environment
- [ ] Camera integration
- [ ] YOLOv8 object detection
- [ ] Face recognition system
- [ ] Path planning algorithm
- [ ] LED control system
- [ ] Audio processing
- [ ] AI assistant integration
- [ ] Voice interaction
- [ ] Testing and optimization

## License
[To be determined]
