#!/bin/bash

# Pathlight Setup Script for NVIDIA Jetson Orin Nano
# This script sets up the complete environment for the Pathlight AI wearable

set -e  # Exit on any error

echo "=========================================="
echo "Pathlight Setup Script for Jetson Orin Nano"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on Jetson
check_jetson() {
    print_status "Checking system..."
    
    if [ ! -f "/etc/nv_tegra_release" ]; then
        print_warning "This script is designed for NVIDIA Jetson devices"
        print_warning "Some features may not work on other systems"
    else
        print_success "NVIDIA Jetson detected"
        cat /etc/nv_tegra_release
    fi
    
    # Check CUDA
    if command -v nvidia-smi &> /dev/null; then
        print_success "NVIDIA drivers found"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
    else
        print_error "NVIDIA drivers not found. Please install JetPack SDK first."
        exit 1
    fi
}

# Update system packages
update_system() {
    print_status "Updating system packages..."
    
    sudo apt update
    sudo apt upgrade -y
    
    print_success "System packages updated"
}

# Install system dependencies
install_system_deps() {
    print_status "Installing system dependencies..."
    
    # Essential tools
    sudo apt install -y \
        git \
        curl \
        wget \
        vim \
        nano \
        htop \
        build-essential \
        cmake \
        pkg-config \
        libssl-dev \
        libffi-dev \
        python3-dev \
        python3-pip \
        python3-venv \
        python3-setuptools \
        python3-wheel
    
    # Audio dependencies
    sudo apt install -y \
        portaudio19-dev \
        python3-pyaudio \
        espeak \
        espeak-data \
        pulseaudio \
        pulseaudio-utils
    
    # Camera and video dependencies (including IMX219 CSI camera support)
    sudo apt install -y \
        v4l-utils \
        libv4l-dev \
        libgstreamer1.0-dev \
        libgstreamer-plugins-base1.0-dev \
        libgstreamer-plugins-bad1.0-dev \
        gstreamer1.0-plugins-base \
        gstreamer1.0-plugins-good \
        gstreamer1.0-plugins-bad \
        gstreamer1.0-plugins-ugly \
        gstreamer1.0-libav \
        gstreamer1.0-tools \
        gstreamer1.0-x \
        gstreamer1.0-alsa \
        gstreamer1.0-gl \
        gstreamer1.0-gtk3 \
        gstreamer1.0-qt5 \
        gstreamer1.0-pulseaudio \
        nvidia-l4t-camera \
        nvidia-l4t-multimedia
    
    # GPIO and I2C dependencies
    sudo apt install -y \
        python3-gpiozero \
        i2c-tools \
        libi2c-dev \
        python3-smbus2
    
    # Additional dependencies
    sudo apt install -y \
        libatlas-base-dev \
        liblapack-dev \
        libblas-dev \
        libhdf5-dev \
        libhdf5-serial-dev \
        libhdf5-103 \
        qtbase5-dev \
        qtchooser \
        qt5-qmake \
        qtbase5-dev-tools \
        libqt5gui5 \
        libqt5widgets5 \
        libqt5core5a \
        libqt5opengl5 \
        libqt5opengl5-dev \
        python3-pyqt5 \
        libgtk-3-dev \
        libcanberra-gtk-module \
        libcanberra-gtk3-module
    
    print_success "System dependencies installed"
}

# Setup Python environment
setup_python_env() {
    print_status "Setting up Python environment..."
    
    # Create project directory
    mkdir -p ~/cursorPathlight
    cd ~/cursorPathlight
    
    # Create virtual environment
    python3 -m venv cursorPathlight_env
    source cursorPathlight_env/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    
    print_success "Python environment created"
}

# Install PyTorch for Jetson
install_pytorch() {
    print_status "Installing PyTorch for Jetson..."
    
    source ~/cursorPathlight/cursorPathlight_env/bin/activate
    
    # Install PyTorch with CUDA support for Jetson
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    
    print_success "PyTorch installed"
}

# Install Python dependencies
install_python_deps() {
    print_status "Installing Python dependencies..."
    
    source ~/cursorPathlight/cursorPathlight_env/bin/activate
    
    # Install core dependencies
    pip install numpy pillow matplotlib scipy
    
    # Install OpenCV
    pip install opencv-python opencv-contrib-python
    
    # Install AI and ML libraries
    pip install ultralytics  # YOLOv8
    pip install face-recognition
    pip install scikit-learn
    pip install transformers
    
    # Install audio libraries
    pip install pyaudio pyttsx3 speechrecognition librosa
    
    # Install AI assistant libraries
    pip install openai googlemaps
    
    # Install hardware control libraries
    pip install smbus2 spidev
    
    # Install web and API libraries
    pip install requests fastapi uvicorn
    
    # Install utility libraries
    pip install python-dotenv pyyaml click tqdm loguru
    
    # Install testing libraries
    pip install pytest pytest-cov pytest-asyncio
    
    # Install development libraries
    pip install black flake8 mypy
    
    print_success "Python dependencies installed"
}

# Setup project files
setup_project() {
    print_status "Setting up project files..."
    
    cd ~/cursorPathlight
    
    # Create necessary directories
    mkdir -p data logs models config
    
    # Create configuration file
    if [ ! -f "config/config.yaml" ]; then
        cp config/config.example.yaml config/config.yaml
        print_success "Configuration file created"
    else
        print_warning "Configuration file already exists"
    fi
    
    # Set permissions
    chmod +x main.py
    chmod +x scripts/*.sh
    
    print_success "Project files setup complete"
}

# Setup system services
setup_services() {
    print_status "Setting up system services..."
    
    # Enable I2C
    if ! grep -q "i2c_arm=on" /boot/firmware/config.txt; then
        echo "i2c_arm=on" | sudo tee -a /boot/firmware/config.txt
        print_success "I2C enabled"
    fi
    
    # Setup auto-start service
    sudo tee /etc/systemd/system/cursorPathlight.service > /dev/null <<EOF
[Unit]
Description=Pathlight AI Navigation Assistant
After=network.target

[Service]
Type=simple
User=nvidia
WorkingDirectory=/home/nvidia/cursorPathlight
Environment=PATH=/home/nvidia/cursorPathlight/cursorPathlight_env/bin
ExecStart=/home/nvidia/cursorPathlight/cursorPathlight_env/bin/python main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
    
    # Enable service
    sudo systemctl enable cursorPathlight.service
    print_success "System service configured"
}

# Performance optimization
optimize_performance() {
    print_status "Optimizing system performance..."
    
    # Set GPU memory allocation
    sudo tee /etc/systemd/system/nvzramconfig.service > /dev/null <<EOF
[Unit]
Description=NVIDIA ZRAM configuration
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/usr/bin/echo 1 > /sys/devices/virtual/block/zram0/disksize
ExecStart=/usr/bin/echo 1 > /sys/devices/virtual/block/zram0/comp_algorithm
ExecStart=/usr/bin/echo 1 > /sys/devices/virtual/block/zram0/max_comp_streams
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF
    
    # Optimize for real-time processing
    echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
    echo 'vm.vfs_cache_pressure=50' | sudo tee -a /etc/sysctl.conf
    
    # Enable services
    sudo systemctl enable nvzramconfig.service
    
    print_success "Performance optimizations applied"
}

# Test installation
test_installation() {
    print_status "Testing installation..."
    
    source ~/cursorPathlight/cursorPathlight_env/bin/activate
    
    # Test Python imports
    python3 -c "
import torch
import cv2
import numpy as np
import face_recognition
import pyttsx3
import speech_recognition
import ultralytics
print('All core libraries imported successfully')
"
    
    # Test CUDA
    python3 -c "
import torch
if torch.cuda.is_available():
    print(f'CUDA available: {torch.cuda.get_device_name(0)}')
    print(f'CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('CUDA not available')
"
    
    # Test IMX219 CSI camera
    print_status "Testing IMX219 CSI camera..."
    v4l2-ctl --list-devices
    
    # Check for IMX219 cameras
    if v4l2-ctl --list-devices | grep -i "imx219" > /dev/null; then
        print_success "IMX219 camera detected"
    else
        print_warning "IMX219 camera not detected in v4l2-ctl output"
        print_warning "This is normal if camera is not connected or drivers not loaded"
    fi
    
    print_success "Installation test completed"
}

# Main setup function
main() {
    print_status "Starting Pathlight setup..."
    
    check_jetson
    update_system
    install_system_deps
    setup_python_env
    install_pytorch
    install_python_deps
    setup_project
    setup_services
    optimize_performance
    test_installation
    
    print_success "Pathlight setup completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Copy your project files to ~/cursorPathlight/"
    echo "2. Edit config/config.yaml with your settings"
    echo "3. Add your API keys to the configuration"
    echo "4. Test the system: cd ~/cursorPathlight && source cursorPathlight_env/bin/activate && python main.py"
    echo "5. Start the service: sudo systemctl start pathlight"
    echo ""
    echo "For more information, see docs/setup.md"
}

# Run main function
main "$@" 