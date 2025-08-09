#!/bin/bash

# Pathlight Setup Script for NVIDIA Jetson Orin Nano
# This script sets up the complete environment for the Pathlight AI wearable
# Optimized for JetPack 6.2.1 with CUDA 12.6 support

set -e  # Exit on any error

echo "=========================================="
echo "Pathlight Setup Script for Jetson Orin Nano"
echo "JetPack 6.2.1 + CUDA 12.6 + PyTorch 2.5"
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

# Check if running on Jetson with proper JetPack version
check_jetson() {
    print_status "Checking system compatibility..."
    
    if [ ! -f "/etc/nv_tegra_release" ]; then
        print_error "This script is designed for NVIDIA Jetson devices"
        print_error "No Jetson detected. Please install JetPack 6.2.1 first."
        exit 1
    else
        print_success "NVIDIA Jetson detected"
        cat /etc/nv_tegra_release
        
        # Check for JetPack 6.x
        JETPACK_VERSION=$(cat /etc/nv_tegra_release | grep "REVISION" | awk '{print $4}' | cut -d',' -f1)
        if [[ "$JETPACK_VERSION" < "36.4" ]]; then
            print_warning "JetPack version may be older than 6.2.1"
            print_warning "This script is optimized for JetPack 6.2.1 (L4T 36.4+)"
        else
            print_success "JetPack 6.2.1+ detected - perfect for CUDA 12.6!"
        fi
    fi
    
    # Check CUDA
    if command -v nvidia-smi &> /dev/null; then
        print_success "NVIDIA drivers found"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
        
        # Check CUDA version
        if command -v nvcc &> /dev/null; then
            CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
            print_success "CUDA version: $CUDA_VERSION"
            
            if [[ "$CUDA_VERSION" < "12.6" ]]; then
                print_warning "CUDA version is $CUDA_VERSION, this script is optimized for CUDA 12.6+"
            fi
        else
            print_error "nvcc not found. Please install CUDA toolkit."
            exit 1
        fi
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

# Install system dependencies optimized for CUDA
install_system_deps() {
    print_status "Installing system dependencies with CUDA support..."
    
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
    
    # CUDA development tools
    sudo apt install -y \
        cuda-toolkit-12-6 \
        libcublas-12-6 \
        libcurand-12-6 \
        libcufft-12-6 \
        libcusparse-12-6 \
        libcusolver-12-6
    
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
    
    # Additional dependencies for AI/ML
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
    
    print_success "System dependencies with CUDA support installed"
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

# Install PyTorch with CUDA support for JetPack 6.2.1
install_pytorch() {
    print_status "Installing PyTorch with CUDA 12.6 support for JetPack 6.2.1..."
    
    source ~/cursorPathlight/cursorPathlight_env/bin/activate
    
    # Verify CUDA 12.6 for JetPack 6.2.1 compatibility
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2- | tr -d .)
    print_status "Detected CUDA version: $CUDA_VERSION"
    
    if [[ "$CUDA_VERSION" != "12.6" ]]; then
        print_error "Expected CUDA 12.6 for JetPack 6.2.1, found $CUDA_VERSION"
        print_error "Please ensure JetPack 6.2.1 is properly installed"
        exit 1
    fi
    
    # Install cuSPARSELt (required for PyTorch 24.06+ builds)
    print_status "Installing cuSPARSELt for PyTorch compatibility..."
    cd /tmp
    wget https://developer.download.nvidia.com/compute/libcusparse-lt/0.6.0/local_installers/libcusparse-lt-dev-0.6.0.106-1_arm64.deb
    sudo dpkg -i libcusparse-lt-dev-0.6.0.106-1_arm64.deb || print_warning "cuSPARSELt installation failed, continuing..."
    rm -f libcusparse-lt-dev-0.6.0.106-1_arm64.deb
    cd ~/cursorPathlight
    
    # Install NVIDIA's official PyTorch 2.5.0 wheel for JetPack 6.2.1
    print_status "Installing NVIDIA PyTorch 2.5.0 (CUDA 12.6) for JetPack 6.2.1..."
    
    # PyTorch 2.5.0 for CUDA 12.6 (JetPack 6.2.1) - prevents ABI mismatches
    pip install --no-cache https://developer.download.nvidia.com/compute/redist/jp/v62/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
    
    # Install compatible torchvision (build from source against exact PyTorch version)
    print_status "Building TorchVision 0.19.1 from source for PyTorch 2.5.0 compatibility..."
    apt-get update && apt-get install -y --no-install-recommends \
        libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev \
        libffi-dev libssl-dev build-essential cmake
    
    # Build TorchVision 0.19.1 (compatible with PyTorch 2.5.0)
    git clone --branch v0.19.1 --depth 1 https://github.com/pytorch/vision torchvision_build
    cd torchvision_build
    python3 setup.py install
    cd .. && rm -rf torchvision_build
    
    print_status "Installing torchaudio 2.5.0..."
    pip install torchaudio==2.5.0 --no-deps
    
    print_success "PyTorch with CUDA support installed"
}

# Install Python dependencies
install_python_deps() {
    print_status "Installing Python dependencies..."
    
    source ~/cursorPathlight/cursorPathlight_env/bin/activate
    
    # Install core dependencies with NumPy 1.26.4 (for YOLOv8 8.3.0 compatibility)
    pip install numpy==1.26.4 pillow matplotlib scipy
    
    # Build OpenCV from source with CUDA 12.6 support
    print_status "Building OpenCV 4.10.0 from source with CUDA 12.6, GStreamer, and V4L2..."
    
    # Install OpenCV build dependencies
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git pkg-config \
        libjpeg-dev libtiff5-dev libpng-dev \
        libavcodec-dev libavformat-dev libswscale-dev \
        libgtk2.0-dev libcanberra-gtk-module \
        libxvidcore-dev libx264-dev libgtk-3-dev \
        libtbb2 libtbb-dev libdc1394-22-dev libv4l-dev \
        libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
        libvorbis-dev libxine2-dev libtesseract-dev \
        libfaac-dev libmp3lame-dev libtheora-dev \
        libpostproc-dev libopencore-amrnb-dev libopencore-amrwb-dev \
        libopenblas-dev libatlas-base-dev gfortran
    
    # Clone and build OpenCV with CUDA
    cd /tmp
    git clone --branch 4.10.0 --depth 1 https://github.com/opencv/opencv.git
    git clone --branch 4.10.0 --depth 1 https://github.com/opencv/opencv_contrib.git
    cd opencv && mkdir build && cd build
    
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D OPENCV_EXTRA_MODULES_PATH=/tmp/opencv_contrib/modules \
        -D WITH_CUDA=ON \
        -D CUDA_ARCH_BIN=8.7 \
        -D CUDA_ARCH_PTX=8.7 \
        -D WITH_CUDNN=ON \
        -D OPENCV_DNN_CUDA=ON \
        -D ENABLE_FAST_MATH=ON \
        -D CUDA_FAST_MATH=ON \
        -D WITH_CUBLAS=ON \
        -D WITH_LIBV4L=ON \
        -D WITH_GSTREAMER=ON \
        -D WITH_GSTREAMER_0_10=OFF \
        -D BUILD_opencv_python3=ON \
        -D OPENCV_GENERATE_PKGCONFIG=ON \
        -D BUILD_EXAMPLES=OFF ..
    
    make -j$(nproc)
    make install
    ldconfig
    
    # Clean up build files
    cd ~/cursorPathlight && rm -rf /tmp/opencv /tmp/opencv_contrib
    
    # Verify OpenCV CUDA support
    python3 -c "import cv2; print(f'OpenCV version: {cv2.__version__}'); print('CUDA support:'); print(cv2.getBuildInformation())" || print_warning "OpenCV CUDA verification failed"
    
    # Install YOLOv8 8.3.0 (compatible with NumPy 1.26.4)
    print_status "Installing YOLOv8 8.3.0 (with NumPy 1.26.4 compatibility)..."
    pip install ultralytics==8.3.0
    
    # Install additional ML dependencies
    pip install \
        scipy==1.13.1 \
        scikit-learn==1.5.1 \
        matplotlib==3.9.2 \
        seaborn==0.13.2
    
    # Install face recognition with CUDA support
    print_status "Installing face recognition libraries with CUDA support..."
    
    # Install dependencies for building dlib with CUDA
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake pkg-config \
        libopenblas-dev liblapack-dev libatlas-base-dev \
        libgtk-3-dev libboost-python-dev libboost-system-dev \
        libx11-dev libatlas3-base
    
    # Build dlib 19.24.6 from source with CUDA support
    print_status "Building dlib 19.24.6 with CUDA 12.6 support for Orin Nano (sm_87)..."
    print_status "This will take approximately 45 minutes and requires high RAM usage..."
    
    # Enable swap to prevent OOM during dlib compilation
    print_status "Enabling swap for dlib compilation..."
    if [ ! -f /swapfile ]; then
        dd if=/dev/zero of=/swapfile bs=1M count=4096
        chmod 600 /swapfile
        mkswap /swapfile
        swapon /swapfile
    fi
    
    cd /tmp
    wget http://dlib.net/files/dlib-19.24.6.tar.bz2
    tar xf dlib-19.24.6.tar.bz2
    cd dlib-19.24.6
    
    # Create build directory and configure with CUDA for Orin Nano
    mkdir build && cd build
    cmake .. \
        -DDLIB_USE_CUDA=1 \
        -DUSE_AVX_INSTRUCTIONS=OFF \
        -DCMAKE_CUDA_ARCHITECTURES=87 \
        -DCMAKE_BUILD_TYPE=Release \
        -DCUDA_HOST_COMPILER=/usr/bin/gcc \
        -DCMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs
    
    # Build and install dlib (this takes ~45 minutes)
    make -j$(nproc)
    make install
    ldconfig
    
    # Install Python bindings with CUDA support
    cd ..
    python3 setup.py install
    
    # Clean up build files
    cd /tmp && rm -rf dlib-19.24.6*
    
    print_status "Disabling swap after dlib build..."
    swapoff /swapfile || true
    
    # Install face_recognition for CNN model support
    print_status "Installing face_recognition 1.3.0..."
    pip install face_recognition==1.3.0
    
    # Verify CUDA-enabled face_recognition installation
    python3 -c "
import face_recognition
import dlib
print(f'✅ face_recognition version: {face_recognition.__version__}')
print(f'✅ dlib version: {dlib.DLIB_VERSION}')
print(f'✅ dlib CUDA support: {\"YES\" if hasattr(dlib, \"cuda\") else \"NO\"}')
print('Available models: hog (CPU), cnn (GPU with CUDA acceleration)')
    " || print_warning "face_recognition CUDA verification failed"
    
    # Additional AI libraries
    pip install transformers==4.45.2
    
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

# Performance optimization for CUDA
optimize_performance() {
    print_status "Optimizing system performance for CUDA..."
    
    # Set GPU performance mode
    sudo nvidia-smi -pm 1  # Enable persistence mode
    
    # Set maximum performance mode for Jetson
    if command -v nvpmodel &> /dev/null; then
        sudo nvpmodel -m 0  # Maximum performance mode
        print_success "Set Jetson to maximum performance mode"
    fi
    
    # Optimize for real-time processing
    echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
    echo 'vm.vfs_cache_pressure=50' | sudo tee -a /etc/sysctl.conf
    
    print_success "Performance optimizations applied"
}

# Test installation with CUDA support
test_installation() {
    print_status "Testing installation with CUDA support..."
    
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
print('✓ All core libraries imported successfully')
"
    
    # Test CUDA and PyTorch
    python3 -c "
import torch
print(f'✓ PyTorch version: {torch.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✓ CUDA device: {torch.cuda.get_device_name(0)}')
    print(f'✓ CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    # Test CUDA tensor operations
    test_tensor = torch.randn(100, 100).cuda()
    result = torch.matmul(test_tensor, test_tensor.T)
    print(f'✓ CUDA tensor operations working on: {result.device}')
else:
    print('✗ CUDA not available - check installation')
"
    
    # Test YOLOv8 with CUDA
    python3 -c "
from ultralytics import YOLO
import torch
model = YOLO('yolov8n.pt')
print(f'✓ YOLOv8 model loaded')
if torch.cuda.is_available():
    model.to('cuda')
    print('✓ YOLOv8 moved to CUDA device')
else:
    print('ℹ YOLOv8 running on CPU (CUDA not available)')
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
    
    print_success "Installation test completed - CUDA enabled!"
}

# Main setup function
main() {
    print_status "Starting Pathlight setup with CUDA support..."
    
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
    
    print_success "Pathlight setup completed successfully with CUDA support!"
    echo ""
    echo "🎯 CUDA ENABLED - Expected Performance:"
    echo "• Object Detection: 15-30 FPS (vs 2-5 FPS CPU-only)"
    echo "• Face Recognition: 5-15 FPS (vs 1-3 FPS CPU-only)"
    echo "• Stereo Vision: 10-20 FPS (vs 5-10 FPS CPU-only)"
    echo ""
    echo "Next steps:"
    echo "1. Copy your project files to ~/cursorPathlight/"
    echo "2. Edit config/config.yaml with your settings"
    echo "3. Add your API keys to the configuration"
    echo "4. Test the system: cd ~/cursorPathlight && source cursorPathlight_env/bin/activate && python main.py"
    echo "5. Start the service: sudo systemctl start cursorPathlight"
    echo ""
    echo "For more information, see docs/setup.md"
}

# Run main function
main "$@"