#!/bin/bash
# Transfer Pathlight project to Jetson SSD
# This script copies all files to the connected SSD

set -e  # Exit on any error

echo "=========================================="
echo "Pathlight SSD Transfer Script"
echo "=========================================="

# Configuration
SOURCE_DIR="/Users/ariyanp/pathlight_project"
SSD_MOUNT_POINT=""
SSD_TARGET_DIR="/home/nvidia/cursorPathlight"

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

# Function to find SSD mount point
find_ssd_mount() {
    print_status "Searching for connected SSD..."
    
    # Check for mounted volumes
    for mount in /Volumes/*; do
        if [ -d "$mount" ] && [ "$mount" != "/Volumes/Macintosh HD" ]; then
            print_status "Found potential SSD at: $mount"
            SSD_MOUNT_POINT="$mount"
            return 0
        fi
    done
    
    # Try to mount Linux partitions
    print_status "Attempting to mount Linux partitions..."
    
    # Check for disk4 (from previous output)
    if [ -e "/dev/disk4" ]; then
        print_status "Found disk4, attempting to mount..."
        
        # Create mount point
        sudo mkdir -p /Volumes/JETSON_SSD 2>/dev/null || true
        
        # Try to mount the largest partition (likely the main filesystem)
        for partition in /dev/disk4s*; do
            if [ -e "$partition" ]; then
                print_status "Trying to mount $partition..."
                if sudo mount -t ext4 "$partition" /Volumes/JETSON_SSD 2>/dev/null; then
                    SSD_MOUNT_POINT="/Volumes/JETSON_SSD"
                    print_success "Successfully mounted SSD at $SSD_MOUNT_POINT"
                    return 0
                fi
            fi
        done
    fi
    
    print_error "Could not find or mount SSD"
    return 1
}

# Function to verify SSD is writable
verify_ssd_access() {
    if [ -z "$SSD_MOUNT_POINT" ]; then
        print_error "No SSD mount point found"
        return 1
    fi
    
    print_status "Verifying SSD access..."
    
    # Test write access
    TEST_FILE="$SSD_MOUNT_POINT/test_write.tmp"
    if echo "test" > "$TEST_FILE" 2>/dev/null; then
        rm "$TEST_FILE"
        print_success "SSD is writable"
        return 0
    else
        print_error "SSD is not writable"
        return 1
    fi
}

# Function to create directory structure
create_directories() {
    print_status "Creating directory structure on SSD..."
    
    # Create main directory
    sudo mkdir -p "$SSD_MOUNT_POINT$SSD_TARGET_DIR"
    
    # Create subdirectories
    sudo mkdir -p "$SSD_MOUNT_POINT$SSD_TARGET_DIR"/{ai,config,core,docs,hardware,scripts,tests,data,logs,models}
    sudo mkdir -p "$SSD_MOUNT_POINT$SSD_TARGET_DIR"/hardware/{audio,camera,imu,leds,microcontroller}
    sudo mkdir -p "$SSD_MOUNT_POINT$SSD_TARGET_DIR"/core/{vision,navigation,memory}
    sudo mkdir -p "$SSD_MOUNT_POINT$SSD_TARGET_DIR"/ai/{assistant,voice}
    
    print_success "Directory structure created"
}

# Function to copy files
copy_files() {
    print_status "Copying Pathlight files to SSD..."
    
    # Copy main files
    sudo cp "$SOURCE_DIR"/main.py "$SSD_MOUNT_POINT$SSD_TARGET_DIR/"
    sudo cp "$SOURCE_DIR"/requirements.txt "$SSD_MOUNT_POINT$SSD_TARGET_DIR/"
    sudo cp "$SOURCE_DIR"/README.md "$SSD_MOUNT_POINT$SSD_TARGET_DIR/"
    sudo cp "$SOURCE_DIR"/MASTER_INSTRUCTIONS.md "$SSD_MOUNT_POINT$SSD_TARGET_DIR/"
    sudo cp "$SOURCE_DIR"/QUICK_REFERENCE.md "$SSD_MOUNT_POINT$SSD_TARGET_DIR/"
    sudo cp "$SOURCE_DIR"/TRANSFER_INSTRUCTIONS.md "$SSD_MOUNT_POINT$SSD_TARGET_DIR/"
    
    # Copy directories
    sudo cp -r "$SOURCE_DIR"/ai "$SSD_MOUNT_POINT$SSD_TARGET_DIR/"
    sudo cp -r "$SOURCE_DIR"/config "$SSD_MOUNT_POINT$SSD_TARGET_DIR/"
    sudo cp -r "$SOURCE_DIR"/core "$SSD_MOUNT_POINT$SSD_TARGET_DIR/"
    sudo cp -r "$SOURCE_DIR"/docs "$SSD_MOUNT_POINT$SSD_TARGET_DIR/"
    sudo cp -r "$SOURCE_DIR"/hardware "$SSD_MOUNT_POINT$SSD_TARGET_DIR/"
    sudo cp -r "$SOURCE_DIR"/scripts "$SSD_MOUNT_POINT$SSD_TARGET_DIR/"
    sudo cp -r "$SOURCE_DIR"/tests "$SSD_MOUNT_POINT$SSD_TARGET_DIR/"
    
    print_success "Files copied successfully"
}

# Function to set permissions
set_permissions() {
    print_status "Setting file permissions..."
    
    # Set ownership to nvidia user (will be created on Jetson)
    sudo chown -R 1000:1000 "$SSD_MOUNT_POINT$SSD_TARGET_DIR"
    
    # Set executable permissions for scripts
    sudo chmod +x "$SSD_MOUNT_POINT$SSD_TARGET_DIR"/scripts/*.py
    sudo chmod +x "$SSD_MOUNT_POINT$SSD_TARGET_DIR"/scripts/*.sh
    
    print_success "Permissions set"
}

# Function to create setup script for Jetson
create_jetson_setup() {
    print_status "Creating Jetson setup script..."
    
    cat > "$SSD_MOUNT_POINT$SSD_TARGET_DIR/setup_jetson.sh" << 'JETSON_EOF'
#!/bin/bash
# Jetson Setup Script
# Run this on the Jetson after transferring files

set -e

echo "=========================================="
echo "Pathlight Jetson Setup"
echo "=========================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

# Update system
print_status "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install dependencies
print_status "Installing system dependencies..."
sudo apt install -y \
    python3-pip \
    python3-venv \
    git \
    cmake \
    build-essential \
    libatlas-base-dev \
    liblapack-dev \
    libblas-dev \
    libhdf5-dev \
    libhdf5-serial-dev \
    libhdf5-103 \
    libqtgui4 \
    libqtwebkit4 \
    libqt4-test \
    python3-pyqt5 \
    libgtk-3-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libatlas-base-dev \
    gfortran \
    libgstreamer1.0-0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-gtk3 \
    gstreamer1.0-qt5 \
    gstreamer1.0-pulseaudio \
    nvidia-l4t-camera \
    nvidia-l4t-multimedia \
    v4l-utils \
    libv4l-dev \
    pulseaudio \
    pulseaudio-utils \
    portaudio19-dev \
    libasound2-dev \
    libportaudio2 \
    libportaudiocpp0 \
    ffmpeg \
    libav-tools \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libcanberra-gtk-module \
    libcanberra-gtk3-module \
    libgtk-3-dev \
    libgtk2.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer1.0-dev \
    libgirepository1.0-dev \
    libcairo2-dev \
    libpango1.0-dev \
    libatk1.0-dev \
    libgdk-pixbuf2.0-dev \
    libgtk-3-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libatlas-base-dev \
    gfortran \
    libgstreamer1.0-0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-gtk3 \
    gstreamer1.0-qt5 \
    gstreamer1.0-pulseaudio \
    nvidia-l4t-camera \
    nvidia-l4t-multimedia

# Create nvidia user if it doesn't exist
if ! id "nvidia" &>/dev/null; then
    print_status "Creating nvidia user..."
    sudo useradd -m -s /bin/bash nvidia
    sudo usermod -aG sudo nvidia
    echo "nvidia:pathlight123" | sudo chpasswd
fi

# Set up Python virtual environment
print_status "Setting up Python virtual environment..."
cd /home/nvidia/cursorPathlight
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Python dependencies
print_status "Installing Python dependencies..."
pip install -r requirements.txt

# Create necessary directories
print_status "Creating data and log directories..."
mkdir -p data logs models
mkdir -p data/faces data/stereo_calibration

# Set up systemd service
print_status "Setting up systemd service..."
sudo cp scripts/cursorPathlight.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable cursorPathlight.service

# Set permissions
print_status "Setting file permissions..."
sudo chown -R nvidia:nvidia /home/nvidia/cursorPathlight
chmod +x scripts/*.py scripts/*.sh

print_success "Jetson setup completed!"
print_status "To start Pathlight: sudo systemctl start cursorPathlight"
print_status "To check status: sudo systemctl status cursorPathlight"
print_status "To view logs: sudo journalctl -u cursorPathlight -f"

JETSON_EOF
    
    chmod +x "$SSD_MOUNT_POINT$SSD_TARGET_DIR/setup_jetson.sh"
    print_success "Jetson setup script created"
}

# Function to create README for Jetson
create_jetson_readme() {
    print_status "Creating Jetson README..."
    
    cat > "$SSD_MOUNT_POINT$SSD_TARGET_DIR/JETSON_README.md" << 'README_EOF'
# Pathlight Jetson Setup Instructions

## Quick Start

1. **Connect SSD to Jetson**
   - Power off Jetson
   - Connect SSD to Jetson
   - Power on Jetson

2. **Run Setup Script**
   ```bash
   cd /home/nvidia/cursorPathlight
chmod +x setup_jetson.sh
./setup_jetson.sh
   ```

3. **Start Pathlight**
   ```bash
   sudo systemctl start cursorPathlight
   ```

4. **Check Status**
   ```bash
   sudo systemctl status cursorPathlight
sudo journalctl -u cursorPathlight -f
   ```

## Hardware Connections

### IMU (MPU6050)
- VCC → 3.3V
- GND → GND
- SCL → I2C1_SCL (Pin 28)
- SDA → I2C1_SDA (Pin 27)

### Dual IMX219 Cameras
- Left Camera CSI → CSI0
- Right Camera CSI → CSI1
- Power via CSI connector

### LED Array
- VCC → 3.3V
- GND → GND
- SCL → I2C1_SCL (Pin 28)
- SDA → I2C1_SDA (Pin 27)
- I2C Address: 0x70

### Small Speakers
- Left Speaker → Audio Out L
- Right Speaker → Audio Out R
- Ground → Audio GND

### Microcontroller
- TX → GPIO14
- RX → GPIO15
- Enable → GPIO18
- VCC → 3.3V
- GND → GND

## Configuration

Edit `config/config.yaml` to customize settings:
- Camera parameters
- Audio settings
- IMU calibration
- Safety thresholds

## Troubleshooting

### Camera Issues
```bash
# Check camera detection
v4l2-ctl --list-devices

# Test camera
python3 scripts/test_camera.py
```

### Audio Issues
```bash
# Check audio devices
aplay -l
arecord -l

# Test audio
python3 scripts/test_audio.py
```

### IMU Issues
```bash
# Check I2C devices
i2cdetect -y 1

# Test IMU
python3 scripts/test_imu.py
```

### System Issues
```bash
# Check system logs
sudo journalctl -u pathlight -f

# Restart service
sudo systemctl restart pathlight
```

## Autonomous Operation

Pathlight will automatically start when:
1. Battery is connected
2. System boots up
3. All hardware is detected

To disable autonomous startup:
```bash
sudo systemctl disable pathlight
```

To enable manual startup:
```bash
sudo systemctl enable pathlight
```

## Safety Features

- Low battery shutdown (20%)
- Critical battery shutdown (10%)
- Temperature monitoring
- Emergency stop detection
- System health monitoring

## Support

For issues, check:
1. Hardware connections
2. System logs
3. Configuration files
4. Test scripts output

README_EOF
    
    print_success "Jetson README created"
}

# Main execution
main() {
    print_status "Starting Pathlight SSD transfer..."
    
    # Check if source directory exists
    if [ ! -d "$SOURCE_DIR" ]; then
        print_error "Source directory not found: $SOURCE_DIR"
        exit 1
    fi
    
    # Find and mount SSD
    if ! find_ssd_mount; then
        print_error "Failed to find or mount SSD"
        exit 1
    fi
    
    # Verify SSD access
    if ! verify_ssd_access; then
        print_error "SSD is not accessible"
        exit 1
    fi
    
    # Create directory structure
    create_directories
    
    # Copy files
    copy_files
    
    # Set permissions
    set_permissions
    
    # Create Jetson setup script
    create_jetson_setup
    
    # Create Jetson README
    create_jetson_readme
    
    print_success "=========================================="
    print_success "Pathlight transfer completed successfully!"
    print_success "=========================================="
    print_status "SSD mounted at: $SSD_MOUNT_POINT"
    print_status "Files copied to: $SSD_MOUNT_POINT$SSD_TARGET_DIR"
    print_status ""
    print_status "Next steps:"
    print_status "1. Safely eject SSD from Mac"
    print_status "2. Connect SSD to Jetson"
    print_status "3. Run: ./setup_jetson.sh"
    print_status "4. Start: sudo systemctl start pathlight"
}

# Run main function
main "$@"
