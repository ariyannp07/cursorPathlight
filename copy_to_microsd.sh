#!/bin/bash
# Copy Pathlight files to microSD when it becomes writable

echo "=========================================="
echo "Pathlight microSD Copy Script"
echo "=========================================="

MICROSD_PATH="/Volumes/JETSON_SSD"
TARGET_DIR="$MICROSD_PATH/home/nvidia/pathlight"

# Check if microSD is writable
if [ ! -w "$MICROSD_PATH" ]; then
    echo "❌ microSD is not writable at $MICROSD_PATH"
    echo "Please ensure microSD has write permissions and try again"
    exit 1
fi

echo "✅ microSD is writable, starting copy..."

# Create directory structure
echo "📁 Creating directories..."
mkdir -p "$TARGET_DIR"/{ai,config,core,docs,hardware,scripts,tests,data,logs,models}
mkdir -p "$TARGET_DIR"/hardware/{audio,camera,imu,leds,microcontroller}
mkdir -p "$TARGET_DIR"/core/{vision,navigation,memory}
mkdir -p "$TARGET_DIR"/ai/{assistant,voice}

# Copy files
echo "📋 Copying files..."
cp main.py "$TARGET_DIR/"
cp requirements.txt "$TARGET_DIR/"
cp README.md "$TARGET_DIR/"
cp MASTER_INSTRUCTIONS.md "$TARGET_DIR/"
cp QUICK_REFERENCE.md "$TARGET_DIR/"
cp TRANSFER_INSTRUCTIONS.md "$TARGET_DIR/"

# Copy directories
cp -r ai "$TARGET_DIR/"
cp -r config "$TARGET_DIR/"
cp -r core "$TARGET_DIR/"
cp -r docs "$TARGET_DIR/"
cp -r hardware "$TARGET_DIR/"
cp -r scripts "$TARGET_DIR/"
cp -r tests "$TARGET_DIR/"

# Set permissions
echo "🔐 Setting permissions..."
chmod +x "$TARGET_DIR"/scripts/*.py
chmod +x "$TARGET_DIR"/scripts/*.sh

echo "✅ Copy completed successfully!"
echo "📁 Files copied to: $TARGET_DIR"
echo ""
echo "Next steps:"
echo "1. Safely eject microSD from Mac"
echo "2. Insert microSD into Jetson"
echo "3. On Jetson, run:"
echo "   cd /home/nvidia/pathlight"
echo "   chmod +x setup_jetson.sh"
echo "   ./setup_jetson.sh"
echo "   sudo systemctl start pathlight"
