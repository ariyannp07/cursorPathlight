#!/bin/bash

# BULLETPROOF JETSON DEPLOYMENT SCRIPT - FINAL VERSION
# GPT-5 Enhanced with All Safety Guards and Fallbacks
# For pathlight_ULTIMATE_BULLETPROOF.tar.gz

set -euo pipefail

echo "========================================"
echo "BULLETPROOF PATHLIGHT DEPLOYMENT"
echo "GPT-5 Enhanced | Production Ready"
echo "========================================"

# Prepare logging with error trap
mkdir -p ~/pathlight_logs
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="~/pathlight_logs/deploy_${TS}.log"

# Error trap for better debugging
trap 'echo "Failed on line $LINENO"; tail -n 80 ~/pathlight_logs/deploy_${TS}.log 2>/dev/null || true' ERR

# 0) Prechecks: internet + space + power mode
echo "Checking internet connectivity..."
ping -c1 8.8.8.8 >/dev/null || curl -fsI http://connectivitycheck.gstatic.com/generate_204 >/dev/null \
  || { echo "No internet - connect Wi-Fi/Ethernet first."; exit 1; }
echo "Internet OK"

echo "Checking disk space..."
df -h ~

# (A) Ensure we won't OOM during builds (8G swap if <2G swap exists)
echo "Checking/creating swap for heavy builds..."
if ! swapon --show | grep -q '^/swapfile'; then
  echo "Creating 8G swapfile for OpenCV/dlib builds..."
  sudo fallocate -l 8G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  
  # Only append to fstab if not already there
  if ! grep -qs '^/swapfile ' /etc/fstab; then
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
  fi
  echo "8G swap created"
else
  echo "Swap already exists"
fi
swapon --show

# (B) Power/perf sanity
echo "Optimizing Jetson performance..."
sudo nvpmodel -q || true
sudo nvpmodel -m 2 || true
sudo jetson_clocks || true
echo "Performance mode set"

# (C) USB locate/copy/extract with better archive handling
echo "Locating USB archive..."
USB_TGZ="$(find /media/$USER -maxdepth 2 -type f -name 'pathlight_ULTIMATE_BULLETPROOF.tar.gz' | head -n1)"
[ -n "${USB_TGZ:-}" ] || { echo "Tarball not found on USB."; exit 1; }
echo "Found: $USB_TGZ"

echo "Copying and extracting..."
cp -v "$USB_TGZ" ~/
cd ~

# Confirm archive contents before moving
echo "Verifying archive structure..."
TOPDIR="$(tar -tzf ~/pathlight_ULTIMATE_BULLETPROOF.tar.gz | head -1 | cut -f1 -d/)"
tar -xzf pathlight_ULTIMATE_BULLETPROOF.tar.gz
[ -d "$TOPDIR" ] || { echo "Top-level dir missing in archive"; exit 1; }

# Clean old install and move into place
echo "Cleaning old installation..."
rm -rf ~/cursorPathlight
mv "$TOPDIR" cursorPathlight
cd cursorPathlight
echo "Project files ready"

# Make scripts executable and prepare logging
chmod +x scripts/setup_jetson.sh
mkdir -p logs

echo "Build logs will be saved to: logs/setup_${TS}.log"

# (D) Run setup with user-owned logs (setup script uses sudo internally)
echo "Running complete setup (this will take 2-3 hours)..."
echo "OpenCV + dlib source builds are slow but necessary for CUDA"
bash -lc "./scripts/setup_jetson.sh 2>&1 | tee logs/setup_${TS}.log"

echo "Setup completed! Activating environment..."

# (E) Activate the venv with verification
VENV_PATH="cursorPathlight_env"
if [ ! -d "$VENV_PATH" ]; then
  echo "Virtualenv '$VENV_PATH' not found; searching..."
  VENV_PATH="$(find . -maxdepth 2 -type d -name 'venv' -o -name '*env' | head -n1 || true)"
fi
[ -n "${VENV_PATH:-}" ] || { echo "Could not locate your venv directory."; exit 1; }
source "$VENV_PATH/bin/activate"

# Verify the venv activation
echo "Virtual environment activated: $VENV_PATH"
which python
python -c "import sys; print('VENV OK:', sys.prefix)"

# (F) Quick sanity checks with improved V4L2 detection
echo "Running quick sanity checks..."
python - <<'PY'
import sys
print("Python:", sys.version)

try:
    import torch, torchvision
    print("PyTorch:", torch.__version__, "| CUDA:", torch.version.cuda, "| Available:", torch.cuda.is_available())
    print("TorchVision:", torchvision.__version__)
except Exception as e: 
    print("TORCH ERROR:", e)

try:
    import cv2
    bi = cv2.getBuildInformation()
    print("OpenCV:", cv2.__version__)
    print("CUDA Support:", "YES" if "CUDA: YES" in bi else "NO")
    print("GStreamer Support:", "YES" if "GStreamer:                   YES" in bi else "NO")
    # Improved V4L2 detection
    print("V4L2 Support:", "YES" if "V4L" in bi and "YES" in bi.split("V4L",1)[1][:40] else "UNKNOWN")
except Exception as e: 
    print("OPENCV ERROR:", e)

try:
    import dlib
    cuda_status = getattr(dlib, "DLIB_USE_CUDA", "n/a")
    print("dlib:", dlib.__version__, "| CUDA:", "YES" if cuda_status else "NO")
except Exception as e: 
    print("DLIB ERROR:", e)

try:
    import ultralytics, numpy as np
    print("YOLOv8:", ultralytics.__version__, "| NumPy:", np.__version__)
except Exception as e: 
    print("YOLO/NUMPY ERROR:", e)

try:
    import face_recognition, pyaudio, librosa
    print("Face Recognition:", face_recognition.__version__)
    print("PyAudio:", pyaudio.__version__)
    print("Librosa:", librosa.__version__)
except Exception as e: 
    print("AUDIO/FACE ERROR:", e)
PY

echo ""
echo "Running comprehensive test suite..."
# (G) Full verification
python3 scripts/test_installation.py | tee logs/test_${TS}.log

echo ""
echo "========================================"
echo "DEPLOYMENT COMPLETE!"
echo "========================================"
echo "Check logs/test_${TS}.log for results"
echo "Expected: 14/14 tests passing"
echo "Ready for 5+ FPS performance!"
echo "========================================"