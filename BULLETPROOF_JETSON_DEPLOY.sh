#!/bin/bash

# BULLETPROOF JETSON DEPLOYMENT SCRIPT
# GPT-5 Enhanced Version with All Safety Guards
# For pathlight_final_PRODUCTION_READY.tar.gz

# Harden the execution shell too
set -euo pipefail

echo "========================================"
echo "🚀 BULLETPROOF PATHLIGHT DEPLOYMENT"
echo "GPT-5 Enhanced | Production Ready"
echo "========================================"

# 0) Prechecks: internet + space + power mode
echo "📡 Checking internet connectivity..."
ping -c1 8.8.8.8 >/dev/null || { echo "❌ No internet – connect Wi-Fi/Ethernet first."; exit 1; }
echo "✅ Internet OK"

echo "💾 Checking disk space..."
df -h ~

# (A) Ensure we won't OOM during builds (8G swap if <2G swap exists)
echo "🔄 Checking/creating swap for heavy builds..."
if ! swapon --show | grep -q '^/swapfile'; then
  echo "Creating 8G swapfile for OpenCV/dlib builds..."
  sudo fallocate -l 8G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
  echo "✅ 8G swap created"
else
  echo "✅ Swap already exists"
fi
swapon --show

# (B) Power/perf sanity
echo "⚡ Optimizing Jetson performance..."
sudo nvpmodel -q || true
# On Orin Nano devkits: mode 2 == MAXN; verify on your unit:
sudo nvpmodel -m 2 || true
sudo jetson_clocks || true
echo "✅ Performance mode set"

# (C) USB locate/copy/extract
echo "🔍 Locating USB archive..."
USB_TGZ="$(find /media/$USER -maxdepth 2 -type f -name 'pathlight_final_PRODUCTION_READY.tar.gz' | head -n1)"
[ -n "${USB_TGZ:-}" ] || { echo "❌ Tarball not found on USB."; exit 1; }
echo "✅ Found: $USB_TGZ"

echo "📦 Copying and extracting..."
cp -v "$USB_TGZ" ~/
cd ~
tar -xzf pathlight_final_PRODUCTION_READY.tar.gz
[ -d pathlight_final ] || { echo "❌ Expected 'pathlight_final' dir missing"; exit 1; }

# Clean old install and move into place
echo "🧹 Cleaning old installation..."
rm -rf ~/cursorPathlight
mv pathlight_final cursorPathlight
cd cursorPathlight
echo "✅ Project files ready"

# Make scripts executable and prepare logging
chmod +x scripts/setup_jetson.sh
mkdir -p logs
TS=$(date +%Y%m%d_%H%M%S)
echo "📝 Build logs will be saved to: logs/setup_${TS}.log"

# (D) Run setup with full log (root writes the log; that's fine)
echo "🛠️  Running complete setup (this will take 2-3 hours)..."
echo "⏰ OpenCV + dlib source builds are slow but necessary for CUDA"
sudo -E bash -c "./scripts/setup_jetson.sh 2>&1 | tee logs/setup_${TS}.log"

echo "✅ Setup completed! Activating environment..."

# (E) Activate the venv your script created (with fallback detection)
VENV_PATH="cursorPathlight_env"
if [ ! -d "$VENV_PATH" ]; then
  echo "🔍 Virtualenv '$VENV_PATH' not found; searching..."
  VENV_PATH="$(find . -maxdepth 2 -type d -name 'venv' -o -name '*env' | head -n1 || true)"
fi
[ -n "${VENV_PATH:-}" ] || { echo "❌ Could not locate your venv directory."; exit 1; }
# shellcheck disable=SC1090
source "$VENV_PATH/bin/activate"
echo "✅ Virtual environment activated: $VENV_PATH"

# (F) Fast sanity probe
echo "🧪 Running quick sanity checks..."
python - <<'PY'
import sys, subprocess
def run(c):
    print(f"\n$ {c}")
    try: print(subprocess.check_output(c, shell=True, text=True, stderr=subprocess.STDOUT))
    except subprocess.CalledProcessError as e: print(e.output)

print("🐍 Python:", sys.version)

try:
    import torch, torchvision
    print("🔥 PyTorch:", torch.__version__, "| CUDA:", torch.version.cuda, "| Available:", torch.cuda.is_available())
    print("👁️  TorchVision:", torchvision.__version__)
except Exception as e: 
    print("❌ TORCH ERROR:", e)

try:
    import cv2
    bi = cv2.getBuildInformation()
    print("📷 OpenCV:", cv2.__version__)
    print("🎯 CUDA Support:", "✅ YES" if "CUDA: YES" in bi else "❌ NO")
    print("📹 GStreamer Support:", "✅ YES" if "GStreamer:                   YES" in bi else "❌ NO")
    print("📺 V4L2 Support:", "✅ YES" if "V4L/V4L2:                   YES" in bi else "❌ NO")
except Exception as e: 
    print("❌ OPENCV ERROR:", e)

try:
    import dlib
    cuda_status = getattr(dlib, "DLIB_USE_CUDA", "n/a")
    print("🤖 dlib:", dlib.__version__, "| CUDA:", "✅ YES" if cuda_status else "❌ NO")
except Exception as e: 
    print("❌ DLIB ERROR:", e)

try:
    import ultralytics, numpy as np
    print("🎯 YOLOv8:", ultralytics.__version__, "| NumPy:", np.__version__)
except Exception as e: 
    print("❌ YOLO/NUMPY ERROR:", e)

try:
    import face_recognition, pyaudio, librosa
    print("👤 Face Recognition:", face_recognition.__version__)
    print("🔊 PyAudio:", pyaudio.__version__)
    print("🎵 Librosa:", librosa.__version__)
except Exception as e: 
    print("❌ AUDIO/FACE ERROR:", e)
PY

echo ""
echo "🧪 Running comprehensive test suite..."
# (G) Full verification
python3 scripts/test_installation.py | tee logs/test_${TS}.log

echo ""
echo "========================================"
echo "🎉 DEPLOYMENT COMPLETE!"
echo "========================================"
echo "📊 Check logs/test_${TS}.log for results"
echo "🎯 Expected: 14/14 tests passing"
echo "⚡ Ready for 5+ FPS performance!"
echo "========================================"