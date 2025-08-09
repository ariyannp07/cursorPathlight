# CPU-Only Setup for Pathlight on Jetson

This document outlines the modifications made to ensure Pathlight runs entirely on CPU without CUDA dependencies.

## Changes Made

### 1. PyTorch Installation
- Updated `scripts/setup_jetson.sh` to install CPU-only PyTorch wheels
- Changed from CUDA 11.8 wheels to CPU-only wheels for better compatibility

### 2. Code Modifications

#### Object Detection (`core/vision/object_detector.py`)
- Force CPU device: `self.device = 'cpu'`
- Removed CUDA device checking
- All YOLOv8 models now run on CPU

#### Configuration Files
- `config/config.example.yaml`: Changed `device: "cuda"` to `device: "cpu"`
- `config/config.example.yaml.backup`: Same change

#### Main Application (`main.py`)
- Added CPU optimization environment variables
- Force CPU-only operation: `os.environ['CUDA_VISIBLE_DEVICES'] = ''`
- Optimized thread usage for Jetson

### 3. Testing Updates
- Updated `scripts/test_installation.py`
- Replaced CUDA tests with CPU operation verification
- Added PyTorch CPU tensor operation tests

### 4. Documentation Updates
- `docs/setup.md`: Updated PyTorch installation instructions
- `docs/development.md`: Updated troubleshooting section
- All references to CUDA installation removed or modified

## Performance Optimizations

### CPU Threading
```python
os.environ['OMP_NUM_THREADS'] = str(max(1, os.cpu_count() // 2))
os.environ['MKL_NUM_THREADS'] = str(max(1, os.cpu_count() // 2))
os.environ['OPENBLAS_NUM_THREADS'] = str(max(1, os.cpu_count() // 2))
```

### Model Selection
- Use lightweight models: `yolov8n.pt` (nano version)
- Face recognition uses HOG method (faster than CNN on CPU)
- Reduced confidence thresholds for better performance

## Verification

Run the installation test to verify CPU-only operation:
```bash
cd ~/cursorPathlight
source cursorPathlight_env/bin/activate
python scripts/test_installation.py
```

Expected output should show:
- ✓ PyTorch CPU Operation: PASS
- ✓ CPU tensor operations working: cpu
- ✓ Number of CPU threads: [number]

## Expected Performance

### CPU Performance vs GPU
- Object detection: ~2-5 FPS (vs 15-30 FPS on GPU)
- Face recognition: ~1-3 FPS (vs 5-10 FPS on GPU)
- Stereo vision: ~5-10 FPS (mostly unaffected)

### Recommendations
1. Process frames at lower resolution (640x480 instead of 1280x720)
2. Reduce detection frequency (every 3rd frame instead of every frame)
3. Use threading for parallel processing of different components
4. Consider model quantization for additional speedup

## Troubleshooting

### If you see CUDA errors:
```bash
# Verify CUDA is disabled
python -c "import os; print('CUDA_VISIBLE_DEVICES:', os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set'))"

# Check PyTorch device
python -c "import torch; print('Available device:', torch.device('cpu'))"
```

### Performance issues:
- Monitor CPU usage: `htop`
- Check temperature: `tegrastats`
- Verify thread settings: `echo $OMP_NUM_THREADS`

## Re-enabling CUDA (if needed)

To re-enable CUDA support in the future:
1. Reinstall PyTorch with CUDA wheels
2. Revert device settings in `object_detector.py`
3. Update configuration files
4. Remove `CUDA_VISIBLE_DEVICES` setting from `main.py`