# 🎯 PATHLIGHT FINAL SPECIFICATIONS

**Complete software stack specifications for CUDA-accelerated Pathlight on Jetson Orin Nano**

---

## 📊 **VERIFIED COMPATIBLE SOFTWARE STACK**

### **Core AI Framework**
- **PyTorch**: `2.4.0a0+07cecf4168` (NVIDIA optimized for Jetson)
- **TorchVision**: `0.19.0` (Source built for compatibility)
- **TorchAudio**: `2.4.0` (Matches PyTorch version)

### **Computer Vision & Detection**
- **OpenCV**: `4.10.0.84` (CUDA enabled)
- **OpenCV-Contrib**: `4.10.0.84` (CUDA enabled)
- **YOLOv8 (ultralytics)**: `8.3.0` (GPU optimized)

### **Face Recognition**
- **face_recognition**: `1.3.0` (CNN model support)
- **dlib**: `19.24.6` (CUDA enabled)

### **Machine Learning Libraries**
- **NumPy**: `1.26.4`
- **SciPy**: `1.13.1`
- **scikit-learn**: `1.5.1`
- **matplotlib**: `3.9.2`
- **seaborn**: `0.13.2`

### **Hardware & System**
- **CUDA**: `12.6` (JetPack 6.2.1 native)
- **cuDNN**: `9.3.0` (JetPack 6.2.1 native)
- **TensorRT**: `10.3.0` (JetPack 6.2.1 native)
- **Python**: `3.10.12` (Ubuntu 22.04 default)

---

## 🔧 **PATHLIGHT COMPONENT COMPATIBILITY**

### **Object Detection (YOLOv8)**
```yaml
Configuration:
  model: "yolov8n.pt"
  confidence: 0.5
  device: "cuda"
  classes: [0, 1, 2, 3, 5, 7]  # person, bicycle, car, motorcycle, bus, truck

Performance:
  - GPU acceleration: ✅ Enabled
  - Expected FPS: 15-25 FPS (yolov8n)
  - CUDA memory usage: ~500MB
  - Compatibility: ✅ PyTorch 2.4.0a0 + ultralytics 8.3.0
```

### **Face Recognition**
```yaml
Configuration:
  model: "cnn"  # GPU accelerated CNN model
  tolerance: 0.6
  upsample: 1   # Higher quality on GPU
  
Performance:
  - GPU acceleration: ✅ Enabled
  - Expected FPS: 10-15 FPS
  - CUDA memory usage: ~300MB
  - Compatibility: ✅ face_recognition 1.3.0 + dlib 19.24.6
```

### **Stereo Vision**
```yaml
Configuration:
  calibration_file: "data/stereo_calibration.pkl"
  camera_separation: 0.12  # meters
  min_depth: 0.1
  max_depth: 10.0

Performance:
  - OpenCV CUDA: ✅ Enabled
  - Expected FPS: 20-30 FPS
  - Compatibility: ✅ OpenCV 4.10.0.84 with CUDA
```

---

## 🚀 **PERFORMANCE SPECIFICATIONS**

### **Target Performance Metrics**
- **Overall System FPS**: 5+ FPS (combined pipeline)
- **Individual Components**:
  - Object Detection: 15-25 FPS
  - Face Recognition: 10-15 FPS  
  - Stereo Vision: 20-30 FPS
- **GPU Utilization**: 40-70% during operation
- **Memory Usage**: 
  - System RAM: ~2GB
  - GPU Memory: ~1GB
- **Power Consumption**: 10-15W (25W mode)

### **Quality Metrics**
- **Object Detection Accuracy**: >90% for common objects
- **Face Recognition Accuracy**: >95% for known faces
- **Depth Estimation Accuracy**: ±10cm at 2m distance
- **Real-time Response**: <200ms latency

---

## 🔬 **TESTING & VALIDATION**

### **Installation Tests (8/8 must pass)**
1. ✅ **System Info Test**: Hardware and OS verification
2. ✅ **CUDA Availability Test**: CUDA 12.6 detection
3. ✅ **PyTorch CUDA Test**: GPU tensor operations
4. ✅ **TorchVision Test**: Computer vision operations
5. ✅ **YOLOv8 Test**: Object detection model loading
6. ✅ **Face Recognition Test**: Face detection functionality
7. ✅ **OpenCV CUDA Test**: OpenCV GPU operations
8. ✅ **Module Import Test**: All Pathlight modules

### **Performance Tests**
```bash
# GPU Memory Test
nvidia-smi  # Should show GPU utilization

# FPS Test  
python3 scripts/test_performance.py

# CUDA Operations Test
python3 -c "
import torch
x = torch.randn(1000, 1000).cuda()
y = torch.mm(x, x)
print('✅ CUDA operations working')
"
```

---

## 🛠 **CONFIGURATION FILES**

### **Key Configuration Settings**
```yaml
# config/config.example.yaml
yolo:
  device: "cuda"           # Enable GPU acceleration
  model: "yolov8n.pt"      # Optimized for real-time

face_recognition:
  model: "cnn"             # GPU accelerated model
  upsample: 1              # Quality setting for GPU

camera:
  fps: 30                  # Input frame rate
  resolution: [640, 480]   # Optimized resolution
```

### **Environment Configuration**
```bash
# main.py environment variables
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4  
export OPENBLAS_NUM_THREADS=4
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1
```

---

## 🔍 **TROUBLESHOOTING GUIDE**

### **Common Issues & Solutions**

#### **"operator torchvision::nms does not exist"**
- **Cause**: PyTorch/TorchVision version mismatch
- **Solution**: Use PyTorch 2.4.0a0 + TorchVision 0.19.0 (source built)

#### **"CUDA out of memory"**  
- **Cause**: GPU memory exhaustion
- **Solution**: Reduce batch size, lower resolution, or use smaller models

#### **Low FPS Performance**
- **Cause**: CPU fallback or inefficient processing
- **Solution**: Verify GPU utilization with `nvidia-smi`

#### **Import errors**
- **Cause**: Missing dependencies or virtual environment issues
- **Solution**: Activate environment: `source cursorPathlight_env/bin/activate`

---

## 📈 **OPTIMIZATION RECOMMENDATIONS**

### **For Maximum Performance**
1. **Use 25W power mode**: `sudo nvpmodel -m 0`
2. **Max CPU frequency**: `sudo jetson_clocks`
3. **Optimize models**: Use TensorRT conversion for production
4. **Memory management**: Clear GPU cache between operations
5. **Frame skipping**: Process every 2nd or 3rd frame for higher throughput

### **For Power Efficiency**
1. **Use 15W power mode**: `sudo nvpmodel -m 2`
2. **Reduce resolution**: 320x240 for object detection
3. **Lower frame rates**: 15 FPS instead of 30 FPS
4. **Model optimization**: Use YOLOv8n instead of larger models

---

## ✅ **VALIDATION CHECKLIST**

Before deployment, verify:

- [ ] All 8 installation tests pass
- [ ] GPU utilization shows during inference
- [ ] System achieves 5+ FPS combined performance
- [ ] No CUDA/PyTorch errors in logs
- [ ] Face recognition uses CNN model
- [ ] YOLOv8 uses CUDA device
- [ ] OpenCV reports CUDA devices > 0
- [ ] Virtual environment activated
- [ ] All Pathlight modules import successfully
- [ ] Real-time navigation responsive

---

## 🎯 **FINAL NOTES**

This software stack represents the **optimal balance** between:
- **Stability**: Using proven, tested versions
- **Performance**: Full CUDA acceleration
- **Compatibility**: No version conflicts
- **Maintainability**: Well-documented and reproducible

**Expected Result**: 5+ FPS real-time performance with full CUDA acceleration on Jetson Orin Nano. 🚀