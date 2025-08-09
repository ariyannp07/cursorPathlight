#!/usr/bin/env python3
"""
Pathlight Installation Test Script
Tests all components to ensure proper installation with CUDA 12.6 compatibility
"""

import sys
import os
import importlib
import logging
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_cuda_environment():
    """Test CUDA environment and versions"""
    print("=" * 60)
    print("TESTING CUDA ENVIRONMENT")
    print("=" * 60)
    
    try:
        # Test CUDA version
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("CUDA Compiler:")
            for line in result.stdout.split('\n'):
                if 'release' in line:
                    print(f"  {line.strip()}")
        else:
            print("✗ CUDA compiler not found")
            return False
            
        # Test TensorRT packages
        result = subprocess.run(['dpkg', '-l'], capture_output=True, text=True)
        tensorrt_packages = [line for line in result.stdout.split('\n') if 'nvinfer' in line]
        if tensorrt_packages:
            print("TensorRT packages:")
            for pkg in tensorrt_packages[:3]:  # Show first 3
                print(f"  {pkg}")
        else:
            print("⚠ TensorRT packages not found")
            
        return True
        
    except Exception as e:
        print(f"✗ CUDA environment test failed: {e}")
        return False

def test_pytorch_cuda():
    """Test PyTorch CUDA compatibility"""
    print("\nTesting PyTorch CUDA compatibility...")
    
    try:
        import torch
        import platform
        
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"GPU count: {torch.cuda.device_count()}")
            print(f"Current GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("✗ CUDA not available in PyTorch")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ PyTorch CUDA test failed: {e}")
        return False

def test_torchvision_compatibility():
    """Test TorchVision ABI compatibility"""
    print("\nTesting TorchVision compatibility...")
    
    try:
        import torchvision
        import torch
        
        print(f"TorchVision version: {torchvision.__version__}")
        print(f"PyTorch version: {torch.__version__}")
        
        # Test a simple operation
        transform = torchvision.transforms.ToTensor()
        print("✓ TorchVision transforms working")
        
        return True
        
    except Exception as e:
        print(f"✗ TorchVision compatibility test failed: {e}")
        return False

def test_opencv_cuda():
    """Test OpenCV CUDA support"""
    print("\nTesting OpenCV CUDA support...")
    
    try:
        import cv2
        
        print(f"OpenCV version: {cv2.__version__}")
        
        # Check build information
        build_info = cv2.getBuildInformation()
        cuda_enabled = "CUDA: YES" in build_info
        gstreamer_enabled = "GStreamer: YES" in build_info
        
        print(f"CUDA support: {'YES' if cuda_enabled else 'NO'}")
        print(f"GStreamer support: {'YES' if gstreamer_enabled else 'NO'}")
        
        if cuda_enabled:
            cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
            print(f"CUDA devices: {cuda_devices}")
        
        return cuda_enabled
        
    except Exception as e:
        print(f"✗ OpenCV CUDA test failed: {e}")
        return False

def test_dlib_cuda():
    """Test dlib CUDA support"""
    print("\nTesting dlib CUDA support...")
    
    try:
        import dlib
        
        print(f"dlib version: {dlib.__version__}")
        
        try:
            cuda_enabled = dlib.DLIB_USE_CUDA
            print(f"DLIB_USE_CUDA: {cuda_enabled}")
            
            if cuda_enabled:
                print("✓ dlib compiled with CUDA support")
            else:
                print("✗ dlib not compiled with CUDA")
                return False
                
        except AttributeError:
            print("✗ DLIB CUDA flag not accessible")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ dlib CUDA test failed: {e}")
        return False

def test_ultralytics_numpy():
    """Test Ultralytics and NumPy compatibility"""
    print("\nTesting Ultralytics and NumPy compatibility...")
    
    try:
        import numpy
        import ultralytics
        
        print(f"Ultralytics version: {ultralytics.__version__}")
        print(f"NumPy version: {numpy.__version__}")
        
        # Test YOLOv8 import
        from ultralytics import YOLO
        print("✓ YOLOv8 import successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Ultralytics/NumPy compatibility test failed: {e}")
        return False

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing module imports...")
    
    modules = [
        'torch',
        'torchvision',
        'cv2',
        'numpy',
        'ultralytics',
        'face_recognition',
        'pyttsx3',
        'speech_recognition',
        'openai',
        'yaml',
        'sqlite3',
        'pyaudio',
        'smbus2',
        # 'RPi.GPIO'  # Not needed for Jetson
    ]
    
    failed_imports = []
    
    for module in modules:
        try:
            importlib.import_module(module)
            print(f"✓ {module}")
        except ImportError as e:
            print(f"✗ {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\nFailed to import: {', '.join(failed_imports)}")
        return False
    
    print("All modules imported successfully!")
    return True

def test_cuda_pytorch():
    """Test PyTorch with CUDA support"""
    print("\nTesting PyTorch with CUDA support...")
    
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"✓ CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            print(f"✓ CUDA version: {torch.version.cuda}")
            
            # Test CUDA tensor operations
            test_tensor = torch.randn(100, 100).cuda()
            result = torch.matmul(test_tensor, test_tensor.T)
            print(f"✓ CUDA tensor operations working on: {result.device}")
            
            # Test cuDNN
            if torch.backends.cudnn.enabled:
                print(f"✓ cuDNN enabled: version {torch.backends.cudnn.version()}")
            else:
                print("⚠ cuDNN not enabled")
            
            return True
        else:
            print("✗ CUDA not available")
            print("ℹ Falling back to CPU operations...")
            # Test CPU as fallback
            test_tensor = torch.randn(10, 10)
            result = torch.matmul(test_tensor, test_tensor.T)
            print(f"✓ CPU tensor operations working: {result.device}")
            return False
            
    except Exception as e:
        print(f"✗ PyTorch CUDA test failed: {e}")
        return False

def test_project_modules():
    """Test that project modules can be imported"""
    print("\nTesting project modules...")
    
    project_modules = [
        'core.vision.object_detector',
        'core.vision.face_recognizer',
        'core.navigation.path_planner',
        'core.audio.audio_manager',
        'core.memory.memory_manager',
        'hardware.camera.camera_manager',
        'hardware.leds.led_controller',
        'hardware.audio.audio_io',
        'ai.assistant.ai_assistant',
        'ai.voice.voice_processor'
    ]
    
    failed_imports = []
    
    for module in project_modules:
        try:
            importlib.import_module(module)
            print(f"✓ {module}")
        except ImportError as e:
            print(f"✗ {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\nFailed to import project modules: {', '.join(failed_imports)}")
        return False
    
    print("All project modules imported successfully!")
    return True

def test_configuration():
    """Test configuration file"""
    print("\nTesting configuration...")
    
    config_file = project_root / "config" / "config.yaml"
    example_config = project_root / "config" / "config.example.yaml"
    
    if config_file.exists():
        print("✓ Configuration file exists")
        try:
            import yaml
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            print("✓ Configuration file is valid YAML")
            return True
        except Exception as e:
            print(f"✗ Configuration file error: {e}")
            return False
    elif example_config.exists():
        print("⚠ Configuration file not found, but example exists")
        print("  Run: cp config/config.example.yaml config/config.yaml")
        return False
    else:
        print("✗ No configuration files found")
        return False

def test_directories():
    """Test that required directories exist"""
    print("\nTesting directories...")
    
    required_dirs = [
        "data",
        "logs",
        "models",
        "config"
    ]
    
    missing_dirs = []
    
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            print(f"✓ {dir_name}/")
        else:
            print(f"✗ {dir_name}/ (missing)")
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        print(f"\nMissing directories: {', '.join(missing_dirs)}")
        print("Creating missing directories...")
        for dir_name in missing_dirs:
            (project_root / dir_name).mkdir(exist_ok=True)
            print(f"  Created {dir_name}/")
    
    return True

def test_yolo_model():
    """Test YOLOv8 model download"""
    print("\nTesting YOLOv8 model...")
    
    try:
        from ultralytics import YOLO
        
        # Try to load the model (will download if not present)
        model = YOLO('yolov8n.pt')
        print("✓ YOLOv8 model loaded successfully")
        return True
    except Exception as e:
        print(f"✗ YOLOv8 model test failed: {e}")
        return False

def test_audio_system():
    """Test audio system"""
    print("\nTesting audio system...")
    
    try:
        import pyttsx3
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        print(f"✓ Audio system initialized with {len(voices)} voices")
        return True
    except Exception as e:
        print(f"✗ Audio system test failed: {e}")
        return False

def test_imx219_camera():
    """Test IMX219 CSI camera"""
    print("\nTesting IMX219 CSI camera...")
    
    try:
        import cv2
        
        # Check for available cameras
        available_cameras = []
        for i in range(4):
            try:
                cap = cv2.VideoCapture(f"/dev/video{i}", cv2.CAP_V4L2)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        available_cameras.append(f"/dev/video{i}")
                    cap.release()
            except:
                continue
        
        if available_cameras:
            print(f"✓ Found {len(available_cameras)} camera(s): {', '.join(available_cameras)}")
            
            # Test the first available camera
            cap = cv2.VideoCapture(available_cameras[0], cv2.CAP_V4L2)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    print(f"✓ Successfully captured frame: {frame.shape}")
                    cap.release()
                    return True
                else:
                    print("✗ Could not capture frame")
                    cap.release()
                    return False
            else:
                print("✗ Could not open camera")
                return False
        else:
            print("⚠ No cameras detected")
            print("  This is normal if IMX219 camera is not connected")
            return True  # Don't fail the test if no camera is connected
            
    except Exception as e:
        print(f"✗ IMX219 camera test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("Pathlight Installation Test")
    print("=" * 50)
    
    tests = [
        ("CUDA Environment", test_cuda_environment),
        ("PyTorch CUDA Support", test_pytorch_cuda),
        ("TorchVision Compatibility", test_torchvision_compatibility),
        ("OpenCV CUDA Support", test_opencv_cuda),
        ("dlib CUDA Support", test_dlib_cuda),
        ("Ultralytics/NumPy Compatibility", test_ultralytics_numpy),
        ("Module Imports", test_imports),
        ("Project Modules", test_project_modules),
        ("Configuration", test_configuration),
        ("Directories", test_directories),
        ("YOLOv8 Model", test_yolo_model),
        ("Audio System", test_audio_system),
        ("IMX219 Camera", test_imx219_camera)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Pathlight is ready to use.")
        print("\nNext steps:")
        print("1. Configure your API keys in config/config.yaml")
        print("2. Connect your camera and LED array")
        print("3. Run: python main.py")
    else:
        print(f"\n⚠ {total - passed} test(s) failed. Please check the errors above.")
        print("\nTroubleshooting:")
        print("1. Run the setup script: ./scripts/setup_jetson.sh")
        print("2. Check the documentation: docs/setup.md")
        print("3. Verify your hardware connections")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 