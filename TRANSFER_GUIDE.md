# Pathlight Transfer Guide

## Current Situation
✅ **All Pathlight files are ready on your Mac**  
✅ **Project archive created: `/tmp/pathlight_project.tar.gz` (85KB)**  
❌ **microSD is read-only on macOS (normal for Linux filesystems)**

## Transfer Options

### Option 1: Direct Transfer (Recommended)
1. **Safely eject microSD from Mac**
2. **Insert microSD into Jetson**
3. **On Jetson, run these commands:**
   ```bash
   # Create project directory
   sudo mkdir -p /home/nvidia/pathlight
   cd /home/nvidia/pathlight
   
   # Download project from Mac (if connected via network)
   # OR copy from USB drive
   # OR manually transfer files
   ```

### Option 2: Network Transfer
1. **On Mac, start a simple web server:**
   ```bash
   cd /tmp
   python3 -m http.server 8000
   ```
2. **On Jetson, download:**
   ```bash
   wget http://[MAC_IP]:8000/pathlight_project.tar.gz
   tar -xzf pathlight_project.tar.gz
   ```

### Option 3: USB Transfer
1. **Copy archive to USB drive on Mac**
2. **Transfer USB drive to Jetson**
3. **Extract on Jetson**

## What's Included

✅ **Complete Pathlight system with:**
- Dual IMX219 camera support
- IMU (MPU6050) integration
- Microcontroller communication
- Enhanced audio for small speakers
- Autonomous startup system
- 3D stereo vision
- All hardware drivers and managers

✅ **Ready-to-run scripts:**
- `setup_jetson.sh` - Complete Jetson setup
- `autonomous_startup.py` - Automatic startup
- `pathlight.service` - Systemd service
- All test and calibration scripts

✅ **Documentation:**
- `MASTER_INSTRUCTIONS.md` - Complete setup guide
- `QUICK_REFERENCE.md` - Quick reference
- `JETSON_README.md` - Jetson-specific instructions

## Next Steps After Transfer

1. **Extract project on Jetson:**
   ```bash
   tar -xzf pathlight_project.tar.gz
   ```

2. **Run setup script:**
   ```bash
   cd /home/nvidia/pathlight
   chmod +x setup_jetson.sh
   ./setup_jetson.sh
   ```

3. **Start Pathlight:**
   ```bash
   sudo systemctl start pathlight
   ```

4. **Check status:**
   ```bash
   sudo systemctl status pathlight
   sudo journalctl -u pathlight -f
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

### LED Array
- VCC → 3.3V
- GND → GND
- SCL → I2C1_SCL (Pin 28)
- SDA → I2C1_SDA (Pin 27)

### Small Speakers
- Left Speaker → Audio Out L
- Right Speaker → Audio Out R

### Microcontroller
- TX → GPIO14
- RX → GPIO15
- Enable → GPIO18

## Autonomous Operation

Once set up, Pathlight will:
✅ **Start automatically when battery is connected**  
✅ **Monitor battery level and system health**  
✅ **Shutdown safely on low battery**  
✅ **Provide audio feedback for all operations**  
✅ **Handle all hardware automatically**

## Support

If you encounter issues:
1. Check hardware connections
2. Review system logs: `sudo journalctl -u pathlight -f`
3. Run test scripts: `python3 scripts/test_*.py`
4. Check configuration: `config/config.yaml`

The system is designed to be fully autonomous and robust!
