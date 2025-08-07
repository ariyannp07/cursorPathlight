"""
IMU Manager Module
Handles IMU sensor for headset orientation and movement tracking
"""

import time
import logging
import math
from typing import Dict, Any, Optional, Tuple
import numpy as np

try:
    import smbus2 as smbus
    SMBUS_AVAILABLE = True
except ImportError:
    SMBUS_AVAILABLE = False
    logging.warning("smbus2 not available, IMU will use simulated data")

try:
    import Jetson.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    logging.warning("Jetson.GPIO not available, using simulated GPIO")


class IMUManager:
    """Manages IMU sensor for headset orientation tracking"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize IMU manager"""
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # IMU configuration
        self.i2c_bus = config.get('i2c_bus', 1)
        self.i2c_address = config.get('i2c_address', 0x68)  # MPU6050 default
        self.sample_rate = config.get('sample_rate', 100)  # Hz
        self.calibration_samples = config.get('calibration_samples', 1000)
        
        # Orientation tracking
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        
        # Movement tracking
        self.acceleration = [0.0, 0.0, 0.0]
        self.angular_velocity = [0.0, 0.0, 0.0]
        self.velocity = [0.0, 0.0, 0.0]
        self.position = [0.0, 0.0, 0.0]
        
        # Calibration data
        self.accel_bias = [0.0, 0.0, 0.0]
        self.gyro_bias = [0.0, 0.0, 0.0]
        self.is_calibrated = False
        
        # Filtering
        self.alpha = 0.98  # Complementary filter coefficient
        self.dt = 1.0 / self.sample_rate
        
        # I2C bus
        self.bus = None
        self.running = False
        
        # Initialize IMU
        self._initialize_imu()
        
    def _initialize_imu(self):
        """Initialize IMU hardware"""
        try:
            if SMBUS_AVAILABLE:
                self.bus = smbus.SMBus(self.i2c_bus)
                self._configure_mpu6050()
                self.logger.info("IMU hardware initialized")
            else:
                self.logger.warning("Using simulated IMU data")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize IMU: {e}")
            self.logger.info("Falling back to simulated IMU")
    
    def _configure_mpu6050(self):
        """Configure MPU6050 IMU sensor"""
        try:
            # Wake up MPU6050
            self.bus.write_byte_data(self.i2c_address, 0x6B, 0x00)
            time.sleep(0.1)
            
            # Set sample rate
            sample_rate_div = int(1000 / self.sample_rate) - 1
            self.bus.write_byte_data(self.i2c_address, 0x19, sample_rate_div)
            
            # Set accelerometer range (±2g)
            self.bus.write_byte_data(self.i2c_address, 0x1C, 0x00)
            
            # Set gyroscope range (±250°/s)
            self.bus.write_byte_data(self.i2c_address, 0x1B, 0x00)
            
            # Enable interrupt
            self.bus.write_byte_data(self.i2c_address, 0x38, 0x01)
            
            self.logger.info("MPU6050 configured successfully")
            
        except Exception as e:
            self.logger.error(f"Error configuring MPU6050: {e}")
    
    def _read_accelerometer(self) -> Tuple[float, float, float]:
        """Read accelerometer data"""
        try:
            if self.bus:
                # Read accelerometer data (6 bytes)
                data = self.bus.read_i2c_block_data(self.i2c_address, 0x3B, 6)
                
                # Convert to signed 16-bit values
                accel_x = self._bytes_to_int(data[0], data[1]) / 16384.0  # ±2g range
                accel_y = self._bytes_to_int(data[2], data[3]) / 16384.0
                accel_z = self._bytes_to_int(data[4], data[5]) / 16384.0
                
                return accel_x, accel_y, accel_z
            else:
                # Simulated data
                return self._simulate_accelerometer()
                
        except Exception as e:
            self.logger.error(f"Error reading accelerometer: {e}")
            return 0.0, 0.0, 0.0
    
    def _read_gyroscope(self) -> Tuple[float, float, float]:
        """Read gyroscope data"""
        try:
            if self.bus:
                # Read gyroscope data (6 bytes)
                data = self.bus.read_i2c_block_data(self.i2c_address, 0x43, 6)
                
                # Convert to signed 16-bit values
                gyro_x = self._bytes_to_int(data[0], data[1]) / 131.0  # ±250°/s range
                gyro_y = self._bytes_to_int(data[2], data[3]) / 131.0
                gyro_z = self._bytes_to_int(data[4], data[5]) / 131.0
                
                return gyro_x, gyro_y, gyro_z
            else:
                # Simulated data
                return self._simulate_gyroscope()
                
        except Exception as e:
            self.logger.error(f"Error reading gyroscope: {e}")
            return 0.0, 0.0, 0.0
    
    def _bytes_to_int(self, msb: int, lsb: int) -> int:
        """Convert two bytes to signed 16-bit integer"""
        value = (msb << 8) | lsb
        if value > 32767:
            value -= 65536
        return value
    
    def _simulate_accelerometer(self) -> Tuple[float, float, float]:
        """Simulate accelerometer data for testing"""
        import random
        # Simulate slight head movement
        x = random.gauss(0, 0.1)
        y = random.gauss(0, 0.1)
        z = random.gauss(1.0, 0.05)  # Gravity
        return x, y, z
    
    def _simulate_gyroscope(self) -> Tuple[float, float, float]:
        """Simulate gyroscope data for testing"""
        import random
        # Simulate slight rotation
        x = random.gauss(0, 0.5)
        y = random.gauss(0, 0.5)
        z = random.gauss(0, 0.5)
        return x, y, z
    
    def calibrate(self):
        """Calibrate IMU sensors"""
        self.logger.info("Starting IMU calibration...")
        
        accel_samples = []
        gyro_samples = []
        
        # Collect calibration samples
        for i in range(self.calibration_samples):
            accel_x, accel_y, accel_z = self._read_accelerometer()
            gyro_x, gyro_y, gyro_z = self._read_gyroscope()
            
            accel_samples.append([accel_x, accel_y, accel_z])
            gyro_samples.append([gyro_x, gyro_y, gyro_z])
            
            time.sleep(1.0 / self.sample_rate)
        
        # Calculate biases
        accel_samples = np.array(accel_samples)
        gyro_samples = np.array(gyro_samples)
        
        self.accel_bias = np.mean(accel_samples, axis=0)
        self.gyro_bias = np.mean(gyro_samples, axis=0)
        
        # Adjust for gravity in Z-axis
        self.accel_bias[2] -= 1.0
        
        self.is_calibrated = True
        self.logger.info(f"IMU calibration completed. Accel bias: {self.accel_bias}, Gyro bias: {self.gyro_bias}")
    
    def update_orientation(self):
        """Update orientation using complementary filter"""
        # Read sensor data
        accel_x, accel_y, accel_z = self._read_accelerometer()
        gyro_x, gyro_y, gyro_z = self._read_gyroscope()
        
        # Apply calibration
        if self.is_calibrated:
            accel_x -= self.accel_bias[0]
            accel_y -= self.accel_bias[1]
            accel_z -= self.accel_bias[2]
            
            gyro_x -= self.gyro_bias[0]
            gyro_y -= self.gyro_bias[1]
            gyro_z -= self.gyro_bias[2]
        
        # Store raw data
        self.acceleration = [accel_x, accel_y, accel_z]
        self.angular_velocity = [gyro_x, gyro_y, gyro_z]
        
        # Calculate roll and pitch from accelerometer
        roll_accel = math.atan2(accel_y, math.sqrt(accel_x**2 + accel_z**2))
        pitch_accel = math.atan2(-accel_x, math.sqrt(accel_y**2 + accel_z**2))
        
        # Integrate gyroscope data
        roll_gyro = self.roll + gyro_y * self.dt
        pitch_gyro = self.pitch + gyro_x * self.dt
        yaw_gyro = self.yaw + gyro_z * self.dt
        
        # Complementary filter
        self.roll = self.alpha * roll_gyro + (1 - self.alpha) * roll_accel
        self.pitch = self.alpha * pitch_gyro + (1 - self.alpha) * pitch_accel
        self.yaw = yaw_gyro  # No accelerometer reference for yaw
        
        # Convert to degrees
        self.roll_deg = math.degrees(self.roll)
        self.pitch_deg = math.degrees(self.pitch)
        self.yaw_deg = math.degrees(self.yaw)
    
    def get_orientation(self) -> Dict[str, float]:
        """Get current orientation"""
        return {
            'roll': self.roll_deg,
            'pitch': self.pitch_deg,
            'yaw': self.yaw_deg,
            'roll_rad': self.roll,
            'pitch_rad': self.pitch,
            'yaw_rad': self.yaw
        }
    
    def get_movement(self) -> Dict[str, Any]:
        """Get current movement data"""
        return {
            'acceleration': self.acceleration,
            'angular_velocity': self.angular_velocity,
            'velocity': self.velocity,
            'position': self.position
        }
    
    def detect_head_movement(self) -> Dict[str, Any]:
        """Detect specific head movements"""
        # Calculate movement thresholds
        accel_magnitude = math.sqrt(sum(a**2 for a in self.acceleration))
        gyro_magnitude = math.sqrt(sum(g**2 for g in self.angular_velocity))
        
        # Detect movement types
        movements = {
            'nod': abs(self.pitch_deg) > 10,  # Head nod (pitch)
            'shake': abs(self.yaw_deg) > 10,  # Head shake (yaw)
            'tilt': abs(self.roll_deg) > 10,  # Head tilt (roll)
            'quick_movement': accel_magnitude > 0.5,
            'rotation': gyro_magnitude > 5.0
        }
        
        return movements
    
    def get_direction_vector(self) -> Tuple[float, float, float]:
        """Get direction vector based on head orientation"""
        # Convert orientation to direction vector
        # This assumes the headset is pointing forward when level
        direction_x = math.cos(self.yaw) * math.cos(self.pitch)
        direction_y = math.sin(self.yaw) * math.cos(self.pitch)
        direction_z = math.sin(self.pitch)
        
        return direction_x, direction_y, direction_z
    
    def start(self):
        """Start IMU monitoring"""
        if not self.is_calibrated:
            self.calibrate()
        
        self.running = True
        self.logger.info("IMU monitoring started")
    
    def stop(self):
        """Stop IMU monitoring"""
        self.running = False
        if self.bus:
            self.bus.close()
        self.logger.info("IMU monitoring stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get IMU system status"""
        return {
            'running': self.running,
            'calibrated': self.is_calibrated,
            'sample_rate': self.sample_rate,
            'i2c_address': self.i2c_address,
            'hardware_available': self.bus is not None
        }
