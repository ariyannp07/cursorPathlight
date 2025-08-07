#!/usr/bin/env python3
"""
Autonomous Startup System
Handles automatic startup when battery is connected
"""

import os
import sys
import time
import logging
import subprocess
import signal
from pathlib import Path
import yaml
import threading

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from hardware.audio.audio_manager import AudioManager
from hardware.imu.imu_manager import IMUManager
from hardware.microcontroller.microcontroller_manager import MicrocontrollerManager


class AutonomousStartup:
    """Handles autonomous startup of Pathlight system"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize autonomous startup system"""
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path
        self.config = self._load_config()
        
        # System components
        self.audio_manager = None
        self.imu_manager = None
        self.microcontroller_manager = None
        
        # Startup state
        self.is_starting = False
        self.is_running = False
        self.startup_complete = False
        
        # Battery monitoring
        self.battery_level = 100
        self.battery_monitoring = True
        
        # Initialize components
        self._initialize_components()
        
    def _load_config(self) -> dict:
        """Load configuration file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.logger.info("Configuration loaded successfully")
            return config
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            return {}
    
    def _initialize_components(self):
        """Initialize hardware components"""
        try:
            # Initialize audio manager
            if 'audio' in self.config:
                self.audio_manager = AudioManager(self.config['audio'])
                self.logger.info("Audio manager initialized")
            
            # Initialize IMU manager
            if 'imu' in self.config:
                self.imu_manager = IMUManager(self.config['imu'])
                self.logger.info("IMU manager initialized")
            
            # Initialize microcontroller manager
            if 'microcontroller' in self.config:
                self.microcontroller_manager = MicrocontrollerManager(self.config['microcontroller'])
                self.logger.info("Microcontroller manager initialized")
                
        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
    
    def check_battery_level(self) -> int:
        """Check current battery level"""
        try:
            if self.microcontroller_manager:
                # Get battery level from microcontroller
                sensor_data = self.microcontroller_manager.get_sensor_data('battery')
                if sensor_data and 'level' in sensor_data:
                    return sensor_data['level']
            
            # Fallback: check system battery
            try:
                # Try to read battery from system
                with open('/sys/class/power_supply/BAT0/capacity', 'r') as f:
                    level = int(f.read().strip())
                    return level
            except:
                pass
            
            # Default to 100% if can't read
            return 100
            
        except Exception as e:
            self.logger.error(f"Error checking battery: {e}")
            return 100
    
    def check_system_health(self) -> bool:
        """Check if system is healthy for startup"""
        try:
            # Check battery level
            battery_level = self.check_battery_level()
            self.battery_level = battery_level
            
            if battery_level < self.config.get('safety', {}).get('critical_battery_threshold', 10):
                self.logger.warning(f"Battery too low: {battery_level}%")
                return False
            
            # Check disk space
            disk_usage = self._check_disk_space()
            if disk_usage > 90:  # More than 90% full
                self.logger.warning(f"Disk space low: {disk_usage}%")
                return False
            
            # Check memory
            memory_usage = self._check_memory_usage()
            if memory_usage > 80:  # More than 80% used
                self.logger.warning(f"Memory usage high: {memory_usage}%")
                return False
            
            # Check temperature
            temperature = self._check_temperature()
            if temperature > 80:  # Above 80°C
                self.logger.warning(f"Temperature high: {temperature}°C")
                return False
            
            self.logger.info("System health check passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in system health check: {e}")
            return False
    
    def _check_disk_space(self) -> float:
        """Check disk space usage"""
        try:
            statvfs = os.statvfs('/')
            total = statvfs.f_blocks * statvfs.f_frsize
            free = statvfs.f_bavail * statvfs.f_frsize
            used_percent = ((total - free) / total) * 100
            return used_percent
        except:
            return 0.0
    
    def _check_memory_usage(self) -> float:
        """Check memory usage"""
        try:
            with open('/proc/meminfo', 'r') as f:
                lines = f.readlines()
                total = int(lines[0].split()[1])
                available = int(lines[2].split()[1])
                used_percent = ((total - available) / total) * 100
                return used_percent
        except:
            return 0.0
    
    def _check_temperature(self) -> float:
        """Check system temperature"""
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp = int(f.read().strip()) / 1000.0  # Convert from millidegrees
                return temp
        except:
            return 0.0
    
    def startup_sequence(self):
        """Execute startup sequence"""
        if self.is_starting:
            self.logger.warning("Startup already in progress")
            return False
        
        self.is_starting = True
        self.logger.info("Starting autonomous startup sequence...")
        
        try:
            # Step 1: System health check
            if not self.check_system_health():
                self.logger.error("System health check failed")
                return False
            
            # Step 2: Initialize audio and play startup sound
            if self.audio_manager:
                self.audio_manager.start()
                self.audio_manager.speak("Pathlight system starting up", priority=True)
                time.sleep(1)
            
            # Step 3: Initialize IMU and calibrate
            if self.imu_manager:
                self.imu_manager.start()
                self.audio_manager.speak("Calibrating sensors", priority=True)
                time.sleep(2)
            
            # Step 4: Initialize microcontroller
            if self.microcontroller_manager:
                self.microcontroller_manager.start_monitoring()
                self.audio_manager.speak("Hardware systems online", priority=True)
                time.sleep(1)
            
            # Step 5: Start main Pathlight system
            self.audio_manager.speak("Starting navigation system", priority=True)
            self._start_main_system()
            
            # Step 6: Startup complete
            self.audio_manager.speak("Pathlight ready for navigation", priority=True)
            self.startup_complete = True
            self.is_running = True
            
            self.logger.info("Autonomous startup completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Startup sequence failed: {e}")
            if self.audio_manager:
                self.audio_manager.speak("Startup failed, please check system", priority=True)
            return False
        finally:
            self.is_starting = False
    
    def _start_main_system(self):
        """Start the main Pathlight system"""
        try:
            # Start main.py in a separate process
            main_script = project_root / "main.py"
            if main_script.exists():
                self.logger.info("Starting main Pathlight system...")
                
                # Start main system process
                self.main_process = subprocess.Popen([
                    sys.executable, str(main_script)
                ], cwd=str(project_root))
                
                # Wait a moment to ensure it starts
                time.sleep(3)
                
                if self.main_process.poll() is None:
                    self.logger.info("Main system started successfully")
                else:
                    self.logger.error("Main system failed to start")
                    
            else:
                self.logger.error("Main script not found")
                
        except Exception as e:
            self.logger.error(f"Error starting main system: {e}")
    
    def shutdown_sequence(self):
        """Execute shutdown sequence"""
        self.logger.info("Starting shutdown sequence...")
        
        try:
            # Stop main system
            if hasattr(self, 'main_process') and self.main_process:
                self.main_process.terminate()
                self.main_process.wait(timeout=10)
            
            # Stop components
            if self.audio_manager:
                self.audio_manager.speak("Shutting down Pathlight", priority=True)
                time.sleep(1)
                self.audio_manager.stop()
            
            if self.imu_manager:
                self.imu_manager.stop()
            
            if self.microcontroller_manager:
                self.microcontroller_manager.stop_monitoring()
            
            self.is_running = False
            self.logger.info("Shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    def monitor_system(self):
        """Monitor system health and battery"""
        while self.battery_monitoring:
            try:
                # Check battery level
                battery_level = self.check_battery_level()
                
                # Check for low battery
                low_threshold = self.config.get('safety', {}).get('low_battery_threshold', 20)
                critical_threshold = self.config.get('safety', {}).get('critical_battery_threshold', 10)
                
                if battery_level <= critical_threshold and self.is_running:
                    self.logger.warning("Critical battery level, initiating shutdown")
                    if self.audio_manager:
                        self.audio_manager.speak("Critical battery level, shutting down", priority=True)
                    self.shutdown_sequence()
                    break
                elif battery_level <= low_threshold and self.is_running:
                    self.logger.warning("Low battery level")
                    if self.audio_manager:
                        self.audio_manager.speak("Low battery warning", priority=True)
                
                # Check system health periodically
                if not self.check_system_health() and self.is_running:
                    self.logger.warning("System health check failed during operation")
                    if self.audio_manager:
                        self.audio_manager.speak("System health issue detected", priority=True)
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in system monitoring: {e}")
                time.sleep(60)  # Wait longer on error
    
    def run(self):
        """Main run loop"""
        try:
            # Start monitoring thread
            monitor_thread = threading.Thread(target=self.monitor_system, daemon=True)
            monitor_thread.start()
            
            # Execute startup sequence
            if self.startup_sequence():
                self.logger.info("Autonomous startup completed, system running")
                
                # Keep running until shutdown
                while self.is_running:
                    time.sleep(1)
                    
            else:
                self.logger.error("Startup failed")
                
        except KeyboardInterrupt:
            self.logger.info("Received shutdown signal")
            self.shutdown_sequence()
        except Exception as e:
            self.logger.error(f"Error in main run loop: {e}")
            self.shutdown_sequence()


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Pathlight Autonomous Startup')
    parser.add_argument('--config', '-c', default='config/config.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('/home/nvidia/pathlight/logs/startup.log'),
            logging.StreamHandler()
        ]
    )
    
    # Create and run autonomous startup
    startup = AutonomousStartup(args.config)
    startup.run()


if __name__ == "__main__":
    main()
