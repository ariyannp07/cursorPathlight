"""
Microcontroller Manager Module
Handles communication with external microcontroller for additional sensors and controls
"""

import time
import logging
import serial
import threading
from typing import Dict, Any, Optional, List
import json

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    logging.warning("RPi.GPIO not available, using simulated GPIO")


class MicrocontrollerManager:
    """Manages communication with external microcontroller"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize microcontroller manager"""
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Communication configuration
        self.serial_port = config.get('serial_port', '/dev/ttyUSB0')
        self.baud_rate = config.get('baud_rate', 115200)
        self.timeout = config.get('timeout', 1.0)
        
        # GPIO pins for direct communication
        self.tx_pin = config.get('tx_pin', 14)
        self.rx_pin = config.get('rx_pin', 15)
        self.enable_pin = config.get('enable_pin', 18)
        
        # Serial connection
        self.serial_conn = None
        self.running = False
        
        # Data buffers
        self.sensor_data = {}
        self.command_queue = []
        self.response_queue = []
        
        # Threading
        self.read_thread = None
        self.write_thread = None
        self.lock = threading.Lock()
        
        # Initialize connection
        self._initialize_connection()
        
    def _initialize_connection(self):
        """Initialize serial connection to microcontroller"""
        try:
            # Try serial connection first
            self.serial_conn = serial.Serial(
                port=self.serial_port,
                baudrate=self.baud_rate,
                timeout=self.timeout,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE
            )
            
            self.logger.info(f"Serial connection established on {self.serial_port}")
            
        except Exception as e:
            self.logger.warning(f"Serial connection failed: {e}")
            self.logger.info("Falling back to GPIO communication")
            self._initialize_gpio()
    
    def _initialize_gpio(self):
        """Initialize GPIO communication"""
        try:
            if GPIO_AVAILABLE:
                GPIO.setmode(GPIO.BCM)
                GPIO.setup(self.tx_pin, GPIO.OUT)
                GPIO.setup(self.rx_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
                GPIO.setup(self.enable_pin, GPIO.OUT)
                
                self.logger.info("GPIO communication initialized")
            else:
                self.logger.warning("GPIO not available, using simulated communication")
                
        except Exception as e:
            self.logger.error(f"GPIO initialization failed: {e}")
    
    def send_command(self, command: str, params: Dict[str, Any] = None) -> bool:
        """Send command to microcontroller"""
        try:
            with self.lock:
                if self.serial_conn and self.serial_conn.is_open:
                    # Format command
                    cmd_data = {
                        'command': command,
                        'params': params or {},
                        'timestamp': time.time()
                    }
                    
                    cmd_json = json.dumps(cmd_data) + '\n'
                    self.serial_conn.write(cmd_json.encode())
                    self.serial_conn.flush()
                    
                    self.logger.debug(f"Sent command: {command}")
                    return True
                    
                elif GPIO_AVAILABLE:
                    # GPIO communication
                    return self._send_gpio_command(command, params)
                else:
                    # Simulated communication
                    self.logger.debug(f"Simulated command: {command}")
                    return True
                    
        except Exception as e:
            self.logger.error(f"Error sending command: {e}")
            return False
    
    def _send_gpio_command(self, command: str, params: Dict[str, Any] = None) -> bool:
        """Send command via GPIO"""
        try:
            # Enable transmission
            GPIO.output(self.enable_pin, GPIO.HIGH)
            time.sleep(0.001)
            
            # Send command (simplified - would need proper protocol)
            # This is a placeholder for actual GPIO communication protocol
            
            # Disable transmission
            GPIO.output(self.enable_pin, GPIO.LOW)
            
            return True
            
        except Exception as e:
            self.logger.error(f"GPIO command failed: {e}")
            return False
    
    def read_response(self) -> Optional[Dict[str, Any]]:
        """Read response from microcontroller"""
        try:
            with self.lock:
                if self.serial_conn and self.serial_conn.is_open:
                    if self.serial_conn.in_waiting > 0:
                        line = self.serial_conn.readline().decode().strip()
                        if line:
                            response = json.loads(line)
                            self.logger.debug(f"Received response: {response}")
                            return response
                            
                elif GPIO_AVAILABLE:
                    # GPIO reading
                    return self._read_gpio_response()
                    
        except Exception as e:
            self.logger.error(f"Error reading response: {e}")
            
        return None
    
    def _read_gpio_response(self) -> Optional[Dict[str, Any]]:
        """Read response via GPIO"""
        try:
            # Check if data is available
            if GPIO.input(self.rx_pin) == GPIO.LOW:
                # Read data (simplified - would need proper protocol)
                # This is a placeholder for actual GPIO reading protocol
                return {'status': 'ok', 'data': 'simulated_gpio_data'}
                
        except Exception as e:
            self.logger.error(f"GPIO read failed: {e}")
            
        return None
    
    def get_sensor_data(self, sensor_type: str) -> Optional[Dict[str, Any]]:
        """Get data from specific sensor"""
        try:
            # Request sensor data
            success = self.send_command('get_sensor', {'type': sensor_type})
            if success:
                # Wait for response
                time.sleep(0.1)
                response = self.read_response()
                if response and response.get('status') == 'ok':
                    return response.get('data', {})
                    
        except Exception as e:
            self.logger.error(f"Error getting sensor data: {e}")
            
        return None
    
    def set_actuator(self, actuator_type: str, value: Any) -> bool:
        """Set actuator value"""
        try:
            return self.send_command('set_actuator', {
                'type': actuator_type,
                'value': value
            })
            
        except Exception as e:
            self.logger.error(f"Error setting actuator: {e}")
            return False
    
    def get_all_sensors(self) -> Dict[str, Any]:
        """Get data from all sensors"""
        sensors = {}
        
        # Common sensor types
        sensor_types = ['temperature', 'humidity', 'pressure', 'light', 'battery']
        
        for sensor_type in sensor_types:
            data = self.get_sensor_data(sensor_type)
            if data:
                sensors[sensor_type] = data
        
        return sensors
    
    def start_monitoring(self):
        """Start continuous monitoring"""
        self.running = True
        
        # Start read thread
        self.read_thread = threading.Thread(target=self._read_loop, daemon=True)
        self.read_thread.start()
        
        # Start write thread
        self.write_thread = threading.Thread(target=self._write_loop, daemon=True)
        self.write_thread.start()
        
        self.logger.info("Microcontroller monitoring started")
    
    def _read_loop(self):
        """Continuous read loop"""
        while self.running:
            try:
                response = self.read_response()
                if response:
                    self.response_queue.append(response)
                    
                    # Update sensor data
                    if 'sensor_data' in response:
                        self.sensor_data.update(response['sensor_data'])
                        
            except Exception as e:
                self.logger.error(f"Read loop error: {e}")
                
            time.sleep(0.01)  # 100Hz read rate
    
    def _write_loop(self):
        """Continuous write loop"""
        while self.running:
            try:
                with self.lock:
                    if self.command_queue:
                        command = self.command_queue.pop(0)
                        self.send_command(command['cmd'], command['params'])
                        
            except Exception as e:
                self.logger.error(f"Write loop error: {e}")
                
            time.sleep(0.01)  # 100Hz write rate
    
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.running = False
        
        if self.read_thread:
            self.read_thread.join(timeout=1.0)
            
        if self.write_thread:
            self.write_thread.join(timeout=1.0)
        
        if self.serial_conn:
            self.serial_conn.close()
            
        if GPIO_AVAILABLE:
            GPIO.cleanup()
            
        self.logger.info("Microcontroller monitoring stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get microcontroller system status"""
        return {
            'running': self.running,
            'serial_connected': self.serial_conn and self.serial_conn.is_open,
            'gpio_available': GPIO_AVAILABLE,
            'sensor_data': self.sensor_data,
            'command_queue_size': len(self.command_queue),
            'response_queue_size': len(self.response_queue)
        }
