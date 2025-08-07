"""
LED Controller Module
Handles LED array control for displaying navigation directions
"""

import logging
import time
import threading
from typing import Dict, Any, List, Optional
from enum import Enum
import smbus2
try:
    import Jetson.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    import logging
    logging.warning("Jetson.GPIO not available, using simulated GPIO")


class LEDPattern(Enum):
    """LED patterns for different directions"""
    FORWARD = "forward"
    LEFT = "left"
    RIGHT = "right"
    BACKWARD = "backward"
    STOP = "stop"
    WARNING = "warning"
    CLEAR = "clear"


class LEDController:
    """LED array controller for navigation display"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the LED controller"""
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Configuration parameters
        self.num_leds = config.get('num_leds', 16)
        self.brightness = config.get('brightness', 0.8)
        self.i2c_address = config.get('i2c_address', 0x70)
        self.i2c_bus = config.get('i2c_bus', 1)
        
        # LED patterns from config
        self.patterns = config.get('pattern', {
            'forward': [0, 1, 2, 3],
            'left': [4, 5, 6, 7],
            'right': [8, 9, 10, 11],
            'backward': [12, 13, 14, 15]
        })
        
        # Initialize hardware interface
        self.bus = None
        self.gpio_initialized = False
        self.current_pattern = None
        self.animation_thread = None
        self.running = False
        
        # Initialize hardware
        self._initialize_hardware()
        
        # Animation patterns
        self.animation_patterns = {
            LEDPattern.WARNING: self._warning_animation,
            LEDPattern.STOP: self._stop_animation,
            LEDPattern.CLEAR: self._clear_animation
        }
        
    def _initialize_hardware(self):
        """Initialize hardware interface"""
        try:
            # Try I2C first (for I2C LED drivers like PCA9685)
            self.bus = smbus2.SMBus(self.i2c_bus)
            self.logger.info(f"Initialized I2C bus {self.i2c_bus}")
            
            # Test I2C communication
            try:
                self.bus.read_byte(self.i2c_address)
                self.logger.info("I2C LED controller detected")
            except:
                self.logger.warning("I2C LED controller not found, falling back to GPIO")
                self.bus = None
                
        except Exception as e:
            self.logger.warning(f"Could not initialize I2C: {e}")
            self.bus = None
        
        # Fallback to GPIO if I2C not available
        if self.bus is None:
            try:
                GPIO.setmode(GPIO.BCM)
                GPIO.setwarnings(False)
                
                # Setup GPIO pins for LEDs (assuming 16 LEDs)
                self.led_pins = list(range(2, 18))  # GPIO pins 2-17
                for pin in self.led_pins:
                    GPIO.setup(pin, GPIO.OUT)
                    GPIO.output(pin, GPIO.LOW)
                
                self.gpio_initialized = True
                self.logger.info("Initialized GPIO LED controller")
                
            except Exception as e:
                self.logger.error(f"Could not initialize GPIO: {e}")
                self.gpio_initialized = False
    
    def start(self):
        """Start the LED controller"""
        self.running = True
        self.logger.info("LED controller started")
    
    def stop(self):
        """Stop the LED controller"""
        self.running = False
        
        # Stop any running animation
        if self.animation_thread and self.animation_thread.is_alive():
            self.animation_thread.join(timeout=1.0)
        
        # Turn off all LEDs
        self.clear_all()
        
        # Cleanup GPIO
        if self.gpio_initialized:
            GPIO.cleanup()
        
        self.logger.info("LED controller stopped")
    
    def display_direction(self, direction: str):
        """
        Display navigation direction using LED pattern
        
        Args:
            direction: Direction string ('forward', 'left', 'right', 'backward', 'stop')
        """
        try:
            # Convert direction to LED pattern
            if direction == 'forward':
                pattern = LEDPattern.FORWARD
            elif direction == 'left':
                pattern = LEDPattern.LEFT
            elif direction == 'right':
                pattern = LEDPattern.RIGHT
            elif direction == 'backward':
                pattern = LEDPattern.BACKWARD
            elif direction == 'stop':
                pattern = LEDPattern.STOP
            else:
                self.logger.warning(f"Unknown direction: {direction}")
                return
            
            # Display the pattern
            self._display_pattern(pattern)
            
        except Exception as e:
            self.logger.error(f"Error displaying direction: {e}")
    
    def _display_pattern(self, pattern: LEDPattern):
        """Display a specific LED pattern"""
        try:
            # Stop any running animation
            if self.animation_thread and self.animation_thread.is_alive():
                self.animation_thread.join(timeout=0.1)
            
            # Check if this is an animated pattern
            if pattern in self.animation_patterns:
                # Start animation in separate thread
                self.animation_thread = threading.Thread(
                    target=self.animation_patterns[pattern],
                    daemon=True
                )
                self.animation_thread.start()
            else:
                # Static pattern
                self._display_static_pattern(pattern)
            
            self.current_pattern = pattern
            
        except Exception as e:
            self.logger.error(f"Error displaying pattern: {e}")
    
    def _display_static_pattern(self, pattern: LEDPattern):
        """Display a static LED pattern"""
        try:
            # Get LED indices for the pattern
            if pattern == LEDPattern.FORWARD:
                led_indices = self.patterns.get('forward', [0, 1, 2, 3])
            elif pattern == LEDPattern.LEFT:
                led_indices = self.patterns.get('left', [4, 5, 6, 7])
            elif pattern == LEDPattern.RIGHT:
                led_indices = self.patterns.get('right', [8, 9, 10, 11])
            elif pattern == LEDPattern.BACKWARD:
                led_indices = self.patterns.get('backward', [12, 13, 14, 15])
            else:
                led_indices = []
            
            # Turn off all LEDs first
            self.clear_all()
            
            # Turn on LEDs for the pattern
            for led_index in led_indices:
                if 0 <= led_index < self.num_leds:
                    self._set_led(led_index, True)
            
            self.logger.debug(f"Displayed static pattern: {pattern.value}")
            
        except Exception as e:
            self.logger.error(f"Error displaying static pattern: {e}")
    
    def _warning_animation(self):
        """Warning animation pattern"""
        try:
            while self.running and self.current_pattern == LEDPattern.WARNING:
                # Blink all LEDs
                for i in range(self.num_leds):
                    self._set_led(i, True)
                time.sleep(0.2)
                
                for i in range(self.num_leds):
                    self._set_led(i, False)
                time.sleep(0.2)
                
        except Exception as e:
            self.logger.error(f"Error in warning animation: {e}")
    
    def _stop_animation(self):
        """Stop animation pattern"""
        try:
            while self.running and self.current_pattern == LEDPattern.STOP:
                # Rapid blinking of all LEDs
                for i in range(self.num_leds):
                    self._set_led(i, True)
                time.sleep(0.1)
                
                for i in range(self.num_leds):
                    self._set_led(i, False)
                time.sleep(0.1)
                
        except Exception as e:
            self.logger.error(f"Error in stop animation: {e}")
    
    def _clear_animation(self):
        """Clear animation pattern"""
        try:
            # Turn off all LEDs
            self.clear_all()
            
        except Exception as e:
            self.logger.error(f"Error in clear animation: {e}")
    
    def _set_led(self, led_index: int, state: bool):
        """
        Set individual LED state
        
        Args:
            led_index: Index of the LED (0 to num_leds-1)
            state: True to turn on, False to turn off
        """
        try:
            if led_index < 0 or led_index >= self.num_leds:
                return
            
            # Use I2C if available
            if self.bus is not None:
                self._set_led_i2c(led_index, state)
            # Fallback to GPIO
            elif self.gpio_initialized:
                self._set_led_gpio(led_index, state)
            else:
                self.logger.warning("No LED hardware interface available")
                
        except Exception as e:
            self.logger.error(f"Error setting LED {led_index}: {e}")
    
    def _set_led_i2c(self, led_index: int, state: bool):
        """Set LED using I2C interface"""
        try:
            # This is a simplified implementation
            # Actual implementation depends on the specific I2C LED driver
            if state:
                # Turn on LED
                self.bus.write_byte_data(self.i2c_address, led_index, int(255 * self.brightness))
            else:
                # Turn off LED
                self.bus.write_byte_data(self.i2c_address, led_index, 0)
                
        except Exception as e:
            self.logger.error(f"Error setting LED via I2C: {e}")
    
    def _set_led_gpio(self, led_index: int, state: bool):
        """Set LED using GPIO interface"""
        try:
            if 0 <= led_index < len(self.led_pins):
                pin = self.led_pins[led_index]
                GPIO.output(pin, GPIO.HIGH if state else GPIO.LOW)
                
        except Exception as e:
            self.logger.error(f"Error setting LED via GPIO: {e}")
    
    def clear_all(self):
        """Turn off all LEDs"""
        try:
            for i in range(self.num_leds):
                self._set_led(i, False)
                
        except Exception as e:
            self.logger.error(f"Error clearing LEDs: {e}")
    
    def set_brightness(self, brightness: float):
        """
        Set LED brightness
        
        Args:
            brightness: Brightness value (0.0 to 1.0)
        """
        self.brightness = max(0.0, min(1.0, brightness))
        self.logger.info(f"Set LED brightness to {self.brightness}")
    
    def test_pattern(self, pattern_name: str = "test"):
        """Test LED pattern for debugging"""
        try:
            self.logger.info(f"Testing LED pattern: {pattern_name}")
            
            # Turn on all LEDs
            for i in range(self.num_leds):
                self._set_led(i, True)
                time.sleep(0.1)
            
            time.sleep(1.0)
            
            # Turn off all LEDs
            for i in range(self.num_leds):
                self._set_led(i, False)
                time.sleep(0.1)
            
            self.logger.info("LED test completed")
            
        except Exception as e:
            self.logger.error(f"Error in LED test: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get LED controller status"""
        return {
            'running': self.running,
            'current_pattern': self.current_pattern.value if self.current_pattern else None,
            'num_leds': self.num_leds,
            'brightness': self.brightness,
            'i2c_available': self.bus is not None,
            'gpio_available': self.gpio_initialized,
            'animation_running': self.animation_thread and self.animation_thread.is_alive()
        } 