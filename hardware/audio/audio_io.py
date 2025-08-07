"""
Audio I/O Module
Handles microphone input and audio output management
"""

import logging
import pyaudio
import numpy as np
from typing import Dict, Any, Optional


class AudioIO:
    """Audio input/output management system"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the audio I/O system"""
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Configuration parameters
        self.input_device = config.get('input_device', 'default')
        self.output_device = config.get('output_device', 'default')
        self.sample_rate = config.get('sample_rate', 16000)
        self.channels = config.get('channels', 1)
        
        # Audio state
        self.audio = None
        self.input_stream = None
        self.output_stream = None
        
        # Initialize audio system
        self._initialize_audio()
        
    def _initialize_audio(self):
        """Initialize PyAudio system"""
        try:
            self.audio = pyaudio.PyAudio()
            self.logger.info("Audio I/O system initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize audio I/O: {e}")
            self.audio = None
    
    def get_audio_info(self) -> Dict[str, Any]:
        """Get audio system information"""
        if self.audio is None:
            return {'status': 'not_initialized'}
        
        try:
            info = {
                'status': 'initialized',
                'sample_rate': self.sample_rate,
                'channels': self.channels,
                'input_device': self.input_device,
                'output_device': self.output_device
            }
            
            # Get device information
            device_count = self.audio.get_device_count()
            info['device_count'] = device_count
            
            return info
            
        except Exception as e:
            self.logger.error(f"Error getting audio info: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def close(self):
        """Close audio system"""
        if self.audio:
            self.audio.terminate()
            self.logger.info("Audio I/O system closed") 