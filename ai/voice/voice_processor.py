"""
Voice Processor Module
Handles speech recognition and voice input
"""

import logging
import speech_recognition as sr
from typing import Dict, Any, Optional


class VoiceProcessor:
    """Speech recognition and voice processing system"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the voice processor"""
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Configuration parameters
        self.input_device = config.get('input_device', 'default')
        self.sample_rate = config.get('sample_rate', 16000)
        self.channels = config.get('channels', 1)
        
        # Speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = None
        
        # Initialize speech recognition
        self._initialize_speech_recognition()
        
    def _initialize_speech_recognition(self):
        """Initialize speech recognition system"""
        try:
            # Initialize microphone
            self.microphone = sr.Microphone()
            
            # Adjust for ambient noise
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source)
            
            self.logger.info("Voice processor initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize voice processor: {e}")
            self.microphone = None
    
    def listen(self) -> Optional[str]:
        """
        Listen for voice input and convert to text
        
        Returns:
            Recognized text or None if failed
        """
        try:
            if self.microphone is None:
                return None
            
            with self.microphone as source:
                self.logger.debug("Listening for voice input...")
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
            
            # Recognize speech
            text = self.recognizer.recognize_google(audio)
            self.logger.debug(f"Recognized: {text}")
            
            return text
            
        except sr.WaitTimeoutError:
            self.logger.debug("No speech detected within timeout")
            return None
        except sr.UnknownValueError:
            self.logger.debug("Could not understand speech")
            return None
        except sr.RequestError as e:
            self.logger.error(f"Speech recognition service error: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error in speech recognition: {e}")
            return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get voice processor status"""
        return {
            'initialized': self.microphone is not None,
            'input_device': self.input_device,
            'sample_rate': self.sample_rate,
            'channels': self.channels
        } 