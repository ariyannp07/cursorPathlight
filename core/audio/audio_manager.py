"""
Audio Manager Module
Handles text-to-speech and audio output
"""

import pyttsx3
import logging
import threading
import time
from typing import Dict, Any, Optional
from queue import Queue


class AudioManager:
    """Text-to-speech and audio management system"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the audio manager"""
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Configuration parameters
        self.tts_engine = config.get('tts_engine', 'espeak')
        self.voice_rate = config.get('voice_rate', 150)
        self.voice_volume = config.get('voice_volume', 0.8)
        
        # Audio state
        self.engine = None
        self.running = False
        self.speech_queue = Queue()
        self.speech_thread = None
        
        # Initialize TTS engine
        self._initialize_tts()
        
    def _initialize_tts(self):
        """Initialize text-to-speech engine"""
        try:
            # Initialize pyttsx3 engine
            self.engine = pyttsx3.init()
            
            # Set voice properties
            self.engine.setProperty('rate', self.voice_rate)
            self.engine.setProperty('volume', self.voice_volume)
            
            # Get available voices
            voices = self.engine.getProperty('voices')
            if voices:
                # Try to set a good default voice
                for voice in voices:
                    if 'en' in voice.id.lower() or 'english' in voice.name.lower():
                        self.engine.setProperty('voice', voice.id)
                        break
                else:
                    # Use first available voice
                    self.engine.setProperty('voice', voices[0].id)
            
            self.logger.info(f"TTS engine initialized: {self.tts_engine}")
            self.logger.info(f"Voice rate: {self.voice_rate}, Volume: {self.voice_volume}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize TTS engine: {e}")
            self.engine = None
            raise
    
    def start(self):
        """Start the audio manager"""
        if self.running:
            self.logger.warning("Audio manager is already running")
            return
        
        if self.engine is None:
            self.logger.error("TTS engine not initialized")
            return
        
        self.running = True
        self.speech_thread = threading.Thread(target=self._speech_loop, daemon=True)
        self.speech_thread.start()
        
        self.logger.info("Audio manager started")
    
    def stop(self):
        """Stop the audio manager"""
        self.running = False
        
        if self.speech_thread and self.speech_thread.is_alive():
            self.speech_thread.join(timeout=2.0)
        
        if self.engine:
            self.engine.stop()
        
        # Clear speech queue
        while not self.speech_queue.empty():
            self.speech_queue.get()
        
        self.logger.info("Audio manager stopped")
    
    def _speech_loop(self):
        """Main speech processing loop"""
        while self.running:
            try:
                # Get next speech request
                try:
                    text = self.speech_queue.get(timeout=1.0)
                except:
                    continue
                
                if text is None:  # Stop signal
                    break
                
                # Speak the text
                self._speak_text(text)
                
            except Exception as e:
                self.logger.error(f"Error in speech loop: {e}")
                time.sleep(0.1)
    
    def _speak_text(self, text: str):
        """Speak text using TTS engine"""
        try:
            if self.engine:
                self.engine.say(text)
                self.engine.runAndWait()
                self.logger.debug(f"Spoke: {text}")
            
        except Exception as e:
            self.logger.error(f"Error speaking text: {e}")
    
    def speak(self, text: str, priority: bool = False):
        """
        Queue text for speech output
        
        Args:
            text: Text to speak
            priority: If True, add to front of queue
        """
        try:
            if not self.running:
                self.logger.warning("Audio manager not running")
                return
            
            if priority and not self.speech_queue.empty():
                # For priority messages, clear queue and add new message
                while not self.speech_queue.empty():
                    try:
                        self.speech_queue.get_nowait()
                    except:
                        break
            
            self.speech_queue.put(text)
            
        except Exception as e:
            self.logger.error(f"Error queuing speech: {e}")
    
    def speak_immediate(self, text: str):
        """
        Speak text immediately (blocking)
        
        Args:
            text: Text to speak
        """
        try:
            if self.engine:
                self._speak_text(text)
            
        except Exception as e:
            self.logger.error(f"Error in immediate speech: {e}")
    
    def set_voice_rate(self, rate: int):
        """
        Set speech rate
        
        Args:
            rate: Speech rate (words per minute)
        """
        try:
            if self.engine:
                self.engine.setProperty('rate', rate)
                self.voice_rate = rate
                self.logger.info(f"Voice rate set to {rate}")
            
        except Exception as e:
            self.logger.error(f"Error setting voice rate: {e}")
    
    def set_voice_volume(self, volume: float):
        """
        Set speech volume
        
        Args:
            volume: Volume level (0.0 to 1.0)
        """
        try:
            volume = max(0.0, min(1.0, volume))
            
            if self.engine:
                self.engine.setProperty('volume', volume)
                self.voice_volume = volume
                self.logger.info(f"Voice volume set to {volume}")
            
        except Exception as e:
            self.logger.error(f"Error setting voice volume: {e}")
    
    def set_voice(self, voice_id: str):
        """
        Set voice
        
        Args:
            voice_id: Voice identifier
        """
        try:
            if self.engine:
                self.engine.setProperty('voice', voice_id)
                self.logger.info(f"Voice set to {voice_id}")
            
        except Exception as e:
            self.logger.error(f"Error setting voice: {e}")
    
    def get_available_voices(self) -> list:
        """Get list of available voices"""
        try:
            if self.engine:
                voices = self.engine.getProperty('voices')
                return [{'id': voice.id, 'name': voice.name} for voice in voices]
            return []
            
        except Exception as e:
            self.logger.error(f"Error getting available voices: {e}")
            return []
    
    def get_current_voice(self) -> Optional[Dict[str, str]]:
        """Get current voice information"""
        try:
            if self.engine:
                voice_id = self.engine.getProperty('voice')
                voices = self.engine.getProperty('voices')
                
                for voice in voices:
                    if voice.id == voice_id:
                        return {'id': voice.id, 'name': voice.name}
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting current voice: {e}")
            return None
    
    def pause(self):
        """Pause speech output"""
        try:
            if self.engine:
                self.engine.stop()
                self.logger.info("Speech paused")
            
        except Exception as e:
            self.logger.error(f"Error pausing speech: {e}")
    
    def resume(self):
        """Resume speech output"""
        try:
            # Speech will resume automatically when new text is queued
            self.logger.info("Speech resumed")
            
        except Exception as e:
            self.logger.error(f"Error resuming speech: {e}")
    
    def clear_queue(self):
        """Clear speech queue"""
        try:
            while not self.speech_queue.empty():
                self.speech_queue.get()
            
            self.logger.info("Speech queue cleared")
            
        except Exception as e:
            self.logger.error(f"Error clearing speech queue: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get audio manager status"""
        return {
            'running': self.running,
            'tts_engine': self.tts_engine,
            'voice_rate': self.voice_rate,
            'voice_volume': self.voice_volume,
            'queue_size': self.speech_queue.qsize(),
            'engine_initialized': self.engine is not None,
            'current_voice': self.get_current_voice()
        }
    
    def test_speech(self, text: str = "Pathlight audio system test"):
        """Test speech functionality"""
        try:
            self.logger.info("Testing speech system...")
            self.speak_immediate(text)
            self.logger.info("Speech test completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Speech test failed: {e}")
            return False 