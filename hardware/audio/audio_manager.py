"""
Enhanced Audio Manager Module
Handles small speakers and advanced audio features for the headset
"""

import time
import logging
import threading
import queue
from typing import Dict, Any, Optional, List
import numpy as np

try:
    import pyaudio
    import wave
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    logging.warning("pyaudio not available, audio will be simulated")

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    logging.warning("pyttsx3 not available, TTS will be simulated")

try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False
    logging.warning("speech_recognition not available, speech input will be simulated")


class AudioManager:
    """Manages audio input/output for the headset"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize audio manager"""
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Audio configuration
        self.sample_rate = config.get('sample_rate', 44100)
        self.channels = config.get('channels', 2)  # Stereo for dual speakers
        self.chunk_size = config.get('chunk_size', 1024)
        self.format = pyaudio.paFloat32 if PYAUDIO_AVAILABLE else None
        
        # Speaker configuration
        self.left_speaker_volume = config.get('left_speaker_volume', 0.8)
        self.right_speaker_volume = config.get('right_speaker_volume', 0.8)
        self.balance = config.get('balance', 0.0)  # -1.0 (left) to 1.0 (right)
        
        # Audio processing
        self.noise_reduction = config.get('noise_reduction', True)
        self.echo_cancellation = config.get('echo_cancellation', True)
        self.volume_control = config.get('volume_control', True)
        
        # TTS configuration
        self.tts_voice = config.get('tts_voice', 'en-us')
        self.tts_rate = config.get('tts_rate', 150)
        self.tts_volume = config.get('tts_volume', 0.8)
        
        # Speech recognition
        self.speech_timeout = config.get('speech_timeout', 5.0)
        self.phrase_time_limit = config.get('phrase_time_limit', 10.0)
        
        # Audio streams
        self.audio = None
        self.output_stream = None
        self.input_stream = None
        
        # TTS engine
        self.tts_engine = None
        
        # Speech recognizer
        self.speech_recognizer = None
        self.microphone = None
        
        # Audio queues
        self.audio_queue = queue.Queue()
        self.tts_queue = queue.Queue()
        
        # Threading
        self.audio_thread = None
        self.tts_thread = None
        self.speech_thread = None
        self.running = False
        
        # Initialize audio system
        self._initialize_audio()
        
    def _initialize_audio(self):
        """Initialize audio hardware and software"""
        try:
            if PYAUDIO_AVAILABLE:
                self.audio = pyaudio.PyAudio()
                self._setup_audio_streams()
                
            if TTS_AVAILABLE:
                self._setup_tts()
                
            if SPEECH_RECOGNITION_AVAILABLE:
                self._setup_speech_recognition()
                
            self.logger.info("Audio system initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize audio: {e}")
    
    def _setup_audio_streams(self):
        """Setup audio input/output streams"""
        try:
            # Output stream for speakers
            self.output_stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                output=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            
            # Input stream for microphone
            self.input_stream = self.audio.open(
                format=self.format,
                channels=1,  # Mono input
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            self.logger.info("Audio streams configured")
            
        except Exception as e:
            self.logger.error(f"Error setting up audio streams: {e}")
    
    def _setup_tts(self):
        """Setup text-to-speech engine"""
        try:
            self.tts_engine = pyttsx3.init()
            
            # Configure TTS
            voices = self.tts_engine.getProperty('voices')
            for voice in voices:
                if self.tts_voice in voice.id.lower():
                    self.tts_engine.setProperty('voice', voice.id)
                    break
            
            self.tts_engine.setProperty('rate', self.tts_rate)
            self.tts_engine.setProperty('volume', self.tts_volume)
            
            self.logger.info("TTS engine configured")
            
        except Exception as e:
            self.logger.error(f"Error setting up TTS: {e}")
    
    def _setup_speech_recognition(self):
        """Setup speech recognition"""
        try:
            self.speech_recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            
            # Adjust for ambient noise
            with self.microphone as source:
                self.speech_recognizer.adjust_for_ambient_noise(source, duration=1)
            
            self.logger.info("Speech recognition configured")
            
        except Exception as e:
            self.logger.error(f"Error setting up speech recognition: {e}")
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Audio output callback for real-time processing"""
        try:
            if not self.audio_queue.empty():
                audio_data = self.audio_queue.get_nowait()
                
                # Apply stereo balance
                if self.channels == 2 and len(audio_data.shape) == 1:
                    # Convert mono to stereo
                    left_channel = audio_data * (1.0 - max(0, self.balance))
                    right_channel = audio_data * (1.0 + min(0, self.balance))
                    audio_data = np.column_stack((left_channel, right_channel))
                
                # Apply volume control
                if self.volume_control:
                    audio_data *= self.left_speaker_volume
                
                return (audio_data.tobytes(), pyaudio.paContinue)
            else:
                # Return silence
                silence = np.zeros((frame_count, self.channels), dtype=np.float32)
                return (silence.tobytes(), pyaudio.paContinue)
                
        except Exception as e:
            self.logger.error(f"Audio callback error: {e}")
            silence = np.zeros((frame_count, self.channels), dtype=np.float32)
            return (silence.tobytes(), pyaudio.paContinue)
    
    def play_audio(self, audio_data: np.ndarray, blocking: bool = False):
        """Play audio through speakers"""
        try:
            if PYAUDIO_AVAILABLE and self.output_stream:
                # Convert to float32 if needed
                if audio_data.dtype != np.float32:
                    audio_data = audio_data.astype(np.float32)
                
                # Normalize audio
                if np.max(np.abs(audio_data)) > 0:
                    audio_data = audio_data / np.max(np.abs(audio_data)) * 0.8
                
                # Add to queue
                self.audio_queue.put(audio_data)
                
                if blocking:
                    # Wait for audio to finish
                    duration = len(audio_data) / self.sample_rate
                    time.sleep(duration)
                    
                self.logger.debug("Audio queued for playback")
                
        except Exception as e:
            self.logger.error(f"Error playing audio: {e}")
    
    def play_tone(self, frequency: float, duration: float, volume: float = 0.5):
        """Play a tone at specified frequency"""
        try:
            # Generate tone
            t = np.linspace(0, duration, int(self.sample_rate * duration), False)
            tone = np.sin(2 * np.pi * frequency * t) * volume
            
            # Apply stereo balance
            if self.channels == 2:
                left_tone = tone * (1.0 - max(0, self.balance))
                right_tone = tone * (1.0 + min(0, self.balance))
                tone = np.column_stack((left_tone, right_tone))
            
            self.play_audio(tone, blocking=True)
            
        except Exception as e:
            self.logger.error(f"Error playing tone: {e}")
    
    def speak(self, text: str, priority: bool = False):
        """Convert text to speech and play through speakers"""
        try:
            if TTS_AVAILABLE and self.tts_engine:
                # Add to TTS queue
                self.tts_queue.put({
                    'text': text,
                    'priority': priority,
                    'timestamp': time.time()
                })
                
                if priority:
                    # Clear queue for priority messages
                    while not self.tts_queue.empty():
                        try:
                            self.tts_queue.get_nowait()
                        except queue.Empty:
                            break
                    self.tts_queue.put({
                        'text': text,
                        'priority': True,
                        'timestamp': time.time()
                    })
                    
            else:
                # Simulated TTS
                self.logger.info(f"TTS (simulated): {text}")
                
        except Exception as e:
            self.logger.error(f"Error in TTS: {e}")
    
    def listen_for_speech(self, timeout: float = None) -> Optional[str]:
        """Listen for speech input"""
        try:
            if SPEECH_RECOGNITION_AVAILABLE and self.microphone:
                with self.microphone as source:
                    self.logger.debug("Listening for speech...")
                    audio = self.speech_recognizer.listen(
                        source,
                        timeout=timeout or self.speech_timeout,
                        phrase_time_limit=self.phrase_time_limit
                    )
                
                # Recognize speech
                text = self.speech_recognizer.recognize_google(audio)
                self.logger.info(f"Speech recognized: {text}")
                return text
                
            else:
                # Simulated speech input
                self.logger.info("Speech recognition not available (simulated)")
                return None
                
        except sr.WaitTimeoutError:
            self.logger.debug("Speech timeout")
            return None
        except sr.UnknownValueError:
            self.logger.debug("Speech not understood")
            return None
        except Exception as e:
            self.logger.error(f"Speech recognition error: {e}")
            return None
    
    def record_audio(self, duration: float) -> Optional[np.ndarray]:
        """Record audio for specified duration"""
        try:
            if PYAUDIO_AVAILABLE and self.input_stream:
                frames = []
                num_frames = int(self.sample_rate * duration / self.chunk_size)
                
                for _ in range(num_frames):
                    data = self.input_stream.read(self.chunk_size)
                    frames.append(data)
                
                # Convert to numpy array
                audio_data = np.frombuffer(b''.join(frames), dtype=np.float32)
                
                # Apply noise reduction if enabled
                if self.noise_reduction:
                    audio_data = self._reduce_noise(audio_data)
                
                return audio_data
                
        except Exception as e:
            self.logger.error(f"Error recording audio: {e}")
            return None
    
    def _reduce_noise(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply simple noise reduction"""
        try:
            # Simple high-pass filter to reduce low-frequency noise
            from scipy import signal
            
            # Design high-pass filter
            nyquist = self.sample_rate / 2
            cutoff = 100  # Hz
            b, a = signal.butter(4, cutoff / nyquist, btype='high')
            
            # Apply filter
            filtered_audio = signal.filtfilt(b, a, audio_data)
            
            return filtered_audio
            
        except ImportError:
            # Fallback without scipy
            return audio_data
        except Exception as e:
            self.logger.error(f"Error in noise reduction: {e}")
            return audio_data
    
    def set_volume(self, left_volume: float = None, right_volume: float = None):
        """Set speaker volumes"""
        if left_volume is not None:
            self.left_speaker_volume = max(0.0, min(1.0, left_volume))
        if right_volume is not None:
            self.right_speaker_volume = max(0.0, min(1.0, right_volume))
        
        self.logger.info(f"Volume set: L={self.left_speaker_volume:.2f}, R={self.right_speaker_volume:.2f}")
    
    def set_balance(self, balance: float):
        """Set stereo balance (-1.0 = left, 0.0 = center, 1.0 = right)"""
        self.balance = max(-1.0, min(1.0, balance))
        self.logger.info(f"Balance set: {self.balance:.2f}")
    
    def _tts_worker(self):
        """TTS worker thread"""
        while self.running:
            try:
                if not self.tts_queue.empty():
                    tts_item = self.tts_queue.get(timeout=0.1)
                    
                    if TTS_AVAILABLE and self.tts_engine:
                        # Generate speech
                        self.tts_engine.say(tts_item['text'])
                        self.tts_engine.runAndWait()
                    else:
                        # Simulated TTS
                        self.logger.info(f"TTS: {tts_item['text']}")
                        time.sleep(len(tts_item['text']) * 0.05)  # Simulate speaking time
                        
                else:
                    time.sleep(0.01)
                    
            except Exception as e:
                self.logger.error(f"TTS worker error: {e}")
                time.sleep(0.1)
    
    def start(self):
        """Start audio system"""
        self.running = True
        
        # Start TTS thread
        self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self.tts_thread.start()
        
        self.logger.info("Audio system started")
    
    def stop(self):
        """Stop audio system"""
        self.running = False
        
        if self.tts_thread:
            self.tts_thread.join(timeout=1.0)
        
        if self.output_stream:
            self.output_stream.stop_stream()
            self.output_stream.close()
            
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()
            
        if self.audio:
            self.audio.terminate()
            
        self.logger.info("Audio system stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get audio system status"""
        return {
            'running': self.running,
            'pyaudio_available': PYAUDIO_AVAILABLE,
            'tts_available': TTS_AVAILABLE,
            'speech_recognition_available': SPEECH_RECOGNITION_AVAILABLE,
            'left_volume': self.left_speaker_volume,
            'right_volume': self.right_speaker_volume,
            'balance': self.balance,
            'tts_queue_size': self.tts_queue.qsize(),
            'audio_queue_size': self.audio_queue.qsize()
        }
