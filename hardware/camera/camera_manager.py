"""
Camera Manager Module
Handles camera input and image processing
"""

import cv2
import numpy as np
import logging
import threading
import time
from typing import Dict, Any, Optional, Tuple
from queue import Queue


class CameraManager:
    """Camera management system for video input"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the camera manager"""
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Configuration parameters
        self.device_id = config.get('device_id', 0)
        self.width = config.get('width', 640)
        self.height = config.get('height', 480)
        self.fps = config.get('fps', 30)
        self.format = config.get('format', 'YUYV')
        self.auto_exposure = config.get('auto_exposure', True)
        self.auto_focus = config.get('auto_focus', True)
        
        # Camera state
        self.camera = None
        self.running = False
        self.frame_queue = Queue(maxsize=3)  # Buffer for latest frames
        self.capture_thread = None
        
        # Performance tracking
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.actual_fps = 0
        
        # Initialize camera
        self._initialize_camera()
    
    def get_stereo_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get synchronized frames from both cameras for stereo vision
        
        Returns:
            Tuple of (left_frame, right_frame) or (None, None) if failed
        """
        try:
            # Get current frame from this camera
            left_frame = self.get_frame()
            
            if left_frame is None:
                return None, None
            
            # Get frame from other camera (right camera)
            right_camera = CameraManager(self.config)
            right_camera.device_id = 1 if self.device_id == 0 else 0
            right_camera._initialize_camera()
            
            right_frame = right_camera.get_frame_sync()
            right_camera.stop()
            
            return left_frame, right_frame
            
        except Exception as e:
            self.logger.error(f"Error getting stereo frames: {e}")
            return None, None
        
    def _initialize_camera(self):
        """Initialize IMX219 CSI camera hardware"""
        try:
            # For IMX219 CSI camera, we need to use the proper device path
            # The dual IMX219 typically uses /dev/video0 and /dev/video1
            # We'll start with the first camera (left camera)
            camera_device = f"/dev/video{self.device_id}"
            
            # Open camera
            self.camera = cv2.VideoCapture(camera_device, cv2.CAP_V4L2)
            
            if not self.camera.isOpened():
                # Try alternative device paths for CSI camera
                alternative_paths = [
                    f"/dev/video{self.device_id}",
                    "/dev/video0",
                    "/dev/video1",
                    "/dev/video2"
                ]
                
                for path in alternative_paths:
                    self.logger.info(f"Trying camera path: {path}")
                    self.camera = cv2.VideoCapture(path, cv2.CAP_V4L2)
                    if self.camera.isOpened():
                        self.logger.info(f"Successfully opened camera at {path}")
                        break
                
                if not self.camera.isOpened():
                    raise Exception(f"Could not open IMX219 CSI camera at any path")
            
            # Set camera properties for IMX219
            # IMX219 supports various resolutions: 3280x2464, 1920x1080, 1640x1232, 1280x720, 640x480
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.camera.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Set format for IMX219 (supports MJPEG, YUYV, etc.)
            if self.format == 'MJPEG':
                self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            elif self.format == 'YUYV':
                self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', 'U', 'Y', 'V'))
            
            # IMX219 specific settings
            # Set auto-exposure
            if self.auto_exposure:
                self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Auto exposure
            else:
                self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # Manual exposure
            
            # Set auto-focus (IMX219 has fixed focus, but we can set the property)
            if self.auto_focus:
                self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            else:
                self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            
            # Additional IMX219 optimizations
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
            
            # Verify camera settings
            actual_width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
            actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
            
            self.logger.info(f"IMX219 CSI camera initialized: {actual_width}x{actual_height} @ {actual_fps}fps")
            
            # Test capture
            ret, test_frame = self.camera.read()
            if not ret:
                raise Exception("Could not capture test frame from IMX219 camera")
            
            self.logger.info("IMX219 camera test capture successful")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize IMX219 CSI camera: {e}")
            self.camera = None
            raise
    
    def start(self):
        """Start camera capture thread"""
        if self.running:
            self.logger.warning("Camera is already running")
            return
        
        if self.camera is None:
            self.logger.error("Camera not initialized")
            return
        
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        self.logger.info("Camera capture started")
    
    def stop(self):
        """Stop camera capture"""
        self.running = False
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        
        if self.camera:
            self.camera.release()
            self.camera = None
        
        # Clear frame queue
        while not self.frame_queue.empty():
            self.frame_queue.get()
        
        self.logger.info("Camera capture stopped")
    
    def _capture_loop(self):
        """Main camera capture loop"""
        while self.running:
            try:
                # Capture frame
                ret, frame = self.camera.read()
                
                if not ret:
                    self.logger.warning("Failed to capture frame")
                    time.sleep(0.01)
                    continue
                
                # Update frame queue (remove old frames if queue is full)
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except:
                        pass
                
                self.frame_queue.put(frame.copy())
                
                # Update FPS counter
                self.frame_count += 1
                current_time = time.time()
                
                if current_time - self.last_fps_time >= 1.0:
                    self.actual_fps = self.frame_count / (current_time - self.last_fps_time)
                    self.frame_count = 0
                    self.last_fps_time = current_time
                    self.logger.debug(f"Camera FPS: {self.actual_fps:.1f}")
                
                # Control capture rate
                time.sleep(1.0 / self.fps)
                
            except Exception as e:
                self.logger.error(f"Error in capture loop: {e}")
                time.sleep(0.1)
    
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get the latest frame from camera
        
        Returns:
            Latest frame or None if no frame available
        """
        try:
            if not self.frame_queue.empty():
                return self.frame_queue.get_nowait()
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting frame: {e}")
            return None
    
    def get_frame_sync(self) -> Optional[np.ndarray]:
        """
        Get frame synchronously (blocking)
        
        Returns:
            Frame or None if failed
        """
        try:
            if self.camera is None:
                return None
            
            ret, frame = self.camera.read()
            if ret:
                return frame
            return None
            
        except Exception as e:
            self.logger.error(f"Error in sync frame capture: {e}")
            return None
    
    def set_resolution(self, width: int, height: int):
        """
        Set camera resolution
        
        Args:
            width: Frame width
            height: Frame height
        """
        try:
            if self.camera:
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                
                # Verify setting
                actual_width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
                actual_height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
                
                self.width = int(actual_width)
                self.height = int(actual_height)
                
                self.logger.info(f"Resolution set to {self.width}x{self.height}")
                
        except Exception as e:
            self.logger.error(f"Error setting resolution: {e}")
    
    def set_fps(self, fps: int):
        """
        Set camera frame rate
        
        Args:
            fps: Frames per second
        """
        try:
            if self.camera:
                self.camera.set(cv2.CAP_PROP_FPS, fps)
                self.fps = fps
                self.logger.info(f"FPS set to {fps}")
                
        except Exception as e:
            self.logger.error(f"Error setting FPS: {e}")
    
    def set_exposure(self, exposure: float):
        """
        Set camera exposure
        
        Args:
            exposure: Exposure value
        """
        try:
            if self.camera:
                self.camera.set(cv2.CAP_PROP_EXPOSURE, exposure)
                self.logger.info(f"Exposure set to {exposure}")
                
        except Exception as e:
            self.logger.error(f"Error setting exposure: {e}")
    
    def set_brightness(self, brightness: float):
        """
        Set camera brightness
        
        Args:
            brightness: Brightness value
        """
        try:
            if self.camera:
                self.camera.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
                self.logger.info(f"Brightness set to {brightness}")
                
        except Exception as e:
            self.logger.error(f"Error setting brightness: {e}")
    
    def set_contrast(self, contrast: float):
        """
        Set camera contrast
        
        Args:
            contrast: Contrast value
        """
        try:
            if self.camera:
                self.camera.set(cv2.CAP_PROP_CONTRAST, contrast)
                self.logger.info(f"Contrast set to {contrast}")
                
        except Exception as e:
            self.logger.error(f"Error setting contrast: {e}")
    
    def get_camera_info(self) -> Dict[str, Any]:
        """Get camera information and status"""
        if self.camera is None:
            return {'status': 'not_initialized'}
        
        try:
            info = {
                'status': 'running' if self.running else 'stopped',
                'device_id': self.device_id,
                'camera_type': 'IMX219 CSI',
                'width': self.width,
                'height': self.height,
                'fps': self.fps,
                'actual_fps': self.actual_fps,
                'format': self.format,
                'auto_exposure': self.auto_exposure,
                'auto_focus': self.auto_focus,
                'frame_queue_size': self.frame_queue.qsize(),
                'is_opened': self.camera.isOpened()
            }
            
            # Get current camera properties
            if self.camera.isOpened():
                info.update({
                    'actual_width': self.camera.get(cv2.CAP_PROP_FRAME_WIDTH),
                    'actual_height': self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT),
                    'actual_fps_camera': self.camera.get(cv2.CAP_PROP_FPS),
                    'brightness': self.camera.get(cv2.CAP_PROP_BRIGHTNESS),
                    'contrast': self.camera.get(cv2.CAP_PROP_CONTRAST),
                    'exposure': self.camera.get(cv2.CAP_PROP_EXPOSURE)
                })
            
            return info
            
        except Exception as e:
            self.logger.error(f"Error getting camera info: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def switch_camera(self, camera_id: int):
        """
        Switch between left and right IMX219 cameras
        
        Args:
            camera_id: 0 for left camera, 1 for right camera
        """
        try:
            if self.running:
                self.stop()
            
            self.device_id = camera_id
            self._initialize_camera()
            
            if self.running:
                self.start()
            
            self.logger.info(f"Switched to IMX219 camera {camera_id}")
            
        except Exception as e:
            self.logger.error(f"Error switching camera: {e}")
    
    def get_available_cameras(self) -> List[str]:
        """Get list of available IMX219 cameras"""
        available_cameras = []
        
        for i in range(4):  # Check first 4 video devices
            try:
                cap = cv2.VideoCapture(f"/dev/video{i}", cv2.CAP_V4L2)
                if cap.isOpened():
                    # Try to read a frame to confirm it's working
                    ret, frame = cap.read()
                    if ret:
                        available_cameras.append(f"/dev/video{i}")
                    cap.release()
            except:
                continue
        
        return available_cameras
    
    def test_camera(self) -> bool:
        """Test camera functionality"""
        try:
            if self.camera is None:
                return False
            
            # Capture test frame
            ret, frame = self.camera.read()
            if not ret:
                return False
            
            # Check frame properties
            if frame is None or frame.size == 0:
                return False
            
            # Save test image
            cv2.imwrite('camera_test.jpg', frame)
            self.logger.info("Camera test successful - saved test image")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Camera test failed: {e}")
            return False 