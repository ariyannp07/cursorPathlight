"""
Object Detection Module using YOLOv8
Handles real-time object detection for obstacle identification
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Tuple
from ultralytics import YOLO
import torch


class ObjectDetector:
    """YOLOv8-based object detector for obstacle identification"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the object detector"""
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Load YOLOv8 model
        self.model = self._load_model()
        
        # Detection parameters
        self.confidence_threshold = config.get('confidence', 0.5)
        self.iou_threshold = config.get('iou_threshold', 0.45)
        self.max_detections = config.get('max_detections', 100)
        self.target_classes = config.get('classes', [0, 1, 2, 3, 5, 7])  # person, vehicle classes
        
        # Device configuration - Force CPU for compatibility
        self.device = 'cpu'  # Force CPU-only operation for Jetson compatibility
        self.logger.info(f"Using device: {self.device} (forced CPU mode)")
        
        # Performance tracking
        self.fps_counter = 0
        self.last_fps_time = 0
        
    def _load_model(self) -> YOLO:
        """Load YOLOv8 model"""
        try:
            model_name = self.config.get('model', 'yolov8n.pt')
            self.logger.info(f"Loading YOLOv8 model: {model_name}")
            
            # Load model
            model = YOLO(model_name)
            
            # Move to CPU device (GPU disabled for compatibility)
            model.to('cpu')
            self.logger.info("Model loaded on CPU device")
            
            self.logger.info("YOLOv8 model loaded successfully")
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load YOLOv8 model: {e}")
            raise
    
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects in the given frame
        
        Args:
            frame: Input image frame (BGR format)
            
        Returns:
            List of detected objects with bounding boxes and metadata
        """
        try:
            # Run inference
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                max_det=self.max_detections,
                classes=self.target_classes,
                verbose=False
            )
            
            # Process results
            detections = self._process_results(results[0], frame.shape)
            
            # Update FPS counter
            self._update_fps()
            
            return detections
            
        except Exception as e:
            self.logger.error(f"Error during object detection: {e}")
            return []
    
    def _process_results(self, result, frame_shape: Tuple[int, int, int]) -> List[Dict[str, Any]]:
        """Process YOLOv8 results into structured format"""
        detections = []
        
        if result.boxes is None:
            return detections
        
        # Get detection data
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
        confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
        class_ids = result.boxes.cls.cpu().numpy().astype(int)  # Class IDs
        
        # Process each detection
        for i in range(len(boxes)):
            if i >= self.max_detections:
                break
                
            x1, y1, x2, y2 = boxes[i]
            confidence = confidences[i]
            class_id = class_ids[i]
            
            # Calculate center point and dimensions
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            
            # Calculate distance estimate (rough approximation)
            distance = self._estimate_distance(width, height, class_id, frame_shape)
            
            # Create detection object
            detection = {
                'bbox': [x1, y1, x2, y2],
                'center': [center_x, center_y],
                'width': width,
                'height': height,
                'confidence': float(confidence),
                'class_id': int(class_id),
                'class_name': self._get_class_name(class_id),
                'distance': distance,
                'area': width * height,
                'is_obstacle': self._is_obstacle(class_id, distance)
            }
            
            detections.append(detection)
        
        return detections
    
    def _get_class_name(self, class_id: int) -> str:
        """Get class name from class ID"""
        class_names = {
            0: 'person',
            1: 'bicycle',
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck',
            8: 'truck',
            9: 'boat',
            10: 'traffic light',
            11: 'fire hydrant',
            12: 'stop sign',
            13: 'parking meter',
            14: 'bench',
            15: 'bird',
            16: 'cat',
            17: 'dog',
            18: 'horse',
            19: 'sheep',
            20: 'cow',
            21: 'elephant',
            22: 'bear',
            23: 'zebra',
            24: 'giraffe',
            25: 'backpack',
            26: 'umbrella',
            27: 'handbag',
            28: 'tie',
            29: 'suitcase',
            30: 'frisbee',
            31: 'skis',
            32: 'snowboard',
            33: 'sports ball',
            34: 'kite',
            35: 'baseball bat',
            36: 'baseball glove',
            37: 'skateboard',
            38: 'surfboard',
            39: 'tennis racket',
            40: 'bottle',
            41: 'wine glass',
            42: 'cup',
            43: 'fork',
            44: 'knife',
            45: 'spoon',
            46: 'bowl',
            47: 'banana',
            48: 'apple',
            49: 'sandwich',
            50: 'orange',
            51: 'broccoli',
            52: 'carrot',
            53: 'hot dog',
            54: 'pizza',
            55: 'donut',
            56: 'cake',
            57: 'chair',
            58: 'couch',
            59: 'potted plant',
            60: 'bed',
            61: 'dining table',
            62: 'toilet',
            63: 'tv',
            64: 'laptop',
            65: 'mouse',
            66: 'remote',
            67: 'keyboard',
            68: 'cell phone',
            69: 'microwave',
            70: 'oven',
            71: 'toaster',
            72: 'sink',
            73: 'refrigerator',
            74: 'book',
            75: 'clock',
            76: 'vase',
            77: 'scissors',
            78: 'teddy bear',
            79: 'hair drier',
            80: 'toothbrush'
        }
        
        return class_names.get(class_id, f'unknown_{class_id}')
    
    def _estimate_distance(self, width: float, height: float, class_id: int, 
                          frame_shape: Tuple[int, int, int]) -> float:
        """
        Estimate distance to object based on bounding box size
        This is a rough approximation - more accurate methods would use depth sensors
        """
        frame_height, frame_width = frame_shape[:2]
        
        # Known average sizes for different object classes (in pixels at 1 meter distance)
        # These values would need calibration for your specific camera setup
        known_sizes = {
            0: 100,  # person height
            1: 60,   # bicycle width
            2: 80,   # car width
            3: 60,   # motorcycle width
            5: 120,  # bus width
            7: 100,  # truck width
        }
        
        if class_id not in known_sizes:
            return 5.0  # Default distance for unknown objects
        
        # Use the larger dimension for distance estimation
        object_size = max(width, height)
        known_size = known_sizes[class_id]
        
        # Simple inverse relationship: distance = known_size / object_size
        distance = known_size / object_size if object_size > 0 else 10.0
        
        # Clamp to reasonable range
        distance = max(0.5, min(20.0, distance))
        
        return distance
    
    def _is_obstacle(self, class_id: int, distance: float) -> bool:
        """Determine if detected object is an obstacle"""
        # Objects that are always obstacles
        obstacle_classes = {1, 2, 3, 5, 7, 8, 10, 11, 12, 13, 14, 57, 58, 59, 60, 61, 62}
        
        # People are obstacles if too close
        if class_id == 0:  # person
            return distance < 2.0
        
        # Other objects are obstacles if they're in the obstacle classes and close enough
        return class_id in obstacle_classes and distance < 5.0
    
    def _update_fps(self):
        """Update FPS counter"""
        import time
        current_time = time.time()
        self.fps_counter += 1
        
        if current_time - self.last_fps_time >= 1.0:
            fps = self.fps_counter / (current_time - self.last_fps_time)
            self.logger.debug(f"Object detection FPS: {fps:.1f}")
            self.fps_counter = 0
            self.last_fps_time = current_time
    
    def get_detection_summary(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a summary of detections for navigation planning"""
        if not detections:
            return {'obstacles': [], 'people': [], 'vehicles': [], 'total_objects': 0}
        
        obstacles = [d for d in detections if d['is_obstacle']]
        people = [d for d in detections if d['class_id'] == 0]
        vehicles = [d for d in detections if d['class_id'] in [1, 2, 3, 5, 7, 8]]
        
        return {
            'obstacles': obstacles,
            'people': people,
            'vehicles': vehicles,
            'total_objects': len(detections),
            'closest_obstacle': min(obstacles, key=lambda x: x['distance']) if obstacles else None,
            'closest_person': min(people, key=lambda x: x['distance']) if people else None
        }
    
    def visualize_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Draw detection bounding boxes on frame for debugging"""
        vis_frame = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            distance = detection['distance']
            is_obstacle = detection['is_obstacle']
            
            # Choose color based on obstacle status
            color = (0, 0, 255) if is_obstacle else (0, 255, 0)  # Red for obstacles, Green for others
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # Draw label
            label = f"{class_name} {confidence:.2f} {distance:.1f}m"
            cv2.putText(vis_frame, label, (int(x1), int(y1) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return vis_frame 