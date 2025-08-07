"""
Face Recognition Module
Handles face detection, recognition, and memory management
"""

import cv2
import numpy as np
import face_recognition
import pickle
import os
import logging
import time
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path


class FaceRecognizer:
    """Face recognition system for identifying familiar faces"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the face recognizer"""
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Configuration parameters
        self.tolerance = config.get('tolerance', 0.6)
        self.model = config.get('model', 'hog')  # 'hog' or 'cnn'
        self.upsample = config.get('upsample', 1)
        
        # Database paths
        self.database_path = Path(config.get('database_path', 'data/faces'))
        self.known_faces_file = self.database_path / config.get('known_faces_file', 'known_faces.pkl')
        
        # Create database directory
        self.database_path.mkdir(parents=True, exist_ok=True)
        
        # Load known faces
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_data = {}
        
        self._load_known_faces()
        
        # Performance tracking
        self.fps_counter = 0
        self.last_fps_time = 0
        
    def _load_known_faces(self):
        """Load known faces from database"""
        try:
            if self.known_faces_file.exists():
                with open(self.known_faces_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data.get('encodings', [])
                    self.known_face_names = data.get('names', [])
                    self.known_face_data = data.get('data', {})
                
                self.logger.info(f"Loaded {len(self.known_face_names)} known faces from database")
            else:
                self.logger.info("No existing face database found, starting fresh")
                
        except Exception as e:
            self.logger.error(f"Error loading known faces: {e}")
            self.known_face_encodings = []
            self.known_face_names = []
            self.known_face_data = {}
    
    def _save_known_faces(self):
        """Save known faces to database"""
        try:
            data = {
                'encodings': self.known_face_encodings,
                'names': self.known_face_names,
                'data': self.known_face_data
            }
            
            with open(self.known_faces_file, 'wb') as f:
                pickle.dump(data, f)
                
            self.logger.debug("Saved face database")
            
        except Exception as e:
            self.logger.error(f"Error saving known faces: {e}")
    
    def recognize(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Recognize faces in the given frame
        
        Args:
            frame: Input image frame (BGR format)
            
        Returns:
            List of recognized faces with metadata
        """
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Find face locations
            face_locations = face_recognition.face_locations(
                rgb_frame, 
                model=self.model, 
                number_of_times_to_upsample=self.upsample
            )
            
            # Get face encodings
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            # Recognize faces
            recognized_faces = []
            
            for i, (face_location, face_encoding) in enumerate(zip(face_locations, face_encodings)):
                # Compare with known faces
                matches = face_recognition.compare_faces(
                    self.known_face_encodings, 
                    face_encoding, 
                    tolerance=self.tolerance
                )
                
                face_name = "Unknown"
                face_data = {}
                confidence = 0.0
                
                if True in matches:
                    # Find the best match
                    face_distances = face_recognition.face_distance(
                        self.known_face_encodings, 
                        face_encoding
                    )
                    best_match_index = np.argmin(face_distances)
                    
                    if matches[best_match_index]:
                        face_name = self.known_face_names[best_match_index]
                        face_data = self.known_face_data.get(face_name, {})
                        confidence = 1.0 - face_distances[best_match_index]
                
                # Create face recognition result
                top, right, bottom, left = face_location
                face_result = {
                    'id': i,
                    'name': face_name,
                    'confidence': confidence,
                    'location': face_location,
                    'bbox': [left, top, right, bottom],
                    'center': [(left + right) // 2, (top + bottom) // 2],
                    'data': face_data,
                    'encoding': face_encoding,
                    'is_known': face_name != "Unknown"
                }
                
                recognized_faces.append(face_result)
            
            # Update FPS counter
            self._update_fps()
            
            return recognized_faces
            
        except Exception as e:
            self.logger.error(f"Error during face recognition: {e}")
            return []
    
    def add_face(self, frame: np.ndarray, name: str, additional_data: Dict[str, Any] = None) -> bool:
        """
        Add a new face to the database
        
        Args:
            frame: Image frame containing the face
            name: Name of the person
            additional_data: Additional information about the person
            
        Returns:
            True if face was added successfully
        """
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Find face locations
            face_locations = face_recognition.face_locations(
                rgb_frame, 
                model=self.model, 
                number_of_times_to_upsample=self.upsample
            )
            
            if not face_locations:
                self.logger.warning("No face detected in the provided image")
                return False
            
            # Get face encoding (use the first face found)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            if not face_encodings:
                self.logger.warning("Could not encode face")
                return False
            
            face_encoding = face_encodings[0]
            
            # Check if face already exists
            if name in self.known_face_names:
                self.logger.warning(f"Face '{name}' already exists in database")
                return False
            
            # Add to database
            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(name)
            
            # Store additional data
            face_data = additional_data or {}
            face_data['added_date'] = time.time()
            face_data['interactions'] = []
            self.known_face_data[name] = face_data
            
            # Save database
            self._save_known_faces()
            
            self.logger.info(f"Added face '{name}' to database")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding face: {e}")
            return False
    
    def update_interaction(self, name: str, interaction_data: Dict[str, Any]):
        """
        Update interaction history for a known face
        
        Args:
            name: Name of the person
            interaction_data: Interaction information
        """
        try:
            if name in self.known_face_data:
                if 'interactions' not in self.known_face_data[name]:
                    self.known_face_data[name]['interactions'] = []
                
                interaction_data['timestamp'] = time.time()
                self.known_face_data[name]['interactions'].append(interaction_data)
                
                # Keep only recent interactions (last 50)
                if len(self.known_face_data[name]['interactions']) > 50:
                    self.known_face_data[name]['interactions'] = \
                        self.known_face_data[name]['interactions'][-50:]
                
                # Save database
                self._save_known_faces()
                
                self.logger.debug(f"Updated interaction for '{name}'")
            
        except Exception as e:
            self.logger.error(f"Error updating interaction: {e}")
    
    def get_face_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a known face
        
        Args:
            name: Name of the person
            
        Returns:
            Face information or None if not found
        """
        if name in self.known_face_data:
            return self.known_face_data[name]
        return None
    
    def remove_face(self, name: str) -> bool:
        """
        Remove a face from the database
        
        Args:
            name: Name of the person to remove
            
        Returns:
            True if face was removed successfully
        """
        try:
            if name not in self.known_face_names:
                self.logger.warning(f"Face '{name}' not found in database")
                return False
            
            # Find index
            index = self.known_face_names.index(name)
            
            # Remove from lists
            del self.known_face_encodings[index]
            del self.known_face_names[index]
            
            # Remove from data
            if name in self.known_face_data:
                del self.known_face_data[name]
            
            # Save database
            self._save_known_faces()
            
            self.logger.info(f"Removed face '{name}' from database")
            return True
            
        except Exception as e:
            self.logger.error(f"Error removing face: {e}")
            return False
    
    def get_known_faces(self) -> List[str]:
        """Get list of all known face names"""
        return self.known_face_names.copy()
    
    def _update_fps(self):
        """Update FPS counter"""
        current_time = time.time()
        self.fps_counter += 1
        
        if current_time - self.last_fps_time >= 1.0:
            fps = self.fps_counter / (current_time - self.last_fps_time)
            self.logger.debug(f"Face recognition FPS: {fps:.1f}")
            self.fps_counter = 0
            self.last_fps_time = current_time
    
    def visualize_faces(self, frame: np.ndarray, faces: List[Dict[str, Any]]) -> np.ndarray:
        """Draw face recognition results on frame for debugging"""
        vis_frame = frame.copy()
        
        for face in faces:
            left, top, right, bottom = face['bbox']
            name = face['name']
            confidence = face['confidence']
            is_known = face['is_known']
            
            # Choose color based on recognition status
            color = (0, 255, 0) if is_known else (0, 0, 255)  # Green for known, Red for unknown
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (left, top), (right, bottom), color, 2)
            
            # Draw label
            label = f"{name} {confidence:.2f}"
            cv2.putText(vis_frame, label, (left, top - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return vis_frame 