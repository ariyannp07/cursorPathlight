"""
Memory Manager Module
Handles storage and retrieval of interaction history and face data
"""

import sqlite3
import logging
import json
import time
from typing import Dict, Any, List, Optional
from pathlib import Path


class MemoryManager:
    """Memory management system for storing interaction history"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the memory manager"""
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Database configuration
        self.db_path = config.get('path', 'data/pathlight.db')
        self.backup_interval = config.get('backup_interval', 3600)
        
        # Initialize database
        self._initialize_database()
        
    def _initialize_database(self):
        """Initialize SQLite database"""
        try:
            # Create database directory
            db_dir = Path(self.db_path).parent
            db_dir.mkdir(parents=True, exist_ok=True)
            
            # Connect to database
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
            
            # Create tables
            self._create_tables()
            
            self.logger.info(f"Database initialized: {self.db_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            raise
    
    def _create_tables(self):
        """Create database tables"""
        cursor = self.conn.cursor()
        
        # Interactions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                face_name TEXT,
                interaction_type TEXT,
                content TEXT,
                timestamp REAL,
                metadata TEXT
            )
        ''')
        
        # Face data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS face_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE,
                encoding_data TEXT,
                added_date REAL,
                last_seen REAL,
                interaction_count INTEGER DEFAULT 0,
                metadata TEXT
            )
        ''')
        
        # Environment data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS environment_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                location TEXT,
                objects_detected TEXT,
                path_taken TEXT,
                metadata TEXT
            )
        ''')
        
        self.conn.commit()
    
    def update(self, faces: List[Dict[str, Any]], detections: List[Dict[str, Any]]):
        """Update memory with current observations"""
        try:
            timestamp = time.time()
            
            # Update face interactions
            for face in faces:
                if face.get('is_known'):
                    self._update_face_interaction(face, timestamp)
            
            # Store environment data
            self._store_environment_data(detections, timestamp)
            
        except Exception as e:
            self.logger.error(f"Error updating memory: {e}")
    
    def _update_face_interaction(self, face: Dict[str, Any], timestamp: float):
        """Update interaction history for a face"""
        try:
            cursor = self.conn.cursor()
            
            # Update last seen time
            cursor.execute('''
                UPDATE face_data 
                SET last_seen = ?, interaction_count = interaction_count + 1
                WHERE name = ?
            ''', (timestamp, face['name']))
            
            # Add interaction record
            cursor.execute('''
                INSERT INTO interactions (face_name, interaction_type, content, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                face['name'],
                'recognition',
                f"Recognized {face['name']} with confidence {face['confidence']:.2f}",
                timestamp,
                json.dumps(face)
            ))
            
            self.conn.commit()
            
        except Exception as e:
            self.logger.error(f"Error updating face interaction: {e}")
    
    def _store_environment_data(self, detections: List[Dict[str, Any]], timestamp: float):
        """Store environment observation data"""
        try:
            cursor = self.conn.cursor()
            
            # Store detection data
            cursor.execute('''
                INSERT INTO environment_data (timestamp, objects_detected, metadata)
                VALUES (?, ?, ?)
            ''', (
                timestamp,
                json.dumps([d['class_name'] for d in detections]),
                json.dumps(detections)
            ))
            
            self.conn.commit()
            
        except Exception as e:
            self.logger.error(f"Error storing environment data: {e}")
    
    def get_face_history(self, name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get interaction history for a specific face"""
        try:
            cursor = self.conn.cursor()
            
            cursor.execute('''
                SELECT * FROM interactions 
                WHERE face_name = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (name, limit))
            
            return [dict(row) for row in cursor.fetchall()]
            
        except Exception as e:
            self.logger.error(f"Error getting face history: {e}")
            return []
    
    def get_recent_interactions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent interactions across all faces"""
        try:
            cursor = self.conn.cursor()
            
            cursor.execute('''
                SELECT * FROM interactions 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
            
            return [dict(row) for row in cursor.fetchall()]
            
        except Exception as e:
            self.logger.error(f"Error getting recent interactions: {e}")
            return []
    
    def close(self):
        """Close database connection"""
        if hasattr(self, 'conn'):
            self.conn.close()
            self.logger.info("Database connection closed") 