"""
Path Planning Module
Handles safe path calculation and obstacle avoidance
"""

import numpy as np
import cv2
import logging
import math
from typing import List, Dict, Any, Tuple, Optional
from enum import Enum


class Direction(Enum):
    """Navigation directions"""
    FORWARD = "forward"
    LEFT = "left"
    RIGHT = "right"
    BACKWARD = "backward"
    STOP = "stop"


class PathPlanner:
    """Path planning system for safe navigation"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the path planner"""
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Configuration parameters
        self.grid_size = config.get('grid_size', 0.5)  # meters
        self.max_range = config.get('max_range', 10.0)  # meters
        self.safety_margin = config.get('safety_margin', 0.5)  # meters
        self.path_smoothing = config.get('path_smoothing', True)
        self.obstacle_cost = config.get('obstacle_cost', 1000)
        self.path_cost = config.get('path_cost', 1)
        
        # Grid dimensions
        self.grid_width = int(self.max_range * 2 / self.grid_size)
        self.grid_height = int(self.max_range * 2 / self.grid_size)
        
        # Initialize cost map
        self.cost_map = np.ones((self.grid_height, self.grid_width)) * self.path_cost
        
        # Direction weights for different scenarios
        self.direction_weights = {
            Direction.FORWARD: 1.0,
            Direction.LEFT: 0.8,
            Direction.RIGHT: 0.8,
            Direction.BACKWARD: 0.3,
            Direction.STOP: 0.0
        }
        
        # Safety thresholds
        self.emergency_stop_distance = 0.3  # meters
        self.warning_distance = 1.0  # meters
        
    def plan_path(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Plan safe navigation path based on detected obstacles
        
        Args:
            detections: List of detected objects from object detector
            
        Returns:
            Path planning result with direction and safety information
        """
        try:
            # Update cost map with obstacles
            self._update_cost_map(detections)
            
            # Analyze safety zones
            safety_analysis = self._analyze_safety_zones(detections)
            
            # Determine best direction
            direction = self._determine_direction(safety_analysis)
            
            # Calculate path confidence
            confidence = self._calculate_confidence(safety_analysis, direction)
            
            # Generate navigation instructions
            instructions = self._generate_instructions(direction, safety_analysis)
            
            return {
                'direction': direction.value,
                'confidence': confidence,
                'safety_level': safety_analysis['overall_safety'],
                'closest_obstacle': safety_analysis['closest_obstacle'],
                'instructions': instructions,
                'safety_zones': safety_analysis['zones'],
                'recommended_speed': self._calculate_speed(safety_analysis)
            }
            
        except Exception as e:
            self.logger.error(f"Error in path planning: {e}")
            return {
                'direction': Direction.STOP.value,
                'confidence': 0.0,
                'safety_level': 'danger',
                'closest_obstacle': None,
                'instructions': "Stop immediately. System error detected.",
                'safety_zones': {},
                'recommended_speed': 0.0
            }
    
    def _update_cost_map(self, detections: List[Dict[str, Any]]):
        """Update cost map with detected obstacles"""
        # Reset cost map
        self.cost_map = np.ones((self.grid_height, self.grid_width)) * self.path_cost
        
        # Add obstacles to cost map
        for detection in detections:
            if detection.get('is_obstacle', False):
                self._add_obstacle_to_cost_map(detection)
    
    def _add_obstacle_to_cost_map(self, detection: Dict[str, Any]):
        """Add obstacle to cost map"""
        try:
            # Get obstacle position in world coordinates
            center_x, center_y = detection['center']
            distance = detection['distance']
            
            # Convert to grid coordinates
            grid_x = int((center_x + self.max_range) / self.grid_size)
            grid_y = int((center_y + self.max_range) / self.grid_size)
            
            # Ensure coordinates are within grid bounds
            if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
                # Add obstacle cost with distance-based decay
                obstacle_radius = max(1, int(self.safety_margin / self.grid_size))
                
                for dy in range(-obstacle_radius, obstacle_radius + 1):
                    for dx in range(-obstacle_radius, obstacle_radius + 1):
                        nx, ny = grid_x + dx, grid_y + dy
                        
                        if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                            # Calculate distance from obstacle center
                            dist = math.sqrt(dx*dx + dy*dy) * self.grid_size
                            
                            # Cost decreases with distance
                            if dist <= self.safety_margin:
                                cost_factor = 1.0 - (dist / self.safety_margin)
                                self.cost_map[ny, nx] = max(
                                    self.cost_map[ny, nx],
                                    self.obstacle_cost * cost_factor
                                )
                                
        except Exception as e:
            self.logger.error(f"Error adding obstacle to cost map: {e}")
    
    def _analyze_safety_zones(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze safety zones in different directions"""
        # Define zones (forward, left, right, backward)
        zones = {
            'forward': {'angle_range': (-30, 30), 'obstacles': []},
            'left': {'angle_range': (30, 90), 'obstacles': []},
            'right': {'angle_range': (-90, -30), 'obstacles': []},
            'backward': {'angle_range': (150, 210), 'obstacles': []}
        }
        
        # Categorize obstacles by zone
        for detection in detections:
            if detection.get('is_obstacle', False):
                center_x, center_y = detection['center']
                
                # Calculate angle from center (assuming center is at origin)
                angle = math.degrees(math.atan2(center_y, center_x))
                
                # Normalize angle to 0-360 range
                angle = (angle + 360) % 360
                
                # Assign to appropriate zone
                for zone_name, zone_data in zones.items():
                    min_angle, max_angle = zone_data['angle_range']
                    
                    if min_angle <= angle <= max_angle:
                        zone_data['obstacles'].append(detection)
                        break
        
        # Calculate safety metrics for each zone
        zone_safety = {}
        closest_obstacle = None
        min_distance = float('inf')
        
        for zone_name, zone_data in zones.items():
            obstacles = zone_data['obstacles']
            
            if not obstacles:
                zone_safety[zone_name] = {
                    'safety_level': 'safe',
                    'closest_distance': float('inf'),
                    'obstacle_count': 0
                }
            else:
                # Find closest obstacle in zone
                closest = min(obstacles, key=lambda x: x['distance'])
                distance = closest['distance']
                
                # Determine safety level
                if distance < self.emergency_stop_distance:
                    safety_level = 'danger'
                elif distance < self.warning_distance:
                    safety_level = 'warning'
                else:
                    safety_level = 'safe'
                
                zone_safety[zone_name] = {
                    'safety_level': safety_level,
                    'closest_distance': distance,
                    'obstacle_count': len(obstacles)
                }
                
                # Track overall closest obstacle
                if distance < min_distance:
                    min_distance = distance
                    closest_obstacle = closest
        
        # Determine overall safety level
        if min_distance < self.emergency_stop_distance:
            overall_safety = 'danger'
        elif min_distance < self.warning_distance:
            overall_safety = 'warning'
        else:
            overall_safety = 'safe'
        
        return {
            'zones': zone_safety,
            'overall_safety': overall_safety,
            'closest_obstacle': closest_obstacle,
            'min_distance': min_distance
        }
    
    def _determine_direction(self, safety_analysis: Dict[str, Any]) -> Direction:
        """Determine the best navigation direction"""
        zones = safety_analysis['zones']
        
        # Check for emergency stop condition
        if safety_analysis['overall_safety'] == 'danger':
            return Direction.STOP
        
        # Score each direction
        direction_scores = {}
        
        for direction in [Direction.FORWARD, Direction.LEFT, Direction.RIGHT, Direction.BACKWARD]:
            zone_name = direction.value
            zone_data = zones.get(zone_name, {})
            
            # Base score from direction preference
            score = self.direction_weights[direction]
            
            # Adjust based on safety
            safety_level = zone_data.get('safety_level', 'safe')
            if safety_level == 'danger':
                score *= 0.0
            elif safety_level == 'warning':
                score *= 0.5
            elif safety_level == 'safe':
                score *= 1.0
            
            # Adjust based on obstacle count
            obstacle_count = zone_data.get('obstacle_count', 0)
            score *= max(0.1, 1.0 - obstacle_count * 0.2)
            
            direction_scores[direction] = score
        
        # Choose direction with highest score
        best_direction = max(direction_scores, key=direction_scores.get)
        
        # If no good direction, stop
        if direction_scores[best_direction] < 0.1:
            return Direction.STOP
        
        return best_direction
    
    def _calculate_confidence(self, safety_analysis: Dict[str, Any], direction: Direction) -> float:
        """Calculate confidence in the chosen direction"""
        zones = safety_analysis['zones']
        zone_name = direction.value
        zone_data = zones.get(zone_name, {})
        
        # Base confidence
        confidence = 1.0
        
        # Adjust based on safety level
        safety_level = zone_data.get('safety_level', 'safe')
        if safety_level == 'danger':
            confidence *= 0.0
        elif safety_level == 'warning':
            confidence *= 0.5
        elif safety_level == 'safe':
            confidence *= 1.0
        
        # Adjust based on obstacle count
        obstacle_count = zone_data.get('obstacle_count', 0)
        confidence *= max(0.1, 1.0 - obstacle_count * 0.1)
        
        # Adjust based on overall safety
        if safety_analysis['overall_safety'] == 'danger':
            confidence *= 0.3
        elif safety_analysis['overall_safety'] == 'warning':
            confidence *= 0.7
        
        return max(0.0, min(1.0, confidence))
    
    def _generate_instructions(self, direction: Direction, safety_analysis: Dict[str, Any]) -> str:
        """Generate navigation instructions"""
        if direction == Direction.STOP:
            return "Stop immediately. Obstacle detected too close."
        
        # Base instruction
        instructions = f"Move {direction.value}"
        
        # Add safety information
        closest_obstacle = safety_analysis.get('closest_obstacle')
        if closest_obstacle:
            distance = closest_obstacle['distance']
            class_name = closest_obstacle['class_name']
            
            if distance < self.warning_distance:
                instructions += f". Caution: {class_name} at {distance:.1f} meters"
            else:
                instructions += f". Clear path ahead"
        
        # Add speed recommendation
        speed = self._calculate_speed(safety_analysis)
        if speed < 0.5:
            instructions += ". Proceed slowly"
        elif speed > 1.0:
            instructions += ". Clear to proceed at normal speed"
        
        return instructions
    
    def _calculate_speed(self, safety_analysis: Dict[str, Any]) -> float:
        """Calculate recommended movement speed"""
        min_distance = safety_analysis.get('min_distance', float('inf'))
        overall_safety = safety_analysis.get('overall_safety', 'safe')
        
        # Base speed
        if overall_safety == 'danger':
            return 0.0
        elif overall_safety == 'warning':
            return 0.3
        else:
            return 1.0
        
        # Adjust based on closest obstacle
        if min_distance < self.emergency_stop_distance:
            return 0.0
        elif min_distance < self.warning_distance:
            return 0.5
        else:
            return 1.0
    
    def get_cost_map_visualization(self) -> np.ndarray:
        """Get cost map visualization for debugging"""
        # Normalize cost map for visualization
        normalized = np.clip(self.cost_map / self.obstacle_cost, 0, 1)
        
        # Convert to 8-bit image
        vis_map = (normalized * 255).astype(np.uint8)
        
        # Apply color map
        vis_map = cv2.applyColorMap(vis_map, cv2.COLORMAP_JET)
        
        # Resize for better visualization
        vis_map = cv2.resize(vis_map, (400, 400))
        
        return vis_map 