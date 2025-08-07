"""
Stereo Vision Module
Handles 3D depth perception using dual IMX219 cameras
"""

import cv2
import numpy as np
import logging
import time
from typing import Dict, Any, Tuple, Optional, List
from pathlib import Path
import pickle


class StereoVision:
    """Stereo vision system for 3D depth perception"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the stereo vision system"""
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Configuration parameters
        self.calibration_file = config.get('calibration_file', 'data/stereo_calibration.pkl')
        self.camera_separation = config.get('camera_separation', 0.12)  # meters (12cm typical)
        self.focal_length = config.get('focal_length', 1000.0)  # pixels
        self.min_depth = config.get('min_depth', 0.1)  # meters
        self.max_depth = config.get('max_depth', 10.0)  # meters
        
        # Stereo processing parameters
        self.block_size = config.get('block_size', 11)
        self.min_disparity = config.get('min_disparity', 0)
        self.num_disparities = config.get('num_disparities', 128)
        self.uniqueness_ratio = config.get('uniqueness_ratio', 15)
        self.speckle_window_size = config.get('speckle_window_size', 100)
        self.speckle_range = config.get('speckle_range', 32)
        
        # Calibration data
        self.calibration_data = None
        self.stereo_matcher = None
        self.is_calibrated = False
        
        # Performance tracking
        self.fps_counter = 0
        self.last_fps_time = time.time()
        
        # Initialize stereo system
        self._initialize_stereo()
        
    def _initialize_stereo(self):
        """Initialize stereo vision system"""
        try:
            # Load calibration data if available
            self._load_calibration()
            
            # Initialize stereo matcher
            self._create_stereo_matcher()
            
            self.logger.info("Stereo vision system initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize stereo vision: {e}")
            raise
    
    def _load_calibration(self):
        """Load stereo camera calibration data"""
        try:
            calibration_path = Path(self.calibration_file)
            if calibration_path.exists():
                with open(calibration_path, 'rb') as f:
                    self.calibration_data = pickle.load(f)
                
                self.is_calibrated = True
                self.logger.info("Loaded stereo calibration data")
                
                # Extract calibration parameters
                self.left_camera_matrix = self.calibration_data['left_camera_matrix']
                self.right_camera_matrix = self.calibration_data['right_camera_matrix']
                self.left_distortion = self.calibration_data['left_distortion']
                self.right_distortion = self.calibration_data['right_distortion']
                self.rotation_matrix = self.calibration_data['rotation_matrix']
                self.translation_vector = self.calibration_data['translation_vector']
                self.essential_matrix = self.calibration_data['essential_matrix']
                self.fundamental_matrix = self.calibration_data['fundamental_matrix']
                
            else:
                self.logger.warning("No calibration file found. Using default parameters.")
                self._create_default_calibration()
                
        except Exception as e:
            self.logger.error(f"Error loading calibration: {e}")
            self._create_default_calibration()
    
    def _create_default_calibration(self):
        """Create default calibration for IMX219 cameras"""
        # Default camera matrix for IMX219 (approximate)
        focal_length = self.focal_length
        principal_point = (640, 360)  # Center of 1280x720 image
        
        self.left_camera_matrix = np.array([
            [focal_length, 0, principal_point[0]],
            [0, focal_length, principal_point[1]],
            [0, 0, 1]
        ], dtype=np.float32)
        
        self.right_camera_matrix = self.left_camera_matrix.copy()
        
        # Default distortion coefficients (minimal for IMX219)
        self.left_distortion = np.zeros((1, 5), dtype=np.float32)
        self.right_distortion = np.zeros((1, 5), dtype=np.float32)
        
        # Default stereo geometry
        self.rotation_matrix = np.eye(3, dtype=np.float32)
        self.translation_vector = np.array([[-self.camera_separation], [0], [0]], dtype=np.float32)
        
        # Calculate essential and fundamental matrices
        self.essential_matrix = cv2.findEssentialMat(
            np.array([[0, 0]]), np.array([[0, 0]]),
            self.left_camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0
        )[0]
        
        self.fundamental_matrix = cv2.findFundamentalMat(
            np.array([[0, 0]]), np.array([[0, 0]]),
            method=cv2.FM_8POINT
        )[0]
        
        self.is_calibrated = False
        self.logger.warning("Using default calibration. Consider running calibration for better accuracy.")
    
    def _create_stereo_matcher(self):
        """Create stereo matcher for depth computation"""
        try:
            # Create stereo block matcher
            self.stereo_matcher = cv2.StereoSGBM_create(
                minDisparity=self.min_disparity,
                numDisparities=self.num_disparities,
                blockSize=self.block_size,
                P1=8 * 3 * self.block_size**2,
                P2=32 * 3 * self.block_size**2,
                disp12MaxDiff=1,
                uniquenessRatio=self.uniqueness_ratio,
                speckleWindowSize=self.speckle_window_size,
                speckleRange=self.speckle_range,
                mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
            )
            
            # Create right matcher for left-right consistency check
            self.right_matcher = cv2.ximgproc.createRightMatcher(self.stereo_matcher)
            
            # Create WLS filter for post-processing
            self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(self.stereo_matcher)
            self.wls_filter.setLambda(8000)
            self.wls_filter.setSigmaColor(1.5)
            
            self.logger.info("Stereo matcher created successfully")
            
        except Exception as e:
            self.logger.error(f"Error creating stereo matcher: {e}")
            raise
    
    def compute_depth(self, left_frame: np.ndarray, right_frame: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute depth map from stereo images
        
        Args:
            left_frame: Left camera image (grayscale)
            right_frame: Right camera image (grayscale)
            
        Returns:
            Dictionary containing depth map and disparity map
        """
        try:
            # Ensure images are grayscale
            if len(left_frame.shape) == 3:
                left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
            else:
                left_gray = left_frame
                
            if len(right_frame.shape) == 3:
                right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
            else:
                right_gray = right_frame
            
            # Rectify images if calibrated
            if self.is_calibrated:
                left_rect, right_rect = self._rectify_images(left_gray, right_gray)
            else:
                left_rect, right_rect = left_gray, right_gray
            
            # Compute disparity maps
            left_disparity = self.stereo_matcher.compute(left_rect, right_rect)
            right_disparity = self.right_matcher.compute(right_rect, left_rect)
            
            # Apply WLS filter
            filtered_disparity = self.wls_filter.filter(
                left_disparity, left_rect, disparity_map_right=right_disparity
            )
            
            # Convert disparity to depth
            depth_map = self._disparity_to_depth(filtered_disparity)
            
            # Update FPS counter
            self._update_fps()
            
            return {
                'depth_map': depth_map,
                'disparity_map': filtered_disparity,
                'left_rectified': left_rect,
                'right_rectified': right_rect
            }
            
        except Exception as e:
            self.logger.error(f"Error computing depth: {e}")
            return {
                'depth_map': None,
                'disparity_map': None,
                'left_rectified': None,
                'right_rectified': None
            }
    
    def _rectify_images(self, left_img: np.ndarray, right_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Rectify stereo images using calibration data"""
        try:
            # Get rectification maps
            left_rect_map, right_rect_map = cv2.stereoRectify(
                self.left_camera_matrix, self.left_distortion,
                self.right_camera_matrix, self.right_distortion,
                left_img.shape[::-1], self.rotation_matrix, self.translation_vector,
                flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.9
            )[1:3]
            
            # Apply rectification
            left_rect = cv2.remap(left_img, left_rect_map[0], left_rect_map[1], cv2.INTER_LINEAR)
            right_rect = cv2.remap(right_img, right_rect_map[0], right_rect_map[1], cv2.INTER_LINEAR)
            
            return left_rect, right_rect
            
        except Exception as e:
            self.logger.error(f"Error rectifying images: {e}")
            return left_img, right_img
    
    def _disparity_to_depth(self, disparity_map: np.ndarray) -> np.ndarray:
        """Convert disparity map to depth map"""
        try:
            # Avoid division by zero
            disparity_map = disparity_map.astype(np.float32) / 16.0
            disparity_map[disparity_map == 0] = 0.1
            
            # Convert to depth using stereo geometry
            depth_map = (self.focal_length * self.camera_separation) / disparity_map
            
            # Apply depth limits
            depth_map[depth_map < self.min_depth] = self.min_depth
            depth_map[depth_map > self.max_depth] = self.max_depth
            
            return depth_map
            
        except Exception as e:
            self.logger.error(f"Error converting disparity to depth: {e}")
            return np.zeros_like(disparity_map)
    
    def get_3d_points(self, depth_map: np.ndarray, color_frame: np.ndarray = None) -> np.ndarray:
        """
        Convert depth map to 3D point cloud
        
        Args:
            depth_map: Depth map from stereo vision
            color_frame: Optional color frame for colored point cloud
            
        Returns:
            3D point cloud (Nx3 or Nx6 if color provided)
        """
        try:
            height, width = depth_map.shape
            
            # Create coordinate grids
            x_grid, y_grid = np.meshgrid(np.arange(width), np.arange(height))
            
            # Convert to 3D coordinates
            Z = depth_map
            X = (x_grid - self.left_camera_matrix[0, 2]) * Z / self.left_camera_matrix[0, 0]
            Y = (y_grid - self.left_camera_matrix[1, 2]) * Z / self.left_camera_matrix[1, 1]
            
            # Stack coordinates
            points_3d = np.stack([X, Y, Z], axis=-1)
            
            # Filter valid points
            valid_mask = (depth_map > self.min_depth) & (depth_map < self.max_depth)
            valid_points = points_3d[valid_mask]
            
            # Add color if provided
            if color_frame is not None:
                valid_colors = color_frame[valid_mask]
                valid_points = np.concatenate([valid_points, valid_colors], axis=1)
            
            return valid_points
            
        except Exception as e:
            self.logger.error(f"Error creating 3D point cloud: {e}")
            return np.array([])
    
    def detect_obstacles_3d(self, depth_map: np.ndarray, min_height: float = 0.1) -> List[Dict[str, Any]]:
        """
        Detect 3D obstacles using depth information
        
        Args:
            depth_map: Depth map from stereo vision
            min_height: Minimum obstacle height in meters
            
        Returns:
            List of detected obstacles with 3D information
        """
        try:
            obstacles = []
            height, width = depth_map.shape
            
            # Create height map (assuming camera is mounted at head height)
            camera_height = 1.7  # meters (typical head height)
            height_map = camera_height - depth_map
            
            # Find obstacles above minimum height
            obstacle_mask = height_map > min_height
            
            # Find connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                obstacle_mask.astype(np.uint8), connectivity=8
            )
            
            # Process each obstacle
            for i in range(1, num_labels):  # Skip background (label 0)
                x, y, w, h, area = stats[i]
                
                if area < 100:  # Minimum obstacle size
                    continue
                
                # Get obstacle region
                obstacle_region = depth_map[y:y+h, x:x+w]
                obstacle_heights = height_map[y:y+h, x:x+w]
                
                # Calculate obstacle properties
                min_distance = np.min(obstacle_region)
                max_height = np.max(obstacle_heights)
                center_distance = depth_map[y + h//2, x + w//2]
                
                # Calculate 3D position
                center_x = x + w//2
                center_y = y + h//2
                
                # Convert to 3D coordinates
                Z = center_distance
                X = (center_x - self.left_camera_matrix[0, 2]) * Z / self.left_camera_matrix[0, 0]
                Y = (center_y - self.left_camera_matrix[1, 2]) * Z / self.left_camera_matrix[1, 1]
                
                obstacle = {
                    'id': i,
                    'position_3d': [X, Y, Z],
                    'distance': center_distance,
                    'height': max_height,
                    'width': w,
                    'area': area,
                    'bbox_2d': [x, y, w, h],
                    'min_distance': min_distance,
                    'is_dangerous': center_distance < 1.0  # Less than 1 meter
                }
                
                obstacles.append(obstacle)
            
            return obstacles
            
        except Exception as e:
            self.logger.error(f"Error detecting 3D obstacles: {e}")
            return []
    
    def calculate_safe_path_3d(self, obstacles: List[Dict[str, Any]], 
                              current_position: List[float] = None) -> Dict[str, Any]:
        """
        Calculate safe 3D navigation path
        
        Args:
            obstacles: List of 3D obstacles
            current_position: Current 3D position [x, y, z]
            
        Returns:
            Safe path information
        """
        try:
            if current_position is None:
                current_position = [0, 0, 0]  # Assume at origin
            
            # Create 3D occupancy grid
            grid_size = 0.1  # 10cm grid cells
            grid_range = 5.0  # 5 meter range
            
            grid_x = int(grid_range / grid_size)
            grid_y = int(grid_range / grid_size)
            grid_z = int(grid_range / grid_size)
            
            occupancy_grid = np.zeros((grid_x, grid_y, grid_z), dtype=bool)
            
            # Mark obstacles in grid
            for obstacle in obstacles:
                pos = obstacle['position_3d']
                grid_pos = [
                    int((pos[0] + grid_range/2) / grid_size),
                    int((pos[1] + grid_range/2) / grid_size),
                    int(pos[2] / grid_size)
                ]
                
                # Mark obstacle region
                if (0 <= grid_pos[0] < grid_x and 
                    0 <= grid_pos[1] < grid_y and 
                    0 <= grid_pos[2] < grid_z):
                    occupancy_grid[grid_pos[0], grid_pos[1], grid_pos[2]] = True
            
            # Find safe directions
            safe_directions = []
            current_grid = [
                int((current_position[0] + grid_range/2) / grid_size),
                int((current_position[1] + grid_range/2) / grid_size),
                int(current_position[2] / grid_size)
            ]
            
            # Check 8 directions in horizontal plane
            directions = [
                (1, 0), (1, 1), (0, 1), (-1, 1),
                (-1, 0), (-1, -1), (0, -1), (1, -1)
            ]
            
            for dx, dy in directions:
                is_safe = True
                for step in range(1, 10):  # Check 1 meter ahead
                    check_x = current_grid[0] + dx * step
                    check_y = current_grid[1] + dy * step
                    check_z = current_grid[2]
                    
                    if (0 <= check_x < grid_x and 
                        0 <= check_y < grid_y and 
                        0 <= check_z < grid_z):
                        if occupancy_grid[check_x, check_y, check_z]:
                            is_safe = False
                            break
                    else:
                        is_safe = False
                        break
                
                if is_safe:
                    angle = np.arctan2(dy, dx)
                    safe_directions.append(angle)
            
            # Determine best direction
            if safe_directions:
                # Prefer forward direction (0 degrees)
                best_direction = min(safe_directions, key=lambda x: abs(x))
                confidence = 1.0
            else:
                best_direction = 0.0
                confidence = 0.0
            
            return {
                'safe_direction': best_direction,
                'confidence': confidence,
                'safe_directions': safe_directions,
                'obstacle_count': len(obstacles),
                'closest_obstacle': min(obstacles, key=lambda x: x['distance']) if obstacles else None
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating 3D safe path: {e}")
            return {
                'safe_direction': 0.0,
                'confidence': 0.0,
                'safe_directions': [],
                'obstacle_count': 0,
                'closest_obstacle': None
            }
    
    def calibrate_stereo(self, calibration_images: List[Tuple[np.ndarray, np.ndarray]], 
                        chessboard_size: Tuple[int, int] = (9, 6),
                        square_size: float = 0.025) -> bool:
        """
        Calibrate stereo cameras using chessboard pattern
        
        Args:
            calibration_images: List of (left_image, right_image) pairs
            chessboard_size: Number of internal corners (width, height)
            square_size: Size of chessboard squares in meters
            
        Returns:
            True if calibration successful
        """
        try:
            self.logger.info("Starting stereo calibration...")
            
            # Prepare calibration data
            obj_points = []
            left_img_points = []
            right_img_points = []
            
            # Create object points
            objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
            objp *= square_size
            
            # Process calibration images
            for left_img, right_img in calibration_images:
                # Convert to grayscale
                left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
                right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
                
                # Find chessboard corners
                left_ret, left_corners = cv2.findChessboardCorners(left_gray, chessboard_size, None)
                right_ret, right_corners = cv2.findChessboardCorners(right_gray, chessboard_size, None)
                
                if left_ret and right_ret:
                    # Refine corners
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    left_corners = cv2.cornerSubPix(left_gray, left_corners, (11, 11), (-1, -1), criteria)
                    right_corners = cv2.cornerSubPix(right_gray, right_corners, (11, 11), (-1, -1), criteria)
                    
                    obj_points.append(objp)
                    left_img_points.append(left_corners)
                    right_img_points.append(right_corners)
            
            if len(obj_points) < 5:
                self.logger.error("Not enough calibration images")
                return False
            
            # Calibrate cameras
            left_ret, left_mtx, left_dist, left_rvecs, left_tvecs = cv2.calibrateCamera(
                obj_points, left_img_points, left_gray.shape[::-1], None, None
            )
            
            right_ret, right_mtx, right_dist, right_rvecs, right_tvecs = cv2.calibrateCamera(
                obj_points, right_img_points, right_gray.shape[::-1], None, None
            )
            
            # Stereo calibration
            ret, left_mtx, left_dist, right_mtx, right_dist, R, T, E, F = cv2.stereoCalibrate(
                obj_points, left_img_points, right_img_points,
                left_mtx, left_dist, right_mtx, right_dist,
                left_gray.shape[::-1], flags=cv2.CALIB_FIX_INTRINSIC
            )
            
            # Save calibration data
            self.calibration_data = {
                'left_camera_matrix': left_mtx,
                'right_camera_matrix': right_mtx,
                'left_distortion': left_dist,
                'right_distortion': right_dist,
                'rotation_matrix': R,
                'translation_vector': T,
                'essential_matrix': E,
                'fundamental_matrix': F,
                'calibration_error': ret
            }
            
            # Save to file
            calibration_path = Path(self.calibration_file)
            calibration_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(calibration_path, 'wb') as f:
                pickle.dump(self.calibration_data, f)
            
            # Update current calibration
            self.left_camera_matrix = left_mtx
            self.right_camera_matrix = right_mtx
            self.left_distortion = left_dist
            self.right_distortion = right_dist
            self.rotation_matrix = R
            self.translation_vector = T
            self.essential_matrix = E
            self.fundamental_matrix = F
            
            self.is_calibrated = True
            
            self.logger.info(f"Stereo calibration completed. Error: {ret:.4f}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during stereo calibration: {e}")
            return False
    
    def _update_fps(self):
        """Update FPS counter"""
        current_time = time.time()
        self.fps_counter += 1
        
        if current_time - self.last_fps_time >= 1.0:
            fps = self.fps_counter / (current_time - self.last_fps_time)
            self.logger.debug(f"Stereo vision FPS: {fps:.1f}")
            self.fps_counter = 0
            self.last_fps_time = current_time
    
    def get_status(self) -> Dict[str, Any]:
        """Get stereo vision system status"""
        return {
            'is_calibrated': self.is_calibrated,
            'camera_separation': self.camera_separation,
            'focal_length': self.focal_length,
            'min_depth': self.min_depth,
            'max_depth': self.max_depth,
            'calibration_file': self.calibration_file
        } 