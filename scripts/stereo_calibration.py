#!/usr/bin/env python3
"""
Stereo Camera Calibration Tool
Calibrates dual IMX219 cameras for accurate 3D depth perception
"""

import cv2
import numpy as np
import sys
import os
import time
from pathlib import Path
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from hardware.camera.camera_manager import CameraManager
from core.vision.stereo_vision import StereoVision


class StereoCalibrator:
    """Stereo camera calibration tool"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the stereo calibrator"""
        import yaml
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize cameras
        self.left_camera = CameraManager(self.config['camera'])
        self.right_camera = CameraManager(self.config['camera'])
        self.right_camera.device_id = 1  # Right camera
        
        # Initialize stereo vision
        self.stereo_vision = StereoVision(self.config.get('stereo_vision', {}))
        
        # Calibration parameters
        self.chessboard_size = (9, 6)  # Internal corners
        self.square_size = 0.025  # 2.5cm squares
        self.required_images = 20  # Number of calibration images needed
        
        # Calibration data
        self.calibration_images = []
        self.calibration_points = []
        
    def start_cameras(self):
        """Start both cameras"""
        print("Starting stereo cameras...")
        
        try:
            self.left_camera.start()
            time.sleep(1)
            self.right_camera.start()
            time.sleep(1)
            
            print("✓ Both cameras started successfully")
            return True
            
        except Exception as e:
            print(f"✗ Error starting cameras: {e}")
            return False
    
    def stop_cameras(self):
        """Stop both cameras"""
        print("Stopping cameras...")
        self.left_camera.stop()
        self.right_camera.stop()
        print("✓ Cameras stopped")
    
    def capture_calibration_frame(self) -> bool:
        """Capture a calibration frame from both cameras"""
        try:
            # Get frames from both cameras
            left_frame = self.left_camera.get_frame()
            right_frame = self.right_camera.get_frame()
            
            if left_frame is None or right_frame is None:
                print("✗ Could not capture frames from both cameras")
                return False
            
            # Convert to grayscale for corner detection
            left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
            
            # Find chessboard corners
            left_ret, left_corners = cv2.findChessboardCorners(left_gray, self.chessboard_size, None)
            right_ret, right_corners = cv2.findChessboardCorners(right_gray, self.chessboard_size, None)
            
            if left_ret and right_ret:
                # Refine corners
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                left_corners = cv2.cornerSubPix(left_gray, left_corners, (11, 11), (-1, -1), criteria)
                right_corners = cv2.cornerSubPix(right_gray, right_corners, (11, 11), (-1, -1), criteria)
                
                # Draw corners on images
                cv2.drawChessboardCorners(left_frame, self.chessboard_size, left_corners, left_ret)
                cv2.drawChessboardCorners(right_frame, self.chessboard_size, right_corners, right_ret)
                
                # Save calibration images
                timestamp = int(time.time())
                left_filename = f"calibration/left_{timestamp}.jpg"
                right_filename = f"calibration/right_{timestamp}.jpg"
                
                os.makedirs("calibration", exist_ok=True)
                cv2.imwrite(left_filename, left_frame)
                cv2.imwrite(right_filename, right_frame)
                
                # Store calibration data
                self.calibration_images.append((left_frame, right_frame))
                self.calibration_points.append((left_corners, right_corners))
                
                print(f"✓ Captured calibration frame {len(self.calibration_images)}/{self.required_images}")
                return True
            else:
                print("✗ Chessboard not detected in both images")
                return False
                
        except Exception as e:
            print(f"✗ Error capturing calibration frame: {e}")
            return False
    
    def run_calibration(self):
        """Run the complete calibration process"""
        print("=" * 60)
        print("Stereo Camera Calibration Tool")
        print("=" * 60)
        print()
        print("Instructions:")
        print("1. Print a chessboard pattern (9x6 internal corners)")
        print("2. Hold the chessboard at different angles and distances")
        print("3. Press 'c' to capture a frame when chessboard is detected")
        print("4. Press 'q' to quit")
        print("5. Press 's' to start calibration when enough frames are captured")
        print()
        
        if not self.start_cameras():
            return False
        
        try:
            while True:
                # Get current frames
                left_frame = self.left_camera.get_frame()
                right_frame = self.right_camera.get_frame()
                
                if left_frame is not None and right_frame is not None:
                    # Convert to grayscale
                    left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
                    right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
                    
                    # Check for chessboard
                    left_ret, left_corners = cv2.findChessboardCorners(left_gray, self.chessboard_size, None)
                    right_ret, right_corners = cv2.findChessboardCorners(right_gray, self.chessboard_size, None)
                    
                    # Draw status on frames
                    status_text = f"Frames: {len(self.calibration_images)}/{self.required_images}"
                    cv2.putText(left_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(right_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    if left_ret and right_ret:
                        # Draw corners
                        cv2.drawChessboardCorners(left_frame, self.chessboard_size, left_corners, left_ret)
                        cv2.drawChessboardCorners(right_frame, self.chessboard_size, right_corners, right_ret)
                        
                        # Draw detection indicator
                        cv2.putText(left_frame, "CHESSBOARD DETECTED", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(right_frame, "CHESSBOARD DETECTED", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        cv2.putText(left_frame, "NO CHESSBOARD", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv2.putText(right_frame, "NO CHESSBOARD", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    # Display frames
                    cv2.imshow('Left Camera', left_frame)
                    cv2.imshow('Right Camera', right_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    if left_ret and right_ret:
                        self.capture_calibration_frame()
                    else:
                        print("✗ Cannot capture - chessboard not detected")
                elif key == ord('s'):
                    if len(self.calibration_images) >= 5:
                        print("Starting calibration...")
                        if self.perform_calibration():
                            print("✓ Calibration completed successfully!")
                            break
                        else:
                            print("✗ Calibration failed")
                    else:
                        print(f"✗ Need at least 5 frames, have {len(self.calibration_images)}")
        
        finally:
            cv2.destroyAllWindows()
            self.stop_cameras()
        
        return True
    
    def perform_calibration(self) -> bool:
        """Perform stereo calibration with captured images"""
        try:
            print(f"Performing calibration with {len(self.calibration_images)} images...")
            
            # Prepare calibration data
            obj_points = []
            left_img_points = []
            right_img_points = []
            
            # Create object points
            objp = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1, 2)
            objp *= self.square_size
            
            # Process calibration images
            for left_frame, right_frame in self.calibration_images:
                # Convert to grayscale
                left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
                right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
                
                # Find chessboard corners
                left_ret, left_corners = cv2.findChessboardCorners(left_gray, self.chessboard_size, None)
                right_ret, right_corners = cv2.findChessboardCorners(right_gray, self.chessboard_size, None)
                
                if left_ret and right_ret:
                    # Refine corners
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    left_corners = cv2.cornerSubPix(left_gray, left_corners, (11, 11), (-1, -1), criteria)
                    right_corners = cv2.cornerSubPix(right_gray, right_corners, (11, 11), (-1, -1), criteria)
                    
                    obj_points.append(objp)
                    left_img_points.append(left_corners)
                    right_img_points.append(right_corners)
            
            if len(obj_points) < 5:
                print("✗ Not enough valid calibration images")
                return False
            
            print(f"Using {len(obj_points)} valid calibration images...")
            
            # Calibrate individual cameras
            print("Calibrating left camera...")
            left_ret, left_mtx, left_dist, left_rvecs, left_tvecs = cv2.calibrateCamera(
                obj_points, left_img_points, left_gray.shape[::-1], None, None
            )
            
            print("Calibrating right camera...")
            right_ret, right_mtx, right_dist, right_rvecs, right_tvecs = cv2.calibrateCamera(
                obj_points, right_img_points, right_gray.shape[::-1], None, None
            )
            
            # Stereo calibration
            print("Performing stereo calibration...")
            ret, left_mtx, left_dist, right_mtx, right_dist, R, T, E, F = cv2.stereoCalibrate(
                obj_points, left_img_points, right_img_points,
                left_mtx, left_dist, right_mtx, right_dist,
                left_gray.shape[::-1], flags=cv2.CALIB_FIX_INTRINSIC
            )
            
            print(f"Calibration error: {ret:.4f}")
            
            # Save calibration data
            calibration_data = {
                'left_camera_matrix': left_mtx,
                'right_camera_matrix': right_mtx,
                'left_distortion': left_dist,
                'right_distortion': right_dist,
                'rotation_matrix': R,
                'translation_vector': T,
                'essential_matrix': E,
                'fundamental_matrix': F,
                'calibration_error': ret,
                'calibration_date': time.time(),
                'num_images': len(obj_points)
            }
            
            # Save to file
            calibration_path = Path("data/stereo_calibration.pkl")
            calibration_path.parent.mkdir(parents=True, exist_ok=True)
            
            import pickle
            with open(calibration_path, 'wb') as f:
                pickle.dump(calibration_data, f)
            
            print(f"✓ Calibration data saved to {calibration_path}")
            
            # Test calibration
            self.test_calibration(calibration_data)
            
            return True
            
        except Exception as e:
            print(f"✗ Error during calibration: {e}")
            return False
    
    def test_calibration(self, calibration_data):
        """Test the calibration with current camera feeds"""
        print("Testing calibration...")
        
        try:
            # Get current frames
            left_frame = self.left_camera.get_frame()
            right_frame = self.right_camera.get_frame()
            
            if left_frame is not None and right_frame is not None:
                # Convert to grayscale
                left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
                right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
                
                # Rectify images
                left_rect_map, right_rect_map = cv2.stereoRectify(
                    calibration_data['left_camera_matrix'], calibration_data['left_distortion'],
                    calibration_data['right_camera_matrix'], calibration_data['right_distortion'],
                    left_gray.shape[::-1], calibration_data['rotation_matrix'], 
                    calibration_data['translation_vector'],
                    flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.9
                )[1:3]
                
                # Apply rectification
                left_rect = cv2.remap(left_gray, left_rect_map[0], left_rect_map[1], cv2.INTER_LINEAR)
                right_rect = cv2.remap(right_gray, right_rect_map[0], right_rect_map[1], cv2.INTER_LINEAR)
                
                # Save test images
                cv2.imwrite("calibration/test_left_rectified.jpg", left_rect)
                cv2.imwrite("calibration/test_right_rectified.jpg", right_rect)
                
                print("✓ Rectification test completed")
                print("  Check calibration/test_left_rectified.jpg")
                print("  Check calibration/test_right_rectified.jpg")
                
        except Exception as e:
            print(f"✗ Error testing calibration: {e}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Stereo Camera Calibration Tool')
    parser.add_argument('--config', '-c', default='config/config.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Create calibrator
    calibrator = StereoCalibrator(args.config)
    
    # Run calibration
    success = calibrator.run_calibration()
    
    if success:
        print("✓ Calibration completed successfully!")
        print("You can now use the stereo vision system with accurate depth perception.")
    else:
        print("✗ Calibration failed")
        sys.exit(1)


if __name__ == "__main__":
    main() 