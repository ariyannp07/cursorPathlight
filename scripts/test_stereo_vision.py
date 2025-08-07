#!/usr/bin/env python3
"""
Stereo Vision Test Script
Comprehensive testing of dual IMX219 camera setup and 3D depth perception
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


class StereoVisionTester:
    """Comprehensive stereo vision testing system"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the stereo vision tester"""
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
        
        # Test results
        self.test_results = {}
        
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
    
    def test_camera_detection(self):
        """Test if both cameras are detected and accessible"""
        print("\n" + "="*50)
        print("TEST 1: Camera Detection")
        print("="*50)
        
        try:
            # Check available cameras
            available_cameras = []
            for i in range(4):
                try:
                    cap = cv2.VideoCapture(f"/dev/video{i}", cv2.CAP_V4L2)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret:
                            available_cameras.append(f"/dev/video{i}")
                        cap.release()
                except:
                    continue
            
            print(f"Available cameras: {available_cameras}")
            
            if len(available_cameras) >= 2:
                print("✓ At least 2 cameras detected")
                self.test_results['camera_detection'] = True
                return True
            else:
                print(f"✗ Only {len(available_cameras)} camera(s) detected, need at least 2")
                self.test_results['camera_detection'] = False
                return False
                
        except Exception as e:
            print(f"✗ Error in camera detection test: {e}")
            self.test_results['camera_detection'] = False
            return False
    
    def test_camera_synchronization(self):
        """Test if both cameras can capture synchronized frames"""
        print("\n" + "="*50)
        print("TEST 2: Camera Synchronization")
        print("="*50)
        
        try:
            # Capture frames from both cameras
            left_frame = self.left_camera.get_frame()
            right_frame = self.right_camera.get_frame()
            
            if left_frame is not None and right_frame is not None:
                print(f"✓ Both cameras captured frames")
                print(f"  Left frame shape: {left_frame.shape}")
                print(f"  Right frame shape: {right_frame.shape}")
                
                # Check if frames are similar size
                if left_frame.shape == right_frame.shape:
                    print("✓ Frame sizes match")
                    self.test_results['camera_sync'] = True
                    return True
                else:
                    print("✗ Frame sizes don't match")
                    self.test_results['camera_sync'] = False
                    return False
            else:
                print("✗ Could not capture frames from both cameras")
                self.test_results['camera_sync'] = False
                return False
                
        except Exception as e:
            print(f"✗ Error in camera synchronization test: {e}")
            self.test_results['camera_sync'] = False
            return False
    
    def test_stereo_calibration(self):
        """Test stereo calibration status"""
        print("\n" + "="*50)
        print("TEST 3: Stereo Calibration")
        print("="*50)
        
        try:
            status = self.stereo_vision.get_status()
            
            if status['is_calibrated']:
                print("✓ Stereo cameras are calibrated")
                print(f"  Camera separation: {status['camera_separation']}m")
                print(f"  Focal length: {status['focal_length']} pixels")
                print(f"  Depth range: {status['min_depth']}-{status['max_depth']}m")
                self.test_results['stereo_calibration'] = True
                return True
            else:
                print("⚠ Stereo cameras are not calibrated")
                print("  Using default parameters (less accurate)")
                print("  Run stereo_calibration.py to improve accuracy")
                self.test_results['stereo_calibration'] = False
                return False
                
        except Exception as e:
            print(f"✗ Error in stereo calibration test: {e}")
            self.test_results['stereo_calibration'] = False
            return False
    
    def test_depth_computation(self):
        """Test depth map computation"""
        print("\n" + "="*50)
        print("TEST 4: Depth Computation")
        print("="*50)
        
        try:
            # Capture stereo frames
            left_frame = self.left_camera.get_frame()
            right_frame = self.right_camera.get_frame()
            
            if left_frame is None or right_frame is None:
                print("✗ Could not capture stereo frames")
                self.test_results['depth_computation'] = False
                return False
            
            # Compute depth
            start_time = time.time()
            stereo_result = self.stereo_vision.compute_depth(left_frame, right_frame)
            computation_time = time.time() - start_time
            
            depth_map = stereo_result['depth_map']
            disparity_map = stereo_result['disparity_map']
            
            if depth_map is not None and disparity_map is not None:
                print("✓ Depth computation successful")
                print(f"  Computation time: {computation_time:.3f}s")
                print(f"  Depth map shape: {depth_map.shape}")
                print(f"  Depth range: {np.min(depth_map):.2f}-{np.max(depth_map):.2f}m")
                print(f"  Valid depth pixels: {np.sum(depth_map > 0)}/{depth_map.size}")
                
                # Save test images
                os.makedirs("test_output", exist_ok=True)
                cv2.imwrite("test_output/left_frame.jpg", left_frame)
                cv2.imwrite("test_output/right_frame.jpg", right_frame)
                
                # Normalize depth map for visualization
                depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                cv2.imwrite("test_output/depth_map.jpg", depth_vis)
                
                # Normalize disparity map for visualization
                disparity_vis = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                cv2.imwrite("test_output/disparity_map.jpg", disparity_vis)
                
                print("  Test images saved to test_output/")
                
                self.test_results['depth_computation'] = True
                return True
            else:
                print("✗ Depth computation failed")
                self.test_results['depth_computation'] = False
                return False
                
        except Exception as e:
            print(f"✗ Error in depth computation test: {e}")
            self.test_results['depth_computation'] = False
            return False
    
    def test_3d_obstacle_detection(self):
        """Test 3D obstacle detection"""
        print("\n" + "="*50)
        print("TEST 5: 3D Obstacle Detection")
        print("="*50)
        
        try:
            # Capture stereo frames
            left_frame = self.left_camera.get_frame()
            right_frame = self.right_camera.get_frame()
            
            if left_frame is None or right_frame is None:
                print("✗ Could not capture stereo frames")
                self.test_results['obstacle_detection'] = False
                return False
            
            # Compute depth
            stereo_result = self.stereo_vision.compute_depth(left_frame, right_frame)
            depth_map = stereo_result['depth_map']
            
            if depth_map is None:
                print("✗ Could not compute depth map")
                self.test_results['obstacle_detection'] = False
                return False
            
            # Detect 3D obstacles
            obstacles = self.stereo_vision.detect_obstacles_3d(depth_map)
            
            print(f"✓ 3D obstacle detection completed")
            print(f"  Detected {len(obstacles)} obstacles")
            
            if obstacles:
                # Find closest obstacle
                closest = min(obstacles, key=lambda x: x['distance'])
                print(f"  Closest obstacle: {closest['distance']:.2f}m away")
                print(f"  Obstacle height: {closest['height']:.2f}m")
                print(f"  Obstacle position: {closest['position_3d']}")
                
                # Create obstacle visualization
                obstacle_vis = left_frame.copy()
                for obstacle in obstacles:
                    x, y, w, h = obstacle['bbox_2d']
                    distance = obstacle['distance']
                    color = (0, 0, 255) if obstacle['is_dangerous'] else (0, 255, 0)
                    
                    cv2.rectangle(obstacle_vis, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(obstacle_vis, f"{distance:.1f}m", (x, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                cv2.imwrite("test_output/obstacles_detected.jpg", obstacle_vis)
                print("  Obstacle visualization saved to test_output/obstacles_detected.jpg")
            
            self.test_results['obstacle_detection'] = True
            return True
            
        except Exception as e:
            print(f"✗ Error in 3D obstacle detection test: {e}")
            self.test_results['obstacle_detection'] = False
            return False
    
    def test_safe_path_planning(self):
        """Test 3D safe path planning"""
        print("\n" + "="*50)
        print("TEST 6: Safe Path Planning")
        print("="*50)
        
        try:
            # Capture stereo frames
            left_frame = self.left_camera.get_frame()
            right_frame = self.right_camera.get_frame()
            
            if left_frame is None or right_frame is None:
                print("✗ Could not capture stereo frames")
                self.test_results['path_planning'] = False
                return False
            
            # Compute depth and detect obstacles
            stereo_result = self.stereo_vision.compute_depth(left_frame, right_frame)
            depth_map = stereo_result['depth_map']
            obstacles = self.stereo_vision.detect_obstacles_3d(depth_map)
            
            # Calculate safe path
            path_result = self.stereo_vision.calculate_safe_path_3d(obstacles)
            
            print("✓ Safe path planning completed")
            print(f"  Safe direction: {np.degrees(path_result['safe_direction']):.1f}°")
            print(f"  Confidence: {path_result['confidence']:.2f}")
            print(f"  Available safe directions: {len(path_result['safe_directions'])}")
            print(f"  Obstacle count: {path_result['obstacle_count']}")
            
            if path_result['closest_obstacle']:
                obstacle = path_result['closest_obstacle']
                print(f"  Closest obstacle: {obstacle['distance']:.2f}m")
            
            # Create path visualization
            path_vis = left_frame.copy()
            
            # Draw safe direction arrow
            center = (left_frame.shape[1] // 2, left_frame.shape[0] // 2)
            angle = path_result['safe_direction']
            length = 100
            end_x = int(center[0] + length * np.cos(angle))
            end_y = int(center[1] + length * np.sin(angle))
            
            color = (0, 255, 0) if path_result['confidence'] > 0.5 else (0, 255, 255)
            cv2.arrowedLine(path_vis, center, (end_x, end_y), color, 3)
            
            # Add text
            cv2.putText(path_vis, f"Safe: {np.degrees(angle):.0f}°", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(path_vis, f"Conf: {path_result['confidence']:.2f}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            cv2.imwrite("test_output/safe_path.jpg", path_vis)
            print("  Path visualization saved to test_output/safe_path.jpg")
            
            self.test_results['path_planning'] = True
            return True
            
        except Exception as e:
            print(f"✗ Error in safe path planning test: {e}")
            self.test_results['path_planning'] = False
            return False
    
    def test_performance(self):
        """Test stereo vision performance"""
        print("\n" + "="*50)
        print("TEST 7: Performance Testing")
        print("="*50)
        
        try:
            print("Running performance test for 10 seconds...")
            
            start_time = time.time()
            frame_count = 0
            total_depth_time = 0
            total_obstacle_time = 0
            total_path_time = 0
            
            while time.time() - start_time < 10:
                # Capture frames
                left_frame = self.left_camera.get_frame()
                right_frame = self.right_camera.get_frame()
                
                if left_frame is not None and right_frame is not None:
                    frame_count += 1
                    
                    # Test depth computation time
                    depth_start = time.time()
                    stereo_result = self.stereo_vision.compute_depth(left_frame, right_frame)
                    depth_time = time.time() - depth_start
                    total_depth_time += depth_time
                    
                    # Test obstacle detection time
                    if stereo_result['depth_map'] is not None:
                        obstacle_start = time.time()
                        obstacles = self.stereo_vision.detect_obstacles_3d(stereo_result['depth_map'])
                        obstacle_time = time.time() - obstacle_start
                        total_obstacle_time += obstacle_time
                        
                        # Test path planning time
                        path_start = time.time()
                        path_result = self.stereo_vision.calculate_safe_path_3d(obstacles)
                        path_time = time.time() - path_start
                        total_path_time += path_time
                
                time.sleep(0.1)  # 10 FPS target
            
            # Calculate performance metrics
            total_time = time.time() - start_time
            fps = frame_count / total_time
            avg_depth_time = total_depth_time / frame_count if frame_count > 0 else 0
            avg_obstacle_time = total_obstacle_time / frame_count if frame_count > 0 else 0
            avg_path_time = total_path_time / frame_count if frame_count > 0 else 0
            
            print("✓ Performance test completed")
            print(f"  Total frames processed: {frame_count}")
            print(f"  Average FPS: {fps:.1f}")
            print(f"  Average depth computation time: {avg_depth_time*1000:.1f}ms")
            print(f"  Average obstacle detection time: {avg_obstacle_time*1000:.1f}ms")
            print(f"  Average path planning time: {avg_path_time*1000:.1f}ms")
            print(f"  Total pipeline time: {(avg_depth_time + avg_obstacle_time + avg_path_time)*1000:.1f}ms")
            
            # Performance assessment
            if fps >= 10:
                print("  ✓ Performance: EXCELLENT (>10 FPS)")
                performance_score = "excellent"
            elif fps >= 5:
                print("  ✓ Performance: GOOD (5-10 FPS)")
                performance_score = "good"
            elif fps >= 2:
                print("  ⚠ Performance: ACCEPTABLE (2-5 FPS)")
                performance_score = "acceptable"
            else:
                print("  ✗ Performance: POOR (<2 FPS)")
                performance_score = "poor"
            
            self.test_results['performance'] = {
                'fps': fps,
                'avg_depth_time': avg_depth_time,
                'avg_obstacle_time': avg_obstacle_time,
                'avg_path_time': avg_path_time,
                'score': performance_score
            }
            
            return True
            
        except Exception as e:
            print(f"✗ Error in performance test: {e}")
            self.test_results['performance'] = None
            return False
    
    def run_all_tests(self):
        """Run all stereo vision tests"""
        print("=" * 60)
        print("STEREO VISION COMPREHENSIVE TEST")
        print("=" * 60)
        
        if not self.start_cameras():
            print("✗ Failed to start cameras")
            return False
        
        try:
            # Run all tests
            tests = [
                ("Camera Detection", self.test_camera_detection),
                ("Camera Synchronization", self.test_camera_synchronization),
                ("Stereo Calibration", self.test_stereo_calibration),
                ("Depth Computation", self.test_depth_computation),
                ("3D Obstacle Detection", self.test_3d_obstacle_detection),
                ("Safe Path Planning", self.test_safe_path_planning),
                ("Performance", self.test_performance)
            ]
            
            results = []
            
            for test_name, test_func in tests:
                try:
                    result = test_func()
                    results.append((test_name, result))
                except Exception as e:
                    print(f"✗ {test_name} test failed with exception: {e}")
                    results.append((test_name, False))
            
            # Summary
            print("\n" + "=" * 60)
            print("TEST SUMMARY")
            print("=" * 60)
            
            passed = 0
            total = len(results)
            
            for test_name, result in results:
                status = "PASS" if result else "FAIL"
                print(f"{test_name}: {status}")
                if result:
                    passed += 1
            
            print(f"\nOverall: {passed}/{total} tests passed")
            
            if passed == total:
                print("\n🎉 All stereo vision tests passed!")
                print("Your dual IMX219 camera setup is working perfectly.")
            elif passed >= total * 0.7:
                print(f"\n✅ {passed}/{total} tests passed - System is mostly functional")
                print("Check failed tests for specific issues.")
            else:
                print(f"\n⚠ {total - passed} test(s) failed - System needs attention")
                print("Review failed tests and address issues.")
            
            # Save detailed results
            self._save_test_results()
            
            return passed >= total * 0.7  # 70% pass rate is acceptable
            
        finally:
            self.stop_cameras()
    
    def _save_test_results(self):
        """Save detailed test results to file"""
        try:
            import json
            from datetime import datetime
            
            results = {
                'timestamp': datetime.now().isoformat(),
                'test_results': self.test_results,
                'system_info': {
                    'opencv_version': cv2.__version__,
                    'numpy_version': np.__version__
                }
            }
            
            os.makedirs("test_output", exist_ok=True)
            with open("test_output/stereo_vision_test_results.json", 'w') as f:
                json.dump(results, f, indent=2)
            
            print("\nDetailed results saved to test_output/stereo_vision_test_results.json")
            
        except Exception as e:
            print(f"Warning: Could not save test results: {e}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Stereo Vision Test Script')
    parser.add_argument('--config', '-c', default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--test', '-t', choices=['all', 'camera', 'depth', 'obstacles', 'path', 'performance'],
                       default='all', help='Specific test to run')
    
    args = parser.parse_args()
    
    # Create tester
    tester = StereoVisionTester(args.config)
    
    # Run tests
    if args.test == 'all':
        success = tester.run_all_tests()
    else:
        # Run specific test
        if not tester.start_cameras():
            print("✗ Failed to start cameras")
            sys.exit(1)
        
        try:
            if args.test == 'camera':
                success = tester.test_camera_detection() and tester.test_camera_synchronization()
            elif args.test == 'depth':
                success = tester.test_depth_computation()
            elif args.test == 'obstacles':
                success = tester.test_3d_obstacle_detection()
            elif args.test == 'path':
                success = tester.test_safe_path_planning()
            elif args.test == 'performance':
                success = tester.test_performance()
        finally:
            tester.stop_cameras()
    
    if success:
        print("\n✅ Stereo vision test completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Stereo vision test failed!")
        sys.exit(1)


if __name__ == "__main__":
    main() 