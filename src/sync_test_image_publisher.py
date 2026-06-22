#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from builtin_interfaces.msg import Time
from cv_bridge import CvBridge
import cv2
from pathlib import Path
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class SyncTestImagePublisher(Node):
    def __init__(self):
        super().__init__('sync_test_image_publisher')
        
        # Path to your pre-undistorted images
        self.image_dir = Path("/home/robot/fastlio_ws/data/2026-06-22-IWLG/camera/right")
        self.image_files = sorted(list(self.image_dir.glob("*.png")) + list(self.image_dir.glob("*.jpg")))
        self.current_image_idx = 0
        
        self.bridge = CvBridge()
        self.pub_img = self.create_publisher(Image, '/camera/image_undistorted', 10)
        self.pub_info = self.create_publisher(CameraInfo, '/camera/camera_info', 10)
        
        # Subscribe to the RAW LiDAR topic from the bag to trigger image publishing
        lidar_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Subscribe to the RAW LiDAR topic using the new QoS profile
        self.sub_lidar = self.create_subscription(
            PointCloud2, 
            '/livox/lidar', 
            self.lidar_callback, 
            lidar_qos
        )
        
        # Camera Info with actual undistorted intrinsics from calibration.json
        # (right camera, already undistorted before fusion)
        # Calibration is for 3040x4032, but actual images are 1520x2016 (half-res).
        # The undistort script scales intrinsics by 0.5, so we use the scaled values.
        self.camera_info = CameraInfo()
        self.camera_info.distortion_model = 'plumb_bob'
        self.camera_info.d = [0.0, 0.0, 0.0, 0.0, 0.0]  # Undistorted = 0
        self.camera_info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        # Intrinsics from calibration.json (right camera), scaled to 1520x2016
        fx, fy, cx, cy = 597.3843593065015, 597.4426138023167, 774.8614840838852, 1013.172720601387
        self.camera_info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        self.camera_info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        
        self.get_logger().info(f"Loaded {len(self.image_files)} images for sync testing.")

    def _time_from_filename(self, img_path: Path):
        """
        Parse filename like 1781168047875925000.jpg as nanoseconds since epoch.
        Returns builtin_interfaces.msg.Time or None on failure.
        """
        try:
            stem = img_path.stem
            ns = int(stem)
            sec = ns // 1_000_000_000
            nsec = ns % 1_000_000_000
            t = Time()
            t.sec = int(sec)
            t.nanosec = int(nsec)
            return t
        except Exception:
            return None

    def lidar_callback(self, lidar_msg):
        if self.current_image_idx >= len(self.image_files):
            self.current_image_idx = 0 # Loop back for continuous testing
            
        img_path = self.image_files[self.current_image_idx]
        self.current_image_idx += 1
        
        try:
            cv_image = cv2.imread(str(img_path))
            if cv_image is None:
                return
            
            # Update dimensions dynamically based on the actual image
            self.camera_info.height = cv_image.shape[0]
            self.camera_info.width = cv_image.shape[1]
                
            img_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            
            # Use timestamp encoded in filename if available; otherwise fallback to LiDAR stamp
            file_time = self._time_from_filename(img_path)
            if file_time is not None:
                img_msg.header.stamp = file_time
                self.camera_info.header.stamp = file_time
            else:
                img_msg.header = lidar_msg.header
                self.camera_info.header = lidar_msg.header

            img_msg.header.frame_id = 'camera_link'
            self.camera_info.header.frame_id = 'camera_link'
            
            self.pub_img.publish(img_msg)
            self.pub_info.publish(self.camera_info)
            
        except Exception as e:
            self.get_logger().error(f"Failed to publish image: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = SyncTestImagePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()