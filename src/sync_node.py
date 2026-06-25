#!/usr/bin/env python3
"""
Sync Node: Directly pairs camera images with LiDAR point clouds by timestamp.

Instead of relying on a separate image publisher node, this node:
  1. Scans an image directory at startup and parses all filenames (nanosecond
     epoch timestamps) into a sorted list.
  2. Subscribes to the LiDAR point cloud topic (/cloud_registered from FAST-LIO).
  3. For each incoming LiDAR message, finds the image whose filename timestamp
     is closest to the LiDAR message timestamp, loads it from disk, and publishes
     a SyncedSensorData message containing the image, camera_info, and pointcloud.
"""
import bisect
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from std_msgs.msg import Header
from builtin_interfaces.msg import Time
from cv_bridge import CvBridge
import cv2
from pathlib import Path
from inspection_grounding.msg import SyncedSensorData


class SyncNode(Node):
    def __init__(self):
        super().__init__('sync_node')

        # --- Parameters ---
        self.declare_parameter('image_dir', '/home/robot/fastlio_ws/data/2026-06-11-HKUMTR/camera/right')
        self.declare_parameter('camera_frame_id', 'camera_link')
        # Undistorted intrinsics (right camera, scaled to 1520x2016 half-res)
        self.declare_parameter('fx', 597.3843593065015)
        self.declare_parameter('fy', 597.4426138023167)
        self.declare_parameter('cx', 774.8614840838852)
        self.declare_parameter('cy', 1013.172720601387)

        image_dir = Path(self.get_parameter('image_dir').value)
        self.camera_frame_id = self.get_parameter('camera_frame_id').value
        fx = self.get_parameter('fx').value
        fy = self.get_parameter('fy').value
        cx = self.get_parameter('cx').value
        cy = self.get_parameter('cy').value

        # --- Build sorted image timestamp list ---
        self.image_files = sorted(list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg")))
        if not self.image_files:
            self.get_logger().error(f"No images found in {image_dir}")
            raise RuntimeError("No images found")

        # Sorted list of (timestamp_ns, filepath) for binary search
        self.image_timestamps = []
        for img_path in self.image_files:
            ns = self._parse_timestamp_ns(img_path)
            if ns is not None:
                self.image_timestamps.append((ns, img_path))

        # Sort by timestamp
        self.image_timestamps.sort(key=lambda x: x[0])
        self._sorted_ns = [t[0] for t in self.image_timestamps]

        self.get_logger().info(
            f"Loaded {len(self.image_timestamps)} images from {image_dir}. "
            f"Timestamp range: {self._sorted_ns[0]} to {self._sorted_ns[-1]} ns"
        )

        # --- Camera Info (static, undistorted) ---
        self.camera_info = CameraInfo()
        self.camera_info.distortion_model = 'plumb_bob'
        self.camera_info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.camera_info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        self.camera_info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        self.camera_info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        self.camera_info.header.frame_id = self.camera_frame_id

        self.bridge = CvBridge()

        # --- Subscriber: LiDAR point cloud from FAST-LIO ---
        # /cloud_registered uses BEST_EFFORT QoS from FAST-LIO
        lidar_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.sub_pc = self.create_subscription(
            PointCloud2,
            '/cloud_registered_body',
            self.lidar_callback,
            lidar_qos
        )

        # --- Publishers ---
        self.pub_synced = self.create_publisher(SyncedSensorData, '/synced_sensor_data', 10)
        # Also publish the image as a standalone topic for visualization
        self.pub_image = self.create_publisher(Image, '/synced_image', 10)

        self.get_logger().info("Sync Node initialized. Waiting for LiDAR data...")

    def _parse_timestamp_ns(self, img_path: Path):
        """Parse filename like 1781168047875925000.jpg as nanoseconds since epoch."""
        try:
            return int(img_path.stem)
        except Exception:
            return None

    def _time_from_ns(self, ns: int) -> Time:
        """Convert nanoseconds to builtin_interfaces.msg.Time."""
        t = Time()
        t.sec = int(ns // 1_000_000_000)
        t.nanosec = int(ns % 1_000_000_000)
        return t

    def _find_closest_image(self, target_ns: int):
        """
        Use binary search to find the image with timestamp closest to target_ns.
        Returns (timestamp_ns, filepath) or None.
        """
        if not self._sorted_ns:
            return None

        idx = bisect.bisect_left(self._sorted_ns, target_ns)

        # Edge cases: target is before first or after last
        if idx == 0:
            return self.image_timestamps[0]
        if idx == len(self._sorted_ns):
            return self.image_timestamps[-1]

        # Compare neighbors to find closest
        before = self._sorted_ns[idx - 1]
        after = self._sorted_ns[idx]
        if abs(target_ns - before) <= abs(target_ns - after):
            return self.image_timestamps[idx - 1]
        else:
            return self.image_timestamps[idx]

    def lidar_callback(self, pc_msg):
        """For each LiDAR message, find the closest image by timestamp and publish synced data."""
        # Convert LiDAR header timestamp to nanoseconds
        lidar_ns = pc_msg.header.stamp.sec * 1_000_000_000 + pc_msg.header.stamp.nanosec

        # Find closest image
        result = self._find_closest_image(lidar_ns)
        if result is None:
            self.get_logger().warn("No images available for matching.")
            return

        img_ns, img_path = result

        try:
            cv_image = cv2.imread(str(img_path))
            if cv_image is None:
                self.get_logger().error(f"Failed to load image: {img_path}")
                return

            # Create Image message
            img_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            img_time = self._time_from_ns(img_ns)
            img_msg.header.stamp = img_time
            img_msg.header.frame_id = self.camera_frame_id

            # Create CameraInfo message with matching timestamp/dimensions
            self.camera_info.header.stamp = img_time
            self.camera_info.height = cv_image.shape[0]
            self.camera_info.width = cv_image.shape[1]

            # Build and publish SyncedSensorData
            synced_msg = SyncedSensorData()
            synced_msg.header = img_msg.header  # Use image timestamp as reference
            synced_msg.image = img_msg
            synced_msg.camera_info = self.camera_info
            synced_msg.pointcloud = pc_msg

            # time_diff_ms = abs(lidar_ns - img_ns) / 1_000_000
            # self.get_logger().info(
            #     f"Synced: lidar_ns={lidar_ns}, img_ns={img_ns}, "
            #     f"diff={time_diff_ms:.1f}ms, file={img_path.name}"
            # )
            self.pub_synced.publish(synced_msg)
            self.pub_image.publish(img_msg)

        except Exception as e:
            self.get_logger().error(f"Failed to process synced data: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = SyncNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()