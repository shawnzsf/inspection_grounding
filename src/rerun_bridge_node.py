#!/usr/bin/env python3
"""
Rerun Bridge Node: Visualize ROS 2 topics in Rerun.

Subscribes to:
  - /Laser_map                              (sensor_msgs/PointCloud2, camera_init frame)
  - /pointcloud/segmented_yaml              (sensor_msgs/PointCloud2, camera_init frame)
  - /pointcloud/segmented_yaml_aggregated   (sensor_msgs/PointCloud2, camera_init frame)
  - TF: camera_init -> body, body -> camera_link

Logs everything to a Rerun viewer for 3D visualization.

Requires: pip install rerun-sdk
"""

import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
import tf2_ros
import rerun as rr


def rpy_deg_to_quaternion(roll_deg, pitch_deg, yaw_deg):
    """Convert roll/pitch/yaw in degrees to a quaternion [x, y, z, w]."""
    r = math.radians(roll_deg)
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    return [
        sr * cp * cy - cr * sp * sy,  # x
        cr * sp * cy + sr * cp * sy,  # y
        cr * cp * sy - sr * sp * cy,  # z
        cr * cp * cy + sr * sp * sy,  # w
    ]


class RerunBridgeNode(Node):
    """Bridge node that logs ROS 2 data to Rerun for visualization."""

    def __init__(self):
        super().__init__('rerun_bridge')

        # Initialize Rerun and spawn the viewer
        rr.init("inspection_grounding_rerun", spawn=True)

        # Set world view coordinates (ROS convention: right-hand Z-up)
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

        # Optional leveling rotation (visualization-only, does not affect
        # actual ROS transforms or calculations).  Set via parameter
        # leveling_rpy_deg: [roll, pitch, yaw] in degrees.
        leveling_rpy = self.declare_parameter(
            'leveling_rpy_deg', [0.0, 0.0, 0.0]
        ).value
        if any(abs(a) > 1e-6 for a in leveling_rpy):
            q = rpy_deg_to_quaternion(*leveling_rpy)
            rr.log(
                "world/leveled",
                rr.Transform3D(rotation=rr.Quaternion(xyzw=q)),
                static=True,
            )
            self.base_path = "world/leveled"
            self.get_logger().info(
                f"Leveling rotation applied: RPY={leveling_rpy} deg"
            )
        else:
            self.base_path = "world"

        # TF buffer for looking up transforms
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # QoS: reliable to match FAST-LIO and fusion_yaml_node publishers
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            durability=DurabilityPolicy.VOLATILE
        )

        # Subscribers
        self.sub_laser_map = self.create_subscription(
            PointCloud2, '/Laser_map', self.laser_map_callback, reliable_qos
        )
        self.sub_segmented = self.create_subscription(
            PointCloud2, '/pointcloud/segmented_yaml',
            self.segmented_callback, reliable_qos
        )
        self.sub_aggregated = self.create_subscription(
            PointCloud2, '/pointcloud/segmented_yaml_aggregated',
            self.aggregated_callback, reliable_qos
        )

        # Timer for TF polling (30 Hz)
        self.tf_timer = self.create_timer(1.0 / 30.0, self.tf_callback)

        self.get_logger().info("Rerun bridge started — viewer spawned")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _stamp_to_ns(stamp):
        """Convert builtin_interfaces/Time to integer nanoseconds."""
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    def _set_time(self, stamp):
        """Set the Rerun timeline to the given ROS stamp.

        Rerun's ``timestamp`` parameter expects seconds since Unix epoch
        as a float, so we convert nanoseconds → seconds.
        """
        rr.set_time("ros_time", timestamp=self._stamp_to_ns(stamp) / 1e9)

    def _log_axes(self, path, length=0.5):
        """Log RGB axis arrows at the given entity path.

        Red=X, Green=Y, Blue=Z, each ``length`` meters long.
        """
        origins = np.zeros((3, 3), dtype=np.float32)
        vectors = np.array(
            [[length, 0, 0], [0, length, 0], [0, 0, length]],
            dtype=np.float32
        )
        colors = np.array(
            [[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8
        )
        rr.log(
            f"{path}/axes",
            rr.Arrows3D(
                vectors=vectors,
                origins=origins,
                colors=colors,
            )
        )

    @staticmethod
    def _extract_xyz(msg):
        """Extract XYZ coordinates from a PointCloud2 message as (N, 3) float32."""
        pts = np.array(
            [(p[0], p[1], p[2]) for p in point_cloud2.read_points(
                msg, field_names=('x', 'y', 'z'), skip_nans=True
            )],
            dtype=np.float32
        )
        return pts.reshape(0, 3) if pts.size == 0 else pts

    @staticmethod
    def _extract_xyz_intensity(msg):
        """Extract XYZ + intensity from a PointCloud2 message.

        Returns:
            xyz: (N, 3) float32 array
            intensity: (N,) float32 array, or None if no intensity field
        """
        has_intensity = any(f.name == 'intensity' for f in msg.fields)
        if has_intensity:
            pts = np.array(
                [(p[0], p[1], p[2], p[3]) for p in point_cloud2.read_points(
                    msg, field_names=('x', 'y', 'z', 'intensity'),
                    skip_nans=True
                )],
                dtype=np.float32
            )
            if pts.size == 0:
                return np.zeros((0, 3), dtype=np.float32), None
            return pts[:, :3], pts[:, 3]
        else:
            return RerunBridgeNode._extract_xyz(msg), None

    # ------------------------------------------------------------------
    # Point cloud callbacks
    # ------------------------------------------------------------------

    def laser_map_callback(self, msg):
        """Log /Laser_map point cloud to Rerun with intensity-based coloring."""
        xyz, intensity = self._extract_xyz_intensity(msg)
        if xyz.shape[0] == 0:
            return

        self._set_time(msg.header.stamp)

        if intensity is not None and intensity.size > 0:
            # Map intensity to grayscale RGBA
            i_max = float(intensity.max())
            if i_max < 1e-6:
                i_max = 1.0
            i_norm = np.clip(intensity / i_max, 0.0, 1.0)
            gray = (i_norm * 255).astype(np.uint8)
            colors = np.stack(
                [gray, gray, gray, np.full_like(gray, 255)], axis=1
            )
            rr.log(f"{self.base_path}/camera_init/Laser_map", rr.Points3D(xyz, colors=colors))
        else:
            rr.log(f"{self.base_path}/camera_init/Laser_map", rr.Points3D(xyz))

    def segmented_callback(self, msg):
        """Log /pointcloud/segmented_yaml with green color."""
        xyz = self._extract_xyz(msg)
        if xyz.shape[0] == 0:
            return
        self._set_time(msg.header.stamp)
        rr.log(f"{self.base_path}/camera_init/segmented_yaml", rr.Points3D(
            xyz, colors=[0, 255, 0, 255]
        ))

    def aggregated_callback(self, msg):
        """Log /pointcloud/segmented_yaml_aggregated with yellow color."""
        xyz = self._extract_xyz(msg)
        if xyz.shape[0] == 0:
            return
        self._set_time(msg.header.stamp)
        rr.log(f"{self.base_path}/camera_init/segmented_yaml_aggregated", rr.Points3D(
            xyz, colors=[255, 255, 0, 255]
        ))

    # ------------------------------------------------------------------
    # TF callback
    # ------------------------------------------------------------------

    def tf_callback(self):
        """Poll TF and log camera_init, body, and camera_link frames.

        Visualizes all three coordinate frames as axes gizmos in Rerun:
          - camera_init: root frame at the base path origin
          - body:        camera_init -> body (published by FAST-LIO)
          - camera_link: body -> camera_link (static transform)
        """
        now = self.get_clock().now()
        rr.set_time("ros_time", timestamp=now.nanoseconds / 1e9)

        # camera_init: root frame — log an axis gizmo at the origin
        rr.log(f"{self.base_path}/camera_init", rr.Transform3D())
        self._log_axes(f"{self.base_path}/camera_init")

        # camera_init -> body (published by FAST-LIO)
        try:
            tf = self.tf_buffer.lookup_transform(
                'camera_init', 'body', rclpy.time.Time()
            )
            t = tf.transform.translation
            q = tf.transform.rotation
            rr.log(
                f"{self.base_path}/camera_init/body",
                rr.Transform3D(
                    translation=[t.x, t.y, t.z],
                    rotation=rr.Quaternion(xyzw=[q.x, q.y, q.z, q.w]),
                )
            )
            self._log_axes(f"{self.base_path}/camera_init/body")
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException,
                tf2_ros.ConnectivityException):
            pass

        # body -> camera_link (static transform from launch file)
        try:
            tf = self.tf_buffer.lookup_transform(
                'body', 'camera_link', rclpy.time.Time()
            )
            t = tf.transform.translation
            q = tf.transform.rotation
            rr.log(
                f"{self.base_path}/camera_init/body/camera_link",
                rr.Transform3D(
                    translation=[t.x, t.y, t.z],
                    rotation=rr.Quaternion(xyzw=[q.x, q.y, q.z, q.w]),
                )
            )
            self._log_axes(f"{self.base_path}/camera_init/body/camera_link")
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException,
                tf2_ros.ConnectivityException):
            pass


def main(args=None):
    rclpy.init(args=args)
    node = RerunBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()