#!/usr/bin/env python3
"""
Rerun Bridge Node: Visualize ROS 2 topics in Rerun.

Subscribes to:
  - /Laser_map                              (sensor_msgs/PointCloud2, camera_init frame)
  - /pointcloud/segmented_yaml              (sensor_msgs/PointCloud2, camera_init frame)
  - /pointcloud/segmented_yaml_aggregated   (sensor_msgs/PointCloud2, camera_init frame)
  - /synced_image                           (sensor_msgs/Image, camera_link frame)
  - /cloud_registered_body                  (sensor_msgs/PointCloud2, body frame)
  - TF: camera_init -> body, body -> camera_link

Logs everything to a Rerun viewer for 3D visualization.
A pinhole camera model is logged at the camera_link entity so that the
2D image is correctly projected into the 3D scene.

The /cloud_registered_body topic (per-frame LiDAR in the body frame) is
projected into the camera image plane to produce a depth image, which is
logged as a rr.DepthImage under the camera_link pinhole entity.  Rerun
back-projects this depth image into the 3D scene as a colored point cloud.

Requires: pip install rerun-sdk
"""

import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import PointCloud2, Image
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

        # Camera intrinsics for pinhole projection.
        # Defaults match the half-resolution images (1520x2016) published by
        # , with intrinsics scaled by 0.5 from
        # the original calibration.json values.
        self.fx = self.declare_parameter('fx', 597.3843593065015).value
        self.fy = self.declare_parameter('fy', 597.4426138023167).value
        self.cx = self.declare_parameter('cx', 774.8614840838852).value
        self.cy = self.declare_parameter('cy', 1013.172720601387).value
        self.img_width = self.declare_parameter('image_width', 1520).value
        self.img_height = self.declare_parameter('image_height', 2016).value

        # Enable / disable the depth image visualization
        self.enable_depth = self.declare_parameter('enable_depth', True).value

        # Track actual image dimensions from received images
        self._actual_img_w = self.img_width
        self._actual_img_h = self.img_height

        # QoS: reliable to match FAST-LIO and fusion_node publishers
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
        # Image subscriber — listens to the standalone image topic published
        # by sync_node.py (timestamp-matched to LiDAR data)
        self.sub_image = self.create_subscription(
            Image, '/synced_image', self.image_callback, 10
        )

        # Per-frame LiDAR in body frame — used to compute the depth image
        # by projecting points into the camera plane.  BEST_EFFORT matches
        # FAST-LIO's /cloud_registered_body publisher.
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.sub_cloud_body = self.create_subscription(
            PointCloud2, '/cloud_registered_body',
            self.cloud_body_callback, sensor_qos
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

    def _get_pinhole(self, width, height):
        """Build a Pinhole component for the given image dimensions.

        Intrinsics are scaled proportionally if dimensions differ from
        the parameter defaults. Uses camera_xyz=RDF (X=Right, Y=Down,
        Z=Forward) optical convention.
        """
        scale_x = width / self.img_width if self.img_width > 0 else 1.0
        scale_y = height / self.img_height if self.img_height > 0 else 1.0
        fx = self.fx * scale_x
        fy = self.fy * scale_y
        cx = self.cx * scale_x
        cy = self.cy * scale_y

        intrinsics = np.array([
            [fx,  0.0, cx ],
            [0.0, fy,  cy ],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        return rr.Pinhole(
            image_from_camera=intrinsics,
            resolution=[width, height],
            camera_xyz=rr.ViewCoordinates.RDF,
        )

    @staticmethod
    def _ros_image_to_numpy(msg):
        """Convert a sensor_msgs/Image to a numpy array.

        Returns (array, color_model) where color_model is one of
        'BGR', 'RGB', 'BGRA', 'RGBA', 'L', or None if unsupported.
        """
        enc = msg.encoding.lower()
        try:
            if enc in ('bgr8', 'rgb8'):
                arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                    msg.height, msg.width, 3)
                return arr, 'BGR' if enc == 'bgr8' else 'RGB'
            elif enc in ('bgra8', 'rgba8'):
                arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                    msg.height, msg.width, 4)
                return arr, 'BGRA' if enc == 'bgra8' else 'RGBA'
            elif enc in ('mono8', '8uc1'):
                arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                    msg.height, msg.width)
                return arr, 'L'
            elif enc in ('mono16', '16uc1'):
                arr = np.frombuffer(msg.data, dtype=np.uint16).reshape(
                    msg.height, msg.width)
                return arr, 'L'
            else:
                return None, None
        except (ValueError, AttributeError):
            return None, None

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

    @staticmethod
    def _quat_to_rotmat(q):
        """Convert a ROS Quaternion (x, y, z, w) to a 3×3 rotation matrix."""
        x, y, z, w = q.x, q.y, q.z, q.w
        return np.array([
            [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
            [    2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
            [    2*x*z - 2*y*w,     2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y],
        ], dtype=np.float64)

    def _lookup_transform_matrix(self, target_frame, source_frame, stamp):
        """Look up a TF transform and return it as a 4×4 homogeneous matrix.

        Args:
            target_frame: destination frame
            source_frame: source frame
            stamp: ROS timestamp (builtin_interfaces/Time)

        Returns:
            (4, 4) numpy array, or None if the transform is unavailable.
        """
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                target_frame, source_frame,
                rclpy.time.Time.from_msg(stamp),
                timeout=rclpy.duration.Duration(seconds=0.1),
            )
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException,
                tf2_ros.ConnectivityException):
            return None

        t = tf_msg.transform.translation
        q = tf_msg.transform.rotation
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self._quat_to_rotmat(q)
        T[:3, 3] = [t.x, t.y, t.z]
        return T

    @staticmethod
    def _transform_points(points_xyz, T):
        """Transform (N, 3) points using a 4×4 homogeneous matrix."""
        if points_xyz.size == 0:
            return points_xyz
        ones = np.ones((points_xyz.shape[0], 1), dtype=np.float64)
        pts_h = np.hstack((points_xyz.astype(np.float64), ones))
        return (T @ pts_h.T).T[:, :3]

    def _project_to_depth_image(self, pts_cam, width, height):
        """Project camera-frame points to a 2D depth image (H, W) float32.

        For each point with z > 0 that falls inside the image bounds, the
        corresponding pixel is set to the point's depth (z).  When multiple
        points map to the same pixel, the nearest (smallest z) is kept.

        Args:
            pts_cam: (N, 3) points in the camera optical frame (RDF)
            width: image width in pixels
            height: image height in pixels

        Returns:
            (height, width) float32 array, 0 where no point projected.
        """
        depth_img = np.zeros((height, width), dtype=np.float32)
        if pts_cam.shape[0] == 0:
            return depth_img

        # Scale intrinsics to match the actual image dimensions, mirroring
        # the logic in _get_pinhole().  The stored fx/fy/cx/cy are for the
        # parameter-default resolution (img_width × img_height); if the
        # actual image is larger (e.g. full-res 3040×4032 vs half-res
        # 1520×2016), the intrinsics must be scaled proportionally so that
        # projection and Rerun's back-projection stay consistent.
        scale_x = width / self.img_width if self.img_width > 0 else 1.0
        scale_y = height / self.img_height if self.img_height > 0 else 1.0
        fx = self.fx * scale_x
        fy = self.fy * scale_y
        cx = self.cx * scale_x
        cy = self.cy * scale_y

        z = pts_cam[:, 2]
        valid = z > 1e-6
        if not np.any(valid):
            return depth_img

        x = pts_cam[valid, 0]
        y = pts_cam[valid, 1]
        z = z[valid]

        u = fx * (x / z) + cx
        v = fy * (y / z) + cy

        in_bounds = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        if not np.any(in_bounds):
            return depth_img

        u = u[in_bounds]
        v = v[in_bounds]
        z = z[in_bounds]

        # Round to nearest pixel and clip to valid range [0, dim-1] to
        # guard against round() pushing a value just under width/height
        # up to exactly width/height (out of bounds).
        u_idx = np.clip(np.round(u).astype(np.int64), 0, width - 1)
        v_idx = np.clip(np.round(v).astype(np.int64), 0, height - 1)

        # Keep the nearest depth when multiple points hit the same pixel:
        # sort by depth ascending, then scatter (first write wins).
        order = np.argsort(z)
        u_idx = u_idx[order]
        v_idx = v_idx[order]
        z_sorted = z[order].astype(np.float32)

        # Only write to pixels that are still zero (uninitialised)
        mask = depth_img[v_idx, u_idx] == 0.0
        depth_img[v_idx[mask], u_idx[mask]] = z_sorted[mask]

        return depth_img

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
                [gray, gray, gray, np.full_like(gray, 128)], axis=1
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
    # Per-frame LiDAR → depth image callback
    # ------------------------------------------------------------------

    def cloud_body_callback(self, msg):
        """Project /cloud_registered_body into the camera and log a depth image.

        The LiDAR points (in the body frame) are transformed to the camera
        optical frame via TF, then projected to the image plane.  The
        resulting depth image is logged as a rr.DepthImage under the
        camera_link pinhole entity, so Rerun back-projects it into 3D.
        """
        if not self.enable_depth:
            return

        xyz = self._extract_xyz(msg)
        if xyz.shape[0] == 0:
            return

        stamp = msg.header.stamp
        source_frame = msg.header.frame_id or 'body'

        # body → camera_link
        T_cam_body = self._lookup_transform_matrix(
            'camera_link', source_frame, stamp
        )
        if T_cam_body is None:
            return

        pts_cam = self._transform_points(xyz, T_cam_body)
        depth_img = self._project_to_depth_image(
            pts_cam, self._actual_img_w, self._actual_img_h
        )

        if not np.any(depth_img > 0):
            return

        self._set_time(stamp)
        rr.log(
            f"{self.base_path}/camera_init/body/camera_link/depth",
            rr.DepthImage(
                depth_img,
                meter=1.0,
                colormap="viridis",
                depth_range=[
                    float(depth_img[depth_img > 0].min()),
                    float(depth_img[depth_img > 0].max()),
                ],
            ),
        )

    # ------------------------------------------------------------------
    # Image callback
    # ------------------------------------------------------------------

    def image_callback(self, msg):
        """Log camera image to Rerun under the camera_link pinhole entity."""
        self._set_time(msg.header.stamp)

        # Track actual image dimensions for tf_callback to use
        if msg.width != self._actual_img_w or msg.height != self._actual_img_h:
            self._actual_img_w = msg.width
            self._actual_img_h = msg.height
            scale_x = msg.width / self.img_width if self.img_width > 0 else 1.0
            scale_y = msg.height / self.img_height if self.img_height > 0 else 1.0
            self.get_logger().info(
                f"Image size {msg.width}x{msg.height} differs from default "
                f"{self.img_width}x{self.img_height}, scaling intrinsics by "
                f"({scale_x:.3f}, {scale_y:.3f})"
            )

        img_array, color_model = self._ros_image_to_numpy(msg)
        if img_array is None:
            self.get_logger().warn(
                f"Unsupported image encoding: {msg.encoding}", throttle_duration_sec=5.0
            )
            return
        rr.log(
            f"{self.base_path}/camera_init/body/camera_link/image",
            rr.Image(image=img_array, color_model=color_model),
        )

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
        # camera_init: root frame — log an axis gizmo at the origin
        rr.log(f"{self.base_path}/camera_init", rr.Transform3D())
        self._log_axes(f"{self.base_path}/camera_init")

        # camera_init -> body (published by FAST-LIO)
        try:
            tf = self.tf_buffer.lookup_transform(
                'camera_init', 'body', rclpy.time.Time()
            )
            rr.set_time("ros_time", timestamp=self._stamp_to_ns(tf.header.stamp) / 1e9)
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
        # Log Transform3D AND Pinhole together so they always coexist at
        # the same entity.  Using static=True for Pinhole separately would
        # get shadowed by the non-static Transform3D logged here.
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
                ),
                self._get_pinhole(self._actual_img_w, self._actual_img_h),
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