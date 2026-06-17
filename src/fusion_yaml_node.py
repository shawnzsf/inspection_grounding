#!/usr/bin/env python3
"""
Fusion YAML Node: 2D Bounding Box to 3D PointCloud Segmentation

This node:
1. Receives timestamped PointCloud2 messages (LiDAR in body frame)
2. Matches them with YAML files containing 2D bounding boxes
3. Projects points to camera image plane using camera intrinsics and TF
4. Segments points that fall within the 2D bbox
5. Publishes segmented cloud and object pose in global frame
6. Saves segmented clouds as PCD files
7. Visualizes bbox in RViz
"""

import os
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Point
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from visualization_msgs.msg import Marker
import tf2_ros
import yaml


def quat_to_rotmat(q):
    """Convert ROS Quaternion (x,y,z,w) to 3x3 rotation matrix."""
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [    2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
        [    2*x*z - 2*y*w,     2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y],
    ])


class FusionYamlNode(Node):
    """Segment PointCloud by matching with 2D bounding box YAMLs."""

    def __init__(self):
        super().__init__('fusion_yaml_node')

        # Parameters
        self.declare_parameter('yaml_dir', '/home/robot/fastlio_ws/masks')
        self.declare_parameter('output_dir', '/home/robot/fastlio_ws/legacy_outputs')
        self.declare_parameter('target_frame', 'camera_init')
        self.declare_parameter('camera_frame', 'camera_link')

        self.yaml_dir = Path(self.get_parameter('yaml_dir').value)
        self.output_dir = Path(self.get_parameter('output_dir').value)
        self.target_frame = self.get_parameter('target_frame').value
        self.camera_frame = self.get_parameter('camera_frame').value

        os.makedirs(self.output_dir, exist_ok=True)

        # Camera intrinsics (undistorted, right camera)
        self.fx = 1194.768718613003
        self.fy = 1194.8852276046334
        self.cx = 1549.7229681677704
        self.cy = 2026.345441202774

        # Bbox visualization depth in camera frame
        self.bbox_vis_depth = 2.0

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Publishers with reliable QoS (republished via timer for RViz)
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE
        )

        self.pub_pc = self.create_publisher(
            PointCloud2, '/pointcloud/segmented_yaml', reliable_qos
        )
        self.pub_pose = self.create_publisher(
            PoseStamped, '/object/global_pose', reliable_qos
        )
        self.pub_bbox = self.create_publisher(
            Marker, '/bbox_marker', reliable_qos
        )

        # Subscribers
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.sub_pc = self.create_subscription(
            PointCloud2, '/cloud_registered_body', self.pc_callback, sensor_qos
        )

        # Cache for republishing (ensures RViz receives data even with single-shot triggers)
        self.last_pc_msg = None
        self.last_pose = None
        self.last_bboxes = []
        self.last_centroid = None
        self.last_stamp = None

        # Unique marker ID counter (so markers accumulate instead of overwriting)
        self._marker_id_counter = 0
        self._last_yaml_name = None

        # Timer to republish at 2 Hz so RViz always has fresh data
        self.republish_timer = self.create_timer(0.5, self._republish_callback)

        self.get_logger().info(
            f"Initialized with frames: target={self.target_frame}, "
            f"camera={self.camera_frame}"
        )

    def _parse_bboxes(self, data):
        """Extract all bounding boxes from YAML data dict.

        Supports two formats:
        - New format with 'bounding_boxes' list (each entry has x_min, y_min,
          x_max, y_max, and optionally 'category')
        - Old format with single 'bounding_box' dict (wrapped into a 1-element
          list for backward compatibility)

        Returns:
            List of (x_min, y_min, x_max, y_max) tuples, or empty list if
            none found.
        """
        bboxes = []

        # New format: bounding_boxes list
        bb_list = data.get('bounding_boxes')
        if isinstance(bb_list, list):
            for entry in bb_list:
                if isinstance(entry, dict):
                    try:
                        bboxes.append((
                            float(entry['x_min']),
                            float(entry['y_min']),
                            float(entry['x_max']),
                            float(entry['y_max']),
                        ))
                    except (KeyError, ValueError):
                        continue

        # Fallback: single bounding_box dict (old format)
        if not bboxes:
            bbox = data.get('bounding_box')
            if isinstance(bbox, dict):
                try:
                    bboxes.append((
                        float(bbox['x_min']),
                        float(bbox['y_min']),
                        float(bbox['x_max']),
                        float(bbox['y_max']),
                    ))
                except (KeyError, ValueError):
                    pass

        return bboxes

    def _lookup_transform_matrix(self, target_frame, source_frame, stamp):
        """
        Get 4x4 transformation matrix from source_frame to target_frame.
        
        Args:
            target_frame: destination frame
            source_frame: source frame
            stamp: ROS time stamp
            
        Returns:
            4x4 numpy array (homogeneous transformation)
        """
        tf_msg = self.tf_buffer.lookup_transform(
            target_frame,
            source_frame,
            Time.from_msg(stamp),
            timeout=rclpy.duration.Duration(seconds=0.5)
        )

        t = tf_msg.transform.translation
        q = tf_msg.transform.rotation
        R = quat_to_rotmat(q)

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = [t.x, t.y, t.z]
        return T

    @staticmethod
    def _transform_points(points_xyz, T):
        """
        Transform 3D points using 4x4 transformation matrix.
        
        Args:
            points_xyz: (N, 3) array of points
            T: (4, 4) transformation matrix
            
        Returns:
            (N, 3) transformed points
        """
        if points_xyz.size == 0:
            return points_xyz

        ones = np.ones((points_xyz.shape[0], 1), dtype=np.float64)
        pts_h = np.hstack((points_xyz.astype(np.float64), ones))
        pts_out = (T @ pts_h.T).T
        return pts_out[:, :3]

    def _read_cloud_xyz(self, msg):
        """Extract (x, y, z) from PointCloud2 message."""
        pts = np.array(
            [(p[0], p[1], p[2]) for p in point_cloud2.read_points(
                msg, field_names=('x', 'y', 'z'), skip_nans=True
            )],
            dtype=np.float64
        )
        return pts.reshape(0, 3) if pts.size == 0 else pts

    def _project_to_image(self, pts_cam):
        """
        Project 3D points (in camera frame) to 2D image plane.
        
        Args:
            pts_cam: (N, 3) points in camera frame
            
        Returns:
            u, v: image coordinates
            valid_z: mask of points with z > 0
        """
        z = pts_cam[:, 2]
        valid = z > 1e-6
        u = np.full(pts_cam.shape[0], np.nan, dtype=np.float64)
        v = np.full(pts_cam.shape[0], np.nan, dtype=np.float64)

        u[valid] = self.fx * (pts_cam[valid, 0] / z[valid]) + self.cx
        v[valid] = self.fy * (pts_cam[valid, 1] / z[valid]) + self.cy

        return u, v, valid

    def _save_pcd(self, pts_xyz, out_path):
        """Save points as ASCII PCD file. Creates parent dirs. No-op if 0 points."""
        pts_xyz = np.asarray(pts_xyz, dtype=np.float32)
        n = pts_xyz.shape[0]
        if n == 0:
            return False

        header = [
            "# .PCD v0.7 - Point Cloud Data file format",
            "VERSION 0.7",
            "FIELDS x y z",
            "SIZE 4 4 4",
            "TYPE F F F",
            "COUNT 1 1 1",
            f"WIDTH {n}",
            "HEIGHT 1",
            "VIEWPOINT 0 0 0 1",
            f"POINTS {n}",
            "DATA ascii",
        ]

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w") as f:
            f.write("\n".join(header) + "\n")
            for x, y, z in pts_xyz:
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
        return True

    # Color palette for multiple bboxes (RGB)
    _BBOX_COLORS = [
        (1.0, 0.2, 0.2),   # red
        (0.2, 1.0, 0.2),   # green
        (0.2, 0.4, 1.0),   # blue
        (1.0, 1.0, 0.2),   # yellow
        (1.0, 0.5, 0.0),   # orange
        (0.8, 0.2, 1.0),   # purple
        (0.2, 1.0, 1.0),   # cyan
        (1.0, 0.2, 0.8),   # magenta
    ]

    def _publish_bbox_marker(self, bbox, stamp, T_target_cam, idx=0):
        """Publish 2D bbox as 3D frustum lines in RViz (in world frame).

        The bbox corners are computed in camera frame at a fixed depth, then
        transformed to the world (target) frame so they stay fixed in space
        instead of moving with the camera.

        Args:
            bbox: (x_min, y_min, x_max, y_max) tuple
            stamp: ROS timestamp
            T_target_cam: 4x4 transform from camera frame to target (world) frame
            idx: color index (cycles through palette)
        """
        x_min, y_min, x_max, y_max = bbox

        def px_to_cam_xyz(u, v):
            """Unproject pixel to 3D point in camera frame at visualization depth."""
            x = (u - self.cx) * self.bbox_vis_depth / self.fx
            y = (v - self.cy) * self.bbox_vis_depth / self.fy
            return np.array([x, y, self.bbox_vis_depth], dtype=np.float64)

        # Bbox corners in camera frame
        corners_cam = np.array([
            px_to_cam_xyz(x_min, y_min),
            px_to_cam_xyz(x_max, y_min),
            px_to_cam_xyz(x_max, y_max),
            px_to_cam_xyz(x_min, y_max),
        ])
        origin_cam = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)

        # Transform to world frame
        corners_world = self._transform_points(corners_cam, T_target_cam)
        origin_world = self._transform_points(origin_cam, T_target_cam)[0]

        def to_point(p):
            return Point(x=float(p[0]), y=float(p[1]), z=float(p[2]))

        p1 = to_point(corners_world[0])
        p2 = to_point(corners_world[1])
        p3 = to_point(corners_world[2])
        p4 = to_point(corners_world[3])
        origin = to_point(origin_world)

        r, g, b = self._BBOX_COLORS[idx % len(self._BBOX_COLORS)]

        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = self.target_frame
        marker.ns = 'bbox'
        marker.id = self._marker_id_counter
        self._marker_id_counter += 1
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.03
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.color.a = 1.0

        # Rectangle edges
        marker.points.extend([p1, p2, p2, p3, p3, p4, p4, p1])
        # Rays from camera origin to corners
        marker.points.extend([origin, p1, origin, p2, origin, p3, origin, p4])

        self.pub_bbox.publish(marker)

    def _delete_bbox_markers(self, stamp, count):
        """Delete stale bbox markers to prevent leftover visuals in RViz.

        Args:
            stamp: ROS timestamp
            count: number of markers to delete (IDs 0..count-1)
        """
        for i in range(count):
            marker = Marker()
            marker.header.stamp = stamp
            marker.header.frame_id = self.camera_frame
            marker.ns = 'bbox'
            marker.id = i
            marker.action = Marker.DELETE
            self.pub_bbox.publish(marker)

    def _publish_centroid_marker(self, centroid, stamp):
        """Publish the centroid as a marker in RViz."""
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = self.target_frame
        marker.ns = 'centroid'
        marker.id = self._marker_id_counter
        self._marker_id_counter += 1
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = centroid[0]
        marker.pose.position.y = centroid[1]
        marker.pose.position.z = centroid[2]
        marker.scale.x = 0.1  # Size of the sphere
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        self.pub_bbox.publish(marker)  # Reusing bbox publisher for centroid marker

    def pc_callback(self, msg):
        """Main callback: match YAML, project, segment, publish."""
        stamp = msg.header.stamp
        stamp_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

        if not self.yaml_dir.exists():
            return

        # Find nearest YAML by timestamp
        candidates = []
        for p in self.yaml_dir.iterdir():
            if not p.is_file() or p.suffix.lower() not in ('.yaml', '.yml'):
                continue
            try:
                candidates.append((p, int(p.stem)))
            except ValueError:
                continue

        if not candidates:
            return

        nearest_path, nearest_ns = min(candidates, key=lambda x: abs(x[1] - stamp_ns))
        if abs(nearest_ns - stamp_ns) > 150_000_000:  # 150 ms tolerance
            return

        filename = nearest_path.name

        # Load YAML
        try:
            with open(nearest_path, 'r') as f:
                data = yaml.safe_load(f)
        except Exception as e:
            self.get_logger().error(f"Failed to read {filename}: {e}")
            return

        bboxes = self._parse_bboxes(data)
        # Filter out invalid bboxes (zero or negative area)
        bboxes = [b for b in bboxes if b[2] > b[0] and b[3] > b[1]]
        if not bboxes:
            self.get_logger().warn(f"No valid bboxes in {filename}")
            return

        try:
            # Read cloud
            pts_src = self._read_cloud_xyz(msg)
            if pts_src.shape[0] == 0:
                return

            # Project to camera frame
            source_frame = msg.header.frame_id
            T_cam_src = self._lookup_transform_matrix(
                self.camera_frame, source_frame, stamp
            )
            pts_cam = self._transform_points(pts_src, T_cam_src)

            # Project to image plane
            u, v, valid_z = self._project_to_image(pts_cam)
            valid_proj = valid_z & np.isfinite(u) & np.isfinite(v)

            # Per-bbox segmentation: compute mask for each bbox individually
            per_bbox_masks = []
            per_bbox_pts = []
            for i, (x_min, y_min, x_max, y_max) in enumerate(bboxes):
                bbox_mask = valid_proj & (u >= x_min) & (u <= x_max) & (v >= y_min) & (v <= y_max)
                bbox_pts = pts_src[bbox_mask]
                per_bbox_masks.append(bbox_mask)
                per_bbox_pts.append(bbox_pts)

            # Union mask for combined visualization
            mask = np.zeros(pts_src.shape[0], dtype=bool)
            for m in per_bbox_masks:
                mask |= m

            segmented_pts = pts_src[mask]
            if segmented_pts.shape[0] == 0:
                # Log per-bbox point counts for debugging
                counts = [len(p) for p in per_bbox_pts]
                self.get_logger().warn(
                    f"No points in any bbox: {filename} "
                    f"(per-bbox counts: {counts})"
                )
                return

            # Publish bbox markers in world frame (accumulate with unique IDs)
            # Only publish once per YAML to avoid duplicate markers
            is_new_yaml = (filename != self._last_yaml_name)
            if is_new_yaml:
                try:
                    T_target_cam = self._lookup_transform_matrix(
                        self.target_frame, self.camera_frame, stamp
                    )
                    for i, bbox in enumerate(bboxes):
                        self._publish_bbox_marker(bbox, stamp, T_target_cam, idx=i)
                except Exception as e:
                    self.get_logger().warn(f"Failed to publish bbox markers: {e}")

            # Publish centroid marker for RViz visualization
            centroid_src = segmented_pts.mean(axis=0)
            # Transform to target frame for publishing
            if source_frame != self.target_frame:
                T_target_src = self._lookup_transform_matrix(
                    self.target_frame, source_frame, stamp
                )
                segmented_pts_pub = self._transform_points(segmented_pts, T_target_src)
                pub_frame = self.target_frame
                centroid_pub = self._transform_points(centroid_src.reshape(1, 3), T_target_src)[0]
            else:
                segmented_pts_pub = segmented_pts
                pub_frame = source_frame
                centroid_pub = centroid_src

            # Publish object centroid pose
            pose = PoseStamped(header=Header(stamp=stamp, frame_id=self.target_frame))
            pose.pose.position.x = float(centroid_pub[0])
            pose.pose.position.y = float(centroid_pub[1])
            pose.pose.position.z = float(centroid_pub[2])
            pose.pose.orientation.w = 1.0
            self.pub_pose.publish(pose)

            # Publish segmented PointCloud2 for visualization
            pc_msg = None
            try:
                pc_header = Header(stamp=stamp, frame_id=pub_frame)
                cloud_out = [[float(x), float(y), float(z)] for x, y, z in segmented_pts_pub]
                pc_msg = point_cloud2.create_cloud_xyz32(pc_header, cloud_out)
                self.pub_pc.publish(pc_msg)
            except Exception as e:
                self.get_logger().warn(f"Failed to publish segmented PointCloud2: {e}")

            # Publish centroid marker (only for new YAML to avoid duplicates)
            if is_new_yaml:
                self._publish_centroid_marker(centroid_pub, stamp)

            # Cache results for republishing (only if pc_msg was successfully created)
            if pc_msg is not None:
                self.last_pc_msg = pc_msg
            self.last_pose = pose
            self.last_bboxes = bboxes
            self.last_centroid = centroid_pub
            self.last_stamp = stamp
            self._last_yaml_name = filename

            # Save combined PCD (union of all bboxes)
            pcd_path = self.output_dir / f"{nearest_path.stem}_segmented.pcd"
            self._save_pcd(segmented_pts, pcd_path)

            # Save per-bbox PCDs
            for i, bbox_pts in enumerate(per_bbox_pts):
                if bbox_pts.shape[0] > 0:
                    bbox_pcd_path = self.output_dir / f"{nearest_path.stem}_bbox{i}_segmented.pcd"
                    self._save_pcd(bbox_pts, bbox_pcd_path)

            # Log
            extent = segmented_pts.max(axis=0) - segmented_pts.min(axis=0)
            per_bbox_counts = [p.shape[0] for p in per_bbox_pts]
            self.get_logger().info(
                f"{filename}: {len(bboxes)} bbox(es), "
                f"{segmented_pts.shape[0]} pts (per-bbox: {per_bbox_counts}), "
                f"extent [{extent[0]:.3f}, {extent[1]:.3f}, {extent[2]:.3f}]"
            )

        except Exception as e:
            self.get_logger().error(f"Segmentation failed: {e}")


    def _republish_callback(self):
        """Republish last known PointCloud2 and PoseStamped so RViz always
        has data to display.

        Markers are NOT republished — they use unique IDs and persist in RViz
        once added. Only PointCloud2 and PoseStamped need refreshing.
        """
        if self.last_pc_msg is not None:
            self.pub_pc.publish(self.last_pc_msg)
        if self.last_pose is not None:
            self.pub_pose.publish(self.last_pose)


def main(args=None):
    rclpy.init(args=args)
    node = FusionYamlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()