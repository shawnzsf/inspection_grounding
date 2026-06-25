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
from inspection_grounding.msg import SyncedSensorData
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
        # Scale factor for bbox coordinates if annotation resolution differs
        # from the camera intrinsics resolution. Set to 1.0 when they match.
        self.declare_parameter('bbox_scale', 1.0)
        # Bbox visualization depth in camera frame (metres)
        self.declare_parameter('bbox_vis_depth', 2.0)
        # Tolerance for matching YAML files to LiDAR timestamps (nanoseconds)
        self.declare_parameter('yaml_match_tolerance_ns', 150000000)
        # Undistorted intrinsics (right camera, default 1520×2016 half-res)
        self.declare_parameter('fx', 597.3843593065015)
        self.declare_parameter('fy', 597.4426138023167)
        self.declare_parameter('cx', 774.8614840838852)
        self.declare_parameter('cy', 1013.172720601387)
        # Set image width to calculate scaling factor for projection        
        self.img_width = self.declare_parameter('image_width', 1520).value
        self.img_height = self.declare_parameter('image_height', 2016).value

        self.yaml_dir = Path(self.get_parameter('yaml_dir').value)
        self.output_dir = Path(self.get_parameter('output_dir').value)
        self.target_frame = self.get_parameter('target_frame').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.bbox_scale = float(self.get_parameter('bbox_scale').value)
        self.bbox_vis_depth = float(self.get_parameter('bbox_vis_depth').value)
        self.yaml_match_tolerance_ns = int(self.get_parameter('yaml_match_tolerance_ns').value)

        os.makedirs(self.output_dir, exist_ok=True)

        # Camera intrinsics (undistorted, right camera)
        self.fx = float(self.get_parameter('fx').value)
        self.fy = float(self.get_parameter('fy').value)
        self.cx = float(self.get_parameter('cx').value)
        self.cy = float(self.get_parameter('cy').value)

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

        self.pub_pc = self.create_publisher(PointCloud2, '/pointcloud/segmented_yaml', reliable_qos)
        self.pub_pc_aggregated = self.create_publisher(PointCloud2, '/pointcloud/segmented_yaml_aggregated', reliable_qos)
        self.pub_pose = self.create_publisher(PoseStamped, '/object/global_pose', reliable_qos)

        # Subscribers
        syncMsg_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.sub_pc = self.create_subscription(SyncedSensorData, '/synced_sensor_data', self.syncMsg_callback, syncMsg_qos)

        # Cache for republishing (ensures RViz receives data even with single-shot triggers)
        self.last_pc_msg = None
        self.last_stamp = None

        # Accumulated points across all frames (in target/global frame)
        self.aggregated_pts = None  # (N, 3) array or None
        self.last_aggregated_msg = None

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
        s = self.bbox_scale

        # New format: bounding_boxes list
        bb_list = data.get('bounding_boxes')
        if isinstance(bb_list, list):
            for entry in bb_list:
                if isinstance(entry, dict):
                    try:
                        bboxes.append((
                            float(entry['x_min']) * s,
                            float(entry['y_min']) * s,
                            float(entry['x_max']) * s,
                            float(entry['y_max']) * s,
                        ))
                    except (KeyError, ValueError):
                        continue

        # Fallback: single bounding_box dict (old format)
        if not bboxes:
            bbox = data.get('bounding_box')
            if isinstance(bbox, dict):
                try:
                    bboxes.append((
                        float(bbox['x_min']) * s,
                        float(bbox['y_min']) * s,
                        float(bbox['x_max']) * s,
                        float(bbox['y_max']) * s,
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

    def syncMsg_callback(self, msg):
        """Main callback: match YAML, project, segment, publish."""
        stamp = msg.header.stamp
        # Convert message timestamp to nanosecond
        stamp_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

        if not self.yaml_dir.exists(): return

        # Find YAML by exact timestamp match (SyncMessage timestamp is using the image timestamp)
        yaml_path = None
        for p in self.yaml_dir.iterdir():
            if not p.is_file() or p.suffix.lower() not in ('.yaml', '.yml'):
                continue
            try:
                if int(p.stem) == stamp_ns:
                    yaml_path = p
                    break
            except ValueError:
                continue

        if yaml_path is None: return
        filename = yaml_path.name

        # Load YAML
        try: 
            with open(yaml_path, 'r') as f: data = yaml.safe_load(f)
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
            if pts_src.shape[0] == 0: return

            # Project to camera frame
            source_frame = msg.header.frame_id
            T_cam_src = self._lookup_transform_matrix(self.camera_frame, source_frame, stamp)
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
            
            if source_frame != self.target_frame:
                T_target_src = self._lookup_transform_matrix(
                    self.target_frame, source_frame, stamp
                )
                segmented_pts_pub = self._transform_points(segmented_pts, T_target_src)
                pub_frame = self.target_frame
            else:
                segmented_pts_pub = segmented_pts
                pub_frame = source_frame

            # Publish segmented PointCloud2 for visualization
            pc_msg = None
            try:
                pc_header = Header(stamp=stamp, frame_id=pub_frame)
                cloud_out = [[float(x), float(y), float(z)] for x, y, z in segmented_pts_pub]
                pc_msg = point_cloud2.create_cloud_xyz32(pc_header, cloud_out)
                self.pub_pc.publish(pc_msg)
            except Exception as e:
                self.get_logger().warn(f"Failed to publish segmented PointCloud2: {e}")

            # Accumulate segmented points (in target frame) and publish aggregated cloud
            try:
                if self.aggregated_pts is None:
                    self.aggregated_pts = segmented_pts_pub.copy()
                else:
                    self.aggregated_pts = np.vstack(
                        (self.aggregated_pts, segmented_pts_pub)
                    )
                agg_header = Header(stamp=stamp, frame_id=self.target_frame)
                agg_cloud_out = [
                    [float(x), float(y), float(z)]
                    for x, y, z in self.aggregated_pts
                ]
                self.last_aggregated_msg = point_cloud2.create_cloud_xyz32(
                    agg_header, agg_cloud_out
                )
                self.pub_pc_aggregated.publish(self.last_aggregated_msg)
            except Exception as e:
                self.get_logger().warn(
                    f"Failed to publish aggregated PointCloud2: {e}"
                )

            # Cache results for republishing (only if pc_msg was successfully created)
            if pc_msg is not None:
                self.last_pc_msg = pc_msg
            self.last_stamp = stamp
            self._last_yaml_name = filename

            # Save combined PCD (union of all bboxes)
            pcd_path = self.output_dir / f"{yaml_path.stem}_segmented.pcd"
            self._save_pcd(segmented_pts, pcd_path)

            # Save per-bbox PCDs
            for i, bbox_pts in enumerate(per_bbox_pts):
                if bbox_pts.shape[0] > 0:
                    bbox_pcd_path = self.output_dir / f"{yaml_path.stem}_bbox{i}_segmented.pcd"
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
        if self.last_aggregated_msg is not None:
            self.pub_pc_aggregated.publish(self.last_aggregated_msg)


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