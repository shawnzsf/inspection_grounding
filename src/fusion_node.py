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

import bisect
import json
import os
from collections import deque
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from inspection_grounding.msg import SyncedSensorData
import tf2_ros
import yaml

from inspection_db import InspectionDB
from pose_interpolator import interpolate_pose


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
        super().__init__('fusion_node')

        # Parameters
        self.declare_parameter('yaml_dir', '/home/robot/fastlio_ws/masks')
        self.declare_parameter('output_dir', '/home/robot/fastlio_ws/legacy_outputs')
        self.declare_parameter('target_frame', 'camera_init')
        self.declare_parameter('camera_frame', 'camera_link')
        # Scale factor for bbox coordinates if annotation resolution differs
        # from the camera intrinsics resolution. Set to 1.0 when they match.
        self.declare_parameter('bbox_scale', 1.0)
        # Depth (metres) used when visualizing bbox markers in RViz
        # self.declare_parameter('bbox_vis_depth', 5.0)
        # Tolerance (nanoseconds) when matching YAML filenames to cloud timestamps
        # self.declare_parameter('yaml_match_tolerance_ns', 100_000_000)
        # Annotation mode: 'bbox' (per-timestamp YAML files) or 'mask' (COCO JSON)
        self.declare_parameter('annotation_mode', 'mask')
        # Path to COCO-format segmentation JSON (used when annotation_mode='mask')
        self.declare_parameter('mask_json_path', '/home/robot/fastlio_ws/sam3_tracking_segmentation.json')
        # Database path for inspection logging (set to empty string to disable)
        self.declare_parameter('db_path', '/home/robot/fastlio_ws/inspection.db')
        self.declare_parameter('enable_db', True)
        # Undistorted intrinsics (left camera, default 3040×4032 full-res)
        self.declare_parameter('fx', 1190.4990185909498)
        self.declare_parameter('fy', 1190.344769418459)
        self.declare_parameter('cx', 1545.2154064008814)
        self.declare_parameter('cy', 1983.9367884159963)
        # Set image width to calculate scaling factor for projection
        self.img_width = self.declare_parameter('image_width', 3040).value
        self.img_height = self.declare_parameter('image_height', 4032).value
        # Max odometry entries to buffer (~30 s at 10 Hz)
        self.declare_parameter('odom_buffer_size', 300)

        self.yaml_dir = Path(self.get_parameter('yaml_dir').value)
        self.output_dir = Path(self.get_parameter('output_dir').value)
        self.target_frame = self.get_parameter('target_frame').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.bbox_scale = float(self.get_parameter('bbox_scale').value)
        self.annotation_mode = str(self.get_parameter('annotation_mode').value).lower()
        self.mask_json_path = str(self.get_parameter('mask_json_path').value)
        # self.bbox_vis_depth = float(self.get_parameter('bbox_vis_depth').value)
        # self.yaml_match_tolerance_ns = int(self.get_parameter('yaml_match_tolerance_ns').value)

        # Pre-load COCO JSON if in mask mode
        self._mask_data = None
        if self.annotation_mode == 'mask':
            try:
                with open(self.mask_json_path, 'r') as f:
                    self._mask_data = json.load(f)
                self.get_logger().info(
                    f"Loaded mask JSON: {self.mask_json_path} "
                    f"({len(self._mask_data.get('images', []))} images, "
                    f"{len(self._mask_data.get('annotations', []))} annotations)"
                )
            except Exception as e:
                self.get_logger().error(
                    f"Failed to load mask JSON '{self.mask_json_path}': {e}"
                )

        os.makedirs(self.output_dir, exist_ok=True)

        # Camera intrinsics (undistorted, left camera)
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

        # Odometry subscription (RELIABLE QoS — FAST-LIO publishes reliably)
        odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=100
        )
        self.sub_odom = self.create_subscription(
            Odometry, '/Odometry', self._odom_callback, odom_qos
        )

        # Odometry buffer: list of (timestamp_ns, translation(3,), quaternion(4,))
        # kept sorted by timestamp_ns, trimmed to odom_buffer_size
        self._odom_buffer = []
        self._odom_buffer_size = int(self.get_parameter('odom_buffer_size').value)

        # Pending sync messages waiting for odometry to catch up
        self._pending_sync_msgs = deque(maxlen=10)

        # Cache for republishing (ensures RViz receives data even with single-shot triggers)
        self.last_pc_msg = None
        self.last_stamp = None
        # Interpolated pose from odometry (used for DB logging)
        self._last_interp_translation = None
        self._last_interp_quaternion = None

        # Accumulated points across all frames (in target/global frame)
        self.aggregated_pts = None  # (N, 3) array or None
        self.last_aggregated_msg = None

        # Unique marker ID counter (so markers accumulate instead of overwriting)
        self._marker_id_counter = 0
        self._last_yaml_name = None

        # Timer to republish at 2 Hz so RViz always has fresh data
        self.republish_timer = self.create_timer(0.5, self._republish_callback)

        # Inspection database (per-observation + per-track aggregation)
        self.db = None
        if self.get_parameter('enable_db').value:
            db_path = str(self.get_parameter('db_path').value)
            try:
                self.db = InspectionDB(db_path)
                self.get_logger().info(f"Opened inspection DB: {db_path}")
            except Exception as e:
                self.get_logger().error(f"Failed to open inspection DB: {e}")

        # Track-level accumulation (track_id → (N, 3) points in target frame)
        self.track_pts = {}          # track_id → (N, 3) array
        self.track_categories = {}   # track_id → category name
        self.track_first_seen = {}   # track_id → first timestamp_ns
        self.track_last_seen = {}    # track_id → last timestamp_ns

        self.get_logger().info(
            f"Initialized with frames: target={self.target_frame}, "
            f"camera={self.camera_frame}"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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

    def _parse_mask(self, data, timestamp_ns):
        """Parse COCO-format segmentation JSON and return annotations for the
        image whose timestamp matches *timestamp_ns*.

        The image is matched by exact timestamp — the stem of the image's
        ``file_name`` (e.g. ``"1782119257554358000.jpg"``) must equal
        ``str(timestamp_ns)``.

        Args:
            data: Parsed COCO JSON dict with keys ``images``, ``annotations``,
                  and ``categories``.
            timestamp_ns: Image timestamp in nanoseconds (int or str).

        Returns:
            List of dicts, one per annotation for the matched image.  Each
            dict contains:

            * ``polygons``       — list of (N, 2) float64 arrays (polygon
                                   vertices, unscaled).
            * ``bbox``           — (x_min, y_min, x_max, y_max) tuple in
                                   stored image resolution.
            * ``track_id``       — COCO track ID (int or None).
            * ``category``       — category name (str or None).
            * ``mask_path``      — relative path to mask PNG (str or None).
            * ``image_id``       — COCO image ID (int).
            * ``image_file_name``— source image filename (str).
            * ``image_width``    — stored image width in pixels (int).
            * ``image_height``   — stored image height in pixels (int).

            Returns an empty list if no image matches *timestamp_ns*.
        """
        ts_str = str(timestamp_ns)

        # --- Build lookup tables ------------------------------------------------
        images = data.get('images', [])
        annotations = data.get('annotations', [])
        categories = data.get('categories', [])

        # category_id → name
        cat_map = {}
        for cat in categories:
            if isinstance(cat, dict) and 'id' in cat:
                cat_map[cat['id']] = cat.get('name')

        # image_id → image dict
        img_map = {}
        matched_image = None
        for img in images:
            if not isinstance(img, dict):
                continue
            img_map[img.get('id')] = img
            file_name = img.get('file_name', '')
            # Stem of file_name (e.g. "1782119257554358000" from ".../....jpg")
            stem = os.path.splitext(os.path.basename(file_name))[0]
            if stem == ts_str:
                matched_image = img

        if matched_image is None:
            return []

        matched_image_id = matched_image.get('id')
        img_w = matched_image.get('width')
        img_h = matched_image.get('height')
        img_file_name = matched_image.get('file_name', '')

        # --- Collect annotations for the matched image -------------------------
        results = []
        for ann in annotations:
            if not isinstance(ann, dict):
                continue
            if ann.get('image_id') != matched_image_id:
                continue

            # --- Parse segmentation polygons ----------------------------------
            polygons = []
            seg = ann.get('segmentation')
            if isinstance(seg, list):
                for poly in seg:
                    if not isinstance(poly, list) or len(poly) < 2:
                        continue
                    # COCO stores flat [x1, y1, x2, y2, ...] — reshape to (N, 2)
                    arr = np.array(poly, dtype=np.float64).reshape(-1, 2)
                    if arr.shape[0] >= 3:
                        polygons.append(arr)

            # --- Convert COCO bbox [x, y, w, h] → (x_min, y_min, x_max, y_max) -
            bbox_raw = ann.get('bbox')
            if isinstance(bbox_raw, (list, tuple)) and len(bbox_raw) == 4:
                bx, by, bw, bh = (float(v) for v in bbox_raw)
                bbox = (bx, by, bx + bw, by + bh)
            else:
                bbox = None

            results.append({
                'polygons': polygons,
                'bbox': bbox,
                'track_id': ann.get('track_id'),
                'category': cat_map.get(ann.get('category_id')),
                'mask_path': ann.get('mask_path'),
                'image_id': matched_image_id,
                'image_file_name': img_file_name,
                'image_width': img_w,
                'image_height': img_h,
            })

        return results

    def _get_scale_factors(self, stored_width, stored_height):
        """Compute scale factors between stored dimensions and intrinsics resolution.

        Mirrors the scaling logic in rerun_bridge_node._get_pinhole():
            scale_x = stored_width  / self.img_width
            scale_y = stored_height / self.img_height

        To convert annotation coordinates (at stored resolution) to the
        intrinsics resolution, divide by the returned scale factors.
        To scale intrinsics up to the stored resolution, multiply by them.

        Args:
            stored_width:  width  of the annotation/image (e.g. 3040)
            stored_height: height of the annotation/image (e.g. 4032)

        Returns:
            (scale_x, scale_y) tuple of floats.
        """
        scale_x = stored_width / self.img_width if self.img_width > 0 else 1.0
        scale_y = stored_height / self.img_height if self.img_height > 0 else 1.0
        return scale_x, scale_y

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

    def _lookup_tf_pose(self, target_frame, source_frame, stamp):
        """
        Look up transform and return raw translation and rotation.

        Args:
            target_frame: destination frame
            source_frame: source frame
            stamp: ROS time stamp

        Returns:
            (translation, rotation) where translation is (x, y, z) and
            rotation is (qx, qy, qz, qw).
        """
        tf_msg = self.tf_buffer.lookup_transform(
            target_frame,
            source_frame,
            Time.from_msg(stamp),
            timeout=rclpy.duration.Duration(seconds=0.5)
        )
        t = tf_msg.transform.translation
        q = tf_msg.transform.rotation
        return (t.x, t.y, t.z), (q.x, q.y, q.z, q.w)

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

    def _project_to_image(self, pts_cam, stored_width=None, stored_height=None):
        """
        Project 3D points (in camera frame) to 2D image plane.

        If stored_width and stored_height are provided, the intrinsics are
        scaled from the default resolution (self.img_width × self.img_height)
        to the stored resolution using _get_scale_factors(), so that the
        projected (u, v) coordinates are in the same coordinate space as
        annotations stored at that resolution.

        Args:
            pts_cam: (N, 3) points in camera frame
            stored_width:  width  of the annotation/image resolution (optional)
            stored_height: height of the annotation/image resolution (optional)

        Returns:
            u, v: image coordinates
            valid_z: mask of points with z > 0
        """
        # Scale intrinsics to match the stored annotation resolution
        if stored_width is not None and stored_height is not None:
            scale_x, scale_y = self._get_scale_factors(stored_width, stored_height)
            fx = self.fx * scale_x
            fy = self.fy * scale_y
            cx = self.cx * scale_x
            cy = self.cy * scale_y
        else:
            fx = self.fx
            fy = self.fy
            cx = self.cx
            cy = self.cy

        z = pts_cam[:, 2]
        valid = z > 1e-6
        u = np.full(pts_cam.shape[0], np.nan, dtype=np.float64)
        v = np.full(pts_cam.shape[0], np.nan, dtype=np.float64)

        u[valid] = fx * (pts_cam[valid, 0] / z[valid]) + cx
        v[valid] = fy * (pts_cam[valid, 1] / z[valid]) + cy

        return u, v, valid

    @staticmethod
    def _points_in_polygon(u, v, polygon):
        """Vectorized point-in-polygon test (ray casting algorithm).

        Args:
            u: (N,) array of x-coordinates (image column)
            v: (N,) array of y-coordinates (image row)
            polygon: (M, 2) array of polygon vertices

        Returns:
            (N,) boolean array — True for points inside (or on edge of) polygon.
        """
        n = len(polygon)
        if n < 3:
            return np.zeros(len(u), dtype=bool)

        inside = np.zeros(len(u), dtype=bool)
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]

            # Check if edge crosses the horizontal ray from (u, v)
            cond1 = (yi > v) != (yj > v)
            # Compute x-coordinate of intersection
            with np.errstate(divide='ignore', invalid='ignore'):
                x_int = (xj - xi) * (v - yi) / (yj - yi) + xi
            cond2 = u < x_int
            inside ^= (cond1 & cond2)
            j = i

        return inside

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

    # ------------------------------------------------------------------
    # Odometry buffering & pose interpolation
    # ------------------------------------------------------------------

    def _odom_callback(self, msg):
        """Buffer odometry messages and retry pending sync messages."""
        t_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        pos = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
        ], dtype=np.float64)
        quat = np.array([
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        ], dtype=np.float64)

        self._odom_buffer.append((t_ns, pos, quat))

        # Trim to max size (drop oldest)
        if len(self._odom_buffer) > self._odom_buffer_size:
            self._odom_buffer = self._odom_buffer[-self._odom_buffer_size:]

        # Retry any pending sync messages that now have bracketing odometry
        self._process_pending_sync_msgs()

    def _get_interpolated_pose(self, target_ns):
        """Return interpolated (translation, quaternion) at *target_ns*.

        Returns:
            (translation(3,), quaternion(4,)) tuple, or None if the second
            bracketing odometry frame has not arrived yet.
        """
        if len(self._odom_buffer) < 2:
            return None

        times = [e[0] for e in self._odom_buffer]

        # target is at or before the first frame → clamp to first
        if target_ns <= times[0]:
            return self._odom_buffer[0][1].copy(), self._odom_buffer[0][2].copy()

        # target is after the last frame → second frame not available yet
        if target_ns > times[-1]:
            return None

        # Binary search for the insertion point
        idx = bisect.bisect_left(times, target_ns)

        # Exact match
        if idx < len(times) and times[idx] == target_ns:
            return self._odom_buffer[idx][1].copy(), self._odom_buffer[idx][2].copy()

        # Interpolate between idx-1 and idx
        t1, p1, q1 = self._odom_buffer[idx - 1]
        t2, p2, q2 = self._odom_buffer[idx]
        return interpolate_pose(t1, p1, q1, t2, p2, q2, target_ns)

    def _process_pending_sync_msgs(self):
        """Process queued sync messages whose odometry has caught up.

        Iterates in FIFO order and stops at the first message that still
        cannot be interpolated, preserving temporal ordering.
        """
        if not self._pending_sync_msgs:
            return

        remaining = deque()
        while self._pending_sync_msgs:
            msg = self._pending_sync_msgs.popleft()
            stamp = msg.header.stamp
            stamp_ns = stamp.sec * 1_000_000_000 + stamp.nanosec
            pose = self._get_interpolated_pose(stamp_ns)
            if pose is None:
                remaining.append(msg)
                break  # stop — later messages are also not ready
            self._last_interp_translation, self._last_interp_quaternion = pose
            self.syncMsg_callback(msg, _skip_interpolation=True)

        # Put unprocessed messages back
        self._pending_sync_msgs.extendleft(reversed(remaining))

    # ------------------------------------------------------------------
    # Main callback
    # ------------------------------------------------------------------

    def syncMsg_callback(self, msg, _skip_interpolation=False):
        """Main callback: match annotation, project, segment, publish.

        Supports two annotation modes controlled by the 'annotation_mode' parameter:
        - 'bbox': per-timestamp YAML files with bounding boxes
        - 'mask': single COCO-format JSON with segmentation polygons

        Args:
            msg: SyncedSensorData message.
            _skip_interpolation: If True, skip pose interpolation (used when
                called from _process_pending_sync_msgs, which has already set
                the interpolated pose).
        """
        stamp = msg.header.stamp
        stamp_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

        # ------------------------------------------------------------------
        # Step 0: Pose interpolation from odometry buffer
        # ------------------------------------------------------------------
        # Try to get the interpolated pose at the message timestamp. If the
        # second bracketing odometry frame hasn't arrived yet, queue the
        # message for deferred processing.
        if not _skip_interpolation:
            pose = self._get_interpolated_pose(stamp_ns)
            if pose is None:
                self._pending_sync_msgs.append(msg)
                self.get_logger().debug(
                    f"Deferring msg {stamp_ns}: odometry not yet available "
                    f"(pending: {len(self._pending_sync_msgs)})"
                )
                return
            self._last_interp_translation, self._last_interp_quaternion = pose

        # ------------------------------------------------------------------
        # Step 1: Load annotations based on mode
        # ------------------------------------------------------------------
        annotations = []       # unified list of annotation dicts
        stored_w = None         # annotation image width (for projection scaling)
        stored_h = None         # annotation image height
        source_id = None        # identifier for logging (filename or timestamp)

        if self.annotation_mode == 'mask':
            if self._mask_data is None:
                return
            masks = self._parse_mask(self._mask_data, stamp_ns)
            if not masks:
                return
            source_id = str(stamp_ns)
            stored_w = masks[0]['image_width']
            stored_h = masks[0]['image_height']
            for m in masks:
                annotations.append({
                    'polygons': m['polygons'],
                    'bbox': m['bbox'],
                    'label': m.get('category') or f"track{m.get('track_id', '?')}",
                    'track_id': m.get('track_id'),
                    'category': m.get('category'),
                    'mask_path': m.get('mask_path'),
                    'image_id': m.get('image_id'),
                    'image_file_name': m.get('image_file_name'),
                })
        else:
            # bbox mode: find YAML by exact timestamp match
            if not self.yaml_dir.exists():
                return
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
            if yaml_path is None:
                return
            source_id = yaml_path.name
            try:
                with open(yaml_path, 'r') as f:
                    data = yaml.safe_load(f)
            except Exception as e:
                self.get_logger().error(f"Failed to read {source_id}: {e}")
                return
            bboxes = self._parse_bboxes(data)
            bboxes = [b for b in bboxes if b[2] > b[0] and b[3] > b[1]]
            if not bboxes:
                self.get_logger().warn(f"No valid bboxes in {source_id}")
                return
            for i, bb in enumerate(bboxes):
                annotations.append({
                    'polygons': [],
                    'bbox': bb,
                    'label': f"bbox{i}",
                })

        if not annotations:
            return

        # ------------------------------------------------------------------
        # Step 2: Read cloud, transform, project
        # ------------------------------------------------------------------
        try:
            pts_src = self._read_cloud_xyz(msg.pointcloud)
            if pts_src.shape[0] == 0:
                return

            source_frame = msg.pointcloud.header.frame_id
            T_cam_src = self._lookup_transform_matrix(self.camera_frame, source_frame, stamp)

            pts_cam = self._transform_points(pts_src, T_cam_src)

            # Project to image plane (scale intrinsics if stored dimensions known)
            u, v, valid_z = self._project_to_image(pts_cam, stored_w, stored_h)
            valid_proj = valid_z & np.isfinite(u) & np.isfinite(v)

            # ------------------------------------------------------------------
            # Step 3: Per-annotation segmentation
            # ------------------------------------------------------------------
            per_ann_masks = []
            per_ann_pts = []
            for ann in annotations:
                if ann['polygons']:
                    # Polygon-based segmentation (mask mode)
                    ann_mask = np.zeros(pts_src.shape[0], dtype=bool)
                    for poly in ann['polygons']:
                        ann_mask |= valid_proj & self._points_in_polygon(u, v, poly)
                elif ann['bbox'] is not None:
                    # Bbox-based segmentation (bbox mode or mask fallback)
                    x_min, y_min, x_max, y_max = ann['bbox']
                    ann_mask = valid_proj & (u >= x_min) & (u <= x_max) & \
                               (v >= y_min) & (v <= y_max)
                else:
                    ann_mask = np.zeros(pts_src.shape[0], dtype=bool)

                per_ann_masks.append(ann_mask)
                per_ann_pts.append(pts_src[ann_mask])

            # Union mask for combined output
            union_mask = np.zeros(pts_src.shape[0], dtype=bool)
            for m in per_ann_masks:
                union_mask |= m

            segmented_pts = pts_src[union_mask]
            if segmented_pts.shape[0] == 0:
                counts = [len(p) for p in per_ann_pts]
                self.get_logger().warn(
                    f"No points in any annotation: {source_id} "
                    f"(per-ann counts: {counts})"
                )
                return

            # ------------------------------------------------------------------
            # Step 4: Transform to target frame, publish, save
            # ------------------------------------------------------------------
            if source_frame != self.target_frame:
                T_target_src = self._lookup_transform_matrix(
                    self.target_frame, source_frame, stamp
                )
                segmented_pts_pub = self._transform_points(segmented_pts, T_target_src)
                pub_frame = self.target_frame
            else:
                segmented_pts_pub = segmented_pts
                pub_frame = source_frame

            # Publish segmented PointCloud2
            pc_msg = None
            try:
                pc_header = Header(stamp=stamp, frame_id=pub_frame)
                cloud_out = [[float(x), float(y), float(z)] for x, y, z in segmented_pts_pub]
                pc_msg = point_cloud2.create_cloud_xyz32(pc_header, cloud_out)
                self.pub_pc.publish(pc_msg)
            except Exception as e:
                self.get_logger().warn(f"Failed to publish segmented PointCloud2: {e}")

            # Accumulate and publish aggregated cloud
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

            # Cache for republishing
            if pc_msg is not None:
                self.last_pc_msg = pc_msg
            self.last_stamp = stamp
            self._last_yaml_name = source_id

            # Save PCDs
            stem = str(stamp_ns)
            pcd_path = self.output_dir / f"{stem}_segmented.pcd"
            self._save_pcd(segmented_pts, pcd_path)

            for i, ann_pts in enumerate(per_ann_pts):
                if ann_pts.shape[0] > 0:
                    ann_pcd_path = self.output_dir / f"{stem}_ann{i}_segmented.pcd"
                    self._save_pcd(ann_pts, ann_pcd_path)

            # ------------------------------------------------------------------
            # Step 5: Database logging (per-observation + per-track aggregation)
            # ------------------------------------------------------------------
            if self.db is not None:
                # Get sensor pose (camera_init → body) for DB storage
                tf_translation = None
                tf_rotation = None
                if self._last_interp_translation is not None:
                    tf_translation = tuple(self._last_interp_translation)
                    tf_rotation = tuple(self._last_interp_quaternion)
                elif source_frame != self.target_frame:
                    try:
                        tf_translation, tf_rotation = self._lookup_tf_pose(
                            self.target_frame, source_frame, stamp
                        )
                    except Exception as e:
                        self.get_logger().warn(f"TF pose lookup failed: {e}")

                for i, (ann, ann_pts) in enumerate(zip(annotations, per_ann_pts)):
                    if ann_pts.shape[0] == 0:
                        continue

                    track_id = ann.get('track_id')
                    category = ann.get('category')

                    # Transform per-annotation points to target frame
                    if source_frame != self.target_frame:
                        ann_pts_target = self._transform_points(ann_pts, T_target_src)
                    else:
                        ann_pts_target = ann_pts

                    # Compute 3D centroid and bbox (in target frame)
                    centroid = tuple(ann_pts_target.mean(axis=0))
                    bbox3d_min = tuple(ann_pts_target.min(axis=0))
                    bbox3d_max = tuple(ann_pts_target.max(axis=0))

                    # Save per-observation PCD (in target frame)
                    if track_id is not None:
                        obs_pcd_path = self.output_dir / f"{stem}_track{track_id}_obs.pcd"
                    else:
                        obs_pcd_path = self.output_dir / f"{stem}_ann{i}_obs.pcd"
                    self._save_pcd(ann_pts_target, obs_pcd_path)

                    # Convert bbox to COCO [x, y, w, h] for DB storage
                    bbox_2d_coco = None
                    if ann['bbox'] is not None:
                        x_min, y_min, x_max, y_max = ann['bbox']
                        bbox_2d_coco = [x_min, y_min, x_max - x_min, y_max - y_min]

                    # Log observation to DB
                    try:
                        self.db.add_observation(
                            timestamp_ns=stamp_ns,
                            track_id=track_id,
                            category=category,
                            image_id=ann.get('image_id'),
                            image_file_name=ann.get('image_file_name'),
                            centroid=centroid,
                            bbox3d_min=bbox3d_min,
                            bbox3d_max=bbox3d_max,
                            point_count=int(ann_pts.shape[0]),
                            pcd_path=str(obs_pcd_path),
                            mask_path=ann.get('mask_path'),
                            bbox_2d=bbox_2d_coco,
                            tf_translation=tf_translation,
                            tf_rotation=tf_rotation,
                        )
                    except Exception as e:
                        self.get_logger().warn(f"DB observation insert failed: {e}")

                    # Track-level accumulation and upsert
                    if track_id is not None:
                        if track_id not in self.track_pts:
                            self.track_pts[track_id] = ann_pts_target.copy()
                            self.track_categories[track_id] = category
                            self.track_first_seen[track_id] = stamp_ns
                        else:
                            self.track_pts[track_id] = np.vstack(
                                (self.track_pts[track_id], ann_pts_target)
                            )
                        self.track_last_seen[track_id] = stamp_ns

                        track_pts = self.track_pts[track_id]
                        agg_centroid = tuple(track_pts.mean(axis=0))
                        agg_bbox_min = tuple(track_pts.min(axis=0))
                        agg_bbox_max = tuple(track_pts.max(axis=0))
                        obs_count = self.db.get_observation_count(track_id)

                        # Save/update aggregated per-track PCD
                        agg_pcd_path = self.output_dir / f"track{track_id}_aggregated.pcd"
                        self._save_pcd(track_pts, agg_pcd_path)

                        try:
                            self.db.upsert_object(
                                track_id=track_id,
                                category=self.track_categories.get(track_id),
                                centroid=agg_centroid,
                                bbox3d_min=agg_bbox_min,
                                bbox3d_max=agg_bbox_max,
                                total_point_count=int(track_pts.shape[0]),
                                observation_count=obs_count,
                                first_seen_ns=self.track_first_seen[track_id],
                                last_seen_ns=self.track_last_seen[track_id],
                                aggregated_pcd_path=str(agg_pcd_path),
                                tf_translation=tf_translation,
                                tf_rotation=tf_rotation,
                            )
                        except Exception as e:
                            self.get_logger().warn(f"DB object upsert failed: {e}")

            # Log
            extent = segmented_pts.max(axis=0) - segmented_pts.min(axis=0)
            per_ann_counts = [p.shape[0] for p in per_ann_pts]
            self.get_logger().info(
                f"{source_id}: {len(annotations)} ann(s) [{self.annotation_mode}], "
                f"{segmented_pts.shape[0]} pts (per-ann: {per_ann_counts}), "
                f"extent [{extent[0]:.3f}, {extent[1]:.3f}, {extent[2]:.3f}]"
            )

        except Exception as e:
            self.get_logger().error(f"Segmentation failed: {e}")


    def destroy_node(self):
        """Save aggregated point cloud and close DB before shutting down."""
        if self.aggregated_pts is not None and self.aggregated_pts.shape[0] > 0:
            agg_pcd_path = self.output_dir / "aggregated_segmented.pcd"
            if self._save_pcd(self.aggregated_pts, agg_pcd_path):
                self.get_logger().info(
                    f"Saved aggregated cloud: {agg_pcd_path} "
                    f"({self.aggregated_pts.shape[0]} pts)"
                )

        # Close inspection database
        if self.db is not None:
            try:
                self.db.close()
                self.get_logger().info("Inspection DB closed.")
            except Exception as e:
                self.get_logger().warn(f"Error closing DB: {e}")

        super().destroy_node()

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