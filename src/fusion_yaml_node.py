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

import json
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
        # Annotation mode: 'bbox' (per-timestamp YAML files) or 'mask' (COCO JSON)
        self.declare_parameter('annotation_mode', 'mask')
        # Path to COCO-format segmentation JSON (used when annotation_mode='mask')
        self.declare_parameter('mask_json_path', '/home/robot/fastlio_ws/sam3_tracking_segmentation.json')
        # Bbox visualization depth in camera frame (metres)
        self.declare_parameter('bbox_vis_depth', 2.0)
        # Tolerance for matching YAML files to LiDAR timestamps (nanoseconds)
        self.declare_parameter('yaml_match_tolerance_ns', 150000000)
        # Undistorted intrinsics (left camera, default 3040×4032 full-res)
        self.declare_parameter('fx', 1190.4990185909498)
        self.declare_parameter('fy', 1190.344769418459)
        self.declare_parameter('cx', 1545.2154064008814)
        self.declare_parameter('cy', 1983.9367884159963)
        # Set image width to calculate scaling factor for projection
        self.img_width = self.declare_parameter('image_width', 3040).value
        self.img_height = self.declare_parameter('image_height', 4032).value

        self.yaml_dir = Path(self.get_parameter('yaml_dir').value)
        self.output_dir = Path(self.get_parameter('output_dir').value)
        self.target_frame = self.get_parameter('target_frame').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.bbox_scale = float(self.get_parameter('bbox_scale').value)
        self.annotation_mode = str(self.get_parameter('annotation_mode').value).lower()
        self.mask_json_path = str(self.get_parameter('mask_json_path').value)
        self.bbox_vis_depth = float(self.get_parameter('bbox_vis_depth').value)
        self.yaml_match_tolerance_ns = int(self.get_parameter('yaml_match_tolerance_ns').value)

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

    def _parse_mask(self, data, timestamp_ns):
        """Parse segmentation masks from COCO-format JSON for a given timestamp.

        Expects a COCO-format dict with 'images', 'annotations', and
        'categories' keys (as produced by SAM3 tracking export).

        The image is matched by exact timestamp: the stem of the image's
        'file_name' (e.g. "1782119257554358000.jpg") must equal str(timestamp_ns).

        Annotation coordinates (polygon vertices and bbox) are returned in
        their original stored image resolution.  The caller should pass the
        same stored dimensions to _project_to_image() so that projected
        points are in the same coordinate space.

        Args:
            data: COCO-format dict (parsed JSON)
            timestamp_ns: integer nanosecond timestamp to match

        Returns:
            List of dicts, one per annotation, each containing:
              - 'polygons': list of (N, 2) float64 numpy arrays (unscaled)
              - 'bbox': (x_min, y_min, x_max, y_max) tuple (unscaled)
              - 'track_id': int or None
              - 'category': str or None
              - 'mask_path': str or None (relative path)
              - 'image_width': int, stored image width (for projection scaling)
              - 'image_height': int, stored image height (for projection scaling)
            Empty list if no matching image or no annotations.
        """
        if not isinstance(data, dict):
            return []

        images = data.get('images', [])
        annotations = data.get('annotations', [])
        categories = data.get('categories', [])

        # Build category lookup
        categories_by_id = {
            cat['id']: cat.get('name')
            for cat in categories if isinstance(cat, dict) and 'id' in cat
        }

        # Find the image whose file_name stem matches timestamp_ns exactly
        target_stem = str(timestamp_ns)
        image_info = None
        for img in images:
            if not isinstance(img, dict):
                continue
            file_name = img.get('file_name', '')
            if Path(file_name).stem == target_stem:
                image_info = img
                break

        if image_info is None:
            return []

        image_id = image_info.get('id')
        if image_id is None:
            return []

        stored_w = image_info.get('width', self.img_width)
        stored_h = image_info.get('height', self.img_height)

        # Collect annotations for this image
        results = []
        for ann in annotations:
            if not isinstance(ann, dict) or ann.get('image_id') != image_id:
                continue

            # Parse segmentation polygons (kept at original stored resolution)
            polygons = []
            seg = ann.get('segmentation')
            if isinstance(seg, list):
                for poly in seg:
                    if isinstance(poly, list) and len(poly) >= 6:  # at least 3 points
                        arr = np.array(poly, dtype=np.float64).reshape(-1, 2)
                        polygons.append(arr)

            # Parse bbox (COCO format: [x, y, w, h] → [x_min, y_min, x_max, y_max])
            bbox = None
            coco_bbox = ann.get('bbox')
            if isinstance(coco_bbox, list) and len(coco_bbox) == 4:
                try:
                    x, y, w, h = coco_bbox
                    bbox = (
                        float(x),
                        float(y),
                        float(x + w),
                        float(y + h),
                    )
                except (ValueError, TypeError):
                    pass

            # Skip annotations with no usable geometry
            if not polygons and bbox is None:
                continue

            results.append({
                'polygons': polygons,
                'bbox': bbox,
                'track_id': ann.get('track_id'),
                'category': categories_by_id.get(ann.get('category_id')),
                'mask_path': ann.get('mask_path'),
                'image_width': stored_w,
                'image_height': stored_h,
            })

        return results

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

    def syncMsg_callback(self, msg):
        """Main callback: match annotation, project, segment, publish.

        Supports two annotation modes controlled by the 'annotation_mode' parameter:
        - 'bbox': per-timestamp YAML files with bounding boxes
        - 'mask': single COCO-format JSON with segmentation polygons
        """
        stamp = msg.header.stamp
        stamp_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

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