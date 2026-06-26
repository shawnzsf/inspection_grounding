# inspection_grounding

LiDAR–Camera fusion and object grounding for tunnel/inspection robotics.  
The package synchronizes camera images with LiDAR point clouds, projects 3D points onto 2D image annotations (bounding boxes or segmentation masks), and segments the corresponding 3D object points for downstream inspection. Segmented points are stored in a SQLite database with per-observation and per-track aggregation.

---

## Table of Contents

1. [Package Structure](#package-structure)
2. [Nodes](#nodes)
   - [sync_node](#1-sync_nodepy)
   - [fusion_node](#2-fusion_nodepy)
   - [rerun_bridge_node](#3-rerun_bridge_nodepy)
3. [Inspection Database](#inspection-database)
4. [Inter-Node Data Flow](#inter-node-data-flow)
5. [Custom Messages](#custom-messages)
6. [Launch Files](#launch-files)
7. [Calibration & Configuration](#calibration--configuration)
8. [Build & Run](#build--run)

---

## Package Structure

```
src/inspection_grounding/
├── CMakeLists.txt                        # ament_cmake build config
├── package.xml                           # ROS 2 package manifest
├── README.md                             # this file
├── msg/
│   └── SyncedSensorData.msg              # bundled image + camera_info + pointcloud
├── src/
│   ├── sync_node.py                      # time-synchronizer node
│   ├── fusion_node.py                    # 2D-annotation → 3D pointcloud segmentation node
│   ├── inspection_db.py                  # SQLite wrapper for per-object inspection data
│   └── rerun_bridge_node.py              # ROS 2 → Rerun visualization bridge
├── config/
│   ├── params.yaml                       # ROS 2 node parameters (passed via --params-file)
│   └── launch_config.yaml                # Launch-level config (bag path, static TF)
├── launch/
│   ├── test_sync.launch.py               # sync pipeline test
│   ├── test_fusion.launch.py             # full fusion pipeline test (RViz)
│   └── test_fusion_rerun.launch.py       # full fusion pipeline test (Rerun)
└── test/
    └── test_db.py                        # unit tests for InspectionDB
```

---

## Nodes

### 1. `sync_node.py`

**Purpose** — Time-synchronize camera images, camera info, and LiDAR point clouds into a single message.

| Direction | Topic | Type | QoS |
|-----------|-------|------|-----|
| Sub | `/camera/image_undistorted` | `sensor_msgs/Image` | default |
| Sub | `/camera/camera_info` | `sensor_msgs/CameraInfo` | default |
| Sub | `/cloud_registered` | `sensor_msgs/PointCloud2` | default |
| Pub | `/synced_sensor_data` | `inspection_grounding/SyncedSensorData` | default |

**Internal workflow:**

1. Three `message_filters.Subscriber` objects wrap the input topics.
2. An `ApproximateTimeSynchronizer` (queue=10, slop=50 ms) aligns messages by header stamp.
3. On each synchronized triple, a `SyncedSensorData` message is assembled (image header is used as the reference stamp) and published.

---

### 2. `fusion_node.py`

**Purpose** — The core fusion node. Matches each LiDAR scan with 2D annotations (bounding-box YAML files or COCO-format segmentation JSON) by timestamp, projects the 3D points onto the image plane, segments points inside the annotation region, publishes the result, and stores per-observation/per-track data in a SQLite database.

| Direction | Topic | Type | QoS |
|-----------|-------|------|-----|
| Sub | `/synced_sensor_data` | `inspection_grounding/SyncedSensorData` | BEST_EFFORT, KEEP_LAST(10) |
| Pub | `/pointcloud/segmented_yaml` | `sensor_msgs/PointCloud2` | RELIABLE, KEEP_LAST(10) |
| Pub | `/pointcloud/segmented_yaml_aggregated` | `sensor_msgs/PointCloud2` | RELIABLE, KEEP_LAST(10) |
| Pub | `/object/global_pose` | `geometry_msgs/PoseStamped` | RELIABLE, KEEP_LAST(10) |

**Parameters**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `yaml_dir` | `/home/robot/fastlio_ws/masks/Ground_Truth_20` | Directory containing bbox YAML files (used when `annotation_mode=bbox`) |
| `output_dir` | `/home/robot/fastlio_ws/pcd_outputs` | Directory for saved PCD files |
| `target_frame` | `camera_init` | Global/world frame (FAST-LIO origin) |
| `camera_frame` | `camera_link` | Camera optical frame |
| `bbox_scale` | `1.0` | Scale factor for bbox coordinates if annotation resolution differs from intrinsics resolution |
| `annotation_mode` | `mask` | Annotation source: `bbox` (per-timestamp YAML files) or `mask` (COCO JSON) |
| `mask_json_path` | `.../sam3_tracking_segmentation.json` | Path to COCO-format segmentation JSON (used when `annotation_mode=mask`) |
| `bbox_vis_depth` | `2.0` | Bbox visualization depth in camera frame (metres) |
| `yaml_match_tolerance_ns` | `100000000` | Tolerance for matching YAML files to LiDAR timestamps (nanoseconds, 100 ms) |
| `fx`, `fy`, `cx`, `cy` | — | Undistorted camera intrinsics (see [Calibration](#calibration--configuration)) |
| `image_width` | `3040` | Image width in pixels (for projection scaling) |
| `image_height` | `4032` | Image height in pixels (for projection scaling) |
| `enable_db` | `true` | Enable SQLite inspection database storage |
| `db_path` | `/home/robot/fastlio_ws/inspection.db` | Path to the SQLite database file |

#### Annotation Modes

The node supports two annotation modes, controlled by the `annotation_mode` parameter:

**`bbox` mode** — Per-timestamp YAML files in `yaml_dir`. Each file is named `<epoch_nanoseconds>.yaml` and contains bounding box coordinates. The `_parse_bboxes()` helper extracts `(x_min, y_min, x_max, y_max)` tuples from either a `bounding_boxes` list (new format) or a single `bounding_box` dict (old format).

**`mask` mode** — A single COCO-format JSON file (`mask_json_path`) containing segmentation polygons and bounding boxes for all images. The `_parse_mask()` helper (see below) matches annotations by timestamp and returns polygon-based segmentation data.

#### `_parse_mask()` Helper

```python
def _parse_mask(self, data, timestamp_ns):
```

Parses a COCO-format segmentation JSON dict and returns all annotations for the image whose timestamp matches `timestamp_ns`.

**Matching logic:** The image is matched by exact timestamp — the stem of the image's `file_name` (e.g. `"1782119257554358000.jpg"`) must equal `str(timestamp_ns)`.

**Returns:** A list of dicts, one per annotation, each containing:

| Key | Type | Description |
|-----|------|-------------|
| `polygons` | `list[np.ndarray]` | List of `(N, 2)` float64 arrays (polygon vertices, unscaled) |
| `bbox` | `tuple` | `(x_min, y_min, x_max, y_max)` in stored image resolution |
| `track_id` | `int \| None` | COCO track ID (groups observations of the same object) |
| `category` | `str \| None` | Category name (e.g. `"Poster"`) |
| `mask_path` | `str \| None` | Relative path to mask PNG file |
| `image_id` | `int` | COCO image ID |
| `image_file_name` | `str` | Source image filename |
| `image_width` | `int` | Stored image width (for projection scaling) |
| `image_height` | `int` | Stored image height (for projection scaling) |

Annotation coordinates are returned in their original stored image resolution. The caller passes the same stored dimensions to `_project_to_image()` so that projected points are in the same coordinate space.

**COCO JSON format** (see `sam3_tracking_segmentation.json`):

```json
{
  "images": [
    {"id": 1, "file_name": "1782119257554358000.jpg", "width": 3040, "height": 4032}
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "track_id": 2947,
      "segmentation": [[x1, y1, x2, y2, ..., xN, yN]],
      "bbox": [x, y, w, h],
      "mask_path": "masks/1782119257554358000_track2947.png"
    }
  ],
  "categories": [
    {"id": 1, "name": "Poster"}
  ]
}
```

#### Internal workflow (`syncMsg_callback`):

```
SyncedSensorData msg
      │
      ▼
┌──────────────────────────┐
│ 1. Timestamp matching    │  bbox mode: scan yaml_dir, parse filename as epoch-ns
│    (exact or tolerance)  │  mask mode: look up image by file_name stem == timestamp
└──────────────────────────┘
      │
      ▼
┌──────────────────────────┐
│ 2. Parse annotations     │  bbox mode: _parse_bboxes() → list of (x_min,y_min,x_max,y_max)
│                          │  mask mode: _parse_mask() → list of {polygons, bbox, track_id, ...}
└──────────────────────────┘
      │
      ▼
┌──────────────────────────┐
│ 3. Read cloud XYZ        │  sensor_msgs_py.read_points → (N,3) array
└──────────────────────────┘
      │
      ▼
┌──────────────────────────┐
│ 4. Transform → camera    │  TF lookup: camera_frame ← cloud frame
│    frame                 │  Apply 4×4 homogeneous transform
└──────────────────────────┘
      │
      ▼
┌──────────────────────────┐
│ 5. Project to image      │  u = fx·x/z + cx
│    plane                 │  v = fy·y/z + cy
│                          │  (intrinsics scaled to match annotation resolution)
└──────────────────────────┘
      │
      ▼
┌──────────────────────────┐
│ 6. Segment inside        │  bbox mode: point-in-bbox test
│    annotation region     │  mask mode: point-in-polygon test (via _point_in_polygon)
└──────────────────────────┘
      │
      ▼
┌──────────────────────────┐
│ 7. Transform → target    │  TF lookup: target_frame ← cloud frame
│    (global) frame        │  Apply transform to segmented points + centroid
└──────────────────────────┘
      │
      ▼
┌──────────────────────────┐
│ 8. Publish, save & DB    │  • PointCloud2 (segmented, in target frame)
│                          │  • PointCloud2 (aggregated, accumulated in target frame)
│                          │  • PoseStamped (centroid, in target frame)
│                          │  • PCD files (per-annotation, per-frame, per-track, aggregated)
│                          │  • SQLite DB (per-observation + per-track aggregation)
└──────────────────────────┘
```

**Aggregated cloud** — The `/pointcloud/segmented_yaml_aggregated` topic accumulates all segmented points (transformed to the target/global frame) across every processed frame. This provides a growing point cloud of all detected objects in the world frame, useful for building up a complete inspection map.

**Republishing** — A 2 Hz timer (`_republish_callback`) re-emits the last segmented cloud and aggregated cloud so RViz always has data to display even when the input is single-shot.

**PCD output files** — The node saves multiple PCD files per frame:

| Pattern | Description |
|---------|-------------|
| `{timestamp}_ann{N}_segmented.pcd` | Per-annotation segmented points (source frame) |
| `{timestamp}_segmented.pcd` | Combined points from all annotations (source frame) |
| `{timestamp}_track{N}_obs.pcd` | Per-track observation points (target frame) |
| `track{N}_aggregated.pcd` | Per-track accumulated points across all frames (target frame) |
| `aggregated_segmented.pcd` | Global accumulated points across all frames (target frame) |

---

### 3. `rerun_bridge_node.py`

**Purpose** — Bridge node that subscribes to ROS 2 topics and logs them to a [Rerun](https://www.rerun.io/) viewer for 3D visualization. Provides an alternative to RViz with a built-in viewer window.

| Direction | Topic | Type | QoS |
|-----------|-------|------|-----|
| Sub | `/Laser_map` | `sensor_msgs/PointCloud2` | RELIABLE, KEEP_LAST(5) |
| Sub | `/pointcloud/segmented_yaml` | `sensor_msgs/PointCloud2` | RELIABLE, KEEP_LAST(5) |
| Sub | `/pointcloud/segmented_yaml_aggregated` | `sensor_msgs/PointCloud2` | RELIABLE, KEEP_LAST(5) |
| Sub | `/camera/image_undistorted` | `sensor_msgs/Image` | default (KEEP_LAST 10) |
| TF | `camera_init` → `body` | `geometry_msgs/TransformStamped` | via tf2 |
| TF | `body` → `camera_link` | `geometry_msgs/TransformStamped` | via tf2 |

**Parameters**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `leveling_rpy_deg` | `[0.0, 0.0, 0.0]` | Optional roll/pitch/yaw (degrees) for visualization-only leveling rotation |
| `camera_fx` | `597.384` | Camera focal length x (pixels) — for pinhole projection |
| `camera_fy` | `597.443` | Camera focal length y (pixels) |
| `camera_cx` | `774.861` | Camera principal point x (pixels) |
| `camera_cy` | `1013.173` | Camera principal point y (pixels) |
| `image_width` | `1520` | Image width in pixels |
| `image_height` | `2016` | Image height in pixels |
| `enable_depth` | `true` | Enable depth image visualization (projects `/cloud_registered_body` into camera plane) |

**Visualization details:**

| Entity Path | Content | Color |
|-------------|---------|-------|
| `world/leveled/camera_init/Laser_map` | `/Laser_map` point cloud | Intensity-based grayscale |
| `world/leveled/camera_init/segmented_yaml` | Per-frame segmented points | Green |
| `world/leveled/camera_init/segmented_yaml_aggregated` | Accumulated segmented points | Yellow |
| `world/leveled/camera_init/axes` | `camera_init` frame axes | RGB (X=red, Y=green, Z=blue) |
| `world/leveled/camera_init/body/axes` | `body` frame axes | RGB |
| `world/leveled/camera_init/body/camera_link` | Pinhole camera model (static) | — |
| `world/leveled/camera_init/body/camera_link/axes` | `camera_link` frame axes | RGB |
| `world/leveled/camera_init/body/camera_link/image` | `/camera/image_undistorted` | Original image colors |

- All point clouds are logged under `camera_init/` so they're correctly positioned in the TF hierarchy.
- TF frames are polled at 30 Hz and visualized as RGB axis arrows (0.5 m length).
- The optional `leveling_rpy_deg` parameter applies a static rotation at the `world/leveled` entity — useful for leveling the scene for visualization without affecting actual ROS transforms.
- A **pinhole camera model** (`rr.Pinhole`) is logged at the `camera_link` entity using the camera intrinsics. This projects the 2D image into the 3D scene as a virtual camera frustum, allowing you to see where the camera is looking and how the image aligns with the point clouds. The camera uses `RDF` convention (X=Right, Y=Down, Z=Forward) matching the standard optical frame. The pinhole model is **dynamically re-logged** if the actual image dimensions differ from the parameter defaults — intrinsics are scaled proportionally to match the incoming image resolution.
- The **camera image** is logged under the pinhole entity, so Rerun displays it both as a 2D image and projected into the 3D frustum. Supported encodings: `bgr8`, `rgb8`, `bgra8`, `rgba8`, `mono8`, `mono16`.
- Requires the `rerun-sdk` Python package: `pip install rerun-sdk`

---

## Inspection Database

The `inspection_db.py` module provides a SQLite wrapper (`InspectionDB` class) for storing per-object inspection data. When `enable_db: true` is set in `params.yaml`, the fusion node writes observation and object records to the database on every processed frame.

### Schema

**`categories`** — Category lookup table:

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER PK | Category ID (from COCO) |
| `name` | TEXT | Category name (e.g. `"Poster"`) |

**`observations`** — One row per annotation per timestamp (per-frame):

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER PK AUTO | Row ID |
| `timestamp_ns` | INTEGER | LiDAR/image timestamp in nanoseconds |
| `track_id` | INTEGER | COCO track ID (groups observations of same object) |
| `category` | TEXT | Category name |
| `image_id` | INTEGER | COCO image ID |
| `image_file_name` | TEXT | Source image filename |
| `centroid_x/y/z` | REAL | 3D centroid in `camera_init` frame |
| `bbox3d_min_x/y/z` | REAL | Min corner of 3D bounding box |
| `bbox3d_max_x/y/z` | REAL | Max corner of 3D bounding box |
| `point_count` | INTEGER | Number of LiDAR points in this observation |
| `pcd_path` | TEXT | Path to per-observation PCD file |
| `mask_path` | TEXT | Path to mask PNG file |
| `bbox_2d` | TEXT | JSON-serialized `[x, y, w, h]` from COCO annotation |
| `tf_translation_x/y/z` | REAL | Sensor position (`camera_init` → `body`) at observation time |
| `tf_rotation_x/y/z/w` | REAL | Sensor orientation quaternion (`camera_init` → `body`) at observation time |
| `created_at` | TEXT | Row creation timestamp |

**`objects`** — One row per unique `track_id` (aggregated across frames):

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER PK AUTO | Row ID |
| `track_id` | INTEGER UNIQUE | Unique object tracker ID |
| `category` | TEXT | Category name |
| `centroid_x/y/z` | REAL | Aggregated 3D centroid |
| `bbox3d_min_x/y/z` | REAL | Overall 3D bbox min corner |
| `bbox3d_max_x/y/z` | REAL | Overall 3D bbox max corner |
| `total_point_count` | INTEGER | Total points across all observations |
| `observation_count` | INTEGER | Number of observations for this track |
| `first_seen_ns` | INTEGER | First timestamp this object was seen |
| `last_seen_ns` | INTEGER | Last timestamp this object was seen |
| `aggregated_pcd_path` | TEXT | Path to merged per-track PCD |
| `tf_translation_x/y/z` | REAL | Sensor position (`camera_init` → `body`) at last observation |
| `tf_rotation_x/y/z/w` | REAL | Sensor orientation quaternion at last observation |
| `created_at` | TEXT | Row creation timestamp |

### Usage

```python
from inspection_db import InspectionDB

db = InspectionDB("/path/to/inspection.db")

# Per-frame observation
db.add_observation(
    timestamp_ns=1782119257554358000,
    track_id=2947,
    category="Poster",
    centroid=(1.2, 3.4, 5.6),
    bbox3d_min=(1.0, 3.0, 5.0),
    bbox3d_max=(1.4, 3.8, 6.2),
    point_count=662,
    pcd_path="/path/to/pcd",
    mask_path="/path/to/mask.png",
    bbox_2d=[100, 200, 300, 400],
    tf_translation=(0.5, 1.0, 2.0),       # sensor position in camera_init
    tf_rotation=(0.0, 0.0, 0.0, 1.0),     # sensor orientation quaternion
)

# Aggregated object (upserted on each new observation)
db.upsert_object(
    track_id=2947,
    category="Poster",
    centroid=(1.2, 3.4, 5.6),
    total_point_count=6626,
    observation_count=10,
    first_seen_ns=1782119257554358000,
    last_seen_ns=1782119457554358000,
    aggregated_pcd_path="/path/to/track2947_aggregated.pcd",
    tf_translation=(0.5, 1.0, 2.0),       # sensor position in camera_init
    tf_rotation=(0.0, 0.0, 0.0, 1.0),     # sensor orientation quaternion
)

# Query
objects = db.get_all_objects()
observations = db.get_observations(track_id=2947)
db.close()
```

### Querying the Database

```bash
# Top objects by point count
sqlite3 inspection.db \
  "SELECT track_id, category, observation_count, total_point_count
   FROM objects ORDER BY total_point_count DESC LIMIT 10;"

# All observations for a specific track
sqlite3 inspection.db \
  "SELECT timestamp_ns, point_count, pcd_path
   FROM observations WHERE track_id=2947 ORDER BY timestamp_ns;"

# Object count by category
sqlite3 inspection.db \
  "SELECT category, COUNT(*) as cnt, SUM(total_point_count) as pts
   FROM objects GROUP BY category ORDER BY cnt DESC;"
```

---

## Inter-Node Data Flow

```
┌───────────────────────────────────┐
│  ROS 2 Bag (sqlite3, --clock)     │
│  /livox/lidar, /livox/imu         │
└──────┬────────────────────────────┘
       │
       ▼
┌─────────────┐         ┌──────────────────────────┐
│  FAST-LIO   │────────►│  /cloud_registered_body  │
│             │         │  /cloud_registered       │
└─────────────┘         └─────────────┬────────────┘
                                      │
                     ┌────────────────┼──────────────────┐
                     │                │                  │
                     ▼                ▼                  ▼
           ┌─────────────────┐ ┌──────────────┐  ┌───────────────────┐
           │ sync_test_image │ │  sync_node   │  │ fusion_node  │
           │   _publisher    │ │              │  │                   │
           │                 │ │ (ApproxTime  │  │ (matches YAML or  │
           │ (triggered by   │ │  Sync, 50ms) │  │  COCO JSON by     │
           │  /livox/lidar)  │ │              │  │  timestamp)       │
           └────────┬────────┘ └──────┬───────┘  └────────┬──────────┘
                    │                 │                   │
                    ▼                 ▼                   ▼
      /camera/image_undistorted  /synced_sensor_data  /pointcloud/segmented_yaml
      /camera/camera_info                              /pointcloud/segmented_yaml_aggregated
                                                        /object/global_pose
                                                        *.pcd files
                                                        inspection.db
```

**Two independent paths:**

- **Sync path**: `sync_node` → `/synced_sensor_data`. Verifies that images and point clouds can be aligned in time.
- **Fusion path**: `fusion_node` subscribes to `/synced_sensor_data` and matches it with 2D annotations (YAML bboxes or COCO masks) — it segments the LiDAR points, publishes results, saves PCDs, and writes to the SQLite database.

---

## Custom Messages

### `SyncedSensorData.msg`

```
std_msgs/Header header          # reference stamp (from image)
sensor_msgs/Image image         # undistorted camera image
sensor_msgs/CameraInfo camera_info
sensor_msgs/PointCloud2 pointcloud
```

---

## Launch Files

### `test_sync.launch.py`

Tests the synchronization pipeline only.

| Component | Description |
|-----------|-------------|
| `ros2 bag play` | Replays the bag with `--clock` (sim time) |
| FAST-LIO | Runs `mapping.launch.py` with `use_sim_time:=true` |
| `sync_node` | Synchronizes and publishes `/synced_sensor_data` |
| RViz2 | Visualization |

### `test_fusion.launch.py`

Full pipeline including segmentation.

| Component | Description |
|-----------|-------------|
| `ros2 bag play` | Replays the bag with `--clock` |
| FAST-LIO | Runs `mapping.launch.py` with `use_sim_time:=true` |
| `sync_node` | Loads images from disk and pairs with LiDAR by timestamp |
| `fusion_node` | Segments point clouds using 2D annotations (bbox or mask mode) |
| `static_transform_publisher` | Publishes `body` → `camera_link` TF |
| RViz2 | Visualization |

### `test_fusion_rerun.launch.py`

Same as `test_fusion.launch.py` but replaces RViz with the Rerun bridge for visualization.

| Component | Description |
|-----------|-------------|
| `ros2 bag play` | Replays the bag with `--clock` |
| FAST-LIO | Runs `mapping.launch.py` with `use_sim_time:=true` |
| `sync_node` | Loads images from disk and pairs with LiDAR by timestamp |
| `fusion_node` | Segments point clouds using 2D annotations (bbox or mask mode) |
| `static_transform_publisher` | Publishes `body` → `camera_link` TF |
| `rerun_bridge_node` | Logs ROS 2 data to Rerun viewer |

**Launch arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `leveling_rpy_deg` | `[0.0, 20.0, 0.0]` | Visualization-only leveling rotation (roll, pitch, yaw in degrees) |
| `camera_fx` | `597.384` | Camera focal length x (pixels) |
| `camera_fy` | `597.443` | Camera focal length y (pixels) |
| `camera_cx` | `774.861` | Camera principal point x (pixels) |
| `camera_cy` | `1013.173` | Camera principal point y (pixels) |
| `image_width` | `1520` | Image width in pixels |
| `image_height` | `2016` | Image height in pixels |

**Static TF (body → camera_link):**

The `static_transform_publisher` with parent=`body`, child=`camera_link` publishes **T_{camera→body}** (the inverse of T_{body→camera}).

Configuration is read from `config/launch_config.yaml`:

```yaml
static_tf_body_to_camera:
  translation: [0.03159816, 0.06295154, -0.11131637]       # x, y, z (metres)
  rotation_xyzw: [-0.65290271, 0.04396615, -0.27557732, 0.70416061]  # qx, qy, qz, qw
  parent_frame: body
  child_frame: camera_link
```

Derived from `calibration.json` (left camera) and FAST-LIO `mid360.yaml`:
- **T_{body→camera}**: `R = R_lidar_camera`, `t = t_lidar_camera − R_lidar_camera · t_body_lidar`
  - `R_body_lidar = I` (from FAST-LIO IMU extrinsic), so body and lidar share orientation
  - `t_body_lidar = [0.011, 0.02329, −0.04412]` (lidar origin in body frame)
  - `t_lidar_camera = [−0.0205, −0.0702, −0.0292]` (lidar origin in left camera frame)
- **T_{camera→body}** (what static TF publishes): `R = R_lidar_camera^T`, `t = t_body_lidar − R_lidar_camera^T · t_lidar_camera`

---

## Calibration & Configuration

### Configuration Files

| File | Description |
|------|-------------|
| `config/params.yaml` | ROS 2 node parameters (passed to nodes via `--params-file`) |
| `config/launch_config.yaml` | Launch-level config: bag path and static TF (read directly by launch files) |

### Camera Intrinsics

Sourced from `rosbags/2026-06-11_16-50-08/info/calibration.json`. The fusion node uses the **left camera** intrinsics at full resolution (3040×4032):

| Parameter | Value |
|-----------|-------|
| `fx` | 1190.499 |
| `fy` | 1190.345 |
| `cx` | 1545.215 |
| `cy` | 1983.937 |
| Image size | 3040 × 4032 (portrait) |
| Distortion model | OPENCV_FISHEYE (already undistorted before fusion) |

The Rerun bridge uses the **right camera** intrinsics at half resolution (1520×2016):

| Parameter | Value |
|-----------|-------|
| `fx` | 597.384 |
| `fy` | 597.443 |
| `cx` | 774.861 |
| `cy` | 1013.173 |
| Image size | 1520 × 2016 (portrait) |

### Annotation Formats

**Bbox YAML mode** (`annotation_mode: bbox`):

Each file in `yaml_dir` is named `<epoch_nanoseconds>.yaml` and contains:

```yaml
# New format (multiple bboxes)
bounding_boxes:
  - x_min: 377
    y_min: 283
    x_max: 617
    y_max: 592
    category: Poster
```

or (old format, single bbox):

```yaml
bounding_box:
  x_min: 377
  y_min: 283
  x_max: 617
  y_max: 592
```

The filename timestamp is matched against the LiDAR cloud stamp (tolerance configurable via `yaml_match_tolerance_ns`).

**Mask JSON mode** (`annotation_mode: mask`):

A single COCO-format JSON file (`mask_json_path`) containing `images`, `annotations`, and `categories` arrays. Images are matched by exact timestamp (file name stem == `str(timestamp_ns)`). Each annotation contains segmentation polygons, a bounding box, and a track ID for cross-frame object tracking. See [`_parse_mask()`](#_parse_mask-helper) above for the returned data structure.

### Key File Paths

| Path | Description |
|------|-------------|
| `rosbags/2026-06-22_17-IWLG/data/bag/bag.db3` | Input ROS 2 bag (IWLG dataset) |
| `masks/Ground_Truth_20/` | Directory of bbox YAML files (bbox mode) |
| `masks/2026-6-22-IWLG/sam3_tracking_segmentation.json` | COCO-format segmentation JSON (mask mode) |
| `pcd_outputs/` | Output directory for segmented PCD files |
| `inspection.db` | SQLite inspection database |
| `data/2026-06-22-IWLG/frames/` | Pre-undistorted images for replay |

---

## Build & Run

### Build

```bash
cd /home/robot/fastlio_ws
colcon build --packages-select inspection_grounding
source install/setup.bash
```

### Run — Sync test

```bash
ros2 launch inspection_grounding test_sync.launch.py
```

### Run — Full fusion test (RViz)

```bash
ros2 launch inspection_grounding test_fusion.launch.py
```

### Run — Full fusion test (Rerun)

```bash
ros2 launch inspection_grounding test_fusion_rerun.launch.py
```

### Verify outputs

```bash
# Segmented point cloud topic
ros2 topic echo /pointcloud/segmented_yaml

# Aggregated point cloud (all frames accumulated in global frame)
ros2 topic echo /pointcloud/segmented_yaml_aggregated

# Object centroid pose
ros2 topic echo /object/global_pose

# Saved PCD files
ls pcd_outputs/

# Query the inspection database
sqlite3 inspection.db \
  "SELECT track_id, category, observation_count, total_point_count
   FROM objects ORDER BY total_point_count DESC LIMIT 10;"
```
