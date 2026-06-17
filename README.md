# inspection_grounding

LiDAR–Camera fusion and object grounding for tunnel/inspection robotics.  
The package synchronizes camera images with LiDAR point clouds, projects 3D points onto 2D image bounding boxes, and segments the corresponding 3D object points for downstream inspection.

---

## Table of Contents

1. [Package Structure](#package-structure)
2. [Nodes](#nodes)
   - [sync_node](#1-sync_nodepy)
   - [sync_test_image_publisher](#2-sync_test_image_publisherpy)
   - [fusion_yaml_node](#3-fusion_yaml_nodepy)
3. [Inter-Node Data Flow](#inter-node-data-flow)
4. [Custom Messages](#custom-messages)
5. [Launch Files](#launch-files)
6. [Calibration & Configuration](#calibration--configuration)
7. [Build & Run](#build--run)

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
│   ├── sync_test_image_publisher.py      # offline image replay node (test only)
│   └── fusion_yaml_node.py               # 2D-bbox → 3D pointcloud segmentation node
└── launch/
    ├── test_sync.launch.py               # sync pipeline test
    └── test_fusion.launch.py             # full fusion pipeline test
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

### 2. `sync_test_image_publisher.py`

**Purpose** — Offline test node that replays pre-undistorted images from disk, triggered by incoming LiDAR messages. Lets you test the sync pipeline without a live camera.

| Direction | Topic | Type | QoS |
|-----------|-------|------|-----|
| Sub | `/livox/lidar` | `sensor_msgs/PointCloud2` | BEST_EFFORT, KEEP_LAST(10) |
| Pub | `/camera/image_undistorted` | `sensor_msgs/Image` | default |
| Pub | `/camera/camera_info` | `sensor_msgs/CameraInfo` | default |

**Internal workflow:**

1. On startup, loads and sorts all `*.png` / `*.jpg` files from `data/2026-06-11-HKUMTR/camera/right`.
2. Each LiDAR message triggers `lidar_callback`:
   - Reads the next image (loops back to index 0 at the end).
   - Parses the nanosecond timestamp from the filename (e.g. `1781168047875925000.jpg` → epoch ns).
   - Converts to an `Image` message via `CvBridge` (encoding `bgr8`).
   - Publishes the image and a `CameraInfo` with the real undistorted intrinsics from `calibration.json` (right camera: `fx=1194.769, fy=1194.885, cx=1549.723, cy=2026.345`).
   - If the filename cannot be parsed as a timestamp, the LiDAR header stamp is used as fallback.

> ℹ️ The intrinsics in this node match those in `fusion_yaml_node` and are sourced from `calibration.json` (see [Calibration](#calibration--configuration)).

---

### 3. `fusion_yaml_node.py`

**Purpose** — The core fusion node. Matches each LiDAR scan with a 2D bounding-box YAML file by timestamp, projects the 3D points onto the image plane, segments points inside the bbox, and publishes the result.

| Direction | Topic | Type | QoS |
|-----------|-------|------|-----|
| Sub | `/cloud_registered_body` | `sensor_msgs/PointCloud2` | BEST_EFFORT, KEEP_LAST(10) |
| Pub | `/pointcloud/segmented_yaml` | `sensor_msgs/PointCloud2` | RELIABLE, KEEP_LAST(10) |
| Pub | `/object/global_pose` | `geometry_msgs/PoseStamped` | RELIABLE, KEEP_LAST(10) |
| Pub | `/bbox_marker` | `visualization_msgs/Marker` | RELIABLE, KEEP_LAST(10) |

**Parameters**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `yaml_dir` | `/home/robot/fastlio_ws/masks` | Directory containing bbox YAML files |
| `output_dir` | `/home/robot/fastlio_ws/legacy_outputs` | Directory for saved PCD files |
| `target_frame` | `camera_init` | Global/world frame (FAST-LIO origin) |
| `camera_frame` | `camera_link` | Camera optical frame |

**Internal workflow (`pc_callback`):**

```
PointCloud2 msg
      │
      ▼
┌──────────────────────────┐
│ 1. Timestamp matching    │  Scan yaml_dir, parse filename as epoch-ns,
│    (150 ms tolerance)    │  pick the closest YAML to the cloud stamp.
└──────────────────────────┘
      │
      ▼
┌──────────────────────────┐
│ 2. Load bbox from YAML   │  bounding_box: {x_min, y_min, x_max, y_max}
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
└──────────────────────────┘
      │
      ▼
┌──────────────────────────┐
│ 6. Segment inside bbox   │  mask = (z>0) & (x_min≤u≤x_max) & (y_min≤v≤y_max)
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
│ 8. Publish & save        │  • PointCloud2 (segmented, in target frame)
│                          │  • PoseStamped (centroid, in target frame)
│                          │  • Marker (bbox frustum + centroid sphere)
│                          │  • PCD file (ASCII, in source frame)
└──────────────────────────┘
```

**Republishing** — A 2 Hz timer (`_republish_callback`) re-emits the last segmented cloud, pose, and markers so RViz always has data to display even when the input is single-shot.

**Bbox visualization** — The bbox is drawn as a `LINE_LIST` marker in the camera frame: a rectangle at `bbox_vis_depth` (2.0 m) plus four rays from the camera origin to each corner, forming a frustum.

---

## Inter-Node Data Flow

```
┌─────────────┐
│  ROS 2 Bag  │  (sqlite3, --clock)
│  /livox/lidar, /livox/imu
└──────┬──────┘
       │
       ▼
┌─────────────┐         ┌──────────────────────────┐
│  FAST-LIO   │────────►│  /cloud_registered_body  │
│             │         │  /cloud_registered       │
└─────────────┘         └─────────────┬────────────┘
                                      │
                    ┌─────────────────┼──────────────────┐
                    │                 │                  │
                    ▼                 ▼                  ▼
          ┌─────────────────┐ ┌──────────────┐  ┌───────────────────┐
          │ sync_test_image │ │  sync_node   │  │ fusion_yaml_node  │
          │   _publisher    │ │              │  │                   │
          │                 │ │ (ApproxTime  │  │ (matches YAML     │
          │ (triggered by   │ │  Sync, 50ms) │  │  by timestamp)    │
          │  /livox/lidar)  │ │              │  │                   │
          └────────┬────────┘ └──────┬───────┘  └────────┬──────────┘
                   │                 │                   │
                   ▼                 ▼                   ▼
     /camera/image_undistorted  /synced_sensor_data  /pointcloud/segmented_yaml
     /camera/camera_info                              /object/global_pose
                                                      /bbox_marker
                                                      *.pcd files
```

**Two independent paths:**

- **Sync path** (blue): `sync_test_image_publisher` → `sync_node` → `/synced_sensor_data`. Verifies that images and point clouds can be aligned in time.
- **Fusion path** (green): `fusion_yaml_node` directly subscribes to `/cloud_registered_body` and matches it with YAML masks — it does **not** depend on `sync_node`. The sync path exists for testing/validation.

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
| `sync_test_image_publisher` | Replays images triggered by LiDAR |
| `sync_node` | Synchronizes and publishes `/synced_sensor_data` |
| RViz2 | Visualization |

### `test_fusion.launch.py`

Full pipeline including segmentation.

| Component | Description |
|-----------|-------------|
| `ros2 bag play` | Replays the bag with `--clock` |
| FAST-LIO | Runs `mapping.launch.py` with `use_sim_time:=true` |
| `sync_test_image_publisher` | Replays images (for sync validation) |
| `sync_node` | Synchronizes (for sync validation) |
| `fusion_yaml_node` | Segments point clouds using YAML bboxes |
| `static_transform_publisher` | Publishes `body` → `camera_link` TF |
| RViz2 | Visualization |

**Static TF (body → camera_link):**

The `static_transform_publisher` with parent=`body`, child=`camera_link` publishes **T_{camera→body}** (the inverse of T_{body→camera}).

| | x | y | z |
|---|---|---|---|
| Translation | 0.0281 | -0.0128 | -0.1120 |

| | qx | qy | qz | qw |
|---|---|---|---|---|
| Rotation | 0.0417 | -0.6498 | 0.7106 | -0.2664 |

Derived from `calibration.json` (right camera) and FAST-LIO `mid360.yaml`:
- **T_{body→camera}**: `R = R_lidar_camera`, `t = t_lidar_camera − R_lidar_camera · t_body_lidar`
  - `R_body_lidar = I` (from FAST-LIO IMU extrinsic), so body and lidar share orientation
  - `t_body_lidar = [0.011, 0.02329, −0.04412]` (lidar origin in body frame)
  - `t_lidar_camera = [−0.0205, −0.0702, −0.0292]` (lidar origin in right camera frame)
- **T_{camera→body}** (what static TF publishes): `R = R_lidar_camera^T`, `t = t_body_lidar − R_lidar_camera^T · t_lidar_camera`

---

## Calibration & Configuration

### Camera Intrinsics (right camera, undistorted)

Hardcoded in `fusion_yaml_node.py` and `sync_test_image_publisher.py`, sourced from `rosbags/2026-06-11_16-50-08/info/calibration.json`:

| Parameter | Value |
|-----------|-------|
| `fx` | 1194.769 |
| `fy` | 1194.885 |
| `cx` | 1549.723 |
| `cy` | 2026.345 |
| Image size | 3040 × 4032 (portrait) |
| Distortion model | OPENCV_FISHEYE (already undistorted before fusion) |

### YAML Mask Format

Each file in `yaml_dir` is named `<epoch_nanoseconds>.yaml` and contains:

```yaml
bounding_box:
  x_min: 377
  y_min: 283
  x_max: 617
  y_max: 592
```

The filename timestamp is matched against the LiDAR cloud stamp (150 ms tolerance).

### Key File Paths

| Path | Description |
|------|-------------|
| `rosbags/2026-06-11_16-50-08/data/bag/bag.db3` | Input ROS 2 bag |
| `masks/` | Directory of bbox YAML files |
| `legacy_outputs/` | Output directory for segmented PCD files |
| `data/2026-06-11-HKUMTR/camera/right/` | Pre-undistorted images for test replay |

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

### Run — Full fusion test

```bash
ros2 launch inspection_grounding test_fusion.launch.py
```

### Verify outputs

```bash
# Segmented point cloud topic
ros2 topic echo /pointcloud/segmented_yaml

# Object centroid pose
ros2 topic echo /object/global_pose

# Saved PCD files
ls legacy_outputs/
```
