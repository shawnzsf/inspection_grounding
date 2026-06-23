import os
import yaml
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory('inspection_grounding')

    # Launch-level config (bag path, static TF) — read directly by launch
    launch_cfg_path = os.path.join(pkg_share, 'config', 'launch_config.yaml')
    with open(launch_cfg_path, 'r') as f:
        launch_cfg = yaml.safe_load(f)

    bag_path = launch_cfg['bag_path']
    tf_cfg = launch_cfg['static_tf_body_to_camera']

    # ROS 2 node parameters — passed to nodes via --params-file
    params_path = os.path.join(pkg_share, 'config', 'params.yaml')

    # 1. Play the ROS 2 bag with simulated clock
    bag_play = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', '-s', 'sqlite3', bag_path, '--clock'],
        output='screen'
    )

    # 2. Launch FAST-LIO
    fastlio_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('fast_lio'), 'launch', 'mapping.launch.py')
        ]),
        launch_arguments={'use_sim_time': 'true'}.items()
    )

    # 3. The Sync Node (directly loads images from disk and pairs with LiDAR by timestamp)
    sync_node = Node(
        package='inspection_grounding',
        executable='sync_node.py',
        name='sync_node',
        output='screen',
        parameters=[params_path]
    )

    # 4. The YAML Fusion Node
    # Subscribes to /cloud_registered, checks for matching YAML, outputs colored PC and TF
    yaml_fusion_node = Node(
        package='inspection_grounding',
        executable='fusion_yaml_node.py',
        name='fusion_yaml_node',
        output='screen',
        parameters=[params_path]
    )

    # 5. RViz2 for visualization
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    # Static TF: body (IMU) → camera_link
    # Publishes T_{camera→body}: parent=body, child=camera_link
    #
    # From calibration.json (right camera):
    #   R_lidar_camera (3x3) and t_lidar_camera (lidar origin in camera frame)
    # From FAST-LIO mid360.yaml (IMU extrinsic):
    #   R_body_lidar = I,  t_body_lidar = [0.011, 0.02329, -0.04412]
    #
    # T_{body→camera} = R_lidar_camera, t = t_lidar_camera - R_lidar_camera * t_body_lidar
    # T_{camera→body} = inv(T_{body→camera}):
    #   R = R_lidar_camera^T
    #   t = t_body_lidar - R_lidar_camera^T * t_lidar_camera
    #     = [0.02810730, -0.01283486, -0.11198965]
    # Quaternion from R_lidar_camera^T
    static_tf_lidar_to_cam = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_body_to_cam',
        arguments=[
            *[str(v) for v in tf_cfg['translation']],
            *[str(v) for v in tf_cfg['rotation_xyzw']],
            tf_cfg['parent_frame'],
            tf_cfg['child_frame']
        ],
        parameters=[{'use_sim_time': True}]
    )

    return LaunchDescription([
        bag_play,
        fastlio_launch,
        sync_node,
        yaml_fusion_node,
        static_tf_lidar_to_cam,
        rviz_node
    ])