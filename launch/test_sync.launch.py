import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # 1. Play the ROS 2 bag with simulated clock
    bag_path = "/home/robot/fastlio_ws/rosbags/2026-06-22_17-IWLG/data/bag/bag.db3"
    bag_play = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', '-s', 'sqlite3', bag_path, '--clock'],
        output='screen'
    )
    
    # 2. Launch FAST-LIO (Adjust the launch file name if yours is different, e.g., mapping_avia.launch.py)
    fastlio_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('fast_lio'), 'launch', 'mapping.launch.py')
        ]),
        launch_arguments={'use_sim_time': 'true'}.items()
    )
    
    # 3. Mock Image Publisher (syncs undistorted images to LiDAR timestamps)
    mock_image_pub = Node(
        package='inspection_grounding',
        executable='sync_test_image_publisher.py',
        name='sync_test_image_publisher',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )
    
    # 4. The Sync Node (to verify it receives both)
    sync_node = Node(
        package='inspection_grounding',
        executable='sync_node.py',
        name='sync_node',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )
    
    # 5. RViz2 for visualization
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    return LaunchDescription([
        bag_play,
        fastlio_launch,
        mock_image_pub,
        sync_node,
        rviz_node
    ])