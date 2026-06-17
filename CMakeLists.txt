cmake_minimum_required(VERSION 3.8)
project(inspection_grounding)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 1. Find ROS 2 dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

# 2. Generate the custom message: SyncedSensorData
#    Used by sync_node to publish synchronized image + LiDAR data
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/SyncedSensorData.msg"
  DEPENDENCIES std_msgs sensor_msgs
)

# 3. Install Python nodes
#    - sync_node:              synchronizes camera images with LiDAR point clouds
#    - sync_test_image_publisher: replays images from a directory for testing
#    - fusion_yaml_node:       segments LiDAR points using 2D bbox YAMLs and publishes results
install(PROGRAMS
  src/sync_node.py
  src/sync_test_image_publisher.py
  src/fusion_yaml_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# 4. Install launch files
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

# 5. Install config files
install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

# 6. Finalize the package
ament_package()
