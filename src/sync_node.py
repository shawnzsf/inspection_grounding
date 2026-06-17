#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from message_filters import Subscriber, ApproximateTimeSynchronizer
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from std_msgs.msg import Header
from inspection_grounding.msg import SyncedSensorData

class SyncNode(Node):
    def __init__(self):
        super().__init__('sync_node')

        # Parameters
        self.declare_parameter('image_topic', '/camera/image_undistorted')
        self.declare_parameter('camera_info_topic', '/camera/camera_info')
        self.declare_parameter('pointcloud_topic', '/cloud_registered')
        self.declare_parameter('output_topic', '/synced_sensor_data')
        self.declare_parameter('sync_queue_size', 10)
        self.declare_parameter('sync_slop', 0.05)

        image_topic = self.get_parameter('image_topic').value
        camera_info_topic = self.get_parameter('camera_info_topic').value
        pointcloud_topic = self.get_parameter('pointcloud_topic').value
        output_topic = self.get_parameter('output_topic').value
        queue_size = self.get_parameter('sync_queue_size').value
        slop = self.get_parameter('sync_slop').value

        # Subscribers
        self.sub_image = Subscriber(self, Image, image_topic)
        self.sub_info = Subscriber(self, CameraInfo, camera_info_topic)
        self.sub_pc = Subscriber(self, PointCloud2, pointcloud_topic)

        # Synchronizer
        self.sync = ApproximateTimeSynchronizer(
            [self.sub_image, self.sub_info, self.sub_pc],
            queue_size=queue_size,
            slop=slop
        )
        self.sync.registerCallback(self.sync_callback)

        # Publisher
        self.pub_synced = self.create_publisher(SyncedSensorData, output_topic, 10)
        self.get_logger().info(
            f"Sync Node initialized. "
            f"Subscribed to: {image_topic}, {camera_info_topic}, {pointcloud_topic}. "
            f"Publishing to: {output_topic}. "
            f"Waiting for data..."
        )

    def sync_callback(self, image_msg, info_msg, pc_msg):
        synced_msg = SyncedSensorData()
        synced_msg.header = image_msg.header  # Use image timestamp as reference
        synced_msg.image = image_msg
        synced_msg.camera_info = info_msg
        synced_msg.pointcloud = pc_msg

        self.pub_synced.publish(synced_msg)

def main(args=None):
    rclpy.init(args=args)
    node = SyncNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()