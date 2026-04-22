import math

import rclpy
from geometry_msgs.msg import Vector3
from rclpy.node import Node

from cf_control_msgs.msg import ContorlerParameters, Flat


class TrajectoryPublisher(Node):
    def __init__(self):
        super().__init__('drone_trajectory_publisher')

        self.trajectory_publisher = self.create_publisher(Flat, 'drone_trajectory', 10)
        self.regulator_publisher = self.create_publisher(
            ContorlerParameters, 'drone_regulator_parameters', 10
        )

        self.start_time = self.get_clock().now()
        self.timer = self.create_timer(0.1, self.publish_messages)

        self.get_logger().info('drone_trajectory_publisher node started')

    def publish_messages(self):
        t = (self.get_clock().now() - self.start_time).nanoseconds * 1e-9
        trajectory_msg = Flat()
        trajectory_msg.timestamp = self.get_clock().now().nanoseconds

        amplitude = 0.5
        omega = 0.5

        trajectory_msg.position = Vector3(
            x=amplitude * math.sin(omega * t),
            y=amplitude * math.cos(omega * t),
            z=1.0,
        )

        trajectory_msg.velocity = Vector3(
            x=amplitude * omega * math.cos(omega * t),
            y=-amplitude * omega * math.sin(omega * t),
            z=0.0,
        )

        trajectory_msg.acceleration = Vector3(
            x=-amplitude * omega**2 * math.sin(omega * t),
            y=-amplitude * omega**2 * math.cos(omega * t),
            z=0.0,
        )

        trajectory_msg.jerk = Vector3(
            x=-amplitude * omega**3 * math.cos(omega * t),
            y=amplitude * omega**3 * math.sin(omega * t),
            z=0.0,
        )

        trajectory_msg.snap = Vector3(
            x=amplitude * omega**4 * math.sin(omega * t),
            y=amplitude * omega**4 * math.cos(omega * t),
            z=0.0,
        )

        trajectory_msg.yaw = 0.0
        trajectory_msg.yaw_dot = 0.0
        trajectory_msg.yaw_ddot = 0.0

        self.trajectory_publisher.publish(trajectory_msg)

        regulator_msg = ContorlerParameters()
        regulator_msg.Kp = Vector3(x=4.0, y=4.0, z=4.0)
        regulator_msg.Kv = Vector3(x=2.0, y=2.0, z=2.0)
        regulator_msg.KR = Vector3(x=2.0, y=2.0, z=2.0)
        regulator_msg.Kw = Vector3(x=0.15, y=0.15, z=0.15)

        self.regulator_publisher.publish(regulator_msg)

    def destroy_node(self):
        self.get_logger().info('drone_trajectory_publisher node shutting down')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryPublisher()

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
