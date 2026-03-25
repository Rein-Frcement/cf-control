import rclpy
from rclpy.node import Node

from cf_control_msgs.msg import ThrustAndTorque


class DroneTester(Node):
    def __init__(self):
        super().__init__('drone_tester')
        self.publisher_ = self.create_publisher(ThrustAndTorque, 'drone_control', 10)

        self.timer = self.create_timer(0.01, self.timer_callback)
        self.start_time = self.get_clock().now()

        self.get_logger().info('Drone Tester started. Sending control signals...')

    def timer_callback(self):
        msg = ThrustAndTorque()

        msg.timestamp = self.get_clock().now().nanoseconds

        now = self.get_clock().now()
        elapsed = (now - self.start_time).nanoseconds / 1e9

        if elapsed < 5.0:
            msg.collective_thrust = 5.0
            msg.torque.x = 0.0
            msg.torque.y = 0.0
            msg.torque.z = 0.0
        elif elapsed < 15.0:
            msg.collective_thrust = 22.0
            msg.torque.x = 0.0
            msg.torque.y = 0.0
            msg.torque.z = 0.0
        else:
            msg.collective_thrust = 0.0
            msg.torque.x = 0.0

        self.publisher_.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = DroneTester()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
