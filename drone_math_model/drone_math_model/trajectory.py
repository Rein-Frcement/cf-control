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
        self.startup_delay = 10.0
        self.started = False
        self.timer = self.create_timer(0.1, self.publish_messages)

        self.get_logger().info(
            'drone_trajectory_publisher node started, waiting 10s before sending commands'
        )

    def publish_messages(self):
        t = (self.get_clock().now() - self.start_time).nanoseconds * 1e-9
        if t < self.startup_delay:
            if not self.started:
                self.get_logger().info(
                    f'Waiting {self.startup_delay:.0f}s before starting motor commands '
                    f'(time={t:.1f}s)'
                )
                self.started = True
            return

        trajectory_msg = Flat()
        trajectory_msg.timestamp = self.get_clock().now().nanoseconds

        ascend_time = 20.0
        target_altitude = 1.2
        initial_altitude = 1.0
        climb_rate = (target_altitude - initial_altitude) / ascend_time

        z = initial_altitude + climb_rate * t
        zdot = climb_rate
        zddot = 0.0

        trajectory_msg.position = Vector3(x=0.0, y=0.0, z=z)
        trajectory_msg.velocity = Vector3(x=0.0, y=0.0, z=zdot)
        trajectory_msg.acceleration = Vector3(x=0.0, y=0.0, z=zddot)
        trajectory_msg.jerk = Vector3(x=0.0, y=0.0, z=0.0)
        trajectory_msg.snap = Vector3(x=0.0, y=0.0, z=0.0)

        trajectory_msg.yaw = 0.0
        trajectory_msg.yaw_dot = 0.0
        trajectory_msg.yaw_ddot = 0.0

        self.trajectory_publisher.publish(trajectory_msg)

        regulator_msg = ContorlerParameters()
        regulator_msg.kp = Vector3(x=1.0, y=1.0, z=1.5)
        regulator_msg.kv = Vector3(x=0.5, y=0.5, z=0.8)
        regulator_msg.kr = Vector3(x=0.1, y=0.1, z=0.1)
        regulator_msg.kw = Vector3(x=0.01, y=0.01, z=0.01)

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
