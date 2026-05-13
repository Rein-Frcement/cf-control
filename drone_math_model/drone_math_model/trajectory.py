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
        self.startup_delay = 10.0
        self.started = False
        self.timer = self.create_timer(0.01, self.publish_messages)

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

        ASCEND_TIME = 20.0
        HOVER_TIME = 5.0
        HOVER_ALT = 1.0
        CIRCLE_R = 0.5
        CIRCLE_W = 0.5

        t_fly = t - self.startup_delay

        if t_fly < ASCEND_TIME:
            climb_rate = HOVER_ALT / ASCEND_TIME
            z = climb_rate * t_fly
            pos = Vector3(x=0.0, y=0.0, z=z)
            vel = Vector3(x=0.0, y=0.0, z=climb_rate)
            acc = Vector3(x=0.0, y=0.0, z=0.0)
            jerk = Vector3(x=0.0, y=0.0, z=0.0)
            snap = Vector3(x=0.0, y=0.0, z=0.0)

        elif t_fly < ASCEND_TIME + HOVER_TIME:
            pos = Vector3(x=0.0, y=0.0, z=HOVER_ALT)
            vel = Vector3(x=0.0, y=0.0, z=0.0)
            acc = Vector3(x=0.0, y=0.0, z=0.0)
            jerk = Vector3(x=0.0, y=0.0, z=0.0)
            snap = Vector3(x=0.0, y=0.0, z=0.0)

        else:
            tc = t_fly - ASCEND_TIME - HOVER_TIME
            R = CIRCLE_R
            w = CIRCLE_W
            c = math.cos(w * tc)
            s = math.sin(w * tc)

            pos = Vector3(x=R * c, y=R * s, z=HOVER_ALT)
            vel = Vector3(x=-R * w * s, y=R * w * c, z=0.0)
            acc = Vector3(x=-R * w**2 * c, y=-R * w**2 * s, z=0.0)
            jerk = Vector3(x=R * w**3 * s, y=-R * w**3 * c, z=0.0)
            snap = Vector3(x=R * w**4 * c, y=R * w**4 * s, z=0.0)

        trajectory_msg.position = pos
        trajectory_msg.velocity = vel
        trajectory_msg.acceleration = acc
        trajectory_msg.jerk = jerk
        trajectory_msg.snap = snap

        trajectory_msg.yaw = 0.0
        trajectory_msg.yaw_dot = 0.0
        trajectory_msg.yaw_ddot = 0.0

        self.trajectory_publisher.publish(trajectory_msg)

        regulator_msg = ContorlerParameters()
        regulator_msg.kp = Vector3(x=0.2, y=0.2, z=1.9)
        regulator_msg.kv = Vector3(x=0.2, y=0.2, z=0.7)
        regulator_msg.kr = Vector3(x=3.73e-3, y=3.73e-3, z=2.0e-3)
        regulator_msg.kw = Vector3(x=3.5e-4, y=3.5e-4, z=3.0e-4)

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
