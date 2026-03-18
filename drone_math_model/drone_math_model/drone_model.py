import numpy as np
import rclpy
from cf_control_msgs.msg import ThrustAndTorque
from numpy.linalg import inv
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quaternion_rotate(q, v):
    q_vec = np.array([0, v[0], v[1], v[2]])
    q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
    res = quaternion_multiply(quaternion_multiply(q, q_vec), q_conj)
    return res[1:]


class DroneNode(Node):
    def __init__(self):
        super().__init__('drone_node')

        self.subscription = self.create_subscription(
            ThrustAndTorque, 'drone_control', self.listener_callback, 10
        )

        self.publisher_ = self.create_publisher(Float64MultiArray, 'drone_state', 10)

        self.m = 1.0
        self.g_vec = np.array([0, 0, 9.81])
        self.J = np.diag([1.0, 1.0, 1.0])
        self.J_inv = inv(self.J)

        self.r = np.zeros(3)
        self.v = np.zeros(3)
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.omega = np.zeros(3)

        self.last_timestamp = None

        self.get_logger().info('Drone Node with ThrustAndTorque started.')

    def listener_callback(self, msg):
        if self.last_timestamp is None:
            self.last_timestamp = msg.timestamp
            return

        dt = (msg.timestamp - self.last_timestamp) / 1e9
        if dt <= 0:
            return

        self.last_timestamp = msg.timestamp

        T_val = msg.collective_thrust
        tau = np.array([msg.torque.x, msg.torque.y, msg.torque.z])
        T_body = np.array([0.0, 0.0, T_val])

        d_omega = self.J_inv @ (tau - np.cross(self.omega, self.J @ self.omega))
        d_v = -self.g_vec + (1.0 / self.m) * quaternion_rotate(self.q, T_body)

        omega_quat = np.array([0.0, self.omega[0], self.omega[1], self.omega[2]])
        d_q = 0.5 * quaternion_multiply(self.q, omega_quat)

        self.omega += d_omega * dt
        self.v += d_v * dt
        self.r += self.v * dt
        self.q += d_q * dt

        self.q /= np.linalg.norm(self.q)

        output_msg = Float64MultiArray()
        output_msg.data = np.concatenate([self.r, self.v, self.q, self.omega]).tolist()
        self.publisher_.publish(output_msg)


def main(args=None):
    rclpy.init(args=args)
    node = DroneNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
