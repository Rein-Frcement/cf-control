import numpy as np
import rclpy
from numpy.linalg import inv
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from cf_control_msgs.msg import Flat, ThrustAndTorque


def rotation_matrix_to_quat(R):
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw, qx, qy, qz = (
            0.25 * S,
            (R[2, 1] - R[1, 2]) / S,
            (R[0, 2] - R[2, 0]) / S,
            (R[1, 0] - R[0, 1]) / S,
        )
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw, qx, qy, qz = (
            (R[2, 1] - R[1, 2]) / S,
            0.25 * S,
            (R[0, 1] + R[1, 0]) / S,
            (R[0, 2] + R[2, 0]) / S,
        )
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw, qx, qy, qz = (
            (R[0, 2] - R[2, 0]) / S,
            (R[0, 1] + R[1, 0]) / S,
            0.25 * S,
            (R[1, 2] + R[2, 1]) / S,
        )
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw, qx, qy, qz = (
            (R[1, 0] - R[0, 1]) / S,
            (R[0, 2] + R[2, 0]) / S,
            (R[1, 2] + R[2, 1]) / S,
            0.25 * S,
        )
    return np.array([qw, qx, qy, qz])


class DroneNode(Node):
    def __init__(self):
        super().__init__("drone_flat_to_state_and_torque_node")

        self.subscription = self.create_subscription(
            Flat, "drone_flat_input", self.listener_callback, 10
        )

        self.state_publisher = self.create_publisher(
            Float64MultiArray, "drone_state_output", 10
        )
        self.torque_publisher = self.create_publisher(
            ThrustAndTorque, "drone_thrust_torque_output", 10
        )

        self.declare_parameter("mass", 1.0)
        self.declare_parameter("gravity", 9.81)
        self.declare_parameter("inertia", [0.01, 0.01, 0.02])

        self.m = self.get_parameter("mass").value
        self.g = self.get_parameter("gravity").value
        self.J = np.diag(self.get_parameter("inertia").value)

    def listener_callback(self, msg):
        pos = np.array([msg.position.x, msg.position.y, msg.position.z])
        vel = np.array([msg.velocity.x, msg.velocity.y, msg.velocity.z])
        acc = np.array([msg.acceleration.x, msg.acceleration.y, msg.acceleration.z])
        jerk = np.array([msg.jerk.x, msg.jerk.y, msg.jerk.z])
        snap = np.array([msg.snap.x, msg.snap.y, msg.snap.z])

        yaw = msg.yaw
        yaw_dot = msg.yaw_dot
        yaw_ddot = msg.yaw_ddot

        a_g = acc + np.array([0.0, 0.0, self.g])
        norm_a_g = np.linalg.norm(a_g)
        thrust = self.m * norm_a_g

        zb = a_g / norm_a_g if norm_a_g > 1e-6 else np.array([0.0, 0.0, 1.0])
        xc = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        yb = np.cross(zb, xc)
        norm_yb = np.linalg.norm(yb)
        yb = yb / norm_yb if norm_yb > 1e-6 else np.array([0.0, 1.0, 0.0])
        xb = np.cross(yb, zb)

        R = np.column_stack((xb, yb, zb))
        q = rotation_matrix_to_quat(R)

        h_omega = (
            (self.m / thrust) * (jerk - np.dot(zb, jerk) * zb)
            if thrust > 1e-6
            else np.zeros(3)
        )
        wx = -np.dot(h_omega, yb)
        wy = np.dot(h_omega, xb)
        wz = yaw_dot * zb[2]
        omega = np.array([wx, wy, wz])

        thrust_dot = self.m * np.dot(zb, jerk)
        if thrust > 1e-6:
            h_alpha = (self.m / thrust) * (
                snap
                - np.dot(zb, snap) * zb
                - 2.0 * (thrust_dot / self.m) * np.cross(omega, zb)
                - np.cross(omega, np.cross(omega, zb)) * (thrust / self.m)
            )
            alphax = -np.dot(h_alpha, yb)
            alphay = np.dot(h_alpha, xb)
            alphaz = yaw_ddot * zb[2]
        else:
            alphax = alphay = alphaz = 0.0

        alpha = np.array([alphax, alphay, alphaz])
        torque = self.J @ alpha + np.cross(omega, self.J @ omega)

        state_msg = Float64MultiArray()
        state_msg.data = np.concatenate([pos, vel, q, omega]).tolist()
        self.state_publisher.publish(state_msg)

        tt_msg = ThrustAndTorque()
        tt_msg.timestamp = msg.timestamp
        tt_msg.collective_thrust = thrust
        tt_msg.torque.x = torque[0]
        tt_msg.torque.y = torque[1]
        tt_msg.torque.z = torque[2]
        self.torque_publisher.publish(tt_msg)


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


if __name__ == "__main__":
    main()
