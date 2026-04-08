import numpy as np
import rclpy
from rclpy.node import Node

from cf_control_msgs.msg import DroneOutput, DroneParameters, Flat


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
        super().__init__('drone_flat_to_state_and_torque_node')

        self.flat_subscription = self.create_subscription(
            Flat, 'drone_flat_input', self.listener_callback, 10
        )
        self.params_subscription = self.create_subscription(
            DroneParameters, 'drone_parameters', self.parameters_callback, 10
        )

        self.output_publisher = self.create_publisher(DroneOutput, 'drone_output', 10)

        self.declare_parameter('mass', 1.0)
        self.declare_parameter('gravity', 9.81)
        self.declare_parameter('inertia', [0.01, 0.01, 0.02])

        self.m = self.get_parameter('mass').value
        self.g = self.get_parameter('gravity').value
        self.J = np.diag(self.get_parameter('inertia').value)

    def parameters_callback(self, msg):
        self.m = msg.in_mass
        self.g = msg.in_gravity
        self.J = np.diag([msg.in_i_xx, msg.in_i_yy, msg.in_i_zz])

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
            (self.m / thrust) * (jerk - np.dot(zb, jerk) * zb) if thrust > 1e-6 else np.zeros(3)
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

        out_msg = DroneOutput()
        out_msg.out_pos_x = pos[0]
        out_msg.out_pos_y = pos[1]
        out_msg.out_pos_z = pos[2]
        out_msg.out_quat_w = q[0]
        out_msg.out_quat_x = q[1]
        out_msg.out_quat_y = q[2]
        out_msg.out_quat_z = q[3]
        out_msg.out_vel_x = vel[0]
        out_msg.out_vel_y = vel[1]
        out_msg.out_vel_z = vel[2]
        out_msg.out_omega_x = omega[0]
        out_msg.out_omega_y = omega[1]
        out_msg.out_omega_z = omega[2]
        out_msg.out_thrust = thrust
        out_msg.out_torque_x = torque[0]
        out_msg.out_torque_y = torque[1]
        out_msg.out_torque_z = torque[2]
        self.output_publisher.publish(out_msg)


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
