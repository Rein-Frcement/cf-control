import csv
import os

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node

from cf_control_msgs.msg import ContorlerParameters, Flat, ThrustAndTorque


class Controller(Node):
    def __init__(self):
        super().__init__('drone_controller')

        self.declare_parameter('mass', 0.025)
        self.declare_parameter('gravity', 9.81)
        self.mass = self.get_parameter('mass').value
        self.g = self.get_parameter('gravity').value

        self.Kp = np.eye(3)
        self.Kv = np.eye(3)
        self.KR = np.eye(3)
        self.Kw = np.eye(3)

        self.current_pos = np.zeros(3)
        self.current_vel = np.zeros(3)
        self.current_R = np.eye(3)
        self.current_omega = np.zeros(3)

        self.e_p = np.zeros(3)
        self.e_v = np.zeros(3)
        self.e_R = np.zeros(3)
        self.e_omega = np.zeros(3)

        # CSV logging setup
        log_dir = '/home/developer/ros2_ws/logs'
        os.makedirs(log_dir, exist_ok=True)
        self.csv_file = open(os.path.join(log_dir, 'drone_state_log.csv'), 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(
            [
                'timestamp',
                'pos_x',
                'pos_y',
                'pos_z',
                'vel_x',
                'vel_y',
                'vel_z',
                'omega_x',
                'omega_y',
                'omega_z',
                'desired_pos_x',
                'desired_pos_y',
                'desired_pos_z',
                'desired_vel_x',
                'desired_vel_y',
                'desired_vel_z',
                'thrust',
                'torque_x',
                'torque_y',
                'torque_z',
                'e_p_x', 'e_p_y', 'e_p_z',
                'e_v_x', 'e_v_y', 'e_v_z',
                'e_R_x', 'e_R_y', 'e_R_z',
                'e_omega_x', 'e_omega_y', 'e_omega_z'
            ]
        )

        self.publisher_ = self.create_publisher(ThrustAndTorque, '/cf_control/control_command', 10)

        self.subscriber_ = self.create_subscription(
            Flat, 'drone_trajectory', self.trajectory_callback, 10
        )

        self.subscriber_state = self.create_subscription(
            Odometry, '/crazyflie/odom', self.state_callback, 10
        )

        self.regulator_params_subscription = self.create_subscription(
            ContorlerParameters,
            'drone_regulator_parameters',
            self.regulator_parameters_callback,
            10,
        )

    def regulator_parameters_callback(self, msg):
        self.Kp = np.diag([msg.kp.x, msg.kp.y, msg.kp.z])
        self.Kv = np.diag([msg.kv.x, msg.kv.y, msg.kv.z])
        self.KR = np.diag([msg.kr.x, msg.kr.y, msg.kr.z])
        self.Kw = np.diag([msg.kw.x, msg.kw.y, msg.kw.z])

    def quaternion_to_rotation_matrix(self, quat):
        x, y, z, w = quat
        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        xz = x * z
        yz = y * z
        wx = w * x
        wy = w * y
        wz = w * z

        return np.array(
            [
                [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
                [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
                [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
            ]
        )

    def state_callback(self, msg):
        self.current_pos = np.array(
            [
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z,
            ]
        )
        self.current_vel = np.array(
            [
                msg.twist.twist.linear.x,
                msg.twist.twist.linear.y,
                msg.twist.twist.linear.z,
            ]
        )
        q = msg.pose.pose.orientation
        self.current_R = self.quaternion_to_rotation_matrix(np.array([q.x, q.y, q.z, q.w]))
        self.current_omega = np.array(
            [
                msg.twist.twist.angular.x,
                msg.twist.twist.angular.y,
                msg.twist.twist.angular.z,
            ]
        )
        self.mass = self.get_parameter('mass').value

    def trajectory_callback(self, msg):
        thrust, torque = self.__control(msg)
        # Log state to CSV
        timestamp = self.get_clock().now().nanoseconds
        self.csv_writer.writerow(
            [
                timestamp,
                self.current_pos[0],
                self.current_pos[1],
                self.current_pos[2],
                self.current_vel[0],
                self.current_vel[1],
                self.current_vel[2],
                self.current_omega[0],
                self.current_omega[1],
                self.current_omega[2],
                msg.position.x,
                msg.position.y,
                msg.position.z,
                msg.velocity.x,
                msg.velocity.y,
                msg.velocity.z,
                thrust,
                torque[0],
                torque[1],
                torque[2],
                self.e_p[0], self.e_p[1], self.e_p[2],
                self.e_v[0], self.e_v[1], self.e_v[2],
                self.e_R[0], self.e_R[1], self.e_R[2],
                self.e_omega[0], self.e_omega[1], self.e_omega[2]
            ]
        )
        self.publish_control(thrust, torque)

    def __control(self, msg):
        try:
            pos_T = np.array([msg.position.x, msg.position.y, msg.position.z])
            vel_T = np.array([msg.velocity.x, msg.velocity.y, msg.velocity.z])
            acc_T = np.array([msg.acceleration.x, msg.acceleration.y, msg.acceleration.z])
            jerk_T = np.array([msg.jerk.x, msg.jerk.y, msg.jerk.z])
            yaw_T = msg.yaw
            yaw_dot_T = msg.yaw_dot

            curr_pos = self.current_pos
            curr_vel = self.current_vel
            curr_R = self.current_R
            curr_omega = self.current_omega

            m = self.mass
            g = self.g
            z_W = np.array([0.0, 0.0, 1.0])
            Kp = self.Kp
            Kv = self.Kv
            KR = self.KR
            Kw = self.Kw

            e_p = curr_pos - pos_T
            e_v = curr_vel - vel_T

            self.e_p = e_p
            self.e_v = e_v

            F_des = -np.dot(Kp, e_p) - np.dot(Kv, e_v) + m * g * z_W + m * acc_T
            F_norm = np.linalg.norm(F_des)
            z_B_des = F_des / F_norm

            z_B = curr_R[:, 2]
            thrust = np.dot(F_des, z_B)
            if thrust <= 0.0:
                thrust = np.dot(F_des, z_W)
                if thrust <= 0.0:
                    return 0.0, [0.0, 0.0, 0.0]
            x_C_des = np.array([np.cos(yaw_T), np.sin(yaw_T), 0.0])

            y_B_des = np.cross(z_B_des, x_C_des)
            if np.linalg.norm(y_B_des) < 1e-6:
                y_B_des = np.array([0.0, 1.0, 0.0])
            else:
                y_B_des = y_B_des / np.linalg.norm(y_B_des)

            x_B_des = np.cross(y_B_des, z_B_des)

            R_des = np.column_stack((x_B_des, y_B_des, z_B_des))

            e_R_matrix = 0.5 * (np.dot(R_des.T, curr_R) - np.dot(curr_R.T, R_des))
            e_R = np.array([e_R_matrix[2, 1], e_R_matrix[0, 2], e_R_matrix[1, 0]])

            self.e_R = e_R
            h_omega = (m / F_norm) * (jerk_T - np.dot(z_B_des, jerk_T) * z_B_des)

            p_des = -np.dot(h_omega, y_B_des)
            q_des = np.dot(h_omega, x_B_des)
            r_des = yaw_dot_T * np.dot(z_W, z_B_des)

            omega_des = np.array([p_des, q_des, r_des])
            e_omega = curr_omega - omega_des

            self.e_omega = e_omega
            torque_array = -np.dot(KR, e_R) - np.dot(Kw, e_omega)
            torque = torque_array.tolist()

            return float(thrust), torque

        except Exception as e:
            self.get_logger().error(f'Error parsing Flat message or computing control: {e}')
            return 0.0, [0.0, 0.0, 0.0]

    def publish_control(self, thrust, torque):

        msg = ThrustAndTorque()
        msg.timestamp = self.get_clock().now().nanoseconds
        msg.collective_thrust = thrust
        msg.torque.x = torque[0]
        msg.torque.y = torque[1]
        msg.torque.z = torque[2]

        self.publisher_.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = Controller()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.csv_file.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
