import numpy as np
from rclpy.node import Node

from cf_control_msgs.msg import ContorlerParameters, Flat, ThrustAndTorque


class Controller(Node):
    def __init__(self):
        super().__init__('drone_controller')

        self.publisher_ = self.create_publisher(ThrustAndTorque, 'drone_control', 10)

        self.subscriber_ = self.create_subscription(
            Flat, 'drone_trajectory', self.trajectory_callback, 10
        )

        self.subscriber_state = self.create_subscription(
            Flat, 'drone_state', self.state_callback, 10
        )

        self.regulator_params_subscription = self.create_subscription(
            ContorlerParameters,
            'drone_regulator_parameters',
            self.regulator_parameters_callback,
            10,
        )

    def regulator_parameters_callback(self, msg):
        self.Kp = np.diag([msg.kp, msg.kp, msg.kp])
        self.Kv = np.diag([msg.kv, msg.kv, msg.kv])
        self.KR = np.diag([msg.kr, msg.kr, msg.kr])
        self.Kw = np.diag([msg.kw, msg.kw, msg.kw])

    def state_callback(self, msg):
        self.current_pos = np.array([msg.position.x, msg.position.y, msg.position.z])
        self.current_vel = np.array([msg.velocity.x, msg.velocity.y, msg.velocity.z])
        self.current_R = np.array(
            [
                [msg.rotation_matrix[0], msg.rotation_matrix[1], msg.rotation_matrix[2]],
                [msg.rotation_matrix[3], msg.rotation_matrix[4], msg.rotation_matrix[5]],
                [msg.rotation_matrix[6], msg.rotation_matrix[7], msg.rotation_matrix[8]],
            ]
        )
        self.current_omega = np.array(
            [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]
        )
        self.mass = msg.mass

    def trajectory_callback(self, msg):
        thrust, torque = self.__control(msg)
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
            g = 9.81
            z_W = np.array([0.0, 0.0, 1.0])
            Kp = self.Kp
            Kv = self.Kv
            KR = self.KR
            Kw = self.Kw

            e_p = curr_pos - pos_T
            e_v = curr_vel - vel_T

            F_des = -np.dot(Kp, e_p) - np.dot(Kv, e_v) + m * g * z_W + m * acc_T

            z_B = curr_R[:, 2]
            thrust = np.dot(F_des, z_B)

            if thrust < 0:
                thrust = 0.0

            z_B_des = F_des / np.linalg.norm(F_des)
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
            h_omega = (m / thrust) * (jerk_T - np.dot(z_B_des, jerk_T) * z_B_des)

            p_des = -np.dot(h_omega, y_B_des)
            q_des = np.dot(h_omega, x_B_des)
            r_des = yaw_dot_T * np.dot(z_W, z_B_des)

            omega_des = np.array([p_des, q_des, r_des])
            e_omega = curr_omega - omega_des
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
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
