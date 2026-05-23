import math

import numpy as np
import rclpy
from casadi import SX, vertcat
from nav_msgs.msg import Odometry
from rclpy.node import Node

from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from cf_control_msgs.msg import ContorlerParameters, Flat, ThrustAndTorque


MASS = 0.025
GRAVITY = 9.81
KF = 1.28192e-8
OMEGA_MAX = 2618.0
F_MAX = 4 * KF * OMEGA_MAX ** 2
A_MAX = F_MAX / MASS
A_MAX_Z_UP = A_MAX - GRAVITY
A_MAX_XY = GRAVITY * math.tan(math.radians(30))


class MPCController(Node):
    def __init__(self):
        super().__init__('mpc_node')

        self.m = MASS
        self.g = GRAVITY

        self.KR = np.diag([0.1, 0.1, 0.1])
        self.Kw = np.diag([0.01, 0.01, 0.01])

        self.N = 20
        self.dt = 0.1

        self.current_pos = np.zeros(3)
        self.current_vel = np.zeros(3)
        self.current_R = np.eye(3)
        self.current_omega = np.zeros(3)

        self.e_R = np.zeros(3)
        self.e_omega = np.zeros(3)

        self.solver = self._build_solver()

        self.pub = self.create_publisher(
            ThrustAndTorque, '/cf_control/control_command', 10)
        self.create_subscription(
            Flat, 'drone_trajectory', self._traj_cb, 10)
        self.create_subscription(
            Odometry, '/crazyflie/odom', self._odom_cb, 10)
        self.create_subscription(
            ContorlerParameters, 'drone_regulator_parameters', self._params_cb, 10)

        self.get_logger().info('MPC controller started.')
    def _build_solver(self):
        nx, nu = 6, 3
        ny = nx + nu

        model = AcadosModel()
        model.name = 'cf_pos'

        x = SX.sym('x', nx)
        u = SX.sym('u', nu)
        model.x = x
        model.u = u
        model.f_expl_expr = vertcat(x[3], x[4], x[5], u[0], u[1], u[2])

        ocp = AcadosOcp()
        ocp.model = model
        ocp.dims.N = self.N
        Q = np.diag([10.0, 10.0, 15.0,
                     2.0,  2.0,  3.0])
        R = np.diag([0.5,  0.5,  0.5])

        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'LINEAR_LS'

        ocp.cost.Vx = np.vstack([np.eye(nx),      np.zeros((nu, nx))])
        ocp.cost.Vu = np.vstack([np.zeros((nx, nu)), np.eye(nu)])
        ocp.cost.W = np.block([[Q, np.zeros((nx, nu))],
                               [np.zeros((nu, nx)), R]])
        ocp.cost.Vx_e = np.eye(nx)
        ocp.cost.W_e = 5.0 * Q
        ocp.cost.yref = np.zeros(ny)
        ocp.cost.yref_e = np.zeros(nx)
        ocp.constraints.lbu = np.array([-A_MAX_XY, -A_MAX_XY, -GRAVITY])
        ocp.constraints.ubu = np.array([ A_MAX_XY,  A_MAX_XY,  A_MAX_Z_UP])
        ocp.constraints.idxbu = np.array([0, 1, 2])
        ocp.constraints.x0 = np.zeros(nx)
        ocp.solver_options.tf = self.N * self.dt
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'

        return AcadosOcpSolver(ocp, json_file='cf_pos_ocp.json')

    def _odom_cb(self, msg):
        self.current_pos = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
        ])
        self.current_vel = np.array([
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z,
        ])
        q = msg.pose.pose.orientation
        self.current_R = self._quat_to_R([q.x, q.y, q.z, q.w])
        self.current_omega = np.array([
            msg.twist.twist.angular.x,
            msg.twist.twist.angular.y,
            msg.twist.twist.angular.z,
        ])

    def _traj_cb(self, msg):
        thrust, torque = self._solve(msg)
        cmd = ThrustAndTorque()
        cmd.timestamp = self.get_clock().now().nanoseconds
        cmd.collective_thrust = thrust
        cmd.torque.x = torque[0]
        cmd.torque.y = torque[1]
        cmd.torque.z = torque[2]
        self.pub.publish(cmd)

    def _params_cb(self, msg):
        self.KR = np.diag([msg.kr.x, msg.kr.y, msg.kr.z])
        self.Kw = np.diag([msg.kw.x, msg.kw.y, msg.kw.z])

    def _solve(self, traj):
        try:
            x0 = np.concatenate([self.current_pos, self.current_vel])

            p_ref = np.array([traj.position.x,     traj.position.y,     traj.position.z])
            v_ref = np.array([traj.velocity.x,     traj.velocity.y,     traj.velocity.z])
            a_ref = np.array([traj.acceleration.x, traj.acceleration.y, traj.acceleration.z])
            j_ref = np.array([traj.jerk.x,         traj.jerk.y,         traj.jerk.z])

            self.solver.set(0, 'lbx', x0)
            self.solver.set(0, 'ubx', x0)

            for k in range(self.N):
                tau = k * self.dt
                p_k = p_ref + v_ref * tau + 0.5 * a_ref * tau**2 + (1.0/6.0) * j_ref * tau**3
                v_k = v_ref + a_ref * tau + 0.5 * j_ref * tau**2
                a_k = a_ref + j_ref * tau
                self.solver.set(k, 'yref', np.concatenate([p_k, v_k, a_k]))

            tau_N = self.N * self.dt
            p_N = p_ref + v_ref * tau_N + 0.5 * a_ref * tau_N**2
            v_N = v_ref + a_ref * tau_N
            self.solver.set(self.N, 'yref', np.concatenate([p_N, v_N]))

            status = self.solver.solve()
            if status not in (0, 2):
                self.get_logger().warn(f'MPC solver failed, status={status}')
                return 0.0, [0.0, 0.0, 0.0]

            u_acc = self.solver.get(0, 'u')

        except Exception as exc:
            self.get_logger().error(f'MPC solve error: {exc}')
            return 0.0, [0.0, 0.0, 0.0]

        return self._attitude_control(u_acc, traj)


    def _attitude_control(self, u_acc, traj):
        try:
            m, g = self.m, self.g
            z_W = np.array([0.0, 0.0, 1.0])
            F_des = m * (u_acc + g * z_W)
            F_norm = np.linalg.norm(F_des)
            if F_norm < 1e-6:
                return 0.0, [0.0, 0.0, 0.0]

            z_B_des = F_des / F_norm
            z_B = self.current_R[:, 2]
            thrust = float(np.dot(F_des, z_B))
            if thrust <= 0.0:
                thrust = float(np.dot(F_des, z_W))
                if thrust <= 0.0:
                    return 0.0, [0.0, 0.0, 0.0]

            yaw_T = traj.yaw
            x_C = np.array([math.cos(yaw_T), math.sin(yaw_T), 0.0])

            y_B_des = np.cross(z_B_des, x_C)
            n = np.linalg.norm(y_B_des)
            y_B_des = y_B_des / n if n > 1e-6 else np.array([0.0, 1.0, 0.0])
            x_B_des = np.cross(y_B_des, z_B_des)
            R_des = np.column_stack((x_B_des, y_B_des, z_B_des))
            eR_mat = 0.5 * (R_des.T @ self.current_R - self.current_R.T @ R_des)
            self.e_R = np.array([eR_mat[2, 1], eR_mat[0, 2], eR_mat[1, 0]])
            jerk_T = np.array([traj.jerk.x, traj.jerk.y, traj.jerk.z])
            h_omega = (m / F_norm) * (jerk_T - np.dot(z_B_des, jerk_T) * z_B_des)
            omega_des = np.array([
                -np.dot(h_omega, y_B_des),
                 np.dot(h_omega, x_B_des),
                 traj.yaw_dot * np.dot(z_W, z_B_des),
            ])

            self.e_omega = self.current_omega - omega_des

            torque = (-self.KR @ self.e_R - self.Kw @ self.e_omega).tolist()
            return thrust, torque

        except Exception as exc:
            self.get_logger().error(f'Attitude control error: {exc}')
            return 0.0, [0.0, 0.0, 0.0]

    def _quat_to_R(self, quat):
        x, y, z, w = quat
        return np.array([
            [1 - 2*(y*y + z*z),  2*(x*y - w*z),    2*(x*z + w*y)],
            [2*(x*y + w*z),      1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y),      2*(y*z + w*x),     1 - 2*(x*x + y*y)],
        ])


def main(args=None):
    rclpy.init(args=args)
    node = MPCController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
