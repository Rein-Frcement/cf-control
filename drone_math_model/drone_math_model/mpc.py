import numpy as np
import rclpy
from acados_template import AcadosOcp, AcadosOcpSolver
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

from cf_control_msgs.msg import ThrustAndTorque


class ModelPredictiveController(Node):
    def __init__(self):
        super().__init__('mpc_node')

        self.N = 20
        self.T = 2.0

        self.ocp = AcadosOcp()
        self._setup_ocp()
        self.solver = AcadosOcpSolver(self.ocp, json_file='acados_ocp.json')

        self.state_sub = self.create_subscription(
            Float64MultiArray, 'drone_state', self.state_callback, 10
        )
        self.cmd_pub = self.create_publisher(ThrustAndTorque, '/cf_control/control_command', 10)

        self.current_state = None

    def _setup_ocp(self):
        from casadi import SX, vertcat

        nx = 6
        nu = 1
        x = SX.sym('x', nx)
        u = SX.sym('u', nu)
        dt = self.T / self.N
        x_next = vertcat(
            x[0] + x[3] * dt,
            x[1] + x[4] * dt,
            x[2] + x[5] * dt,
            x[3],
            x[4],
            x[5] + (u[0] - 9.81) * dt,
        )
        self.ocp.model.x = x
        self.ocp.model.u = u
        self.ocp.model.f_expl_expr = x_next
        self.ocp.model.f_impl_expr = x_next - x
        self.ocp.model.name = 'drone_mpc_model'

        self.ocp.dims.N = self.N
        self.ocp.dims.nx = nx
        self.ocp.dims.nu = nu
        self.ocp.dims.ny = nx + nu
        self.ocp.dims.ny_e = nx

        Q = np.diag([10, 10, 10, 1, 1, 1])
        R = np.diag([0.1])
        self.ocp.cost.cost_type = 'LINEAR_LS'
        self.ocp.cost.cost_type_e = 'LINEAR_LS'
        self.ocp.cost.W = np.block([[Q, np.zeros((6, 1))], [np.zeros((1, 6)), R]])
        self.ocp.cost.W_e = Q
        self.ocp.cost.Vx = np.hstack([np.eye(nx), np.zeros((nx, nu))])
        self.ocp.cost.Vu = np.hstack([np.zeros((nu, nx)), np.eye(nu)])
        self.ocp.cost.Vx_e = np.eye(nx)
        self.ocp.cost.yref = np.zeros(nx + nu)
        self.ocp.cost.yref_e = np.zeros(nx)

        self.ocp.constraints.lbu = np.array([0.0])
        self.ocp.constraints.ubu = np.array([2 * 9.81])
        self.ocp.constraints.idxbu = np.array([0])

        self.ocp.solver_options.tf = self.T
        self.ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
        self.ocp.solver_options.integrator_type = 'ERK'
        self.ocp.solver_options.nlp_solver_type = 'SQP_RTI'

    def state_callback(self, msg):
        state = np.array(msg.data[:6])
        self.current_state = state
        goal = np.array([0, 0, 1, 0, 0, 0])
        self.solve_and_publish(state, goal)

    def solve_and_publish(self, state, goal):
        hover_thrust = 9.81
        self.solver.set(0, 'lbx', state)
        self.solver.set(0, 'ubx', state)
        for k in range(self.N):
            yref = np.concatenate([goal, [hover_thrust]])
            self.solver.set(k, 'yref', yref)
        self.solver.set(self.N, 'yref', goal)
        status = self.solver.solve()
        if status != 0:
            self.get_logger().warn(f'MPC solver failed, status {status}')
            return
        u0 = self.solver.get(0, 'u')
        if u0 is None or not np.isfinite(u0[0]):
            self.get_logger().warn('MPC returned invalid thrust')
            return
        cmd = ThrustAndTorque()
        cmd.collective_thrust = float(u0[0])
        cmd.torque.x = 0.0
        cmd.torque.y = 0.0
        cmd.torque.z = 0.0
        cmd.timestamp = self.get_clock().now().nanoseconds
        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    mpc = ModelPredictiveController()
    rclpy.spin(mpc)
    mpc.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
