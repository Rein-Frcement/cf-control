"""CSV test harness for drone_flat_to_state equations.

This script reads input data from trajectory_from_flat_output_test_data.csv and
computes theoretical outputs using the same formulas as drone_flat_to_state.py.
It then compares the computed output with the expected out_* columns in the CSV.

Equations used from drone_flat_to_state.py:

    a_g = a + [0, 0, g]
    thrust = m * ||a_g||
    zb = normalize(a_g)
    xc = [cos(yaw), sin(yaw), 0]
    yb = normalize(cross(zb, xc))
    xb = cross(yb, zb)
    R = [xb, yb, zb]
    q = rotation_matrix_to_quat(R)

    h_omega = (m/thrust) * (jerk - (zb·jerk) * zb)
    wx = -h_omega·yb
    wy = h_omega·xb
    wz = yaw_dot * zb[2]
    omega = [wx, wy, wz]

    thrust_dot = m * (zb·jerk)
    h_alpha = (m/thrust) * (
        snap
        - (zb·snap) * zb
        - 2 * (thrust_dot/m) * cross(omega, zb)
        - cross(omega, cross(omega, zb)) * (thrust/m)
    )
    alphax = -h_alpha·yb
    alphay = h_alpha·xb
    alphaz = yaw_ddot * zb[2]
    alpha = [alphax, alphay, alphaz]

    torque = J @ alpha + cross(omega, J @ omega)
"""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CSV_FILE = Path(__file__).resolve().parent / 'trajectory_from_flat_output_test_data.csv'
TOLERANCE = 1e-6


def parse_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-12 else v


def rotation_matrix_to_quat(R: np.ndarray) -> np.ndarray:
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    return np.array([qw, qx, qy, qz])


def quaternion_equivalent(q1: np.ndarray, q2: np.ndarray) -> bool:
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    if np.dot(q1, q2) < 0:
        q2 = -q2
    return np.allclose(q1, q2, atol=TOLERANCE, rtol=1e-9)


@dataclass
class TestRow:
    name: str
    in_pos: np.ndarray
    in_vel: np.ndarray
    in_acc: np.ndarray
    in_jerk: np.ndarray
    in_snap: np.ndarray
    in_yaw: float
    in_yaw_rate: float
    in_yaw_acceleration: float
    in_mass: float
    in_gravity: float
    in_I_xx: float
    in_I_yy: float
    in_I_zz: float
    out_pos: np.ndarray
    out_quat: np.ndarray
    out_vel: np.ndarray
    out_omega: np.ndarray
    out_thrust: float
    out_torque: np.ndarray


def load_tests():
    if not CSV_FILE.exists():
        raise FileNotFoundError(f'CSV file not found: {CSV_FILE}')

    tests = []
    with CSV_FILE.open(newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            tests.append(
                TestRow(
                    name=row['test_name'],
                    in_pos=np.array(
                        [
                            parse_float(row['in_pos_x']),
                            parse_float(row['in_pos_y']),
                            parse_float(row['in_pos_z']),
                        ]
                    ),
                    in_vel=np.array(
                        [
                            parse_float(row['in_vel_x']),
                            parse_float(row['in_vel_y']),
                            parse_float(row['in_vel_z']),
                        ]
                    ),
                    in_acc=np.array(
                        [
                            parse_float(row['in_acc_x']),
                            parse_float(row['in_acc_y']),
                            parse_float(row['in_acc_z']),
                        ]
                    ),
                    in_jerk=np.array(
                        [
                            parse_float(row['in_jerk_x']),
                            parse_float(row['in_jerk_y']),
                            parse_float(row['in_jerk_z']),
                        ]
                    ),
                    in_snap=np.array(
                        [
                            parse_float(row['in_snap_x']),
                            parse_float(row['in_snap_y']),
                            parse_float(row['in_snap_z']),
                        ]
                    ),
                    in_yaw=parse_float(row['in_yaw']),
                    in_yaw_rate=parse_float(row['in_yaw_rate']),
                    in_yaw_acceleration=parse_float(row['in_yaw_acceleration']),
                    in_mass=parse_float(row['in_mass']),
                    in_gravity=parse_float(row['in_gravity']),
                    in_I_xx=parse_float(row['in_I_xx']),
                    in_I_yy=parse_float(row['in_I_yy']),
                    in_I_zz=parse_float(row['in_I_zz']),
                    out_pos=np.array(
                        [
                            parse_float(row['out_pos_x']),
                            parse_float(row['out_pos_y']),
                            parse_float(row['out_pos_z']),
                        ]
                    ),
                    out_quat=np.array(
                        [
                            parse_float(row['out_quat_w']),
                            parse_float(row['out_quat_x']),
                            parse_float(row['out_quat_y']),
                            parse_float(row['out_quat_z']),
                        ]
                    ),
                    out_vel=np.array(
                        [
                            parse_float(row['out_vel_x']),
                            parse_float(row['out_vel_y']),
                            parse_float(row['out_vel_z']),
                        ]
                    ),
                    out_omega=np.array(
                        [
                            parse_float(row['out_omega_x']),
                            parse_float(row['out_omega_y']),
                            parse_float(row['out_omega_z']),
                        ]
                    ),
                    out_thrust=parse_float(row['out_thrust']),
                    out_torque=np.array(
                        [
                            parse_float(row['out_torque_x']),
                            parse_float(row['out_torque_y']),
                            parse_float(row['out_torque_z']),
                        ]
                    ),
                )
            )
    return tests


def compute_theoretical_output(test: TestRow):
    m = test.in_mass
    g = test.in_gravity
    J = np.diag([test.in_I_xx, test.in_I_yy, test.in_I_zz])

    a_g = test.in_acc + np.array([0.0, 0.0, g])
    thrust = m * np.linalg.norm(a_g)
    zb = normalize(a_g) if np.linalg.norm(a_g) > 1e-12 else np.array([0.0, 0.0, 1.0])
    xc = np.array([np.cos(test.in_yaw), np.sin(test.in_yaw), 0.0])
    yb = np.cross(zb, xc)
    yb = normalize(yb) if np.linalg.norm(yb) > 1e-12 else np.array([0.0, 1.0, 0.0])
    xb = np.cross(yb, zb)
    R = np.column_stack((xb, yb, zb))
    q = rotation_matrix_to_quat(R)

    h_omega = (
        (m / thrust) * (test.in_jerk - np.dot(zb, test.in_jerk) * zb)
        if thrust > 1e-6
        else np.zeros(3)
    )
    wx = -np.dot(h_omega, yb)
    wy = np.dot(h_omega, xb)
    wz = test.in_yaw_rate * zb[2]
    omega = np.array([wx, wy, wz])

    thrust_dot = m * np.dot(zb, test.in_jerk)
    if thrust > 1e-6:
        h_alpha = (m / thrust) * (
            test.in_snap
            - np.dot(zb, test.in_snap) * zb
            - 2.0 * (thrust_dot / m) * np.cross(omega, zb)
            - np.cross(omega, np.cross(omega, zb)) * (thrust / m)
        )
        alphax = -np.dot(h_alpha, yb)
        alphay = np.dot(h_alpha, xb)
        alphaz = test.in_yaw_acceleration * zb[2]
    else:
        alphax = alphay = alphaz = 0.0

    alpha = np.array([alphax, alphay, alphaz])
    torque = J @ alpha + np.cross(omega, J @ omega)

    return {
        'out_pos': test.in_pos,
        'out_quat': q,
        'out_vel': test.in_vel,
        'out_omega': omega,
        'out_thrust': thrust,
        'out_torque': torque,
    }


def compare_vectors(name: str, actual: np.ndarray, expected: np.ndarray):
    if actual.shape != expected.shape:
        return False, float('inf')
    diff = np.abs(actual - expected)
    return np.all(diff <= TOLERANCE), float(np.max(diff))


def main():
    tests = load_tests()
    failures = []

    for idx, test in enumerate(tests, start=1):
        expected = compute_theoretical_output(test)
        q_calc = expected['out_quat']
        q_ref = test.out_quat

        if np.dot(q_calc, q_ref) < 0:
            q_calc = -q_calc

        results = {
            'out_pos': compare_vectors('out_pos', test.out_pos, expected['out_pos']),
            'out_quat': (
                quaternion_equivalent(test.out_quat, q_calc),
                np.max(np.abs(test.out_quat - q_calc)),
            ),
            'out_vel': compare_vectors('out_vel', test.out_vel, expected['out_vel']),
            'out_omega': compare_vectors('out_omega', test.out_omega, expected['out_omega']),
            'out_thrust': (
                abs(test.out_thrust - expected['out_thrust']) <= TOLERANCE,
                abs(test.out_thrust - expected['out_thrust']),
            ),
            'out_torque': compare_vectors('out_torque', test.out_torque, expected['out_torque']),
        }

        failed = [k for k, (ok, _) in results.items() if not ok]
        if failed:
            failures.append((idx, test.name, results, expected))
            print(f'FAIL test {idx} {test.name}')
            for key, (ok, err) in results.items():
                if not ok:
                    print(f'  {key}: error={err:.6g}')
        else:
            print(f'PASS test {idx} {test.name}')

    print(
        f'\nSummary: {len(tests) - len(failures)} passed, {len(failures)} failed, total {len(tests)}'
    )
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
