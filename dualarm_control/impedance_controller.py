"""Joint-space impedance controller and reference trajectory."""

import numpy as np


def desired_trajectory(t: float, q0: np.ndarray, q_goal: np.ndarray, T_move: float):
    """
    Linear interpolation from q0 to q_goal over [0, T_move]; constant after T_move.

    Returns:
        q_des, dq_des, ddq_des
    """
    if t >= T_move:
        return q_goal.copy(), np.zeros_like(q_goal), np.zeros_like(q_goal)
    s = t / T_move
    q_des = (1 - s) * q0 + s * q_goal
    dq_des = (q_goal - q0) / T_move
    ddq_des = np.zeros_like(q0)
    return q_des, dq_des, ddq_des


class ImpedanceController:
    """
    Joint-space impedance control: M(q) * (ddq_des + Kd*de + Kp*e) + nle.
    """

    def __init__(self, nq: int, Kp: float = 100.0, Kd: float = 20.0):
        self.nq = nq
        self.Kp = np.diag([Kp] * nq)
        self.Kd = np.diag([Kd] * nq)

    def compute_torque(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        q_des: np.ndarray,
        dq_des: np.ndarray,
        ddq_des: np.ndarray,
        M: np.ndarray,
        nle: np.ndarray,
    ) -> np.ndarray:
        """
        Compute control torque.

        Args:
            q, dq: Current joint position and velocity.
            q_des, dq_des, ddq_des: Desired trajectory.
            M, nle: Mass matrix and nonlinear effects from compute_pin_dynamics.
        """
        if ddq_des is None:
            ddq_des = np.zeros_like(q)
        e = q_des - q
        de = dq_des - dq
        tau = M @ (ddq_des + self.Kd @ de + self.Kp @ e) + nle
        return tau
