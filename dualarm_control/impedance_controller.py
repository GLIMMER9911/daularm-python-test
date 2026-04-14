"""Joint-space and Cartesian-space impedance controllers and reference trajectories."""

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import pinocchio as pin

from .pinocchio_dynamics import rotation_error


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


def desired_cartesian_trajectory(
    t: float,
    ini_pos: np.ndarray,
    des_pos: np.ndarray,
    ini_quat: np.ndarray,
    des_quat: np.ndarray,
    T_move: float = 4.0,
):
    """
    Cartesian trajectory with 5th-order polynomial interpolation for position
    and SLERP for orientation.
    
    Args:
        t: Current time
        ini_pos, des_pos: Initial and desired position (3D)
        ini_quat, des_quat: Initial and desired quaternion (x,y,z,w format)
        T_move: Movement duration
    
    Returns:
        x_des: Stacked [pos, quat] (7D)
        dot_x_des: Stacked [lin_vel, ang_vel] (6D)
        ddot_x_des: Stacked [lin_acc, ang_acc] (6D)
    """
    if t >= T_move:
        return (
            np.hstack([des_pos, des_quat]),
            np.zeros(6),
            np.zeros(6),
        )
    
    tau = t / T_move
    tau2 = tau * tau
    tau3 = tau2 * tau
    tau4 = tau3 * tau
    tau5 = tau4 * tau
    
    # 5th-order polynomial: s(τ) = 10τ³ - 15τ⁴ + 6τ⁵
    s = 10*tau3 - 15*tau4 + 6*tau5
    ds = (30*tau2 - 60*tau3 + 30*tau4) / T_move
    dds = (60*tau - 180*tau2 + 120*tau3) / (T_move * T_move)
    
    # Position interpolation
    x_des = ini_pos + s * (des_pos - ini_pos)
    dot_x_des = ds * (des_pos - ini_pos)
    ddot_x_des = dds * (des_pos - ini_pos)
    
    # Quaternion SLERP (scipy expects x,y,z,w format)
    slerp = Slerp([0.0, 1.0], Rotation.from_quat([ini_quat, des_quat]))
    quat_des = slerp(s).as_quat()
    
    return (
        np.hstack([x_des, quat_des]),
        np.hstack([dot_x_des, np.zeros(3)]),
        np.hstack([ddot_x_des, np.zeros(3)]),
    )


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


class CartesianImpedanceController:
    """
    Cartesian-space impedance control for multi-arm systems.
    Computes wrench (force + torque) in task space.
    """
    
    def __init__(self, task_dim: int, Kp: float = 30.0, Kd: float = 60.0):
        """
        Args:
            task_dim: Task space dimension (typically 6*num_ee for multiple end-effectors)
            Kp, Kd: Proportional and derivative gains (applied to all dimensions)
        """
        self.task_dim = task_dim
        self.Kp = np.diag([Kp] * task_dim)
        self.Kd = np.diag([Kd] * task_dim)
    
    def compute_wrench(
        self,
        x: np.ndarray,
        dot_x: np.ndarray,
        x_des: np.ndarray,
        dot_x_des: np.ndarray,
        ddot_x_des: np.ndarray,
        Lambda: np.ndarray,
        mu: np.ndarray,
        J_sharp: np.ndarray,
    ) -> np.ndarray:
        """
        Compute Cartesian wrench from impedance control law.
        
        Args:
            x: Current Cartesian state [pos_L, quat_L, pos_R, quat_R, ...]
            dot_x: Current Cartesian velocity [lin_vel_L, ang_vel_L, ...]
            x_des, dot_x_des, ddot_x_des: Desired trajectory
            Lambda: Operational-space inertia
            mu: Nonlinear effects in task space
            J_sharp: Jacobian pseudoinverse
        
        Returns:
            F: Task wrench (force + torque)
        """
        # Split position and orientation for dual-arm case (or generalize as needed)
        n_ee = self.task_dim // 6  # Assuming 6D per end-effector
        
        e = np.zeros(self.task_dim)
        de = np.zeros(self.task_dim)
        
        for i in range(n_ee):
            # Position error
            pos_idx = i * 7
            vel_idx = i * 6
            e[vel_idx:vel_idx+3] = x_des[pos_idx:pos_idx+3] - x[pos_idx:pos_idx+3]
            
            # Orientation error (quaternion: x,y,z,w)
            quat_idx = pos_idx + 3
            R_des = pin.Quaternion(x_des[quat_idx:quat_idx+4]).toRotationMatrix()
            R = pin.Quaternion(x[quat_idx:quat_idx+4]).toRotationMatrix()
            e[vel_idx+3:vel_idx+6] = rotation_error(R_des, R)
            
            # Velocity error
            de[vel_idx:vel_idx+6] = dot_x_des[vel_idx:vel_idx+6] - dot_x[vel_idx:vel_idx+6]
        
        # Impedance control law: F = Lambda * ddot_x_des + mu + Kd * de + Kp * e
        F = Lambda @ ddot_x_des + mu + self.Kd @ de + self.Kp @ e
        return F
