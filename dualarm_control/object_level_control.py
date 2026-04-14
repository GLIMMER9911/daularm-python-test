"""Object-level control: spatial spring forces for dual-arm manipulation."""

import numpy as np
import pinocchio as pin
from .pinocchio_dynamics import rotation_error


def end_effector_spatial_spring_force(
    x: np.ndarray,
    e_d: np.ndarray,
    K_c: np.ndarray,
    n_ee: int = 2,
) -> np.ndarray:
    """
    Compute spatial spring force between end-effectors (dual-arm case).
    
    This function computes the desired relative pose between end-effectors
    and generates restoring forces to maintain that spatial relationship.
    
    Args:
        x: Cartesian state [pos_L, quat_L, pos_R, quat_R, ...] (14D for 2 EE)
        e_d: Desired relative pose [pos_x, pos_y, pos_z, roll, pitch, yaw] (6D)
        K_c: Spatial spring stiffness matrix (6x6)
        n_ee: Number of end-effectors (default 2 for dual-arm)
    
    Returns:
        F: Spatial spring wrench [F_left, F_right] (12D for 2 EE)
    """
    if n_ee != 2:
        raise NotImplementedError("Currently only supports dual-arm (n_ee=2)")
    
    # Extract left and right end-effector positions
    left_pos = x[0:3]
    left_quat = x[3:7]
    right_pos = x[7:10]
    right_quat = x[10:14]
    
    # Relative position error
    e_p0 = left_pos - right_pos
    
    # Relative rotation error
    R_left = pin.Quaternion(left_quat).toRotationMatrix()
    R_right = pin.Quaternion(right_quat).toRotationMatrix()
    e_r0 = rotation_error(R_right, R_left)
    
    # Current relative pose
    e_0 = np.hstack([e_p0, e_r0])
    
    # Spring force: F_spring = K_c * (e_d - e_0)
    F_spring = K_c @ (e_d - e_0)
    
    # Wrench matrix for dual-arm: [I, -I]^T to distribute spring force
    # Left EE gets +F_spring, Right EE gets -F_spring
    F = np.vstack([np.eye(6), -np.eye(6)]) @ F_spring
    
    return F


def object_level_impedance_control(x: np.ndarray, object_mass: float = 1.0) -> np.ndarray:
    """
    Object-level impedance control for symmetric grasping.
    
    Computes the desired wrench based on object dynamics.
    
    Args:
        x: Cartesian state [pos_L, quat_L, pos_R, quat_R, ...]
        object_mass: Mass of the grasped object
    
    Returns:
        F: Object-level wrench
    """
    # Placeholder: can be extended with object dynamics
    # For now, returns zero wrench (object-level control handled by impedance)
    return np.zeros(12)
