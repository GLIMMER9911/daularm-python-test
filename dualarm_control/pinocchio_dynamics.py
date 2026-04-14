"""Pinocchio dynamics: model building, mass matrix, nonlinear effects, and Cartesian-space utilities."""

import numpy as np
import pinocchio as pin


def build_model(urdf_path: str):
    """Build Pinocchio model and data from URDF path."""
    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()
    return model, data


def compute_pin_dynamics(model, data, q: np.ndarray, dq: np.ndarray):
    """
    Compute mass matrix M(q) and nonlinear effects (Coriolis + gravity).

    Returns:
        M: (nq, nq) mass matrix (symmetrized)
        nle: (nq,) nonlinear effects from pin.rnea(model, data, q, dq, 0)
    """
    M = pin.crba(model, data, q)
    M = 0.5 * (M + M.T)
    nle = pin.rnea(model, data, q, dq, np.zeros_like(dq))
    return M, nle


def get_ee_frame_id(model, frame_name: str = "lewis_fr3_link7") -> int:
    """Return Pinocchio frame ID for the given end-effector frame name."""
    return model.getFrameId(frame_name)


def rotation_error(R_des: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Compute orientation error as a 3D rotation vector.
    
    Args:
        R_des: Desired rotation matrix (3x3)
        R: Current rotation matrix (3x3)
    
    Returns:
        e_o: 3D rotation vector representing orientation error
    """
    R_err = R_des.T @ R
    e_o = 0.5 * pin.log3(R_err)
    return e_o


def compute_task_state(model, data, q: np.ndarray, dq: np.ndarray, ee_frame_ids: list):
    """
    Compute Cartesian task state (position, orientation, velocity) for multiple end-effectors.
    
    Args:
        model, data: Pinocchio model and data
        q, dq: Joint position and velocity
        ee_frame_ids: List of end-effector frame IDs
    
    Returns:
        ee_pos: Stacked [pos_1, quat_1, pos_2, quat_2, ...] for all EEs (quaternion: x,y,z,w)
        ee_vel: Stacked [lin_vel_1, ang_vel_1, lin_vel_2, ang_vel_2, ...] for all EEs
    """
    pin.forwardKinematics(model, data, q, dq)
    pin.updateFramePlacements(model, data)
    
    pos_list = []
    vel_list = []
    
    for ee_id in ee_frame_ids:
        ee_SE3 = data.oMf[ee_id].copy()
        
        # Position
        pos = ee_SE3.translation.copy()
        quat = np.array(pin.Quaternion(ee_SE3.rotation).coeffs())  # x,y,z,w
        pos_list.append(pos)
        pos_list.append(quat)
        
        # Velocity (linear + angular)
        ee_vel = pin.getFrameVelocity(model, data, ee_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        vel_list.append(ee_vel.linear)
        vel_list.append(ee_vel.angular)
    
    ee_pos = np.hstack(pos_list)
    ee_vel = np.hstack(vel_list)
    return ee_pos, ee_vel


def compute_task_jacobian(model, data, q: np.ndarray, dq: np.ndarray, ee_frame_ids: list):
    """
    Compute stacked 6D Jacobians and their time derivatives for multiple end-effectors.
    
    Args:
        model, data: Pinocchio model and data
        q, dq: Joint position and velocity
        ee_frame_ids: List of end-effector frame IDs
    
    Returns:
        Ja: Stacked Jacobian matrix (6*n_ee × nv)
        dJa: Stacked Jacobian time derivative (6*n_ee × nv)
    """
    pin.forwardKinematics(model, data, q, dq)
    pin.updateFramePlacements(model, data)
    
    J_list = []
    dJ_list = []
    
    for ee_id in ee_frame_ids:
        pin.computeFrameJacobian(model, data, q, ee_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        J = pin.getFrameJacobian(model, data, ee_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        dJ = pin.frameJacobianTimeVariation(model, data, q, dq, ee_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        J_list.append(J)
        dJ_list.append(dJ)
    
    Ja = np.vstack(J_list)
    dJa = np.vstack(dJ_list)
    return Ja, dJa


def compute_cartesian_space_dynamics(model, data, q: np.ndarray, dq: np.ndarray, ee_frame_ids: list):
    """
    Compute Cartesian space inertia (Lambda), nonlinear effects (mu), and Jacobian pseudoinverse.
    
    Args:
        model, data: Pinocchio model and data
        q, dq: Joint position and velocity
        ee_frame_ids: List of end-effector frame IDs
    
    Returns:
        Lambda: Operational-space inertia matrix (6*n_ee × 6*n_ee)
        mu: Nonlinear effects in task space (6*n_ee,)
        J_sharp: Jacobian pseudoinverse
        g: Gravity vector
    """
    pin.crba(model, data, q)
    M = data.M
    M = 0.5 * (M + M.T)
    
    h = pin.nonLinearEffects(model, data, q, dq)
    g = pin.computeGeneralizedGravity(model, data, q)
    C = pin.computeCoriolisMatrix(model, data, q, dq)
    
    Ja, dJa = compute_task_jacobian(model, data, q, dq, ee_frame_ids)
    
    # Operational-space inertia: Lambda = (J M^{-1} J^T)^{-1}
    MinvJt = np.linalg.solve(M, Ja.T)
    eps = 1e-6
    A = Ja @ MinvJt
    A = 0.5 * (A + A.T)
    Lambda = np.linalg.inv(A + eps * np.eye(A.shape[0]))
    
    # Nonlinear effects in task space: mu = Lambda (J M^{-1} C - dJ) dq
    JM_inv_C = Ja @ np.linalg.solve(M, C)
    mu = Lambda @ (JM_inv_C - dJa) @ dq
    
    # Jacobian pseudoinverse
    J_sharp = np.linalg.solve(M, Ja.T) @ Lambda
    
    return Lambda, mu, J_sharp, g
