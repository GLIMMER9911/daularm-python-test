"""Pinocchio dynamics: model building, mass matrix, and nonlinear effects."""

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
