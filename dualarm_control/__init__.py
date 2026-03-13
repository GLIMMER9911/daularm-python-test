"""Dual-arm control package: simulation, impedance control, and dynamics."""

from .pinocchio_dynamics import build_model, compute_pin_dynamics, get_ee_frame_id
from .mujoco_interface import MuJoCoSim
from .impedance_controller import ImpedanceController, desired_trajectory
from .plotting import plot_joint_trajectories
from .main_simulation import main

__all__ = [
    "build_model",
    "compute_pin_dynamics",
    "get_ee_frame_id",
    "MuJoCoSim",
    "ImpedanceController",
    "desired_trajectory",
    "plot_joint_trajectories",
    "main",
    "mujoco_viewer",
]
