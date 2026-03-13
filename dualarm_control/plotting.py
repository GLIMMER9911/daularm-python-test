"""Plotting utilities for simulation logs."""

import matplotlib.pyplot as plt
import numpy as np


def plot_joint_trajectories(
    log_t: list,
    log_q: np.ndarray,
    joint_indices: list = None,
    labels: list = None,
    figsize=(10, 5),
):
    """
    Plot selected joint positions over time.

    Args:
        log_t: List of time values.
        log_q: (n_steps, nq) array of joint positions.
        joint_indices: Which joint indices to plot (default [1, 8]).
        labels: Legend labels (default "Joint 1", "Joint 8", etc.).
    """
    if joint_indices is None:
        joint_indices = [1, 8]
    if labels is None:
        labels = [f"Joint {j}" for j in joint_indices]
    plt.figure(figsize=figsize)
    for j, label in zip(joint_indices, labels):
        plt.plot(log_t, log_q[:, j], label=label)
    plt.grid()
    plt.legend()
    plt.show()
