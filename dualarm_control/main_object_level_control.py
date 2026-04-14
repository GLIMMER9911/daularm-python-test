"""
Main simulation script: dual-arm Cartesian impedance control with MuJoCo + Pinocchio.

Usage:
    python -m dualarm_control.main_object_level_control
or
    python dualarm_control/main_object_level_control.py
"""

import os
import sys
import numpy as np

# Allow running as script
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from dualarm_control.pinocchio_dynamics import (
        build_model,
        compute_cartesian_space_dynamics,
        compute_task_state,
        compute_task_jacobian,
        get_ee_frame_id,
    )
    from dualarm_control.mujoco_interface import MuJoCoSim
    from dualarm_control.impedance_controller import (
        CartesianImpedanceController,
        desired_cartesian_trajectory,
    )
    from dualarm_control.object_level_control import end_effector_spatial_spring_force
    from dualarm_control.plotting import plot_joint_trajectories
else:
    from .pinocchio_dynamics import (
        build_model,
        compute_cartesian_space_dynamics,
        compute_task_state,
        compute_task_jacobian,
        get_ee_frame_id,
    )
    from .mujoco_interface import MuJoCoSim
    from .impedance_controller import (
        CartesianImpedanceController,
        desired_cartesian_trajectory,
    )
    from .object_level_control import end_effector_spatial_spring_force
    from .plotting import plot_joint_trajectories


def main(
    model_dir: str = None,
    urdf_name: str = "bifrank_robot.urdf",
    scene_name: str = "scene.xml",
    joint_initial_pos: list = None,
    left_offset_pos: list = None,
    right_offset_pos: list = None,
    left_offset_rpy: list = None,
    right_offset_rpy: list = None,
    T_move: float = 4.0,
    sim_time: float = 10.0,
    Kp: float = 30.0,
    Kd: float = 60.0,
    K_c_stiffness: float = 10.0,
    show_viewer: bool = True,
    show_plot: bool = True,
):
    """
    Run dual-arm Cartesian impedance control simulation with spatial spring forces.
    
    Args:
        model_dir: Directory containing URDF and scene files
        urdf_name: URDF filename
        scene_name: MuJoCo scene XML filename
        joint_initial_pos: Initial joint configuration (15D for dual-arm FR3)
        left_offset_pos: Left EE desired position offset [x, y, z]
        right_offset_pos: Right EE desired position offset [x, y, z]
        left_offset_rpy: Left EE desired rotation offset [roll, pitch, yaw]
        right_offset_rpy: Right EE desired rotation offset [roll, pitch, yaw]
        T_move: Movement duration (seconds)
        sim_time: Total simulation time (seconds)
        Kp: Cartesian proportional gain
        Kd: Cartesian derivative gain
        K_c_stiffness: Spatial spring stiffness between end-effectors
        show_viewer: Whether to launch MuJoCo viewer
        show_plot: Whether to plot trajectories after simulation
    """
    
    # Paths
    if model_dir is None:
        model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
    urdf_path = os.path.join(model_dir, urdf_name)
    scene_path = os.path.join(model_dir, scene_name)
    
    # Pinocchio model
    model, data = build_model(urdf_path)
    nq = model.nq
    print("Pinocchio DoF:", model.nq, model.nv)
    
    # MuJoCo simulator
    sim = MuJoCoSim(scene_path, nq)
    assert model.nq == sim.model.nq, "Mismatch in DoF between Pinocchio and MuJoCo"
    print("MuJoCo nq =", sim.model.nq, " nv =", sim.model.nv, " nu =", sim.model.nu)
    
    # Initial configuration
    if joint_initial_pos is None:
        joint_initial_pos = [
            0.0, 0.0, -0.7854, 0.0, -2.35621, -0.7854, 1.5708, 0.0, 
                 0.0, -0.7854, 0.0, -2.35621, 0.7854, 1.5708, -1.5708
        ]
    sim.set_joint_positions(np.array(joint_initial_pos))
    
    if show_viewer:
        sim.launch_viewer()
    
    # Get initial state and end-effector frame IDs
    q0, dq0 = sim.get_joint_state()
    left_ee_frame_id = get_ee_frame_id(model, "lewis_fr3_link7")
    right_ee_frame_id = get_ee_frame_id(model, "richard_fr3_link7")
    ee_frame_ids = [left_ee_frame_id, right_ee_frame_id]
    print(f"Left EE frame ID: {left_ee_frame_id}, Right EE frame ID: {right_ee_frame_id}")
    
    # Compute initial end-effector state
    import pinocchio as pin
    pin.forwardKinematics(model, data, q0, dq0)
    pin.updateFramePlacements(model, data)
    
    ini_left_ee_pos = data.oMf[left_ee_frame_id].translation.copy()
    ini_left_ee_quat = np.array(pin.Quaternion(data.oMf[left_ee_frame_id].rotation).coeffs())
    ini_right_ee_pos = data.oMf[right_ee_frame_id].translation.copy()
    ini_right_ee_quat = np.array(pin.Quaternion(data.oMf[right_ee_frame_id].rotation).coeffs())
    
    # Desired end-effector positions and orientations
    if left_offset_pos is None:
        left_offset_pos = [0.1, -0.0, -0.45]
    if right_offset_pos is None:
        right_offset_pos = [0.1, 0.0, -0.45]
    if left_offset_rpy is None:
        left_offset_rpy = [0.0, 0.0, 0.0]
    if right_offset_rpy is None:
        right_offset_rpy = [0.0, 0.0, 0.0]
    
    des_left_ee_pos = ini_left_ee_pos + np.array(left_offset_pos)
    des_right_ee_pos = ini_right_ee_pos + np.array(right_offset_pos)
    
    # Compute desired rotations
    ini_left_R = data.oMf[left_ee_frame_id].rotation.copy()
    ini_right_R = data.oMf[right_ee_frame_id].rotation.copy()
    left_R_offset = pin.rpy.rpyToMatrix(np.array(left_offset_rpy))
    right_R_offset = pin.rpy.rpyToMatrix(np.array(right_offset_rpy))
    des_left_R = ini_left_R @ left_R_offset
    des_right_R = ini_right_R @ right_R_offset
    des_left_ee_quat = np.array(pin.Quaternion(des_left_R).coeffs())
    des_right_ee_quat = np.array(pin.Quaternion(des_right_R).coeffs())
    
    # Spatial spring force parameters
    K_c = K_c_stiffness * np.eye(6)
    e_d = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])  # Desired relative pose [pos_rel, rot_rel]
    
    # Controllers
    cartesian_controller = CartesianImpedanceController(task_dim=12, Kp=Kp, Kd=Kd)
    
    # Simulation loop
    DT = sim.dt
    steps = int(sim_time / DT)
    log_t = []
    log_x = []
    log_dx = []
    
    t = 0.0
    for k in range(steps):
        # Get current state
        q, dq = sim.get_joint_state()
        
        # Compute Cartesian state
        x, dot_x = compute_task_state(model, data, q, dq, ee_frame_ids)
        
        # Compute Jacobians and dynamics
        Ja, dotJa = compute_task_jacobian(model, data, q, dq, ee_frame_ids)
        Lambda, mu, J_sharp, g = compute_cartesian_space_dynamics(model, data, q, dq, ee_frame_ids)
        
        # Desired trajectory
        x_des_L, dot_x_des_L, ddot_x_des_L = desired_cartesian_trajectory(
            t, ini_left_ee_pos, des_left_ee_pos, ini_left_ee_quat, des_left_ee_quat, T_move
        )
        x_des_R, dot_x_des_R, ddot_x_des_R = desired_cartesian_trajectory(
            t, ini_right_ee_pos, des_right_ee_pos, ini_right_ee_quat, des_right_ee_quat, T_move
        )
        x_des = np.hstack([x_des_L, x_des_R])
        dot_x_des = np.hstack([dot_x_des_L, dot_x_des_R])
        ddot_x_des = np.hstack([ddot_x_des_L, ddot_x_des_R])
        
        # Cartesian impedance wrench
        F = cartesian_controller.compute_wrench(x, dot_x, x_des, dot_x_des, ddot_x_des, Lambda, mu, J_sharp)
        
        # Spatial spring force between end-effectors
        F_spring = end_effector_spatial_spring_force(x, e_d, K_c, n_ee=2)
        
        # Total wrench and convert to joint torques
        tau = g + Ja.T @ (F + F_spring)
        sim.set_control(tau)
        sim.step()
        
        if show_viewer:
            sim.sync_viewer()
        
        # Log data
        log_t.append(t)
        log_x.append(x.copy())
        log_dx.append(dot_x.copy())
        
        t += DT
    
    log_x = np.array(log_x)
    log_dx = np.array(log_dx)
    
    print(f"Simulation completed. Logged {len(log_t)} steps.")
    
    if show_plot:
        # Plot joint trajectories
        q_log = np.array([
            np.hstack([np.arcsin(2*x[3:7][3]*x[3:7][3] - 1) if i % 2 == 0 else 0 for i in range(nq)])
            for x in log_x
        ])  # Simplified: just plotting end-effector positions instead
        plot_joint_trajectories(log_t, log_x[:, :6], joint_indices=[0, 1, 2])
    
    return log_t, log_x, log_dx


if __name__ == "__main__":
    main(show_viewer=True, show_plot=True)
