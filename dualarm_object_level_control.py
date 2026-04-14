#!/usr/bin/env python3
"""
Dual-arm object-level Cartesian impedance control with spatial spring forces.

Converted from: dualarm_object_level_control.ipynb

This script implements:
  - Multi-arm forward kinematics and Jacobians with Pinocchio
  - Cartesian-space impedance control
  - Spatial spring forces between end-effectors
  - MuJoCo simulation with passive viewer
"""

import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

# Add path for imports
file_path = os.path.abspath(".")
sys.path.insert(0, file_path)

import pinocchio as pin
import mujoco
import mujoco.viewer


def rotation_error(R_des: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Compute orientation error as 3D rotation vector."""
    R_err = R_des.T @ R
    e_o = 0.5 * pin.log3(R_err)
    return e_o


def get_pin_state_from_mujoco(data_mj, joint_indices):
    """Extract joint state from MuJoCo data."""
    q = data_mj.qpos[joint_indices].copy()
    dq = data_mj.qvel[joint_indices].copy()
    return q, dq


def compute_task_state(model, data, q, dq, ee_frame_ids):
    """Compute Cartesian task state (position, orientation, velocity)."""
    pin.forwardKinematics(model, data, q, dq)
    pin.updateFramePlacements(model, data)
    
    pos_list = []
    vel_list = []
    
    for ee_id in ee_frame_ids:
        ee_SE3 = data.oMf[ee_id].copy()
        pos = ee_SE3.translation.copy()
        quat = np.array(pin.Quaternion(ee_SE3.rotation).coeffs())
        pos_list.append(pos)
        pos_list.append(quat)
        
        ee_vel = pin.getFrameVelocity(model, data, ee_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        vel_list.append(ee_vel.linear)
        vel_list.append(ee_vel.angular)
    
    ee_pos = np.hstack(pos_list)
    ee_vel = np.hstack(vel_list)
    return ee_pos, ee_vel


def compute_task_jacobian(model, data, q, dq, ee_frame_ids):
    """Compute Jacobians and their time derivatives."""
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


def compute_Cartesian_space_dynamics(model, data, q, dq, ee_frame_ids):
    """Compute operational-space inertia, nonlinear effects, and Jacobian pseudoinverse."""
    pin.crba(model, data, q)
    M = data.M
    M = 0.5 * (M + M.T)
    
    h = pin.nonLinearEffects(model, data, q, dq)
    g = pin.computeGeneralizedGravity(model, data, q)
    C = pin.computeCoriolisMatrix(model, data, q, dq)
    
    Ja, dJa = compute_task_Jacobian(model, data, q, dq, ee_frame_ids)
    
    # Operational-space inertia
    MinvJt = np.linalg.solve(M, Ja.T)
    eps = 1e-6
    A = Ja @ MinvJt
    A = 0.5 * (A + A.T)
    Lambda = np.linalg.inv(A + eps * np.eye(A.shape[0]))
    
    # Nonlinear effects in task space
    JM_inv_C = Ja @ np.linalg.solve(M, C)
    mu = Lambda @ (JM_inv_C - dJa) @ dq
    
    # Jacobian pseudoinverse
    J_sharp = np.linalg.solve(M, Ja.T) @ Lambda
    
    return Lambda, mu, J_sharp, g


def compute_task_Jacobian(model, data, q, dq, ee_frame_ids):
    """Wrapper for compute_task_jacobian for compatibility."""
    return compute_task_jacobian(model, data, q, dq, ee_frame_ids)


def cartesian_impedance_control(x, dot_x, x_des, dot_x_des, ddot_x_des, Lambda, mu, Ja_sharp, K_p, K_d):
    """Cartesian impedance control law."""
    # Split for dual-arm (left: [0:7], right: [7:14])
    e_pos_left = x_des[:3] - x[:3]
    e_pos_right = x_des[7:10] - x[7:10]
    
    # Orientation errors
    R_des_left = pin.Quaternion(x_des[3:7]).toRotationMatrix()
    R_left = pin.Quaternion(x[3:7]).toRotationMatrix()
    e_rot_left = rotation_error(R_des_left, R_left)
    
    R_des_right = pin.Quaternion(x_des[10:14]).toRotationMatrix()
    R_right = pin.Quaternion(x[10:14]).toRotationMatrix()
    e_rot_right = rotation_error(R_des_right, R_right)
    
    # 12D error
    e = np.hstack([e_pos_left, e_rot_left, e_pos_right, e_rot_right])
    
    # Velocity error
    de = dot_x_des - dot_x
    
    # Impedance control
    F = Lambda @ ddot_x_des + mu + K_d @ de + K_p @ e
    return F


def end_effector_spatial_spring_force(x, e_d, K_c):
    """Compute spatial spring force between end-effectors."""
    e_p0 = x[0:3] - x[7:10]
    R_left = pin.Quaternion(x[3:7]).toRotationMatrix()
    R_right = pin.Quaternion(x[10:14]).toRotationMatrix()
    e_r0 = rotation_error(R_right, R_left)
    
    e_0 = np.hstack([e_p0, e_r0])
    F_spring = K_c @ (np.array(e_d) - np.array(e_0))
    
    F = np.vstack([np.eye(6), -np.eye(6)]) @ F_spring
    return F


def desired_trajectory(t, data, left_ee_frame_id, right_ee_frame_id, model, 
                       left_offset_ee_pos, right_offset_ee_pos, left_offset_ee_rpy, 
                       right_offset_ee_rpy, T_move=4.0):
    """Generate Cartesian desired trajectory with 5th-order polynomial interpolation."""
    if t == 0.0 or not hasattr(desired_trajectory, "initialized"):
        desired_trajectory.ini_left_ee_pos = data.oMf[left_ee_frame_id].translation.copy()
        desired_trajectory.ini_right_ee_pos = data.oMf[right_ee_frame_id].translation.copy()
        desired_trajectory.ini_left_ee_quat = np.array(
            pin.Quaternion(data.oMf[left_ee_frame_id].rotation).coeffs()
        )
        desired_trajectory.ini_right_ee_quat = np.array(
            pin.Quaternion(data.oMf[right_ee_frame_id].rotation).coeffs()
        )
        
        ini_left_R = data.oMf[left_ee_frame_id].rotation.copy()
        ini_right_R = data.oMf[right_ee_frame_id].rotation.copy()
        
        left_R_offset = pin.rpy.rpyToMatrix(np.array(left_offset_ee_rpy))
        right_R_offset = pin.rpy.rpyToMatrix(np.array(right_offset_ee_rpy))
        
        desired_trajectory.des_left_ee_pos = desired_trajectory.ini_left_ee_pos + np.array(left_offset_ee_pos)
        desired_trajectory.des_right_ee_pos = desired_trajectory.ini_right_ee_pos + np.array(right_offset_ee_pos)
        desired_trajectory.des_left_R = ini_left_R @ left_R_offset
        desired_trajectory.des_right_R = ini_right_R @ right_R_offset
        desired_trajectory.des_left_ee_quat = np.array(pin.Quaternion(desired_trajectory.des_left_R).coeffs())
        desired_trajectory.des_right_ee_quat = np.array(pin.Quaternion(desired_trajectory.des_right_R).coeffs())
        
        desired_trajectory.initialized = True
    
    ini_left_ee_pos = desired_trajectory.ini_left_ee_pos
    ini_right_ee_pos = desired_trajectory.ini_right_ee_pos
    ini_left_ee_quat = desired_trajectory.ini_left_ee_quat
    ini_right_ee_quat = desired_trajectory.ini_right_ee_quat
    
    des_left_ee_pos = desired_trajectory.des_left_ee_pos
    des_right_ee_pos = desired_trajectory.des_right_ee_pos
    des_left_ee_quat = desired_trajectory.des_left_ee_quat
    des_right_ee_quat = desired_trajectory.des_right_ee_quat
    
    if t >= T_move:
        x_des = np.hstack([des_left_ee_pos, des_left_ee_quat, des_right_ee_pos, des_right_ee_quat])
        return x_des, np.zeros(12), np.zeros(12)
    
    tau = t / T_move
    tau2 = tau * tau
    tau3 = tau2 * tau
    tau4 = tau3 * tau
    tau5 = tau4 * tau
    
    # Position (5th-order poly)
    left_x_des = ini_left_ee_pos + (des_left_ee_pos - ini_left_ee_pos) * (10*tau3 - 15*tau4 + 6*tau5)
    right_x_des = ini_right_ee_pos + (des_right_ee_pos - ini_right_ee_pos) * (10*tau3 - 15*tau4 + 6*tau5)
    
    # Velocity
    poly_dot = (30*tau2 - 60*tau3 + 30*tau4) / T_move
    dot_left_lin = (des_left_ee_pos - ini_left_ee_pos) * poly_dot
    dot_right_lin = (des_right_ee_pos - ini_right_ee_pos) * poly_dot
    dot_x_des = np.hstack([dot_left_lin, np.zeros(3), dot_right_lin, np.zeros(3)])
    
    # Acceleration
    poly_ddot = (60*tau - 180*tau2 + 120*tau3) / (T_move * T_move)
    ddot_left_lin = (des_left_ee_pos - ini_left_ee_pos) * poly_ddot
    ddot_right_lin = (des_right_ee_pos - ini_right_ee_pos) * poly_ddot
    ddot_x_des = np.hstack([ddot_left_lin, np.zeros(3), ddot_right_lin, np.zeros(3)])
    
    # Quaternion SLERP
    s = 10*tau3 - 15*tau4 + 6*tau5
    slerp_left = Slerp([0.0, 1.0], Rotation.from_quat([ini_left_ee_quat, des_left_ee_quat]))
    slerp_right = Slerp([0.0, 1.0], Rotation.from_quat([ini_right_ee_quat, des_right_ee_quat]))
    left_x_quat_des = slerp_left(s).as_quat()
    right_x_quat_des = slerp_right(s).as_quat()
    
    x_des = np.hstack([left_x_des, left_x_quat_des, right_x_des, right_x_quat_des])
    return x_des, dot_x_des, ddot_x_des


def main():
    """Run the simulation."""
    # Setup paths
    urdf = os.path.join(file_path, "model", "bifrank_robot.urdf")
    mjcf_path = os.path.join(file_path, "model", "scene.xml")
    
    # Build Pinocchio model
    model_full = pin.buildModelFromUrdf(urdf)
    data_full = model_full.createData()
    
    # Initial config
    joint_initial_pos = [
        0.0, 
        0.0, -0.7854, 0.0, -2.35621, -0.7854, 1.5708, 0.0, 
        0.0, -0.7854, 0.0, -2.35621, 0.7854, 1.5708, -1.5708
    ]
    q0 = pin.neutral(model_full)
    model = pin.buildReducedModel(model_full, [], q0)
    data = model.createData()
    
    # Get end-effector frame IDs
    left_ee_frame_id = model.getFrameId("lewis_fr3_link7")
    right_ee_frame_id = model.getFrameId("richard_fr3_link7")
    ee_frame_ids = [left_ee_frame_id, right_ee_frame_id]
    
    print("Pinocchio DoF:", model.nq, model.nv)
    print("End-effector frame IDs:", left_ee_frame_id, right_ee_frame_id)
    
    # Load MuJoCo model
    model_mj = mujoco.MjModel.from_xml_path(mjcf_path)
    data_mj = mujoco.MjData(model_mj)
    mujoco.mj_forward(model_mj, data_mj)
    print("MuJoCo nq =", model_mj.nq, " nv =", model_mj.nv, " nu =", model_mj.nu)
    
    # Setup initial state
    joint_indices = np.arange(0, model.nq)
    data_mj.qpos[joint_indices] = joint_initial_pos
    data_mj.qvel[joint_indices] = np.zeros(model.nv)
    mujoco.mj_forward(model_mj, data_mj)
    mujoco.mj_step(model_mj, data_mj)
    
    # Launch viewer
    viewer = mujoco.viewer.launch_passive(model_mj, data_mj)
    
    # Impedance control gains
    K_p_diag = 30.0
    K_d_diag = 60.0
    K_p = np.diag([K_p_diag] * 12)
    K_d = np.diag([K_d_diag] * 12)
    
    # Spatial spring force
    K_c = 10 * np.eye(6)
    e_d = np.array([0.0, 1.0, 0.0, 1.03764157, -1.03764174, -2.19134163e-01])
    
    # Desired offsets
    left_offset_ee_pos = [0.1, -0.0, -0.45]
    right_offset_ee_pos = [0.1, 0.0, -0.45]
    left_offset_ee_rpy = [0.0, 0.0, 0.0]
    right_offset_ee_rpy = [0.0, 0.0, 0.0]
    
    # Control step function
    def control_step(t):
        q, dq = get_pin_state_from_mujoco(data_mj, joint_indices)
        x, dot_x = compute_task_state(model, data, q, dq, ee_frame_ids)
        
        Ja, dotJa = compute_task_jacobian(model, data, q, dq, ee_frame_ids)
        Lambda, mu, J_sharp, g = compute_Cartesian_space_dynamics(model, data, q, dq, ee_frame_ids)
        
        F_spring = end_effector_spatial_spring_force(x, e_d, K_c)
        
        x_des, dot_x_des, ddot_x_des = desired_trajectory(
            t, data, left_ee_frame_id, right_ee_frame_id, model,
            left_offset_ee_pos, right_offset_ee_pos, left_offset_ee_rpy, right_offset_ee_rpy
        )
        
        F = cartesian_impedance_control(x, dot_x, x_des, dot_x_des, ddot_x_des, Lambda, mu, J_sharp, K_p, K_d)
        
        tau = g + Ja.T @ (F + F_spring)
        data_mj.ctrl[:] = tau
        
        return x, dot_x
    
    # Simulation loop
    DT = model_mj.opt.timestep
    sim_time = 10.0
    steps = int(sim_time / DT)
    
    log_t = []
    log_x = []
    log_dx = []
    
    t = 0.0
    mujoco.mj_step(model_mj, data_mj)
    
    for k in range(steps):
        x, dot_x = control_step(t)
        mujoco.mj_step(model_mj, data_mj)
        viewer.sync()
        
        t += DT
        
        log_t.append(k)
        log_x.append(x.copy())
        log_dx.append(dot_x.copy())
        
        if k % 100 == 0:
            print(f"Step {k}/{steps}")
    
    print(f"Simulation completed. Total steps: {len(log_t)}")


if __name__ == "__main__":
    main()
