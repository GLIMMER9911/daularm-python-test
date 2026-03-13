"""
Main simulation script: dual-arm joint-space impedance control with MuJoCo + Pinocchio.

Run from project root:
    python -m dualarm_control.main_simulation
or
    python dualarm_control/main_simulation.py
"""

import os
import sys
import numpy as np

# Allow running as script: python dualarm_control/main_simulation.py
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from dualarm_control.pinocchio_dynamics import build_model, compute_pin_dynamics, get_ee_frame_id
    from dualarm_control.mujoco_interface import MuJoCoSim
    from dualarm_control.impedance_controller import ImpedanceController, desired_trajectory
    from dualarm_control.plotting import plot_joint_trajectories
else:
    from .pinocchio_dynamics import build_model, compute_pin_dynamics, get_ee_frame_id
    from .mujoco_interface import MuJoCoSim
    from .impedance_controller import ImpedanceController, desired_trajectory
    from .plotting import plot_joint_trajectories


def main(
    model_dir: str = None,
    urdf_name: str = "bifrank_robot.urdf",
    scene_name: str = "scene.xml",
    joint_initial_pos: list = None,
    T_move: float = 4.0,
    sim_time: float = 7.0,
    Kp: float = 100.0,
    Kd: float = 20.0,
    show_plot: bool = True,
):
    if model_dir is None:
        model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
    urdf_path = os.path.join(model_dir, urdf_name)
    scene_path = os.path.join(model_dir, scene_name)

    # Pinocchio
    model, data = build_model(urdf_path)
    nq = model.nq
    print("Pinocchio DoF:", model.nq, model.nv)

    # MuJoCo
    sim = MuJoCoSim(scene_path, nq)
    assert model.nq == sim.model.nq, "Mismatch in DoF between Pinocchio and MuJoCo"
    print("MuJoCo nq =", sim.model.nq, " nv =", sim.model.nv, " nu =", sim.model.nu)

    if joint_initial_pos is None:
        joint_initial_pos = [
            0.0, 0.0, -0.7854, 0.0, -2.35621, 0.0, 1.5708, 0.785398,
            0.0, -0.7854, 0.0, -2.35621, 0.0, 1.5708, 0.785398,
        ]
    sim.set_joint_positions(np.array(joint_initial_pos))
    sim.launch_viewer()

    q0, dq0 = sim.get_joint_state()
    print("Initial q0:", q0)
    print("Initial dq0:", dq0)

    q_goal = q0.copy()
    q_goal[1] += 0.5
    q_goal[8] -= 0.5

    controller = ImpedanceController(nq, Kp=Kp, Kd=Kd)

    DT = sim.dt
    steps = int(sim_time / DT)
    log_t = []
    log_q = []
    log_dq = []
    t = 0.0

    for k in range(steps):
        q, dq = sim.get_joint_state()
        q_des, dq_des, ddq_des = desired_trajectory(t, q0, q_goal, T_move)
        M, nle = compute_pin_dynamics(model, data, q, dq)
        tau = controller.compute_torque(q, dq, q_des, dq_des, ddq_des, M, nle)
        sim.set_control(tau)
        sim.step()
        sim.sync_viewer()

        log_t.append(t)
        log_q.append(q.copy())
        log_dq.append(dq.copy())
        t += DT

    log_q = np.array(log_q)
    log_dq = np.array(log_dq)

    if show_plot:
        plot_joint_trajectories(
            log_t,
            log_q,
            joint_indices=[1, 8],
            labels=["L joint 1", "R joint 1"],
        )

    return {
        "log_t": log_t,
        "log_q": log_q,
        "log_dq": log_dq,
        "model": model,
        "data": data,
        "sim": sim,
    }


if __name__ == "__main__":
    main()
