"""MuJoCo simulation interface: load model, viewer, state sync, and stepping."""

import numpy as np
import mujoco
import mujoco.viewer


class MuJoCoSim:
    """Wrapper for MuJoCo model, data, viewer, and joint index mapping."""

    def __init__(self, scene_path: str, nq: int):
        """
        Load MuJoCo scene and allocate state.

        Args:
            scene_path: Path to scene XML (e.g. model/scene.xml).
            nq: Number of position DOFs (must match joint indices used for control).
        """
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)
        self.nq = nq
        self.joint_indices = np.arange(0, nq)
        self._viewer = None

        mujoco.mj_forward(self.model, self.data)

    def launch_viewer(self):
        """Start passive viewer. Call sync() after each step to update."""
        self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        return self._viewer

    def get_joint_state(self):
        """Return (q, dq) for the controlled joints from current MuJoCo state."""
        q = self.data.qpos[self.joint_indices].copy()
        dq = self.data.qvel[self.joint_indices].copy()
        return q, dq

    def set_joint_positions(self, q: np.ndarray):
        """Set qpos for controlled joints and run mj_forward."""
        self.data.qpos[self.joint_indices] = q
        mujoco.mj_forward(self.model, self.data)

    def set_control(self, tau: np.ndarray):
        """Set control (joint torques) for the next step."""
        self.data.ctrl[:] = tau

    def step(self):
        """Advance simulation by one timestep."""
        mujoco.mj_step(self.model, self.data)

    def sync_viewer(self):
        """Sync passive viewer with current state."""
        if self._viewer is not None:
            self._viewer.sync()

    @property
    def dt(self) -> float:
        """Simulation timestep."""
        return self.model.opt.timestep
