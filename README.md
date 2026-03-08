# Dual-Arm Control (MuJoCo + Pinocchio)

Simulation and control of a dual-arm robot using MuJoCo for physics and Pinocchio for rigid-body dynamics.

## Contents

| File | Description |
|------|-------------|
| `dualarm_joint_space_control.ipynb` | Joint-space impedance control using mass matrix \(M(q)\) |
| `dualarm_impedance_control.ipynb` | Cartesian-space impedance control |
| `mujoco_viewer.py` | Custom MuJoCo viewer wrapper |

## Setup

### Create environment

```bash
conda env create -f environment.yml
conda activate dualarm-control
```

### Model dependencies

The notebooks expect the robot URDF and MJCF models from the `mujoco_dualarm` package. Set the path in the notebook to match your workspace:

```python
# Example paths used in the notebooks:
mesh_path = " "
urdf = mesh_path + ""
mjcf_path = ""
```

## Running

1. Ensure the robot model path is set correctly in the notebook.
2. Run the desired notebook:
   ```bash
   jupyter notebook dualarm_joint_space_control.ipynb
   # or
   jupyter notebook dualarm_impedance_control.ipynb
   ```
# daularm-python-test
