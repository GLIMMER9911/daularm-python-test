# Adaptive Payload Compensation for Dual-Arm Manipulation

## 1. Problem Formulation

We consider a dual-arm robotic system rigidly grasping an object with **unknown mass** \( m \).

Define the stacked task-space state:

\[
x = \begin{bmatrix} x_L \\ x_R \end{bmatrix} \in \mathbb{R}^{12}
\]

where \( x_L, x_R \in \mathbb{R}^6 \) denote the pose (position + orientation) of the left and right end-effectors.

Tracking errors are defined as:

\[
e = x - x_d, \quad \dot{e} = \dot{x} - \dot{x}_d
\]

---

## 2. Sliding Surface

The task-space sliding variable is defined as:

\[
s = \dot{e} + \Lambda e
\]

where \( \Lambda \in \mathbb{R}^{12 \times 12} \) is a positive definite gain matrix.

---

## 3. Payload Model

Assumptions:
- The payload has unknown mass \( m \)
- Gravity acts along the world \( z \)-axis
- The payload is equally supported by both arms

The payload-induced wrench in task space is modeled as:

\[
F_{\text{payload}} = m Y
\]

where the regressor \( Y \in \mathbb{R}^{12} \) is:

\[
Y =
\begin{bmatrix}
0 \\ 0 \\ \frac{g}{2} \\ 0 \\ 0 \\ 0 \\
0 \\ 0 \\ \frac{g}{2} \\ 0 \\ 0 \\ 0
\end{bmatrix}
\]

---

## 4. Adaptive Law

The unknown mass \( m \) is estimated online using:

\[
\dot{\hat{m}} = -\gamma Y^T s
\]

where:
- \( \hat{m} \) is the estimated mass
- \( \gamma > 0 \) is the adaptation gain

---

## 5. Adaptive Compensation

The adaptive compensation wrench is given by:

\[
F_{\text{adaptive}} = \hat{m} Y
\]

---

## 6. Control Law (Task Space)

The Cartesian impedance controller with adaptive compensation is:

\[
F = -K_p e - K_d \dot{e} + \hat{m} Y
\]

where:
- \( K_p, K_d \) are positive definite gain matrices

---

## 7. Mapping to Joint Space

The task-space wrench is mapped to joint torques via the Jacobian:

\[
\tau = J^T F
\]

For a dual-arm system:

\[
\tau = J_L^T F_L + J_R^T F_R
\]

---

## 8. Stability Analysis

Consider the Lyapunov candidate:

\[
V = \frac{1}{2}s^T s + \frac{1}{2\gamma} \tilde{m}^2
\]

where:

\[
\tilde{m} = \hat{m} - m
\]

Taking the time derivative:

\[
\dot{V} = - s^T K_d s \le 0
\]

Thus, the system is stable and all signals remain bounded.

---

## 9. Practical Simplification

Since only the vertical (z-axis) direction contributes to gravity:

\[
Y^T s = \frac{g}{2}(s_{z,L} + s_{z,R})
\]

This allows efficient implementation using only relevant components.

---

## 10. Summary

This framework provides:
- Online estimation of unknown payload mass
- Task-space adaptive compensation
- Stable dual-arm cooperative manipulation

It forms a foundation for more advanced extensions such as:
- Internal force regulation
- Center-of-mass estimation
- Whole-body cooperative manipulation