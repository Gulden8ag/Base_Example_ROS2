#!/usr/bin/env python3
"""
XYZServerGazebo_TODO.py
────────────────────────────────────────────────────────────────────────────
Starting point: XYZServerGazebo.py  →  Target: TrajectoryServer.py

Your task: extend this file to expose TWO new trajectory services.

  /ee_traj_joint      → joint-space trapezoidal profile      [T0_1 theory]
  /ee_traj_cartesian  → Cartesian straight-line + trapezoidal [T0_2 theory]

Legend for TODO markers
  [BOTH]      needed by both services — implement once
  [JOINT]     needed only for /ee_traj_joint
  [CARTESIAN] needed only for /ee_traj_cartesian

Steps 4, 5, 11, 14 belong to the Cartesian service.
Steps 2, 3, 10, 13 belong to the joint-space service.
Steps 1, 6, 7, 8, 9, 12, 15, 16 are shared infrastructure for both.
────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

# TODO [BOTH] — Step 1
# Add the missing import needed to subscribe to the robot's joint state topic.
# Refer to Part 2 — Step 1 in your notes.

from my_robot_interfaces.srv import EEpose


# ═══════════════════════════════════════════════════════════════════════════
# SECTION A — DH-BASED KINEMATICS  (no changes needed here)
# ═══════════════════════════════════════════════════════════════════════════

def dh_transform(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    """Standard Denavit-Hartenberg homogeneous transform."""
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0,      sa,      ca,      d],
        [0.0,     0.0,     0.0,    1.0],
    ], dtype=float)


def fk_position_from_dh(q: np.ndarray, dh_table: list, joint_types: list) -> np.ndarray:
    """Forward kinematics — returns end-effector XYZ position."""
    T = np.eye(4, dtype=float)
    for i, (row, jt) in enumerate(zip(dh_table, joint_types)):
        a     = float(row["a"])
        alpha = float(row["alpha"])
        d0    = float(row["d"])
        th0   = float(row["theta"])
        if jt.upper() == "R":
            T = T @ dh_transform(a, alpha, d0, th0 + float(q[i]))
        else:
            T = T @ dh_transform(a, alpha, d0 + float(q[i]), th0)
    return T[:3, 3].copy()


def jacobian_position_geometric_from_dh(
    q: np.ndarray, dh_table: list, joint_types: list
) -> np.ndarray:
    """Geometric position Jacobian (3 x n)."""
    n  = len(dh_table)
    T  = np.eye(4, dtype=float)
    Ts = [T.copy()]
    for i, (row, jt) in enumerate(zip(dh_table, joint_types)):
        a     = float(row["a"])
        alpha = float(row["alpha"])
        d0    = float(row["d"])
        th0   = float(row["theta"])
        if jt.upper() == "R":
            T = T @ dh_transform(a, alpha, d0, th0 + float(q[i]))
        else:
            T = T @ dh_transform(a, alpha, d0 + float(q[i]), th0)
        Ts.append(T.copy())

    p_e = Ts[-1][:3, 3]
    Jp  = np.zeros((3, n), dtype=float)
    for i in range(n):
        z_i = Ts[i][:3, 2]
        o_i = Ts[i][:3, 3]
        if joint_types[i].upper() == "R":
            Jp[:, i] = np.cross(z_i, p_e - o_i)
        else:
            Jp[:, i] = z_i
    return Jp


def clamp(q, qmin, qmax):
    return np.minimum(np.maximum(q, qmin), qmax)


def limit_step(dq, dq_max):
    n = float(np.linalg.norm(dq))
    return dq if n <= dq_max else dq * (dq_max / n)


def ik_lm_position_dh(
    p_des, q0, dh_table, joint_types, qmin, qmax,
    eps=1e-3, max_iters=200, dq_max=np.deg2rad(5.0), lam0=0.05,
):
    """Levenberg-Marquardt numerical IK (position only)."""
    q = clamp(q0.astype(float).copy(), qmin, qmax)
    lam = float(lam0)
    best_q, best_err = q.copy(), float("inf")

    for k in range(max_iters):
        p   = fk_position_from_dh(q, dh_table, joint_types)
        e   = p_des - p
        err = float(np.linalg.norm(e))
        if err < best_err:
            best_err, best_q = err, q.copy()
        if err <= eps:
            return q, True, {"iters": k, "final_error": err, "lambda": lam}

        J  = jacobian_position_geometric_from_dh(q, dh_table, joint_types)
        A  = J.T @ J + lam * np.eye(q.size)
        dq = limit_step(np.linalg.solve(A, J.T @ e), dq_max)

        q_cand = clamp(q + dq, qmin, qmax)
        if np.linalg.norm(p_des - fk_position_from_dh(q_cand, dh_table, joint_types)) < err:
            q   = q_cand
            lam = max(lam / 2.0, 1e-6)
        else:
            lam = min(lam * 2.0, 1e3)

    return best_q, False, {"iters": max_iters, "final_error": best_err, "lambda": lam}


# ═══════════════════════════════════════════════════════════════════════════
# SECTION B — JOINT-SPACE TRAPEZOIDAL TRAJECTORY  [JOINT]
# ═══════════════════════════════════════════════════════════════════════════

# TODO [JOINT] — Step 2
# Implement _trapezoidal_scalar_profile(total_disp, v_max, a_max, dt)
# Builds the 1-D trapezoidal (or triangular) speed profile for the leader joint.
# Returns: lam, lam_dot, times
# Refer to Part 2 — Step 2 in your notes.


# TODO [JOINT] — Step 3
# Implement generate_joint_space_trajectory(q_start, q_end, v_max, a_max, dt)
# Uses _trapezoidal_scalar_profile to generate a synchronized multi-joint trajectory.
# Returns: list of JointTrajectoryPoint
# Refer to Part 2 — Step 3 in your notes.


# ═══════════════════════════════════════════════════════════════════════════
# SECTION C — CARTESIAN LINEAR TRAJECTORY  [CARTESIAN]
# ═══════════════════════════════════════════════════════════════════════════

# TODO [CARTESIAN] — Step 4
# Implement _trapezoidal_speed_value(t, t_a, t_c, v_max, a_max)
# Returns the scalar Cartesian speed at a single instant t.
# Refer to Part 3 — Step 4 in your notes.


# TODO [CARTESIAN] — Step 5
# Implement generate_cartesian_trajectory(
#     p0, pf, q0, dh_table, qmin, qmax, v_max, a_max, dt, lambda_damp=1e-4)
# Generates a joint trajectory that moves the end-effector along a straight
# Cartesian line using Jacobian-based velocity integration.
# Returns: (list of JointTrajectoryPoint, list of warning strings)
# Refer to Part 3 — Step 5 in your notes.


# ═══════════════════════════════════════════════════════════════════════════
# SECTION D — ROS 2 NODE
# ═══════════════════════════════════════════════════════════════════════════

class MoveToXYZServer(Node):
    def __init__(self):

        # TODO [BOTH] — Step 6
        # Rename the node string to "trajectory_server".
        super().__init__("move_to_xyz_server")

        # ── Joint names (unchanged) ─────────────────────────────────────────
        self.joint_names_ = [
            "base_link_link_0_joint",
            "link_0_link_1_joint",
            "link_1_link_2_joint",
            "link_2_link_3_joint",
            "link_3_link_4_joint",
            "link_4_link_5_joint",
        ]

        # ── DH parameters (unchanged) ───────────────────────────────────────
        L1 =  0.1625
        L2 = -0.425
        L3 = -0.3922
        L4 =  0.1333
        L5 =  0.0997
        L6 =  0.0996

        self.dh_table_ = [
            {"a": 0.0, "alpha":  np.pi / 2, "d": L1, "theta":  0.0},
            {"a": L2,  "alpha":  0.0,        "d": 0.0, "theta": -np.pi / 2},
            {"a": L3,  "alpha":  0.0,        "d": 0.0, "theta":  0.0},
            {"a": 0.0, "alpha":  np.pi / 2,  "d": L4, "theta": -np.pi / 2},
            {"a": 0.0, "alpha": -np.pi / 2,  "d": L5, "theta":  0.0},
            {"a": 0.0, "alpha":  0.0,         "d": L6, "theta": -np.pi},
        ]

        self.joint_types_ = ["R", "R", "R", "R", "R", "R"]

        self.qmin_ = np.full(6, -np.pi)
        self.qmax_ = np.full(6,  np.pi)

        # ── Trajectory action client (unchanged) ────────────────────────────
        self.trajectory_client_ = ActionClient(
            self,
            FollowJointTrajectory,
            "/arm_controller/follow_joint_trajectory",
        )

        # TODO [BOTH] — Step 7
        # Replace self.q_last_ik_ with self.q_current_ and self.q_received_.
        # Add the /joint_states subscriber that calls self._joint_state_cb.
        # Refer to Part 2 — Step 7 in your notes.
        self.q_last_ik_ = np.zeros(6)          # ← replace this

        # TODO [BOTH] — Step 8
        # Add motion parameters: v_max_joint_, a_max_joint_, v_max_cart_,
        # a_max_cart_, dt_, lambda_damp_cart_
        # Refer to Part 2 — Step 8 in your notes.

        # TODO [BOTH] — Step 9
        # Rename self.motion_in_progress_ to self.busy_ everywhere.
        self.motion_in_progress_ = False

        # ── Original /ee_pose service — keep while building, remove when done ─
        self.server_ = self.create_service(
            EEpose, "ee_pose", self.ee_pose_callback
        )

        # TODO [JOINT] — Step 10
        # Register the /ee_traj_joint service with callback self._cb_joint_traj.
        # Refer to Part 2 — Step 10 in your notes.

        # TODO [CARTESIAN] — Step 11
        # Register the /ee_traj_cartesian service with callback self._cb_cartesian_traj.
        # Refer to Part 3 — Step 11 in your notes.

        self.get_logger().info("Node is ready.")

    # ── /joint_states subscriber callback ───────────────────────────────────

    # TODO [BOTH] — Step 12
    # Implement _joint_state_cb(self, msg).
    # Map joint names to positions safely — do not read by index directly.
    # Refer to Part 2 — Step 12 in your notes.

    # ══════════════════════════════════════════════════════════════════════════
    # JOINT-SPACE SERVICE CALLBACK  [JOINT]
    # ══════════════════════════════════════════════════════════════════════════

    # TODO [JOINT] — Step 13
    # Implement _cb_joint_traj(self, request, response).
    # Guard → snapshot q_current_ → IK → generate trajectory → send.
    # Refer to Part 2 — Step 13 in your notes.

    # ══════════════════════════════════════════════════════════════════════════
    # CARTESIAN SERVICE CALLBACK  [CARTESIAN]
    # ══════════════════════════════════════════════════════════════════════════

    # TODO [CARTESIAN] — Step 14
    # Implement _cb_cartesian_traj(self, request, response).
    # Guard → snapshot q_current_ → FK for p_start → generate trajectory → send.
    # Note: no IK call here. Refer to Part 3 — Step 14 in your notes.

    # ══════════════════════════════════════════════════════════════════════════
    # ACTION CLIENT HELPERS
    # ══════════════════════════════════════════════════════════════════════════

    # TODO [BOTH] — Step 15
    # Replace send_trajectory_goal() below with _send_trajectory(self, points).
    # The new version accepts a full list of JointTrajectoryPoint objects
    # instead of building a single point internally.
    # Refer to Part 2 — Step 15 in your notes.

    def send_trajectory_goal(self, q_target: list, duration_sec: float) -> bool:
        """
        Original single-point sender — replace with _send_trajectory() in Step 15.
        Kept here so the file runs while you build incrementally.
        """
        if len(q_target) != len(self.joint_names_):
            return False

        if not self.trajectory_client_.wait_for_server(timeout_sec=3.0):
            self.get_logger().error("Trajectory action server not available.")
            return False

        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = self.joint_names_

        point = JointTrajectoryPoint()
        point.positions = list(q_target)
        point.time_from_start.sec     = int(duration_sec)
        point.time_from_start.nanosec = int((duration_sec - int(duration_sec)) * 1e9)
        goal_msg.trajectory.points.append(point)

        self.motion_in_progress_ = True
        future = self.trajectory_client_.send_goal_async(goal_msg)
        future.add_done_callback(self.goal_response_callback)
        return True

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("Trajectory goal rejected.")
            self.motion_in_progress_ = False
            return
        goal_handle.get_result_async().add_done_callback(self.result_callback)

    def result_callback(self, future):
        try:
            result = future.result().result
            self.get_logger().info(f"Trajectory finished. Error: {result.error_code}")
        except Exception as e:
            self.get_logger().error(f"Result error: {e}")
        self.motion_in_progress_ = False

    # TODO [BOTH] — Step 16
    # Rename goal_response_callback → _goal_response_cb
    #         result_callback        → _result_cb
    # Replace every self.motion_in_progress_ with self.busy_ inside both.

    # ══════════════════════════════════════════════════════════════════════════
    # ORIGINAL /ee_pose CALLBACK  (keep for reference, remove when done)
    # ══════════════════════════════════════════════════════════════════════════

    def ee_pose_callback(self, request, response):
        x, y, z = request.x, request.y, request.z
        self.get_logger().info(f"Incoming /ee_pose request: x={x}, y={y}, z={z}")

        if self.motion_in_progress_:
            response.success = False
            response.message = "Motion in progress."
            return response

        p_des = np.array([x, y, z], dtype=float)
        q_sol, ok, info = ik_lm_position_dh(
            p_des, self.q_last_ik_, self.dh_table_, self.joint_types_,
            self.qmin_, self.qmax_,
        )

        if not ok:
            response.success = False
            response.message = "IK failed."
            return response

        self.q_last_ik_ = q_sol.copy()
        ok_sent = self.send_trajectory_goal(list(q_sol.astype(float)), 3.0)
        response.success = ok_sent
        response.message = f"Sent: {list(q_sol)}" if ok_sent else "Send failed."
        return response


def main(args=None):
    rclpy.init(args=args)
    node = MoveToXYZServer()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()