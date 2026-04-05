#!/usr/bin/env python3

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from my_robot_interfaces.srv import EEpose


# =========================
# 1) DH-BASED FORWARD KINEMATICS
# =========================

def dh_transform(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)

    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0,      sa,      ca,      d],
        [0.0,     0.0,     0.0,    1.0]
    ], dtype=float)


def fk_from_dh(q: np.ndarray, dh_table: list[dict], joint_types: list[str]) -> np.ndarray:
    if len(dh_table) != len(joint_types) or len(q) != len(joint_types):
        raise ValueError("Lengths of q, dh_table, and joint_types must match.")

    T = np.eye(4, dtype=float)

    for i, (row, jt) in enumerate(zip(dh_table, joint_types)):
        a = float(row["a"])
        alpha = float(row["alpha"])
        d0 = float(row["d"])
        th0 = float(row["theta"])

        if jt.upper() == "R":
            theta = th0 + float(q[i])
            d = d0
        elif jt.upper() == "P":
            theta = th0
            d = d0 + float(q[i])
        else:
            raise ValueError("joint_types must contain only 'R' or 'P'.")

        T = T @ dh_transform(a, alpha, d, theta)

    return T


def fk_position_from_dh(q: np.ndarray, dh_table: list[dict], joint_types: list[str]) -> np.ndarray:
    T = fk_from_dh(q, dh_table, joint_types)
    return T[0:3, 3].copy()


# =========================
# 2) POSITION JACOBIAN
# =========================

def jacobian_position_geometric_from_dh(q: np.ndarray, dh_table: list[dict], joint_types: list[str]) -> np.ndarray:
    n = len(dh_table)

    if len(q) != n or len(joint_types) != n:
        raise ValueError("Lengths of q, dh_table, and joint_types must match.")

    T = np.eye(4, dtype=float)
    Ts = [T.copy()]

    for i, (row, jt) in enumerate(zip(dh_table, joint_types)):
        a = float(row["a"])
        alpha = float(row["alpha"])
        d0 = float(row["d"])
        th0 = float(row["theta"])

        if jt.upper() == "R":
            theta = th0 + float(q[i])
            d = d0
        else:
            theta = th0
            d = d0 + float(q[i])

        T = T @ dh_transform(a, alpha, d, theta)
        Ts.append(T.copy())

    o_n = Ts[-1][0:3, 3].copy()
    Jp = np.zeros((3, n), dtype=float)

    for i in range(n):
        T_i = Ts[i]
        z = T_i[0:3, 2].copy()
        o = T_i[0:3, 3].copy()

        if joint_types[i].upper() == "R":
            Jp[:, i] = np.cross(z, (o_n - o))
        else:
            Jp[:, i] = z

    return Jp


# =========================
# 3) NUMERICAL IK (POSITION ONLY)
# =========================

def clamp(q: np.ndarray, qmin: np.ndarray, qmax: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(q, qmin), qmax)


def limit_step(dq: np.ndarray, dq_max: float) -> np.ndarray:
    n = float(np.linalg.norm(dq))
    if n <= dq_max:
        return dq
    return dq * (dq_max / n)


def ik_lm_position_dh(
    p_des: np.ndarray,
    q0: np.ndarray,
    dh_table: list[dict],
    joint_types: list[str],
    qmin: np.ndarray,
    qmax: np.ndarray,
    eps: float = 1e-3,
    max_iters: int = 200,
    dq_max: float = np.deg2rad(5.0),
    lam0: float = 0.05,
) -> tuple[np.ndarray, bool, dict]:
    if p_des.shape != (3,):
        p_des = np.array(p_des, dtype=float).reshape(3)

    q = clamp(q0.astype(float).copy(), qmin, qmax)
    lam = float(lam0)

    best_q = q.copy()
    best_err = float("inf")

    for k in range(max_iters):
        p = fk_position_from_dh(q, dh_table, joint_types)
        e = p_des - p
        err = float(np.linalg.norm(e))

        if err < best_err:
            best_err = err
            best_q = q.copy()

        if err <= eps:
            return q, True, {"iters": k, "final_error": err, "lambda": lam}

        J = jacobian_position_geometric_from_dh(q, dh_table, joint_types)

        A = J.T @ J + lam * np.eye(q.size)
        b = J.T @ e
        dq = np.linalg.solve(A, b)

        dq = limit_step(dq, dq_max)
        q_candidate = clamp(q + dq, qmin, qmax)

        p_candidate = fk_position_from_dh(q_candidate, dh_table, joint_types)
        err_candidate = float(np.linalg.norm(p_des - p_candidate))

        if err_candidate < err:
            q = q_candidate
            lam = max(lam / 2.0, 1e-6)
        else:
            lam = min(lam * 2.0, 1e3)

    return best_q, False, {"iters": max_iters, "final_error": best_err, "lambda": lam}


# =========================
# 4) ROS2 SERVICE SERVER
# =========================

class MoveToXYZServer(Node):
    def __init__(self):
        super().__init__("move_to_xyz_server")

        self.server_ = self.create_service(
            EEpose,
            "ee_pose",
            self.ee_pose_callback
        )

        # Use the trajectory action server instead of publishing /q_cmd.
        self.trajectory_client_ = ActionClient(
            self,
            FollowJointTrajectory,
            "/arm_controller/follow_joint_trajectory"
        )

        self.num_joints_ = 6
        self.q_last_ik_ = np.zeros(self.num_joints_, dtype=float)

        # Exact joint names used by the controller.
        self.joint_names_ = [
            "base_link_link_0_joint",
            "link_0_link_1_joint",
            "link_1_link_2_joint",
            "link_2_link_3_joint",
            "link_3_link_4_joint",
            "link_4_link_5_joint",
        ]

        # Trajectory duration in seconds.
        self.default_motion_time_ = 3.0

        # Prevent overlapping service requests while the robot is moving.
        self.motion_in_progress_ = False

        # ---------------------------------------
        # REPLACE THESE WITH YOUR REAL DIMENSIONS
        # Units: meters
        # ---------------------------------------
        L1 = 0.1625
        L2 = -0.425
        L3 = -0.3922
        L4 = 0.1333
        L5 = 0.0997
        L6 = 0.0996

        self.dh_table_ = [
            {"a": 0.0, "alpha": np.pi / 2,  "d": L1, "theta": 0.0},
            {"a": L2,  "alpha": 0.0,        "d": 0.0, "theta": -np.pi / 2},
            {"a": L3,  "alpha": 0.0,        "d": 0.0, "theta": 0.0},
            {"a": 0.0, "alpha": np.pi / 2,  "d": L4, "theta": -np.pi / 2},
            {"a": 0.0, "alpha": -np.pi / 2, "d": L5, "theta": 0.0},
            {"a": 0.0, "alpha": 0.0,        "d": L6, "theta": -np.pi},
        ]

        self.joint_types_ = ["R", "R", "R", "R", "R", "R"]

        # ---------------------------------------
        # REPLACE THESE WITH YOUR REAL LIMITS
        # Units: radians
        # ---------------------------------------
        self.qmin_ = np.array([
            -np.pi, -np.pi, -np.pi,
            -np.pi, -np.pi, -np.pi
        ], dtype=float)

        self.qmax_ = np.array([
             np.pi,  np.pi,  np.pi,
             np.pi,  np.pi,  np.pi
        ], dtype=float)

        self.get_logger().info("Service server '/ee_pose' is ready.")
        self.get_logger().info("Waiting for x, y, z requests...")
        self.get_logger().info(
            "IK result will be sent to /arm_controller/follow_joint_trajectory"
        )

    def ee_pose_callback(self, request: EEpose.Request, response: EEpose.Response):
        x = request.x
        y = request.y
        z = request.z

        self.get_logger().info(
            f"Incoming request: x={x:.4f}, y={y:.4f}, z={z:.4f}"
        )

        # Reject new requests while a previous trajectory is still executing.
        if self.motion_in_progress_:
            response.success = False
            response.message = "A trajectory is already in progress. Wait until it finishes."
            self.get_logger().warn(response.message)
            return response

        q_sol = self.compute_ik(x, y, z)

        if q_sol is None:
            response.success = False
            response.message = "IK failed: no valid solution found."
            self.get_logger().warn(response.message)
            return response

        if len(q_sol) != self.num_joints_:
            response.success = False
            response.message = (
                f"Internal error: q_sol has {len(q_sol)} values but expected {self.num_joints_}."
            )
            self.get_logger().error(response.message)
            return response

        # This avoids hanging the service callback.
        ok = self.send_trajectory_goal(q_sol, self.default_motion_time_)

        if not ok:
            response.success = False
            response.message = "IK solved, but failed to send trajectory goal."
            self.get_logger().error(response.message)
            return response

        response.success = True
        response.message = (
            f"IK solved and trajectory sent to arm_controller: {q_sol}"
        )
        self.get_logger().info(response.message)
        return response

    def send_trajectory_goal(self, q_target: list[float], duration_sec: float) -> bool:
        """
        Send the IK solution as a single-point trajectory.
        NON-BLOCKING: send the goal and return immediately.
        """
        if len(q_target) != len(self.joint_names_):
            self.get_logger().error(
                f"Expected {len(self.joint_names_)} target joints, got {len(q_target)}"
            )
            return False

        self.get_logger().info("Waiting for trajectory action server...")
        if not self.trajectory_client_.wait_for_server(timeout_sec=3.0):
            self.get_logger().error(
                "Trajectory action server /arm_controller/follow_joint_trajectory not available."
            )
            return False

        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = self.joint_names_

        point = JointTrajectoryPoint()
        point.positions = list(q_target)
        point.time_from_start.sec = int(duration_sec)
        point.time_from_start.nanosec = int((duration_sec - int(duration_sec)) * 1e9)

        goal_msg.trajectory.points.append(point)

        self.get_logger().info(
            f"Sending trajectory goal with duration {duration_sec:.2f} s"
        )

        # MODIFICATION:
        # Set the busy flag before sending the goal.
        self.motion_in_progress_ = True

        # NON-BLOCKING send
        send_goal_future = self.trajectory_client_.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.goal_response_callback)

        return True

    def goal_response_callback(self, future):
        """
        MODIFICATION:
        Called when the controller accepts or rejects the trajectory goal.
        """
        goal_handle = future.result()

        if goal_handle is None:
            self.get_logger().error("No response received from trajectory action server.")
            self.motion_in_progress_ = False
            return

        if not goal_handle.accepted:
            self.get_logger().error("Trajectory goal was rejected by the controller.")
            self.motion_in_progress_ = False
            return

        self.get_logger().info("Trajectory goal accepted by the controller.")

        # Request the result asynchronously.
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def result_callback(self, future):
        """
        MODIFICATION:
        Called when the trajectory finishes execution.
        """
        try:
            result = future.result().result
            self.get_logger().info(
                f"Trajectory finished. Error code: {result.error_code}, "
                f"message: {result.error_string}"
            )
        except Exception as e:
            self.get_logger().error(f"Error while receiving trajectory result: {e}")

        # Clear the busy flag after the trajectory is done.
        self.motion_in_progress_ = False

    def compute_ik(self, x: float, y: float, z: float):
        p_des = np.array([x, y, z], dtype=float)

        try:
            q_sol, ok, info = ik_lm_position_dh(
                p_des=p_des,
                q0=self.q_last_ik_,
                dh_table=self.dh_table_,
                joint_types=self.joint_types_,
                qmin=self.qmin_,
                qmax=self.qmax_,
                eps=1e-3,
                max_iters=200,
                dq_max=np.deg2rad(5.0),
                lam0=0.05,
            )
        except Exception as e:
            self.get_logger().error(f"IK exception: {e}")
            return None

        self.get_logger().info(
            f"IK finished: converged={ok}, iters={info['iters']}, "
            f"final_error={info['final_error']:.6f}, lambda={info['lambda']:.6f}"
        )

        if not ok:
            self.get_logger().warn("IK did not converge within tolerance.")
            return None

        self.q_last_ik_ = q_sol.copy()
        return list(q_sol.astype(float))


def main(args=None):
    rclpy.init(args=args)
    my_node = MoveToXYZServer()
    rclpy.spin(my_node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()