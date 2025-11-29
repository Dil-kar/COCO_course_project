from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
import casadi as ca
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import CubicSpline

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Constants
MASS = 0.027
GRAVITY = 9.81
THRUST_HOVER = MASS * GRAVITY

class KoopmanLMPC:
    """
    Koopman Linear MPC solver using CasADi.
    
    This class implements a Linear MPC which can be viewed as a Koopman MPC
    with identity observables (or linearized dynamics).
    
    To implement a full Koopman MPC with lifted states:
    1. Modify the observable size (_nx_lifted).
    2. Update the A, B matrices to represent the dynamics in the lifted space.
    3. Update the lifting function `lift_state`.
    """
    def __init__(self, horizon: int, dt: float, q_diag: list, r_diag: list):
        self._N = horizon
        self._dt = dt
        
        # Dimensions
        self._nx = 9  # Original state: [x, y, z, vx, vy, vz, r, p, y]
        self._nu = 4  # Input: [roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd]
        
        # For now, lifted state is same as original state (Identity lifting)
        self._nx_lifted = self._nx 
        
        # Weights
        self.Q = np.diag(q_diag)
        self.R = np.diag(r_diag)
        
        # --- Dynamics Model (Linear/Koopman) ---
        # State: [x, y, z, vx, vy, vz, r, p, y]
        # Input: [roll_cmd, pitch_cmd, yaw_cmd, thrust_delta]
        
        A = np.eye(9)
        A[0:3, 3:6] = np.eye(3) * self._dt # pos += vel * dt
        
        # Angle to Acceleration coupling
        # ax = g * pitch (approx)
        # ay = -g * roll (approx)
        A[3, 7] = GRAVITY * self._dt   # Pitch -> vx
        A[4, 6] = -GRAVITY * self._dt  # Roll -> vy
        
        # Angle dynamics: Instant tracking (phi_next = u)
        A[6:9, 6:9] = np.zeros((3, 3))
        
        self.A = ca.DM(A)

        B = np.zeros((9, 4))
        # Thrust -> vz
        B[5, 3] = self._dt / MASS
        
        # Input -> Angle (Instant tracking)
        B[6:9, 0:3] = np.eye(3)
        
        self.B = ca.DM(B)
        
        # CasADi Variables
        self._u = ca.MX.sym("u", self._nu)
        self._U = ca.MX.sym("U", self._nu, self._N)
        self._P = ca.MX.sym("P", self._nx_lifted)
        self._Ref = ca.MX.sym("Ref", self._nx_lifted, self._N)

        self._build_solver()

    def _build_solver(self):
        cost = 0
        g = []
        X = [self._P]
        
        for k in range(self._N):
            # Dynamics: x_{k+1} = A x_k + B u_k
            x_next = ca.mtimes(self.A, X[-1]) + ca.mtimes(self.B, self._U[:, k])
            X.append(x_next)

            # Cost
            x_ref = self._Ref[:, k]
            x_err = X[-2] - x_ref
            cost += ca.mtimes([x_err.T, self.Q, x_err])
            
            u_err = self._U[:, k]
            cost += ca.mtimes([u_err.T, self.R, u_err])

        # Terminal cost
        x_err_term = X[-1] - self._Ref[:, self._N-1]
        cost += ca.mtimes([x_err_term.T, self.Q, x_err_term])

        # Solver
        p = ca.vertcat(self._P, ca.reshape(self._Ref, -1, 1))
        nlp = {'x': ca.vertcat(ca.reshape(self._U, -1, 1)), 'f': cost, 'p': p}
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        self.solver = ca.nlpsol("solver", "ipopt", nlp, opts)
        
        # Constraints
        max_angle = 1.0
        max_yaw = 2.0
        max_thrust_delta = 0.8 * THRUST_HOVER
        
        lbu_step = [-max_angle, -max_angle, -max_yaw, -max_thrust_delta]
        ubu_step = [max_angle, max_angle, max_yaw, max_thrust_delta]
        
        self.lbx = np.tile(lbu_step, self._N)
        self.ubx = np.tile(ubu_step, self._N)

    def lift_state(self, x: np.ndarray) -> np.ndarray:
        """Lift state to Koopman observables space. Currently Identity."""
        return x

    def solve(self, x0: np.ndarray, ref_traj: np.ndarray) -> np.ndarray:
        """
        Solve MPC problem.
        x0: Current state (9,)
        ref_traj: Reference trajectory (9, N)
        Returns: First control action (4,)
        """
        z0 = self.lift_state(x0)
        
        # Flatten ref_traj column-major
        p_val = np.concatenate([z0, ref_traj.flatten(order='F')])
        
        try:
            res = self.solver(x0=np.zeros(self._nu * self._N), p=p_val, lbx=self.lbx, ubx=self.ubx)
            # CasADi flattens column-major (F-order). We must reshape with order='F' to get back (nu, N)
            U_opt = np.array(res['x']).reshape(self._nu, self._N, order='F')
            return U_opt[:, 0] # Return first action
        except Exception as e:
            print(f"Solver failed: {e}")
            return np.zeros(self._nu)


class KQLMPCController(Controller):
    """Wrapper around the Koopman LMPC solver to match the env Controller API."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)

        # Config
        self._dt = 1.0 / config.env.freq
        self._horizon = 40
        
        # Tuning
        q_diag = [50, 50, 200, 2, 2, 2, 1, 1, 1]
        r_diag = [1, 1, 1, 2]
        
        # Create Solver
        self._kq = KoopmanLMPC(self._horizon, self._dt, q_diag, r_diag)
        
        # Trajectory
        self._waypoints_pos, self._waypoints_yaw = self._generate_trajectory(config)
        self._tick = 0
        self._tick_max = len(self._waypoints_pos) - 1 - self._N_eff(self._horizon) # Safe indexing

    def _N_eff(self, h):
        return h

    def _generate_trajectory(self, config):
        """Generate cubic spline trajectory from predefined waypoints."""
        waypoints = np.array(
            [
                [1.0, 1.5, 0.05],
                [0.8, 1.0, 0.2],
                [0.55, -0.3, 0.5],
                [0.2, -1.3, 0.65],
                [1.1, -0.85, 1.1],
                [0.2, 0.5, 0.65],
                [0.0, 1.2, 0.525],
                [0.0, 1.2, 1.1],
                [-0.5, 0.0, 1.1],
                [-0.5, -0.5, 1.1],
            ]
        )

        des_completion_time = 10
        ts = np.linspace(0, des_completion_time, np.shape(waypoints)[0])

        cs_x = CubicSpline(ts, waypoints[:, 0])
        cs_y = CubicSpline(ts, waypoints[:, 1])
        cs_z = CubicSpline(ts, waypoints[:, 2])

        ts_dense = np.linspace(0, des_completion_time, int(config.env.freq * des_completion_time))
        x_des = cs_x(ts_dense)
        y_des = cs_y(ts_dense)
        z_des = cs_z(ts_dense)

        # Extend last point
        x_des = np.concatenate((x_des, [x_des[-1]] * self._horizon))
        y_des = np.concatenate((y_des, [y_des[-1]] * self._horizon))
        z_des = np.concatenate((z_des, [z_des[-1]] * self._horizon))

        z_des = np.maximum(z_des, 0.5)

        waypoints_pos = np.stack((x_des, y_des, z_des)).T
        waypoints_yaw = np.zeros_like(x_des)

        return waypoints_pos, waypoints_yaw

    def _build_state_for_kq(self, obs: dict) -> np.ndarray:
        pos = np.asarray(obs["pos"], dtype=np.float64)
        vel = np.asarray(obs["vel"], dtype=np.float64)
        rpy = R.from_quat(obs["quat"]).as_euler("xyz")
        return np.concatenate((pos, vel, rpy))

    def compute_control(self, obs: dict[str, NDArray[np.floating]], info: dict | None = None) -> NDArray[np.floating]:
        # 1) Build state
        x_current = self._build_state_for_kq(obs)

        # 2) Build reference
        ref_traj = np.zeros((9, self._horizon))
        idx_start = min(self._tick, len(self._waypoints_pos) - self._horizon - 1)
        
        for k in range(self._horizon):
            idx = idx_start + k
            ref_traj[0:3, k] = self._waypoints_pos[idx]
            # Vel=0, Angles=0 (except yaw)
            ref_traj[8, k] = self._waypoints_yaw[idx]

        # 3) Solve
        u0 = self._kq.solve(x_current, ref_traj)

        # 4) Map action
        # u0 is [roll, pitch, yaw, thrust_delta]
        # Env expects [thrust, roll, pitch, yaw]
        
        thrust_cmd = u0[3] + THRUST_HOVER
        thrust_cmd = np.clip(thrust_cmd, 0.0, 3.0 * THRUST_HOVER)
        
        roll_cmd = np.clip(u0[0], -1.0, 1.0)
        pitch_cmd = np.clip(u0[1], -1.0, 1.0)
        yaw_cmd = np.clip(u0[2], -2.0, 2.0)

        action = np.array([thrust_cmd, roll_cmd, pitch_cmd, yaw_cmd], dtype=np.float32)
        return action

    def step_callback(self, action, obs, reward, terminated, truncated, info):
        self._tick += 1
        return False

    def episode_callback(self):
        self._tick = 0
