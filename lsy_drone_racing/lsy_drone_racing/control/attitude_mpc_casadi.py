from __future__ import annotations

import numpy as np
from casadi import MX, vertcat, Function, sumsqr, Opti, cos, sin
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R
from typing import TYPE_CHECKING, Callable

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray

# --- System Constants ---
PARAMS_RPY = np.array([[-12.7, 10.15], [-12.7, 10.15], [-8.117, 14.36]])
PARAMS_ACC = np.array([0.1906, 0.4903])
MASS = 0.027
GRAVITY = 9.81
THRUST_MIN = 0.02
THRUST_MAX = 0.1125

# Weights
Q_POS = 10.0
R_RPY = 5.0
R_THRUST = 8.0

# --- Quadrotor Dynamics ---
def quadrotor_ode(x: MX, u: MX) -> MX:
    pos = x[0:3]
    vel = x[3:6]
    rpy = x[6:9]

    r_cmd, p_cmd, y_cmd, thrust_cmd = u[0], u[1], u[2], u[3]

    pos_dot = vel

    # Simple rotation to world frame Z-axis
    z_axis = vertcat(
        cos(rpy[0]) * sin(rpy[1]) * cos(rpy[2]) + sin(rpy[0]) * sin(rpy[2]),
        cos(rpy[0]) * sin(rpy[1]) * sin(rpy[2]) - sin(rpy[0]) * cos(rpy[2]),
        cos(rpy[0]) * cos(rpy[1]),
    )

    thrust = PARAMS_ACC[0] + PARAMS_ACC[1] * thrust_cmd
    vel_dot = thrust * z_axis / MASS - vertcat(0.0, 0.0, GRAVITY)

    rpy_dot = vertcat(
        PARAMS_RPY[0, 0] * rpy[0] + PARAMS_RPY[0, 1] * r_cmd,
        PARAMS_RPY[1, 0] * rpy[1] + PARAMS_RPY[1, 1] * p_cmd,
        PARAMS_RPY[2, 0] * rpy[2] + PARAMS_RPY[2, 1] * y_cmd,
    )

    return vertcat(pos_dot, vel_dot, rpy_dot)

# --- RK4 Integrator ---
def setup_rk4_integrator(f: Callable[[MX, MX], MX], dt: float, nx: int, nu: int) -> Callable[[MX, MX], MX]:
    x = MX.sym('x', nx)
    u = MX.sym('u', nu)
    k1 = f(x, u)
    k2 = f(x + dt/2 * k1, u)
    k3 = f(x + dt/2 * k2, u)
    k4 = f(x + dt * k3, u)
    x_next = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    return Function('F_RK4', [x, u], [x_next])

# --- Attitude MPC ---
class AttitudeMPCCasadi(Controller):
    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)
        self._N = 30
        self._dt = 1 / config.env.freq
        self._nx = 9
        self._nu = 4

        # --- Trajectory ---
        waypoints = np.array([
            [1.0, 1.5, 0.05], [0.8, 1.0, 0.2], [0.55, -0.3, 0.5],
            [0.2, -1.3, 0.65], [1.1, -0.85, 1.1], [0.2, 0.5, 0.65],
            [0.0, 1.2, 0.525], [0.0, 1.2, 1.1], [-0.5, 0.0, 1.1], [-0.5, -0.5, 1.1],
        ])
        des_time = 8
        ts = np.linspace(0, des_time, waypoints.shape[0])
        ts_interp = np.linspace(0, des_time, int(config.env.freq * des_time))
        cs_x, cs_y, cs_z = CubicSpline(ts, waypoints[:, 0]), CubicSpline(ts, waypoints[:, 1]), CubicSpline(ts, waypoints[:, 2])
        x_des, y_des, z_des = cs_x(ts_interp), cs_y(ts_interp), cs_z(ts_interp)
        self._waypoints_pos = np.stack([np.concatenate([x_des, [x_des[-1]]*self._N]),
                                        np.concatenate([y_des, [y_des[-1]]*self._N]),
                                        np.concatenate([z_des, [z_des[-1]]*self._N])], axis=1)
        self._waypoints_yaw = np.concatenate([np.zeros_like(x_des), np.zeros(self._N)])

        self._tick = 0
        self._tick_max = len(x_des) - 1
        self._finished = False
        self._config = config

        # Warm start
        self._x_init = np.zeros((self._nx, self._N + 1))
        self._u_init = np.zeros((self._nu, self._N))
        self._u_init[3, :] = MASS * GRAVITY / (PARAMS_ACC[0] + PARAMS_ACC[1]*1.0)
        self._u_init = np.clip(self._u_init, 0.0, 1.0)

        self._setup_casadi_ocp()

    def _setup_casadi_ocp(self):
        self._opti = Opti()
        X = self._opti.variable(self._nx, self._N + 1)
        U = self._opti.variable(self._nu, self._N)
        P_X0 = self._opti.parameter(self._nx, 1)
        P_YREF = self._opti.parameter(self._nx*(self._N+1), 1)

        F_RK4 = setup_rk4_integrator(quadrotor_ode, self._dt, self._nx, self._nu)

        J = 0
        for k in range(self._N):
            self._opti.subject_to(X[:, k+1] == F_RK4(X[:, k], U[:, k]))
            yref_k = P_YREF[k*self._nx:(k+1)*self._nx]
            pos_err = X[0:3, k] - yref_k[0:3]
            rpy_cmd = U[0:3, k]
            thrust_cmd = U[3, k]
            J += Q_POS * sumsqr(pos_err) + R_RPY * sumsqr(rpy_cmd) + R_THRUST * sumsqr(thrust_cmd)

        # Terminal cost
        yref_e = P_YREF[self._N*self._nx:(self._N+1)*self._nx]
        pos_err_e = X[0:3, self._N] - yref_e[0:3]
        J += Q_POS * sumsqr(pos_err_e)

        # Initial state
        self._opti.subject_to(X[:, 0] == P_X0)

        # State and input bounds
        self._opti.subject_to(self._opti.bounded(-1.0, X[6:9, :], 1.0))
        self._opti.subject_to(self._opti.bounded(-1.0, U[0:3, :], 1.0))
        self._opti.subject_to(self._opti.bounded(0.0, U[3, :], 1.0))

        self._opti.minimize(J)
        opts = {'ipopt.print_level':0,'print_time':0,'ipopt.max_iter':50,'ipopt.tol':1e-5,
                'ipopt.warm_start_init_point':'yes','ipopt.mu_strategy':'adaptive',
                'ipopt.hessian_approximation':'limited-memory'}
        self._opti.solver('ipopt', opts)

        self._X, self._U, self._P_X0, self._P_YREF = X, U, P_X0, P_YREF

    def compute_control(self, obs: dict[str, NDArray[np.floating]], info: dict | None = None) -> NDArray[np.floating]:
        i = min(self._tick, self._tick_max)
        if self._tick >= self._tick_max:
            self._finished = True

        # Current state
        obs["rpy"] = R.from_quat(obs["quat"]).as_euler("xyz")
        x0 = np.concatenate([obs["pos"], obs["vel"], obs["rpy"]])
        self._opti.set_value(self._P_X0, x0)

        # Reference
        yref_full = np.zeros(self._nx*(self._N+1))
        indices = np.arange(i, i+self._N+1)
        for k, idx in enumerate(indices):
            yref_full[k*self._nx:k*self._nx+3] = self._waypoints_pos[idx, :]
            yref_full[k*self._nx+8] = self._waypoints_yaw[idx]
        self._opti.set_value(self._P_YREF, yref_full)

        # Warm start
        self._opti.set_initial(self._X, self._x_init)
        self._opti.set_initial(self._U, self._u_init)

        # Solve
        try:
            sol = self._opti.solve()
            u0 = sol.value(self._U)[:, 0]

            # Update warm start
            u_sol = sol.value(self._U)
            self._u_init[:, :-1] = u_sol[:, 1:]
            self._u_init[:, -1] = u_sol[:, -1]

            x_sol = sol.value(self._X)
            self._x_init[:, :-1] = x_sol[:, 1:]
            self._x_init[:, -1] = x_sol[:, -1]

        except RuntimeError as e:
            print(f"CasADi/IPOPT solver failed: {e}. Using previous control input.")
            u0 = self._u_init[:, 0]

        # Reorder for environment [thrust, roll, pitch, yaw]
        u0_reordered = np.array([u0[3], u0[0], u0[1], u0[2]], dtype=np.float32)

        # # reorder for env
        # u0_reordered = np.array([u0[3], u0[0], u0[1], u0[2]], dtype=np.float32)

        # wrap into full control tensor
        controls = np.zeros((1, 1, 13), dtype=np.float32)
        controls[0, 0, 0:4] = u0_reordered

        # return controls
        return u0_reordered

    def step_callback(self, action: NDArray[np.floating], obs: dict[str, NDArray[np.floating]], reward: float, terminated: bool, truncated: bool, info: dict) -> bool:
        self._tick += 1
        return self._finished

    def episode_callback(self):
        self._tick = 0
        self._x_init = np.zeros((self._nx, self._N+1))
        self._u_init = np.zeros((self._nu, self._N))
        self._u_init[3, :] = MASS * GRAVITY / (PARAMS_ACC[0] + PARAMS_ACC[1]*1.0)
        self._u_init = np.clip(self._u_init, 0.0, 1.0)
