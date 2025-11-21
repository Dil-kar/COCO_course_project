import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R
from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.attitude_mpc import create_ocp_solver, MASS


class IterativeLearningMPC(Controller):
    """Iterative Learning MPC for drone racing based on arXiv:2508.01103."""

    def __init__(self, obs, info, config):
        super().__init__(obs, info, config)
        self._N = 30
        self._dt = 1 / config.env.freq
        self._T_HORIZON = self._N * self._dt
        self._config = config

        # Generate desired trajectory
        self._waypoints_pos, self._waypoints_yaw = self._generate_trajectory()

        # Create Acados OCP solver
        self._acados_ocp_solver, self._ocp = create_ocp_solver(self._T_HORIZON, self._N)
        self._nx = self._ocp.model.x.rows()
        self._nu = self._ocp.model.u.rows()
        self._ny = self._nx + self._nu
        self._ny_e = self._nx

        # Iteration bookkeeping
        self._tick = 0
        self._tick_max = len(self._waypoints_pos) - 1 - self._N
        self._finished = False
        self._past_trajectories = []  # Store previous successful trajectories

    def _generate_trajectory(self):
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

        des_completion_time = 8
        ts = np.linspace(0, des_completion_time, np.shape(waypoints)[0])

        cs_x = CubicSpline(ts, waypoints[:, 0])
        cs_y = CubicSpline(ts, waypoints[:, 1])
        cs_z = CubicSpline(ts, waypoints[:, 2])

        ts_dense = np.linspace(0, des_completion_time, int(self._config.env.freq * des_completion_time))
        x_des = cs_x(ts_dense)
        y_des = cs_y(ts_dense)
        z_des = cs_z(ts_dense)

        # Extend last point for horizon
        x_des = np.concatenate((x_des, [x_des[-1]] * self._N))
        y_des = np.concatenate((y_des, [y_des[-1]] * self._N))
        z_des = np.concatenate((z_des, [z_des[-1]] * self._N))

        waypoints_pos = np.stack((x_des, y_des, z_des)).T
        waypoints_yaw = np.zeros_like(x_des)  # keep yaw zero

        return waypoints_pos, waypoints_yaw

    def compute_control(self, obs, info=None):
        i = min(self._tick, self._tick_max)
        if self._tick >= self._tick_max:
            self._finished = True

        # Convert quaternion to RPY
        obs["rpy"] = R.from_quat(obs["quat"]).as_euler("xyz")
        x0 = np.concatenate((obs["pos"], obs["vel"], obs["rpy"]))

        # Set initial state
        self._acados_ocp_solver.set(0, "lbx", x0)
        self._acados_ocp_solver.set(0, "ubx", x0)

        # Reference including iterative learning term from past trajectories
        yref = np.zeros((self._N, self._ny))
        for j in range(self._N):
            idx = min(i + j, len(self._waypoints_pos) - 1)
            yref[j, 0:3] = self._waypoints_pos[idx]
            yref[j, 5] = self._waypoints_yaw[idx]
            yref[j, 9] = MASS * 9.81  # hover thrust

            # Incorporate safe set / past trajectory penalty (ILMPC)
            if self._past_trajectories:
                yref[j, 0:3] += 0.1 * (self._past_trajectories[-1][idx, 0:3] - yref[j, 0:3])

            self._acados_ocp_solver.set(j, "yref", yref[j])

        yref_e = np.zeros((self._ny_e))
        yref_e[0:3] = self._waypoints_pos[min(i + self._N, len(self._waypoints_pos) - 1)]
        yref_e[5] = self._waypoints_yaw[min(i + self._N, len(self._waypoints_yaw) - 1)]
        self._acados_ocp_solver.set(self._N, "yref", yref_e)

        # Solve OCP
        self._acados_ocp_solver.solve()
        u0 = self._acados_ocp_solver.get(0, "u")

        # Convert to [thrust, roll, pitch, yaw] format expected by environment
        u0 = np.array([u0[3], *u0[:3]], dtype=np.float32)
        return u0

    def step_callback(self, action, obs, reward, terminated, truncated, info):
        self._tick += 1
        return self._finished

    def episode_callback(self):
        """Store successful trajectory for ILMPC iteration."""
        trajectory = np.zeros((len(self._waypoints_pos), self._nx))
        # TODO: fill trajectory from logged states
        self._past_trajectories.append(trajectory)
        self._tick = 0
