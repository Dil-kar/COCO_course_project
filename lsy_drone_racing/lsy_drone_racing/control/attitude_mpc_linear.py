import numpy as np
import casadi as ca
from lsy_drone_racing.control import Controller
from scipy.spatial.transform import Rotation as R

MASS = 0.027
GRAVITY = 9.81
THRUST_HOVER = MASS * GRAVITY

class LinearMPC(Controller):
    """Linear MPC for quadrotor hover/trajectory tracking."""

    def __init__(self, obs, info, config):
        super().__init__(obs, info, config)
        self._N = 20  # prediction horizon
        self._dt = 1 / config.env.freq
        self._nx = 9  # [x, y, z, vx, vy, vz, r, p, y]
        self._nu = 4  # [roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd]

        # Weight matrices
        self.Q = np.diag([10, 10, 10, 1, 1, 1, 1, 1, 1])
        self.R = np.diag([5, 5, 5, 8])

        # Set up linearized discrete-time dynamics around hover
        A = np.zeros((9, 9))
        A[0:3, 3:6] = np.eye(3) * self._dt
        A[3:6, 6:9] = np.zeros((3, 3))  # small-angle approx
        A[3:6, 3:6] = np.eye(3)
        A[6:9, 6:9] = np.eye(3)
        self.A = ca.DM(A)

        B = np.zeros((9, 4))
        B[3:6, 0:3] = np.eye(3) * self._dt  # roll/pitch/yaw effect on acceleration
        B[3:6, 3] = np.array([0, 0, self._dt / MASS])  # thrust affects z acceleration
        B[6:9, 0:3] = np.eye(3) * self._dt
        self.B = ca.DM(B)

        # Initial reference (hover at current pos)
        self.ref_pos = obs["pos"].copy()
        self.ref_yaw = 0.0

        # Set up CasADi variables for MPC
        self._x = ca.MX.sym("x", self._nx)
        self._u = ca.MX.sym("u", self._nu)

        self._U = ca.MX.sym("U", self._nu, self._N)
        self._X = ca.MX.sym("X", self._nx, self._N + 1)

        self._x0 = np.zeros(self._nx)

        # Build MPC QP solver
        self._build_mpc_solver()

        self._tick = 0
        self._finished = False

    def _build_mpc_solver(self):
        """Construct MPC QP solver using CasADi."""
        cost = 0
        g = []

        X = self._X
        U = self._U
        x0 = self._x0

        # Constraints and dynamics
        g.append(X[:, 0] - x0)  # initial condition

        for k in range(self._N):
            x_next = ca.mtimes(self.A, X[:, k]) + ca.mtimes(self.B, U[:, k])
            g.append(X[:, k + 1] - x_next)

            # Cost: track reference hover
            x_ref = ca.DM([self.ref_pos[0], self.ref_pos[1], self.ref_pos[2], 0, 0, 0, 0, 0, self.ref_yaw])
            cost += ca.mtimes([(X[:, k] - x_ref).T, self.Q, (X[:, k] - x_ref)]) \
                    + ca.mtimes([U[:, k].T, self.R, U[:, k]])

        # Terminal cost
        cost += ca.mtimes([(X[:, self._N] - x_ref).T, self.Q, (X[:, self._N] - x_ref)])

        # Flatten constraints
        g = ca.vertcat(*g)

        # Create solver
        nlp = {'x': ca.vertcat(ca.reshape(U, -1, 1)), 'f': cost, 'g': g}
        opts = {'ipopt.print_level': 0, 'print_time': 0}
        self.solver = ca.nlpsol("solver", "ipopt", nlp, opts)

        # Constraint bounds (dynamics equality)
        self.lbg = np.zeros(g.shape[0])
        self.ubg = np.zeros(g.shape[0])

    def compute_control(self, obs, info=None):
        # Update initial state
        rpy = R.from_quat(obs["quat"]).as_euler("xyz")
        x0 = np.concatenate([obs["pos"], obs["vel"], rpy])
        self._x0 = x0

        # Solve MPC
        res = self.solver(x0=np.zeros(self._nu * self._N), lbg=self.lbg, ubg=self.ubg)
        U_opt = np.array(res['x']).reshape(self._nu, self._N)
        u0 = U_opt[:, 0]

        # Convert to [thrust, roll, pitch, yaw] for environment
        u0 = np.array([u0[3], u0[0], u0[1], u0[2]], dtype=np.float32)
        return u0

    def step_callback(self, action, obs, reward, terminated, truncated, info):
        self._tick += 1
        return False

    def episode_callback(self):
        self._tick = 0
