"""
This module implements an MPC using Real-Time Iteration (RTI) for aggressive drone maneuvers.

It explicitly separates the preparation and feedback phases of the RTI scheme (SQP_RTI) 
to minimize latency, utilizing the acados framework.
"""

from __future__ import annotations  # Python 3.10 type hints

import os
from typing import TYPE_CHECKING

import numpy as np
import scipy
import scipy.linalg
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from casadi import MX, cos, sin, vertcat
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

# Attempt to import the base Controller class. Provide a placeholder if not found.
try:
    from lsy_drone_racing.control import Controller
except ImportError:
    print("Info: lsy_drone_racing.control not found. Using placeholder Controller class.")
    class Controller:
        """Base class for controllers (Placeholder)."""
        def __init__(self, obs: dict, info: dict, config: dict):
            pass
        
        def compute_control(self, obs: dict, info: dict | None = None):
            raise NotImplementedError
            
        def step_callback(self, action, obs, reward, terminated, truncated, info):
            return False
            
        def episode_callback(self):
            pass


if TYPE_CHECKING:
    from numpy.typing import NDArray


# Drone parameters (as provided in attitude_mpc.py)
PARAMS_RPY = np.array([[-12.7, 10.15], [-12.7, 10.15], [-8.117, 14.36]])
PARAMS_ACC = np.array([0.1906, 0.4903])
MASS = 0.027
GRAVITY = 9.81
THRUST_MIN = 0.02
THRUST_MAX = 0.1125


def export_quadrotor_ode_model() -> AcadosModel:
    """Symbolic Quadrotor Model."""
    model_name = "lsy_rti_mpc"

    # Define state vector: [pos(3), vel(3), rpy(3)] -> 9 states
    pos = vertcat(MX.sym("x"), MX.sym("y"), MX.sym("z"))
    vel = vertcat(MX.sym("vx"), MX.sym("vy"), MX.sym("vz"))
    rpy = vertcat(MX.sym("r"), MX.sym("p"), MX.sym("y"))
    states = vertcat(pos, vel, rpy)

    # Define input vector: [r_cmd, p_cmd, y_cmd, thrust_cmd] -> 4 inputs
    r_cmd, p_cmd, y_cmd = MX.sym("r_cmd"), MX.sym("p_cmd"), MX.sym("y_cmd")
    thrust_cmd = MX.sym("thrust_cmd")
    inputs = vertcat(r_cmd, p_cmd, y_cmd, thrust_cmd)

    # Define nonlinear system dynamics (identical to attitude_mpc.py)
    pos_dot = vel
    # Body z-axis in world frame
    z_axis = vertcat(
        cos(rpy[0]) * sin(rpy[1]) * cos(rpy[2]) + sin(rpy[0]) * sin(rpy[2]),
        cos(rpy[0]) * sin(rpy[1]) * sin(rpy[2]) - sin(rpy[0]) * cos(rpy[2]),
        cos(rpy[0]) * cos(rpy[1]),
    )
    # Thrust model (linear approximation)
    thrust = PARAMS_ACC[0] + PARAMS_ACC[1] * inputs[3]
    vel_dot = thrust * z_axis / MASS - np.array([0.0, 0.0, GRAVITY])
    # Attitude dynamics (simplified linear model)
    rpy_dot = PARAMS_RPY[:, 0] * rpy + PARAMS_RPY[:, 1] * inputs[:3]
    f = vertcat(pos_dot, vel_dot, rpy_dot)

    # Initialize the model for Acados
    model = AcadosModel()
    model.name = model_name
    model.f_expl_expr = f
    model.x = states
    model.u = inputs

    return model


def create_ocp_solver_rti(
    Tf: float, N: int, verbose: bool = False
) -> tuple[AcadosOcpSolver, AcadosOcp]:
    """Creates an acados OCP Solver configured for RTI."""
    ocp = AcadosOcp()
    ocp.model = export_quadrotor_ode_model()

    # Dimensions
    nx = ocp.model.x.rows()
    nu = ocp.model.u.rows()
    ny = nx + nu
    ny_e = nx
    ocp.dims.N = N

    # Cost Function (LINEAR_LS is required for efficient GAUSS_NEWTON Hessian)
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    # Weights (Using weights from the original attitude_mpc.py)
    Q = np.diag(
        [
            10.0, 10.0, 10.0,  # pos
            0.0, 0.0, 0.0,     # vel
            0.0, 0.0, 0.0,     # rpy
        ]
    )
    R = np.diag([5.0, 5.0, 5.0, 8.0]) # Input regularization

    ocp.cost.W = scipy.linalg.block_diag(Q, R)
    ocp.cost.W_e = Q

    # Selection matrices (Mapping states/inputs to the cost function, as in original)
    Vx = np.zeros((ny, nx))
    Vx[:3, :3] = np.eye(3)  # Select position states
    ocp.cost.Vx = Vx

    Vu = np.zeros((ny, nu))
    Vu[nx : nx + nu, :] = np.eye(nu)  # Select all inputs
    ocp.cost.Vu = Vu

    Vx_e = np.zeros((ny_e, nx))
    Vx_e[:3, :3] = np.eye(3)
    ocp.cost.Vx_e = Vx_e

    # Initial references
    ocp.cost.yref, ocp.cost.yref_e = np.zeros((ny,)), np.zeros((ny_e,))

    # Constraints
    # State Constraints (rpy < 60 deg ≈ 1.0 rad)
    ocp.constraints.lbx = np.array([-1.0, -1.0, -1.0])
    ocp.constraints.ubx = np.array([1.0, 1.0, 1.0])
    ocp.constraints.idxbx = np.array([6, 7, 8]) # Indices for rpy

    # Input Constraints
    ocp.constraints.lbu = np.array([-1.0, -1.0, -1.0, THRUST_MIN * 4])
    ocp.constraints.ubu = np.array([1.0, 1.0, 1.0, THRUST_MAX * 4])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])

    ocp.constraints.x0 = np.zeros((nx))

    # --- Solver Options: Configuration for RTI ---
    ocp.solver_options.tf = Tf
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
    
    # Use Gauss-Newton Hessian approximation (standard for RTI)
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    
    # KEY CONFIGURATION: Enable Real-Time Iteration scheme
    ocp.solver_options.nlp_solver_type = "SQP_RTI" 
    
    ocp.solver_options.tol = 1e-5
    ocp.solver_options.qp_solver_cond_N = N
    # Warm starting is crucial for RTI efficiency
    ocp.solver_options.qp_solver_warm_start = 1
    ocp.solver_options.qp_solver_iter_max = 20

    # Create the solver
    # Ensure the generation directory exists
    code_export_dir = "c_generated_code"
    if not os.path.exists(code_export_dir):
        os.makedirs(code_export_dir)
        
    json_file = os.path.join(code_export_dir, "lsy_rti_mpc.json")
    
    try:
        acados_ocp_solver = AcadosOcpSolver(
            ocp, json_file=json_file, verbose=verbose
        )
    except Exception as e:
        print(f"Error creating AcadosOcpSolver. Ensure Acados is installed and configured.")
        raise e

    return acados_ocp_solver, ocp


class AttitudeMPCRTI(Controller):
    """MPC using the RTI scheme, explicitly separating preparation and feedback phases."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)
        self._N = 30  # Prediction horizon steps
        
        # Determine environment frequency
        try:
            self.env_freq = config.env.freq
        except (AttributeError, TypeError):
            print("Warning: config.env.freq not found or invalid. Defaulting to 100Hz.")
            self.env_freq = 100.0

        self._dt = 1 / self.env_freq
        self._T_HORIZON = self._N * self._dt

        # Trajectory Generation (same waypoints as attitude_mpc.py)
        waypoints = np.array(
            [
                [1.0, 1.5, 0.6], 
                [0.8, 1.0, 0.6], 
                [0.55, -0.3, 0.6], 
                [0.0, -1.3, 1.5],
                [1.1, -0.85, 1.1], 
                [0.2, 0.5, 0.65], 
                [0.0, 1.2, 0.6], 
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

        num_points = int(self.env_freq * des_completion_time)
        ts_dense = np.linspace(0, des_completion_time, num_points)
        x_des = cs_x(ts_dense)
        y_des = cs_y(ts_dense)
        z_des = cs_z(ts_dense)

        # Append end point to cover the horizon
        x_des = np.concatenate((x_des, [x_des[-1]] * self._N))
        y_des = np.concatenate((y_des, [y_des[-1]] * self._N))
        z_des = np.concatenate((z_des, [z_des[-1]] * self._N))
        self._waypoints_pos = np.stack((x_des, y_des, z_des)).T
        self._waypoints_yaw = np.zeros_like(x_des)

        # Initialize Solver
        self._acados_ocp_solver, self._ocp = create_ocp_solver_rti(self._T_HORIZON, self._N)
        self._nx = self._ocp.model.x.rows()
        self._nu = self._ocp.model.u.rows()
        self._ny = self._nx + self._nu
        self._ny_e = self._nx

        self._tick = 0
        self._tick_max = len(x_des) - 1 - self._N
        self._finished = False
        
        # Calculate reference hover thrust input (u_ref) for regularization
        # Model: F = PARAMS_ACC[0] + PARAMS_ACC[1] * U
        # Hover: M*G = PARAMS_ACC[0] + PARAMS_ACC[1] * U_hover
        self.u_hover_ref = (MASS * GRAVITY - PARAMS_ACC[0]) / PARAMS_ACC[1]

        # Flag to track if this is the first control call
        self._first_call = True

    def _preparation_phase(self):
        """
        RTI Preparation Phase (rti_phase=1): 
        Update references, linearize dynamics, and formulate/factorize the QP.
        """
        
        i = min(self._tick, self._tick_max)
        
        # 1. Set Reference Trajectory (yref) for the horizon
        for j in range(self._N):
            idx = min(i + j, len(self._waypoints_pos) - 1)
            yref_j = np.zeros(self._ny)
            
            # State references (mapped by Vx in OCP setup)
            yref_j[0:3] = self._waypoints_pos[idx]      # position
            
            # Input references (mapped by Vu in OCP setup)
            # Regularize thrust command (index nx+3) towards hover.
            yref_j[self._nx + 3] = self.u_hover_ref
            
            self._acados_ocp_solver.set(j, "yref", yref_j)

        # Terminal reference (yref_e)
        idx_e = min(i + self._N, len(self._waypoints_pos) - 1)
        yref_e = np.zeros((self._ny_e))
        yref_e[0:3] = self._waypoints_pos[idx_e]
        self._acados_ocp_solver.set(self._N, "yref", yref_e)

        # 2. Trigger the preparation phase
        # rti_phase = 1 triggers linearization and QP preparation.
        self._acados_ocp_solver.options_set("rti_phase", 1)
        status = self._acados_ocp_solver.solve()
        
        # Status 0: success, 2: max iterations reached (acceptable in RTI QP solver)
        # Status 5: ACADOS_READY (solver created/ready) - this is expected on first call
        if status not in [0, 2, 5]: 
             print(f"Warning: Preparation phase failed at tick {self._tick} with status {status}")

    def _process_obs(self, obs: dict[str, NDArray[np.floating]]) -> NDArray[np.floating]:
        """Helper to convert observation dictionary to state vector x0."""
        # Convert quaternion observation to Euler angles for the state vector
        if "quat" in obs:
            rpy = R.from_quat(obs["quat"]).as_euler("xyz")
        elif "rpy" in obs:
             rpy = obs["rpy"]
        else:
             raise ValueError("Observation must contain 'quat' or 'rpy'.")
             
        x0 = np.concatenate((obs["pos"], obs["vel"], rpy))
        return x0

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """RTI Feedback Phase (rti_phase=2): Update initial state and solve the prepared QP."""
        
        if self._tick >= self._tick_max:
            self._finished = True

        # 1. Update the initial state constraint (x0)
        x0 = self._process_obs(obs)
        
        # Set the initial constraint
        self._acados_ocp_solver.set(0, "lbx", x0)
        self._acados_ocp_solver.set(0, "ubx", x0)

        # On first call, do a full solve to initialize the solver
        if self._first_call:
            # Set up references for the first solve
            i = min(self._tick, self._tick_max)
            for j in range(self._N):
                idx = min(i + j, len(self._waypoints_pos) - 1)
                yref_j = np.zeros(self._ny)
                yref_j[0:3] = self._waypoints_pos[idx]
                yref_j[self._nx + 3] = self.u_hover_ref
                self._acados_ocp_solver.set(j, "yref", yref_j)
            
            idx_e = min(i + self._N, len(self._waypoints_pos) - 1)
            yref_e = np.zeros((self._ny_e))
            yref_e[0:3] = self._waypoints_pos[idx_e]
            self._acados_ocp_solver.set(self._N, "yref", yref_e)
            
            # Full solve on first call
            status = self._acados_ocp_solver.solve()
            self._first_call = False
        else:
            # === FEEDBACK PHASE ===
            # rti_phase = 2 triggers the QP solution using the new x0.
            self._acados_ocp_solver.options_set("rti_phase", 2)
            status = self._acados_ocp_solver.solve()
        
        # 2. Get the control input (with fallback for solver failures)
        if status in [0, 2]:  # Success or max iterations (acceptable)
            u0 = self._acados_ocp_solver.get(0, "u")
        else:
            # Solver failed - use hover command as fallback
            print(f"Warning: Solver failed at tick {self._tick} with status {status}, using hover command")
            # Fallback: hover command [r_cmd=0, p_cmd=0, y_cmd=0, thrust_cmd=hover_thrust]
            u0 = np.array([0.0, 0.0, 0.0, self.u_hover_ref])

        # === PREPARATION PHASE (for the next step) ===
        # Start preparation for the next iteration immediately to maximize available time.
        self._tick += 1
        if not self._finished:
            self._preparation_phase()

        # Format the control command (assuming the environment expects [thrust, r, p, y])
        # Solver output u0 is [r_cmd, p_cmd, y_cmd, thrust_cmd]
        u0_formatted = np.array([u0[3], u0[0], u0[1], u0[2]], dtype=np.float32)

        return u0_formatted

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Callback after environment step."""
        # Tick is incremented within compute_control to optimize timing.
        return self._finished

    def episode_callback(self):
        """Reset the controller at the end of an episode."""
        self._tick = 0
        self._finished = False
        self._first_call = True
        # Reset the solver internal state (warm start trajectory, linearizations)
        self._acados_ocp_solver.reset()