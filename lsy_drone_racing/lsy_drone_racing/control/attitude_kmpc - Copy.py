from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from scipy.spatial.transform import Rotation as R

# Replace with actual import(s) once you inspect the package:
# from kq_lmpc_quadrotor import KQ_LMPC, create_kqlmpc  # <---- example / placeholder

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray

class KQLMPCController(Controller):
    """Wrapper around the kq_lmpc_quadrotor LMPC solver to match the env Controller API."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)

        # --- 1) create / configure KQ-LMPC solver here ---
        # TODO: replace the following placeholder with the actual constructor/factory provided by the package.
        # Example ideas (the real API may differ):
        #   from kq_lmpc_quadrotor import kqlmpc_demo, create_kqlmpc
        #   self.kq = create_kqlmpc(horizon=..., dt=1.0/config.env.freq, acados_options={...})
        #
        # If you only want LQR (no acados), the package README says Koopman LQR works without acados.
        #
        self._kq = None  # <-- set to actual LMPC object

        # horizon and dt (match your env frequency)
        self._dt = 1.0 / config.env.freq
        self._horizon = 30  # tune or read from package default

        # store config for later
        self._config = config
        self._tick = 0

        # If kq object has an init parameter set, initialize it from current obs
        # e.g., self._kq.init_state(x0)  # (example)

    def _build_state_for_kq(self, obs: dict) -> np.ndarray:
        """Convert environment obs -> state vector format expected by kq_lmpc_quadrotor.

        Typical mapping: pos(3), vel(3), orientation (quaternion or euler)
        The package works on SE(3) lifted observables; you will likely pass pos, vel, and quaternion or rotation matrix.
        """
        pos = np.asarray(obs["pos"], dtype=np.float64)     # shape (3,)
        vel = np.asarray(obs["vel"], dtype=np.float64)     # shape (3,)
        quat = np.asarray(obs["quat"], dtype=np.float64)   # [x,y,z,w] or env format
        # The repo likely expects a rotation matrix or quaternion in a particular order.
        # Convert quat -> euler or rotation matrix as needed:
        rpy = R.from_quat(quat).as_euler("xyz")  # roll,pitch,yaw
        # Example flattened state:
        x_kq = np.concatenate((pos, vel, rpy))
        return x_kq

    def compute_control(self, obs: dict[str, NDArray[np.floating]], info: dict | None = None) -> NDArray[np.floating]:
        """Compute 1-step control using the KQ LMPC solver and return action in env format.

        Returns an array [collective_thrust, roll, pitch, yaw] or the format your env expects.
        """
        # 1) Build state in kq format
        x_kq = self._build_state_for_kq(obs)

        # 2) If the kq solver needs the current reference trajectory, create it (positions & yaw)
        #    You can reuse the spline-based waypoint generation you used earlier OR the kq package may provide trajectory helpers.
        #    Example: ref_positions = self._waypoints_pos[self._tick : self._tick + horizon]
        #
        # 3) Query LMPC solver
        if self._kq is None:
            # first time - try to auto-create the solver from package; raises clearer errors
            raise RuntimeError("KQ LMPC object not created. Replace placeholder create call in __init__.")

        # Example (pseudo): u_seq = self._kq.solve(x_kq, ref_traj)
        # The real API might be: u0 = self._kq.get_control(x_kq) or self._kq.step(x_kq)
        # Replace the next two lines with the real solver calls:
        u0 = None
        try:
            # If the package returns a sequence of body/collective thrust/attitude commands,
            # take the first control.
            u0 = self._kq.get_action(x_kq)  # <<-- replace with actual method
        except Exception:
            # fallback: try another common API shape
            u_seq = self._kq.solve(x_kq)  # <<-- replace / adapt
            u0 = u_seq[0]

        # 4) Map package's action to the environment action ordering
        #    Your NMPC returned [thrust, r, p, y] after converting from rpyt->trpy
        #    The kq output might be thrust + desired body rates or thrust + attitude.
        #    Align order and types here:
        # Example mapping (if u0 == [r_cmd, p_cmd, y_cmd, thrust_cmd] like your NMPC used)
        u0 = np.asarray(u0, dtype=np.float32)
        # If kq returns [thrust, r, p, y] already, just return it.
        # But if it returns [r,p,y,thrust], reorder like your NMPC did earlier:
        if u0.shape[0] == 4:
            # If environment expects [thrust, r, p, y]:
            # detect current ordering (common packages use r,p,y,thrust); adapt:
            # try both possibilities conservatively:
            # if thrust likely to be the last element (common), reorder:
            maybe_thrust_last = u0[-1]
            if 0.0 <= maybe_thrust_last <= 10.0:  # crude check (thrust magnitude)
                action = np.array([u0[-1], u0[0], u0[1], u0[2]], dtype=np.float32)
            else:
                action = u0  # assume already correct
        else:
            # unexpected shape: raise or pad
            raise ValueError(f"unexpected LMPC output shape: {u0.shape}")

        return action

    def step_callback(self, action, obs, reward, terminated, truncated, info):
        self._tick += 1
        # update LMPC internal state if needed
        # e.g., self._kq.shift_horizon()
        return False

    def episode_callback(self):
        self._tick = 0
