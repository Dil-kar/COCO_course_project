from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, List, Dict

import fire
import gymnasium
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy
import numpy as np
import mujoco
import imageio
import os

# Plotting Libraries
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from scipy.interpolate import splprep, splev

# Ensure MuJoCo uses the correct backend for off-screen rendering
os.environ['MUJOCO_GL'] = 'egl' 

# Assuming lsy_drone_racing is installed/available
from lsy_drone_racing.utils import load_config, load_controller 

if TYPE_CHECKING:
    from ml_collections import ConfigDict
    from lsy_drone_racing.control.controller import Controller
    from lsy_drone_racing.envs.drone_race import DroneRaceEnv

logger = logging.getLogger(__name__)

# --- CONFIGURATION CONSTANTS ---
# The waypoints provided in your prompt, defining the desired path.
WAYPOINTS = np.array([
    [1.0, 1.5, 0.05], [0.8, 1.0, 0.2], [0.55, -0.3, 0.5],
    [0.2, -1.3, 0.65], [1.1, -0.85, 1.1], [0.2, 0.5, 0.65],
    [0.0, 1.2, 0.525], [0.0, 1.2, 1.1], [-0.5, 0.0, 1.1], [-0.5, -0.5, 1.1],
])

# --- METRIC AND PLOTTING CLASSES ---

class MetricTracker:
    """Collects and summarizes performance data for one episode."""
    def __init__(self, waypoints):
        self.waypoints = waypoints
        self.reset()

    def reset(self):
        self.positions = []
        self.velocities = []
        self.solver_times = []
        self.ref_errors = []
        self.success = False
        self.gates_passed = 0
        self.lap_time = 0.0

    def update(self, pos, vel, solver_dt):
        """Records data for a single time step."""
        self.positions.append(pos)
        self.velocities.append(np.linalg.norm(vel)) # Magnitude of velocity
        self.solver_times.append(solver_dt * 1000) # Convert to ms
        
        # Tracking Error Proxy: Distance to the nearest waypoint point
        dist_to_path = np.min(np.linalg.norm(self.waypoints - pos, axis=1))
        self.ref_errors.append(dist_to_path)

    def get_summary(self):
        """Returns aggregated metrics for the episode."""
        return {
            "pos_history": np.array(self.positions),
            "vel_history": np.array(self.velocities),
            "avg_solver_time_ms": np.mean(self.solver_times) if self.solver_times else 0,
            "max_solver_time_ms": np.max(self.solver_times) if self.solver_times else 0,
            # RMSE of the tracking error
            "rmse_tracking_error": np.sqrt(np.mean(np.array(self.ref_errors)**2)) if self.ref_errors else 0,
            "lap_time": self.lap_time,
            "success": self.success,
            "gates_passed": self.gates_passed
        }



# --- MAIN SIMULATION FUNCTION ---

def simulate(
    config: str = "level0.toml",
    controller: str | None = None,
    n_runs: int = 3,
    gui: bool | None = None,
    record_video: bool = False,
    video_path: str = "race_video.mp4",
    disturbance_scale: float = 0.0,
    plot_mode: str = "evolution" # 'comparison' or 'evolution'
) -> List[Dict]:
    """Evaluate the drone controller over multiple episodes and log metrics."""
    
    config_path = Path(__file__).parents[1] / "config" / config
    config_obj = load_config(config_path)
    

    # ... [GUI setup and Controller/Env loading remain similar] ...
    if gui is None:
        gui = config_obj.sim.gui
    else:
        config_obj.sim.gui = gui

    control_path = Path(__file__).parents[1] / "lsy_drone_racing/control"
    controller_path = control_path / (controller or config_obj.controller.file)
    controller_cls = load_controller(controller_path)

    env: DroneRaceEnv = gymnasium.make(
        config_obj.env.id,
        freq=config_obj.env.freq,
        sim_config=config_obj.sim,
        sensor_range=config_obj.env.sensor_range,
        control_mode=config_obj.env.control_mode,
        track=config_obj.env.track,
        disturbances=config_obj.env.get("disturbances"),
        randomizations=config_obj.env.get("randomizations"),
        seed=config_obj.env.seed,
    )
    env = JaxToNumpy(env)
    
    # Video Setup (omitted for brevity, assume your original logic works)
    # ...

    tracker = MetricTracker(WAYPOINTS)
    ep_results = []
    
    all_runs_pos = []
    all_runs_vel = []

    for run_idx in range(n_runs):
        tracker.reset()
        obs, info = env.reset()
        # Re-instantiate controller for each run if it's IL MPC, 
        # or handle learning updates via its internal logic
        controller_obj: Controller = controller_cls(obs, info, config_obj) 
        i = 0

        print(f"--- Starting Run {run_idx + 1}/{n_runs} for {controller or 'Default'} ---")

        while True:
            curr_time = i / config_obj.env.freq

            # --- MEASURE SOLVER TIME ---
            t_start = time.perf_counter()
            action = controller_obj.compute_control(obs, info)
            t_end = time.perf_counter()
            solver_dt = t_end - t_start
            
            # Update Metrics
            tracker.update(obs["pos"], obs["vel"], solver_dt) # Assuming single drone [0]
            
            # Step Env
            obs, reward, terminated, truncated, info = env.step(action)
            controller_finished = controller_obj.step_callback(action, obs, reward, terminated, truncated, info)

            # ... [Video capture logic] ...
            
            if terminated or truncated or controller_finished:
                # Log success logic
                gates_passed = obs["target_gate"]
                if gates_passed == -1: 
                    gates_passed = len(config_obj.env.track.gates)
                    tracker.success = True
                else:
                    tracker.success = False
                
                tracker.gates_passed = gates_passed
                tracker.lap_time = curr_time
                break

            # ... [GUI render logic] ...
            i += 1

        controller_obj.episode_callback()
        controller_obj.episode_reset()
        
        # Save metrics and trajectory history for this run
        summary = tracker.get_summary()
        ep_results.append(summary)
        all_runs_pos.append(summary["pos_history"])
        all_runs_vel.append(summary["vel_history"])

        # Print Quick Stats
        print(f"Run {run_idx+1}: Time={summary['lap_time']:.2f}s | "
              f"Success={summary['success']} | "
              f"RMSE={summary['rmse_tracking_error']:.3f} | "
              f"AvgSolver={summary['avg_solver_time_ms']:.2f}ms")

    # --- FINAL REPORT AND PLOTTING ---
    print("\n" + "="*40)
    print(f"FINAL AGGREGATE RESULTS FOR: {controller or 'Default'}")
    print("="*40)
    
    # Calculate aggregate metrics
    success_runs = [r for r in ep_results if r['success']]
    
    avg_lap = np.mean([r['lap_time'] for r in success_runs]) if success_runs else np.nan
    success_rate = np.mean([1 if r['success'] else 0 for r in ep_results]) * 100
    avg_rmse = np.mean([r['rmse_tracking_error'] for r in ep_results])
    avg_solver = np.mean([r['avg_solver_time_ms'] for r in ep_results])
    
    print(f"Success Rate:         {success_rate:.1f}%")
    print(f"Avg Lap Time (Valid): {avg_lap:.4f} s")
    print(f"Avg Tracking RMSE:    {avg_rmse:.4f} m")
    print(f"Avg Solver Time:      {avg_solver:.4f} ms")
    print("="*40 + "\n")

    # print(ep_results)
    

    np.save("3lap.npy", np.array(ep_results))

    env.close()
    return ep_results


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger("lsy_drone_racing").setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
    
    # Example usage:
    # 1. To run IL MPC for 10 laps and plot the evolution (Like image_ca1ae8.png):
    # python analysis_script.py simulate --controller="il_mpc_controller.py" --n_runs=10 
    
    # 2. To run Linear MPC (1 lap) and plot XY/XZ path comparison (Like image_ca6ca3.png):
    # python analysis_script.py simulate --controller="linear_mpc_controller.py" --n_runs=1
    
    # 3. To test robustness of SQP MPC:
    # python analysis_script.py simulate --controller="sqp_mpc_controller.py" --n_runs=5 
    
    fire.Fire(simulate, serialize=lambda _: None)