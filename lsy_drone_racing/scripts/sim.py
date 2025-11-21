"""Simulate the competition as in the IROS 2022 Safe Robot Learning competition."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import fire
import gymnasium
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy
import numpy as np
import mujoco # Required for rendering
import imageio # Required for saving video
import os
os.environ['MUJOCO_GL'] = 'egl'  # <--- Add this line
from lsy_drone_racing.utils import load_config, load_controller

if TYPE_CHECKING:
    from ml_collections import ConfigDict
    from lsy_drone_racing.control.controller import Controller
    from lsy_drone_racing.envs.drone_race import DroneRaceEnv


logger = logging.getLogger(__name__)


def simulate(
    config: str = "level0.toml",
    controller: str | None = None,
    n_runs: int = 1,
    gui: bool | None = None,
    record_video: bool = False, # New argument
    video_path: str = "race_video.mp4" # New argument
) -> list[float]:
    """Evaluate the drone controller over multiple episodes.

    Args:
        config: The path to the configuration file.
        controller: The name of the controller file.
        n_runs: The number of episodes.
        gui: Enable/disable the simulation GUI.
        record_video: Whether to save a video of the simulation.
        video_path: Output path for the video file.
    """
    # Load configuration
    config = load_config(Path(__file__).parents[1] / "config" / config)
    if gui is None:
        gui = config.sim.gui
    else:
        config.sim.gui = gui

    # Load controller
    control_path = Path(__file__).parents[1] / "lsy_drone_racing/control"
    controller_path = control_path / (controller or config.controller.file)
    controller_cls = load_controller(controller_path)

    # Create environment
    env: DroneRaceEnv = gymnasium.make(
        config.env.id,
        freq=config.env.freq,
        sim_config=config.sim,
        sensor_range=config.env.sensor_range,
        control_mode=config.env.control_mode,
        track=config.env.track,
        disturbances=config.env.get("disturbances"),
        randomizations=config.env.get("randomizations"),
        seed=config.env.seed,
    )
    env = JaxToNumpy(env)

    # --- VIDEO RECORDING SETUP ---
    renderer = None
    mj_data_cpu = None
    frames = []
    recording_fps = 30
    # Calculate how many simulation steps to skip to achieve ~30 FPS video
    # e.g., if sim freq is 100Hz and video is 30fps, skip ~3 steps per frame
    record_interval = max(1, int(config.env.freq / recording_fps))

    if record_video:
        print(f"Initializing Renderer for recording to {video_path}...")
        # Access the underlying MuJoCo model from the unwrapped env
        mj_model = env.unwrapped.sim.mj_model
        mj_data_cpu = mujoco.MjData(mj_model) # Create a CPU data structure
        renderer = mujoco.Renderer(mj_model, height=480, width=640)
    # -----------------------------

    ep_times = []
    for run_idx in range(n_runs):
        obs, info = env.reset()
        controller: Controller = controller_cls(obs, info, config)
        i = 0
        sim_fps = 60 # Used for GUI throttling

        while True:
            curr_time = i / config.env.freq

            action = controller.compute_control(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            
            controller_finished = controller.step_callback(
                action, obs, reward, terminated, truncated, info
            )

            # --- CAPTURE FRAME ---
            if record_video and (i % record_interval == 0):
                # 1. Get JAX/GPU state
                raw_sim_data = env.unwrapped.sim.data
                # Assuming single environment (index 0)
                jax_pos = raw_sim_data.states.pos[0]
                jax_quat = raw_sim_data.states.quat[0]
                jax_mocap_pos = raw_sim_data.mjx_data.mocap_pos
                jax_mocap_quat = raw_sim_data.mjx_data.mocap_quat

                # 2. Sync to CPU MuJoCo Data
                # Sync Drone (Assumes drone is the first set of joints)
                mj_data_cpu.qpos[:3] = np.array(jax_pos)
                mj_data_cpu.qpos[3:7] = np.array(jax_quat)
                
                # Sync Gates/Obstacles (Mocap bodies)
                # Mocap data usually covers all envs in MJX, we just need the slice for the active objects
                # However, usually the layout is consistent. We simply copy the arrays.
                if jax_mocap_pos is not None:
                    mj_data_cpu.mocap_pos[:] = np.array(jax_mocap_pos)
                    mj_data_cpu.mocap_quat[:] = np.array(jax_mocap_quat)

                # 3. Update Geometry and Render
                mujoco.mj_forward(env.unwrapped.sim.mj_model, mj_data_cpu)
                renderer.update_scene(mj_data_cpu)
                frames.append(renderer.render())
            # ---------------------

            if terminated or truncated or controller_finished:
                break

            # Synchronize the GUI (Viewer window)
            if config.sim.gui:
                if ((i * sim_fps) % config.env.freq) < sim_fps:
                    env.render()
            i += 1

        controller.episode_callback()
        log_episode_stats(obs, info, config, curr_time)
        controller.episode_reset()
        ep_times.append(curr_time if obs["target_gate"] == -1 else None)
    
    # Save video after all runs (or move inside loop to save per run)
    if record_video and len(frames) > 0:
        print(f"Saving video with {len(frames)} frames to {video_path}...")
        imageio.mimsave(video_path, frames, fps=recording_fps)
        print("Video saved!")

    env.close()
    return ep_times


def log_episode_stats(obs: dict, info: dict, config: ConfigDict, curr_time: float):
    """Log the statistics of a single episode."""
    gates_passed = obs["target_gate"]
    if gates_passed == -1:
        gates_passed = len(config.env.track.gates)
    finished = gates_passed == len(config.env.track.gates)
    logger.info(
        f"Flight time (s): {curr_time}\nFinished: {finished}\nGates passed: {gates_passed}\n"
    )


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger("lsy_drone_racing").setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
    fire.Fire(simulate, serialize=lambda _: None)