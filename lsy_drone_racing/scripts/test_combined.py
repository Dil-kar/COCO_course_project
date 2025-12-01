import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from scipy.interpolate import CubicSpline
from typing import List, Dict, Any

# --- CONFIGURATION CONSTANTS (Required for Plotting) ---

# The waypoints provided in your prompt, defining the desired path.
WAYPOINTS = np.array([
    [1.0, 1.5, 0.05], [0.8, 1.0, 0.2], [0.55, -0.3, 0.5],
    [0.2, -1.3, 0.65], [1.1, -0.85, 1.1], [0.2, 0.5, 0.65],
    [0.0, 1.2, 0.525], [0.0, 1.2, 1.1], [-0.5, 0.0, 1.1], [-0.5, -0.5, 1.1],
])

GATE_WIDTH = 0.5
GATE_HEIGHT = 0.5
# Default Gate Data for plotting demonstration
GATE_DATA = [
    {'pos': np.array([0.45, -0.5, 0.56]), 'yaw': 2.35},
    {'pos': np.array([1.0, -1.05, 1.11]), 'yaw': -0.78},
    {'pos': np.array([0.0, 1.0, 0.56]), 'yaw': 0.0},
    {'pos': np.array([-0.5, 0.0, 1.11]), 'yaw': 3.14},
]

# Mock config for cubic spline
class MockConfig:
    def __init__(self, freq):
        self.env = self.Env(freq)
    class Env:
        def __init__(self, freq):
            self.freq = freq
MOCK_CONFIG = MockConfig(freq=100.0) 

# --- PLOTTING CLASSES AND FUNCTIONS ---

class RacingPlotter:
    """Generates visualizations based on collected trajectory data and gates."""
    def __init__(self, waypoints, gate_data: list[dict], config=MOCK_CONFIG):
        self.waypoints = waypoints
        self.gate_data = gate_data
        
        # --- CUBIC SPLINE REFERENCE PATH IMPLEMENTATION ---
        des_time = 8
        ts = np.linspace(0, des_time, waypoints.shape[0])
        ts_interp = np.linspace(0, des_time, int(config.env.freq * des_time))
        
        cs_x = CubicSpline(ts, waypoints[:, 0])
        cs_y = CubicSpline(ts, waypoints[:, 1])
        cs_z = CubicSpline(ts, waypoints[:, 2])
        
        self.ref_x = cs_x(ts_interp)
        self.ref_y = cs_y(ts_interp)
        self.ref_z = cs_z(ts_interp)
            
    def _plot_gates_as_lines(self, ax, plane='xy'):
        """Draws gates as lines, projected onto the XY or XZ plane (Color is black)."""
        GATE_COLOR = 'black'
        
        for gate in self.gate_data:
            pos = gate['pos']
            yaw = gate.get('yaw', 0.0)
            
            R = np.array([[np.cos(yaw), -np.sin(yaw)],
                          [np.sin(yaw),  np.cos(yaw)]])
            
            p_local = np.array([[0, -GATE_WIDTH/2], [0, GATE_WIDTH/2]])
            p_rotated = (R @ p_local.T).T
            
            gate_x = p_rotated[:, 0] + pos[0]
            gate_y = p_rotated[:, 1] + pos[1]
            gate_z = pos[2]
            
            if plane == 'xy':
                # Plot the line segment (gate opening edge)
                ax.plot(gate_x, gate_y, color=GATE_COLOR, linestyle='-', linewidth=4, alpha=0.8, zorder=5)
                # Plot the center marker
                ax.plot(pos[0], pos[1], marker='.', color=GATE_COLOR, markersize=8, zorder=6)
                
            elif plane == 'xz':
                # For XZ projection, plot vertical bar
                ax.plot([pos[0], pos[0]], 
                        [gate_z - GATE_HEIGHT/2, gate_z + GATE_HEIGHT/2], 
                        color=GATE_COLOR, linestyle='-', linewidth=4, alpha=0.8, zorder=5)
                # Plot the center marker
                ax.plot(pos[0], pos[2], marker='.', color=GATE_COLOR, markersize=8, zorder=6)


def plot_trajectory_with_velocity_heatmap_3d(
    run_data_list: list[dict], 
    plotter: RacingPlotter,
    run_index_to_color: int = -1,
    fig_title: str = "Trajectory and Velocity Heatmap (XY & XZ)",
    save_file: str = "trajectory_heatmap_3d.png"
):
    """
    Plots the trajectory with velocity heatmap in both XY (Top-down) and 
    XZ (Side) plane subplots.
    """
    
    if not run_data_list:
        return

    # Create figure with two subplots side-by-side
    fig, (ax_xy, ax_xz) = plt.subplots(1, 2, figsize=(16, 8), sharex=False, sharey=False)
    fig.suptitle(fig_title, fontsize=16)

    # Ensure the run_index_to_color is valid
    idx_to_color = run_index_to_color if run_index_to_color >= 0 else len(run_data_list) + run_index_to_color
    
    # 3. Plot all trajectories
    for i, run_data in enumerate(run_data_list):
        pos = run_data.get('pos_history')
        vel = run_data.get('vel_history')

        if pos is None or pos.ndim != 2 or pos.shape[1] < 2 or len(pos) < 2:
            continue

        # Plot this run's trajectory
        if i == idx_to_color:
            # --- Velocity Heatmap Plot Setup ---
            
            # Normalize velocity for the colormap (common to both plots)
            vel_min = np.min(vel)
            vel_max = np.max(vel)
            norm = Normalize(vmin=vel_min, vmax=vel_max)
            cmap = 'jet' 
            
            # --- XY Plot (Top-down) ---
            ax = ax_xy
            
            # 1. Plot Reference Path
            ax.plot(plotter.ref_x, plotter.ref_y, 'k--', linewidth=2.0, alpha=0.2, label='Reference Path')
            # 2. Plot Gates
            plotter._plot_gates_as_lines(ax, 'xy')
            
            # 3. Plot Heatmap
            points_xy = pos[:, :2].reshape(-1, 1, 2)
            segments_xy = np.concatenate([points_xy[:-1], points_xy[1:]], axis=1)
            lc_xy = LineCollection(segments_xy, cmap=cmap, norm=norm) 
            lc_xy.set_array(vel)
            lc_xy.set_linewidth(3)
            ax.add_collection(lc_xy)
            
            # 4. Add Start/End Markers (XY)
            start_pos_xy = pos[0, :2]
            end_pos_xy = pos[-1, :2]
            ax.plot(start_pos_xy[0], start_pos_xy[1], 'ko', markersize=10, zorder=10, label='Start/End')
            ax.text(start_pos_xy[0], start_pos_xy[1] + 0.05, 'Start', color='black', fontsize=12, ha='center', zorder=11)
            ax.plot(end_pos_xy[0], end_pos_xy[1], 'ko', markersize=10, zorder=10)
            ax.text(end_pos_xy[0], end_pos_xy[1] + 0.05, 'End', color='black', fontsize=12, ha='center', zorder=11)
            
            ax.set_xlabel('X [m]', fontsize=12)
            ax.set_ylabel('Y [m]', fontsize=12)
            ax.set_title("XY Plane (Top-down View)")
            ax.set_aspect('equal')
            ax.grid(True)
            
            # --- XZ Plot (Side View) ---
            ax = ax_xz
            
            # 1. Plot Reference Path
            ax.plot(plotter.ref_x, plotter.ref_z, 'k--', linewidth=2.0, alpha=0.2, label='Reference Path')
            # 2. Plot Gates
            plotter._plot_gates_as_lines(ax, 'xz')
            
            # 3. Plot Heatmap (X and Z coordinates)
            points_xz = pos[:, [0, 2]].reshape(-1, 1, 2)
            segments_xz = np.concatenate([points_xz[:-1], points_xz[1:]], axis=1)
            lc_xz = LineCollection(segments_xz, cmap=cmap, norm=norm) 
            lc_xz.set_array(vel)
            lc_xz.set_linewidth(3)
            line = ax.add_collection(lc_xz) # Store this for the colorbar
            
            # 4. Add Start/End Markers (XZ)
            start_pos_xz = pos[0, [0, 2]]
            end_pos_xz = pos[-1, [0, 2]]
            ax.plot(start_pos_xz[0], start_pos_xz[1], 'ko', markersize=10, zorder=10, label='Start/End')
            ax.text(start_pos_xz[0], start_pos_xz[1] + 0.05, 'Start', color='black', fontsize=12, ha='center', zorder=11)
            ax.plot(end_pos_xz[0], end_pos_xz[1], 'ko', markersize=10, zorder=10)
            ax.text(end_pos_xz[0], end_pos_xz[1] + 0.05, 'End', color='black', fontsize=12, ha='center', zorder=11)
            
            ax.set_xlabel('X [m]', fontsize=12)
            ax.set_ylabel('Z [m]', fontsize=12)
            ax.set_title("XZ Plane (Side View)")
            ax.set_aspect('equal')
            ax.grid(True)
            
            # Add a single colorbar below the two subplots
            cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.03]) # [left, bottom, width, height]
            cbar = fig.colorbar(line, cax=cbar_ax, orientation='horizontal')
            cbar.set_label('Velocity Magnitude [m/s]', fontsize=14)
            
        else:
            # --- Gray History Plot ---
            # Plot historical paths in both subplots
            ax_xy.plot(pos[:, 0], pos[:, 1], color='gray', alpha=0.3, linewidth=1, label='History Path')
            ax_xz.plot(pos[:, 0], pos[:, 2], color='gray', alpha=0.3, linewidth=1, label='History Path')

    # Final Legend for the entire figure
    # Use the XY axes to collect unique handles/labels
    handles, labels = ax_xy.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    
    # Remove 'Start/End' dummy marker from the main legend (it clutters the plot)
    if 'Start/End' in unique_labels:
        del unique_labels['Start/End']
    
    # Add a proxy for the velocity heatmap to the legend
    unique_labels['Velocity Heatmap'] = ax_xy.plot([], [], color='darkorange', linewidth=3, label='Velocity Heatmap')[0]

    fig.legend(unique_labels.values(), unique_labels.keys(), loc='lower center', bbox_to_anchor=(0.5, 0.1), ncol=3)

    plt.tight_layout(rect=[0, 0.1, 1, 0.95]) # Adjust rect for the bottom colorbar and legend
    plt.savefig(save_file)
    plt.close()
    
    print(f"Plot saved to {save_file}")
    
    
def generate_performance_table(all_runs_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Extracts key performance metrics from a list of lap results and creates a table.
    """
    
    table_data = []
    for i, run_data in enumerate(all_runs_data):
        lap_data = {
            "Lap": i + 1,
            "Lap Time [s]": f"{run_data.get('lap_time', np.nan):.2f}",
            "Success": run_data.get('success', False),
            "Gates Passed": int(run_data.get('gates_passed', 0)), # Ensure int for table
            "RMSE Tracking Error": f"{run_data.get('rmse_tracking_error', np.nan):.4f}",
            "Avg Solver Time [ms]": f"{run_data.get('avg_solver_time_ms', np.nan):.2f}",
        }
        table_data.append(lap_data)

    df = pd.DataFrame(table_data)
    df = df.set_index("Lap")
    
    return df

# --- EXECUTION BLOCK: Load Data and Generate Outputs ---


all_runs_data = np.load("3lap.npy", allow_pickle=True).tolist()
print(f"Successfully loaded {len(all_runs_data)} runs from 'my_array.npy'.")

    
# 2. Generate and display the performance table
if all_runs_data:
    performance_df = generate_performance_table(all_runs_data)
    
    print("\n##  Performance Metrics Per Lap ")
    print("-" * 70)
    print(performance_df.to_markdown(numalign="left", stralign="left"))
    print("-" * 70)
else:
    print("No data available to generate the table or plots.")
    
# 3. Generate the 3D velocity heatmap plot (XY and XZ planes)
if all_runs_data:
    plotter_example = RacingPlotter(WAYPOINTS, GATE_DATA, config=MOCK_CONFIG)
    plot_trajectory_with_velocity_heatmap_3d(
        all_runs_data, 
        plotter_example, 
        run_index_to_color=-1, # Plot the last run as the heatmap
        fig_title="Final Iteration Velocity Heatmap (XY and XZ Planes)",
        save_file="final_trajectory_heatmap_3d.png"
    )