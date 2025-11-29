# Controller Documentation

This directory contains various controllers for the LSY Drone Racing challenge.

## SQP NMPC Controller (`sqp_nmpc.py`)

This controller implements a Nonlinear Model Predictive Control (NMPC) using a Sequential Quadratic Programming (SQP) Real-Time Iteration (RTI) scheme. It uses the **acados** framework for efficient solving.

### Recent Fixes & Improvements

If you are running `sqp_nmpc.py`, please note the following critical fixes that were applied to make it robust:

#### 1. Solver Initialization & Crash Prevention
*   **Issue:** The RTI scheme requires a valid linearization point. Originally, the preparation phase was called in `__init__` before any sensor data was received. This caused the solver to initialize with invalid states, leading to immediate failures. When the solver failed, it returned zero/invalid commands, causing the drone to crash (typically around 1.86s).
*   **Fix:** 
    *   Removed the premature initialization.
    *   Added logic to perform a **full solve** (not just an RTI step) on the very first control call when valid observation data is available.
    *   **Safety Fallback:** If the solver returns a non-success status (anything other than `0` or `2`), the controller now outputs a safe "hover" command instead of passing invalid solver outputs to the drone.

#### 2. Trajectory Safety
*   **Issue:** The original trajectory waypoints commanded the drone to fly extremely low (Z=0.05m), causing it to hit the floor or gate frames. Additionally, the path cut too close to obstacles.
*   **Fix:** 
    *   Raised the initial waypoints to **Z=0.6m** (and up to **1.5m** for obstacle avoidance) to ensure clearance of the floor and gate frames.
    *   Adjusted specific waypoints (e.g., shifting from `X=0.2` to `X=0.0`) to fly "high and wide" around obstacles to prevent collisions.

#### 3. Control Mode Configuration
*   **Issue:** This controller outputs **Attitude Commands** (Thrust, Roll, Pitch, Yaw). The default simulation configuration (`level0.toml`) often expects **State Commands** (Pos, Vel, Acc, etc.), leading to shape mismatch errors.
*   **Fix:** Ensure your configuration file (e.g., `config/level0.toml`) sets:
    ```toml
    control_mode = "attitude"
    ```

#### 4. macOS Library Dependencies (Environment)
*   **Issue:** On some macOS systems, the `libacados.dylib` fails to load `libqpOASES_e.dylib` due to missing rpath settings.
*   **Fix:** If you encounter `OSError: Library not loaded`, use `install_name_tool` to add the library path to the dylib:
    ```bash
    install_name_tool -add_rpath /path/to/lsy_drone_racing/acados/lib /path/to/lsy_drone_racing/acados/lib/libacados.dylib
    ```

### Usage

To run this controller:

1.  Ensure `control_mode = "attitude"` in your config file.
2.  Export the acados source directory:
    ```bash
    export ACADOS_SOURCE_DIR=/path/to/lsy_drone_racing/acados
    ```
3.  Run the simulation:
    ```bash
    python scripts/sim.py --config level0.toml --controller sqp_nmpc.py --gui
    ```

