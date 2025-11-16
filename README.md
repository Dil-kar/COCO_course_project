# LSY Drone Racing Control System

This repository provides a complete framework for simulating and testing drone control strategies designed for the **LSY Drone Racing** environment. It includes configurable racing setups, modular controller implementations, and the main simulation pipeline.

---

## 1. Prerequisites: Acados

This project depends on **Acados**, a high-performance library for real-time optimal control.

Before setting up the Python environment:

- Install and configure Acados using the official documentation.
- Ensure the Acados Python interface is available in your environment.

---

## 2. Installation and Setup

Follow the steps below to install the project and its dependencies.

```bash
git clone https://github.com/Dil-kar/COCO_course_project.git
cd lsy_drone_racing
pip install -e .
```

## 3. Configuration

The config/ directory stores all simulation and racing parameters in .toml files (e.g., level0.toml). These files define drone models, environment settings, and default controller parameters.

You can switch between different racing setups simply by changing the specified .toml file at runtime.

## 4. Controller Implementation

The core control logic is located in the lsy_drone_racing\control folder. This directory contains the implementation of all available controllers (e.g., trajectory_controller.py).

The controller to be used for a simulation can be specified either within the active .toml configuration file or via a command-line argument.

## 5. Running the Simulation

Execute the simulation script from the root of the project directory (lsy_drone_racing/).

Option A: Specifying the Configuration and Controller

Use this command to explicitly define both the configuration file and the controller script:
```bash
python scripts/sim.py --config level0.toml --controller trajectory_controller.py
```

Option B: Using the Default Controller (from TOML)

If the controller path is already defined inside the specified .toml configuration file, you can omit the --controller argument:
```bash
python scripts/sim.py --config level0.toml
```
