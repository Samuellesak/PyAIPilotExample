# PyAIPilotExample

Python autopilot client for the OCTOPUS quadcopter simulator. Connects to the sim over UDP MAVLink, estimates state with a 13-state EKF (optionally fused with vision-based gate detection), and flies a waypoint path using a cascade P-P-PI controller.

---

## Overview

```
Simulator (UDP 14550)
        │
        ▼
  comms/mavlink_rx.py  ──►  ekf/imu_ekf.py  ──►  ekf/ekf.py   (IMU → attitude, velocity, position)
        │                         ▲
        │                         │ vision-corrected pose (optional)
        │                   vision/vision_rx.py  (YOLO pose model → PnP → gate lock/mode)
        ▼
  control/controller.py  ──►  control/carrot_tracker.py   (velocity reference from path)
   outer loop: velocity PI → tilt angle P → rate P
   inner loop: body-rate P → motor mixer → thrust commands
        │
        ▼
  MAVLink motor commands → Simulator
```

`main.py` is the entry point; `setup.py` wires all of the above together (see [Project Layout](#project-layout)).

---

## Project Layout

The codebase is a flat set of scripts grouped into packages by concern. There is no `pyproject.toml` — this is not a pip-installable package, just a folder of scripts meant to be run from the repo root.

| Folder | Concern |
|---|---|
| *(root)* | Entry point (`main.py`), component wiring (`setup.py`), flight data logger (`log.py`), and all config/data/weight files |
| [`flight_model/`](flight_model/) | Rigid-body dynamics, rotation math, offline linearization/LQI design |
| [`ekf/`](ekf/) | 13-state Extended Kalman Filter and its live wrapper |
| [`comms/`](comms/) | MAVLink transport, time sync, message diagnostics |
| [`control/`](control/) | Cascade flight controller and path/carrot tracker |
| [`vision/`](vision/) | Live YOLO+PnP gate detection, pose disambiguation, lock/mode state machines, offline vision tooling |
| [`sysid/`](sysid/) | System-identification and calibration scripts (ground-rig and in-flight) |
| [`tests/`](tests/) | Regression tests for the framework-free modules |
| [`YOLO/`](YOLO/) | YOLO pose-model fine-tuning script and trained weight checkpoints |

**Note:** `setup.py` here is *not* a Python packaging file — despite the name, it's an application-wiring module (`setup_components()`) that constructs the MAVLink connection, EKF, controller, logger, and vision receiver used by `main.py`.

Because these packages use plain absolute imports (`from flight_model.dyn import load_params`) rather than relative imports, any script inside a subfolder must be run as a module from the repo root — see [Quick Start](#quick-start).

---

## File Reference

### Root

| File | Purpose |
|------|---------|
| `main.py` | Entry point — waits for keypress, arms the drone, runs the 250 Hz control loop, saves logs/crash traceback on exit |
| `setup.py` | Wires components together: MAVLink connection, `Logger`, `MAVLinkRX`, `IMUEKFHandler`, `TimeSync`, `VisionRX`, `Controller` |
| `log.py` | Thread-safe flight data logger — writes `logs/<timestamp>/cascade.csv`, `ekf.csv`, `mavlink.txt`, `motors.csv` |
| `params.yaml` | All tunable parameters (gains, physical constants, waypoints) |

### `flight_model/`

| File | Purpose |
|------|---------|
| `dyn.py` | Rigid-body dynamics model and `load_params()` — reads `params.yaml` and derives the inertia tensor for the X-config quad |
| `rotations.py` | Single source of truth for quaternion/Euler/rotation-matrix conversions, used throughout the codebase |
| `sim_convention.py` | Isolates sim-vs-FRD sign-convention quirks (gyro axis flips, etc.) in one place |
| `linearize.py` | Numerically linearizes the dynamics at trim conditions across airspeeds; writes `linearization.npz` |
| `lqi.py` | Designs LQR+Integral controller gains off the linearized model; writes `lqi_gains.npz` (alternative controller, unused unless `controller_type: 2` in `params.yaml`) |

### `ekf/`

| File | Purpose |
|------|---------|
| `ekf.py` | `QuadEKF` — the 13-state EKF: predict from raw IMU, gravity-alignment update, zero-velocity update (ZUPT) |
| `imu_ekf.py` | `IMUEKFHandler` — live wrapper around `QuadEKF` handling calibration, hover-entry reset, model-aided predict blending, and vision-correction fusion |

### `comms/`

| File | Purpose |
|------|---------|
| `mavlink_rx.py` | Pure MAVLink transport layer — receives messages, dispatches to registered handlers, tracks message rates |
| `timesync.py` | MAVLink TIMESYNC request loop (measures round-trip latency) |
| `msg_probe.py` | Standalone diagnostic — connects to the sim and prints every unique MAVLink message type/fields seen |

### `control/`

| File | Purpose |
|------|---------|
| `controller.py` | Cascade P-P-PI flight controller, launch sequence (WAIT/BLIP/TRACK), motor mixing, MAVLink command sending |
| `carrot_tracker.py` | Look-ahead "carrot" path tracker with smooth waypoint blending |

### `vision/`

| File | Purpose |
|------|---------|
| `vision_rx.py` | Live vision receiver — UDP camera frame ingestion, YOLO pose-model inference, orange-mask preprocessing, PnP pose solve, gate identification |
| `pose_estimate.py` | Dependency-free dataclass defining the vision→control data contract (`PoseEstimate`, `LockState`) |
| `pose_disambiguation.py` | Resolves PnP pose ambiguities (front/back flip, corner relabeling) without depending on the EKF |
| `gate_lock.py` | Gate lock/confidence state machine (UNLOCKED → ACQUIRING → LOCKED) |
| `vision_mode.py` | Mode/confidence state machine (`Mode`, `VisionModeTracker`, `VerticalAssist`, `PursuitGuidance`, `RecoveryGuard`) consumed by the controller |
| `vision_debug.py` | One-shot standalone visual debugger — captures frames from the sim camera, runs YOLO, dumps annotated debug images |
| `vision_param_opt.py` | Offline grid-search optimizer for gate-detection parameters (sharpen/gauss/confidence/orange-fraction sweep) |
| `gate_vision_calib.py` | Full-flight calibration pipeline fitting a linear velocity correction (YOLO+PnP vs. EKF dead-reckoning) |
| `check_yolo.py` | Offline replay of saved flight JPEG frames through YOLO+PnP, compared against EKF ground truth |

### `sysid/`

| File | Purpose |
|------|---------|
| `sysid.py` | Ground-rig sysid — excites motors via MAVLink, fits `{Dv, Dw, kappa, m}` against `dyn.py` |
| `flight_sysid.py` | In-flight sysid — launch/blip/pitch/roll/yaw test sequence, diagnoses EKF divergence from launch-blip attitude error |
| `flight_sysid_gt.py` | Ground-truth-mode sysid for rate-loop sign/gain identification per axis |
| `flight_sysid_rot.py` | In-flight rotational sysid — fits `[Dw_xy, Dw_z, kappa, m_motor]` from per-motor thrust telemetry |
| `flight_sysid_vel.py` | In-flight sysid for tilt→velocity transfer function per NED axis; computes PI gains |
| `fit_shadow.py` | Offline fit of translational physics params (`Dv`, `T_max_motor`, `tau_motor`, accel bias) from a "shadow" CSV log |
| `ekf_shadow.py` | Offline EKF replay/comparison tool (integration-only vs. +model vs. +vision vs. full) against sim ground truth; fits EKF blend-weight sigmas |
| `thrust_test.py` | Open-loop thrust sweep to identify hover throttle / max thrust via liftoff detection |
| `ned_calibration.py` | Post-flight diagnostic verifying NED frame/heading correctness from logged CSVs |
| `plot_calib.py` | Offline plotter for `gate_vision_calib.py` output logs |

### `YOLO/`

| File | Purpose |
|------|---------|
| `YOLO.py` | Fine-tunes the YOLO pose model (`ultralytics` `model.train()`) against a local `Training/dataset.yaml` (not tracked in git) |
| `best.pt`, `best2.pt`, `best3.pt` | Trained checkpoints from fine-tuning runs |

---

## Quick Start

**Requirements:** Python 3.10+, simulator running and listening on UDP 14550.

```bash
# Install dependencies
pip install -r requirements.txt




# Run (from the repo root)
python main.py
# Press 's' to arm and start
# Press Ctrl+C to stop and save logs
```

The simulator must be reachable at `127.0.0.1:14550` (edit `SIM_SERVER_UDP_IP` / `SIM_SERVER_UDP_PORT` in `main.py` to change).

### Running scripts inside a subfolder

Everything inside `flight_model/`, `ekf/`, `comms/`, `control/`, `vision/`, `sysid/`, and `tests/` imports its siblings with absolute package imports (e.g. `from flight_model.dyn import load_params`), so those scripts must be invoked as modules **from the repo root**, not run directly by path:

```bash
python -m sysid.sysid --plot-only
python -m vision.vision_debug
python -m tests.test_rotations
```

`main.py` is unaffected — it stays a plain `python main.py`.

**Running from an IDE:** the editor's "Run ▶" button executes a script directly by path, which breaks these package imports the same way a bare `python sysid/ekf_shadow.py` does. Use the debug configurations in `.vscode/launch.json` instead (Run and Debug panel) — one per runnable script, each running it as `python -m <package>.<module>` with the working directory set to the repo root.

---

## Controller Architecture

The cascade controller (`control/controller.py`) has three nested loops with separated bandwidths:

```
Velocity loop  (Kp_vel ≈ 1 rad/s)
  └─ Attitude loop  (K_att = 6 rad/s)
       └─ Rate loop  (rate_bandwidth = 20 rad/s)
```

**Outer loop** — NED velocity PI → desired tilt angle:
- Velocity error → desired NED acceleration → `phi_des`, `theta_des` (small-angle inversion)
- Desired yaw rate from `K_psi` / `Ki_psi`
- Collective thrust: `T_coll = 4·T_hover · (1 - a_z/g) / R22`, low-pass filtered (τ = 0.3 s)

**Middle loop** — tilt error P → desired body rates:
- `p_des = K_att · (phi_des − phi_meas)`

**Inner loop** — rate error P → motor torques → motor thrusts:
- `tau_x = K_rate_roll · (p_des − p)` (gains derived from `rate_bandwidth · Ixx`)
- Thrust allocation via inverted mixer matrix (X-config, arm length L)

Roll/pitch measurements use the simulator's ATTITUDE ground truth to avoid EKF attitude bias from motor-vibration DC offsets on the IMU. Velocity and position use EKF dead-reckoning (`ekf/imu_ekf.py`, `ekf/ekf.py`), optionally corrected by vision (`vision/vision_rx.py`).

---

## EKF

State vector `x = [pN, pE, pD, vN, vE, vD, qw, qx, qy, qz, bgx, bgy, bgz]` (13 states), implemented in `ekf/ekf.py`. The live wrapper `ekf/imu_ekf.py` (`IMUEKFHandler`) handles calibration, hover-entry reset, model-aided predict blending, and vision-correction fusion, and is registered as a MAVLink handler by `comms/mavlink_rx.py`.

- **Predict**: integrates raw IMU at ~250 Hz; motor-vibration spikes above 42 m/s² (physical max = 41.9 m/s²) are replaced by the gravity vector in body frame, guaranteeing zero velocity change on bad frames.
- **Update (gravity alignment)**: rejects frames where `|acc_norm − g| > 2 m/s²`.
- **ZUPT**: zero-velocity update, active on the ground, disabled at hover entry.
- **Hover entry**: position, velocity, and roll are reset; ZUPT is disabled.
- **Vision fusion**: when `vision/vision_rx.py` produces a locked gate pose estimate, `IMUEKFHandler` blends it into the state per the `vision_authority` parameter in `params.yaml`.

---

## Vision

`vision/vision_rx.py` runs in its own thread, ingesting camera frames over UDP and running a YOLO pose model to detect gates, then solving PnP for a camera-relative pose. Three small, dependency-light modules downstream form the vision→control contract:

- `vision/pose_estimate.py` — the `PoseEstimate`/`LockState` dataclasses passed between vision and control, deliberately free of camera/EKF dependencies so it's unit-testable in isolation.
- `vision/pose_disambiguation.py` — resolves PnP's inherent front/back pose ambiguity without feeding back into the EKF.
- `vision/gate_lock.py` and `vision/vision_mode.py` — confidence/state machines (UNLOCKED → ACQUIRING → LOCKED, and the higher-level `Mode`/guidance states) that gate when a vision pose is trusted enough to influence the controller.

Offline tooling for tuning and validating this pipeline lives alongside it: `vision_debug.py` (visual debugger), `vision_param_opt.py` (detection-parameter grid search), `gate_vision_calib.py` (velocity-correction fit from a full flight), and `check_yolo.py` (replay saved frames against EKF ground truth).

---

## Key Parameters (`params.yaml`)

```yaml
# Physical
m: 5.49          # total mass [kg]
L: 0.14          # arm length [m]
T_max_motor: 57.53  # max per-motor thrust [N]

# Cascade gains
rate_bandwidth: 20.0   # inner rate loop BW [rad/s]
K_att: 6.0             # attitude P [rad/s]
Kp_vel: 1              # velocity P [1/s]
Ki_vel: 0.00128        # velocity I [1/s²]
Kp_vz: 1              # vertical velocity P [1/s]
Ki_vz: 0.00128        # vertical velocity I [1/s²]
MAX_TILT_DEG: 35.0     # outer loop tilt limit [deg]

# Path
hover_only: true       # set false to follow waypoints
waypoints:             # NED [m]; first entry = world origin
  - [0, 0, 0]
  - [10, -10, -5]
  - ...
```

---

## System Identification

The `sysid/` package holds two families of scripts:

- **Ground-rig**: `sysid.py` sends predefined motor excitation sequences, records IMU response, and fits drag (`Dv`, `Dw`) and torque-to-thrust ratio (`kappa`) by minimising one-step prediction error of `flight_model/dyn.py`:

  ```bash
  python -m sysid.sysid            # collect new data and fit
  python -m sysid.sysid --fit-only # refit from existing sysid_data.npy
  python -m sysid.sysid --plot-only
  ```

  Results are written to `sysid_data.npy` and comparison plots to `sysid_comparison.png` / `sysid_input.png` (all at the repo root).

- **In-flight**: `flight_sysid.py`, `flight_sysid_gt.py`, `flight_sysid_rot.py`, and `flight_sysid_vel.py` run structured maneuvers against a live (or ground-truth-mode) sim connection to identify mass, per-axis rate gains, rotational drag, and the tilt→velocity transfer function respectively. Run any of them the same way, e.g. `python -m sysid.flight_sysid_rot`.

Supporting offline analysis: `fit_shadow.py` and `ekf_shadow.py` refit physics/EKF parameters from logged CSVs; `thrust_test.py` and `ned_calibration.py` are standalone diagnostics; `plot_calib.py` visualizes `gate_vision_calib.py` output.

---

## Log Files

Each run creates `logs/<YYYYMMDD_HHMMSS>/`:

| File | Contents |
|------|---------|
| `cascade.csv` | Controller state at every loop iteration (references, measurements, commands, integrators) |
| `ekf.csv` | Full EKF state, covariance diagonal, IMU inputs, update flags |
| `mavlink.txt` | Raw MAVLink message log |
| `motors.csv` | Actuator output status |

---

## Testing

The `tests/` folder holds regression tests for the modules with no live-sim dependency (`rotations`, `carrot_tracker`, `gate_lock`, `pose_disambiguation`, `vision_mode`). They're bare scripts, not pytest — each defines a `main()` with a local `check(name, cond)` helper that prints `[OK]`/`[FAIL]` per assertion and exits non-zero on failure. Run each directly as a module from the repo root:

```bash
python -m tests.test_rotations
python -m tests.test_carrot_tracker
python -m tests.test_gate_lock
python -m tests.test_pose_disambiguation
python -m tests.test_vision_mode
```

---

## Conventions

- **Frame**: FRD body frame, NED world frame
- **Quaternion**: `[qw, qx, qy, qz]`
- **Motor ordering** (controller): `[BR, BL, FL, FR]`; sim expects `[FL, FR, BL, BR]` — reordering applied on send
- **Gyro sign**: sim `+xgyro` = roll LEFT, `+ygyro` = nose DOWN; both are negated in `comms/mavlink_rx.py` to standard FRD
