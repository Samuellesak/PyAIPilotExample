# PyAIPilotExample

Python autopilot client for the OCTOPUS quadcopter simulator. Connects to the sim over UDP MAVLink, estimates state with a 13-state EKF, and flies a waypoint path using a cascade P-P-PI controller.

---

## Overview

```
Simulator (UDP 14550)
        │
        ▼
  mavlink_rx.py  ──►  ekf.py          (IMU → attitude, velocity, position)
        │
        ▼
  controller.py  ──►  carrot_tracker.py   (velocity reference from path)
   outer loop: velocity PI → tilt angle P → rate P
   inner loop: body-rate P → motor mixer → thrust commands
        │
        ▼
  MAVLink motor commands → Simulator
```

---

## File Reference

| File | Purpose |
|------|---------|
| `main.py` | Entry point — arms the drone and runs the control loop |
| `setup.py` | Wires components (MAVLink connection, EKF, controller, logger, vision) |
| `controller.py` | Cascade P-P-PI flight controller + carrot path follower |
| `mavlink_rx.py` | MAVLink receiver, EKF integration, IMU spike filtering |
| `ekf.py` | 13-state Extended Kalman Filter (position, velocity, quaternion, gyro bias) |
| `dyn.py` | Physics model and `load_params()` — reads `params.yaml` and derives inertia |
| `carrot_tracker.py` | Look-ahead carrot tracker with smooth waypoint blending |
| `log.py` | Flight data logger — writes `logs/<timestamp>/cascade.csv`, `ekf.csv`, `mavlink.txt` |
| `lqi.py` | LQI controller (alternative; unused when `controller_type: 1` in `params.yaml`) |
| `sysid.py` | System identification — excites motors and fits drag/inertia from IMU response |
| `timesync.py` | MAVLink TIMESYNC loop (measures round-trip latency) |
| `vision_rx.py` | Vision / YOLO object detection receiver |
| `params.yaml` | All tunable parameters (gains, physical constants, waypoints) |

---

## Quick Start

**Requirements:** Python 3.10+, simulator running and listening on UDP 14550.

```bash
# Install dependencies
pip install -r requirements.txt

# Run
python main.py
# Press 's' to arm and start
# Press Ctrl+C to stop and save logs
```

The simulator must be reachable at `127.0.0.1:14550` (edit `SIM_SERVER_UDP_IP` / `SIM_SERVER_UDP_PORT` in `main.py` to change).

---

## Controller Architecture

The cascade controller has three nested loops with separated bandwidths:

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

Roll/pitch measurements use the simulator's ATTITUDE ground truth to avoid EKF attitude bias from motor-vibration DC offsets on the IMU. Velocity and position use EKF dead-reckoning.

---

## EKF

State vector `x = [pN, pE, pD, vN, vE, vD, qw, qx, qy, qz, bgx, bgy, bgz]` (13 states).

- **Predict**: integrates raw IMU at ~250 Hz; motor-vibration spikes above 42 m/s² (physical max = 41.9 m/s²) are replaced by the gravity vector in body frame, guaranteeing zero velocity change on bad frames.
- **Update (gravity alignment)**: rejects frames where `|acc_norm − g| > 2 m/s²`.
- **ZUPT**: zero-velocity update, active on the ground, disabled at hover entry.
- **Hover entry**: position, velocity, and roll are reset; ZUPT is disabled.

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

`sysid.py` sends predefined motor excitation sequences, records IMU response, and fits drag (`Dv`, `Dw`) and torque-to-thrust ratio (`kappa`) by minimising one-step prediction error of `dyn.py`:

```bash
python sysid.py            # collect new data and fit
python sysid.py --fit-only # refit from existing sysid_data.npy
python sysid.py --plot-only
```

Results are written to `sysid_data.npy` and comparison plots to `sysid_comparison.png`.

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

## Conventions

- **Frame**: FRD body frame, NED world frame
- **Quaternion**: `[qw, qx, qy, qz]`
- **Motor ordering** (controller): `[BR, BL, FL, FR]`; sim expects `[FL, FR, BL, BR]` — reordering applied on send
- **Gyro sign**: sim `+xgyro` = roll LEFT, `+ygyro` = nose DOWN; both are negated in `mavlink_rx.py` to standard FRD
