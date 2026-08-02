"""
thrust_test.py
==============
Open-loop thrust sweep to identify hover throttle and max thrust in the sim.

Liftoff detection uses the HIGHRES_IMU accelerometer (no ODOMETRY needed):

  On the ground: acc_body_z ≈ −g (constant — ground absorbs the motor thrust)
  After liftoff: net thrust > weight → drone accelerates upward →
                  zacc drops below baseline by more than LIFTOFF_Z_DELTA

The baseline is calibrated during the IDLE phase so slope/tilt are compensated
automatically.  LIFTOFF_CONSEC_MIN consecutive samples below the threshold are
required to avoid triggering on motor-vibration spikes.

Protocol
--------
1. Idle (u=IDLE_NORM) for IDLE_SEC — calibrates zacc baseline.
2. Ramp all 4 motors equally from START_NORM to END_NORM over RAMP_SEC.
3. Hold at END_NORM for HOLD_SEC.
4. Kill motors.

Outputs
-------
- Console: 1 Hz lines with throttle, zacc, acc_norm, gyro_mag
- logs/<session>/mavlink.txt, imu.png, position.png, control.png (+ CSVs)
- logs/thrust_test_<timestamp>/thrust_test.csv
- logs/thrust_test_<timestamp>/thrust_test.png

Usage (from the repo root)
-----
  python -m sysid.thrust_test
"""

import time
import msvcrt
import os
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pymavlink import mavutil

from flight_model.dyn import load_params
from log import Logger

# ── Test parameters ──────────────────────────────────────────────────────────
SIM_IP       = "127.0.0.1"
SIM_PORT     = 14550
CONTROL_HZ   = 250
DT           = 1.0 / CONTROL_HZ

IDLE_SEC     = 4.0    # calibration window before ramp
IDLE_NORM    = 0.02   # very low throttle during IDLE — motors barely spin
RAMP_SEC     = 25.0   # time to sweep from START_NORM to END_NORM
HOLD_SEC     = 3.0    # hold at peak after ramp
START_NORM   = 0.18   # normalized throttle at ramp start
END_NORM     = 0.80   # normalized throttle at ramp end

# Liftoff detection thresholds
# Uses acc_norm (|a|) rather than az alone: on the ground acc_norm ≈ g regardless
# of platform tilt or motor throttle (ground reaction absorbs thrust).  Only after
# true liftoff does net upward acceleration push acc_norm above g.  This makes
# detection incline-invariant — az-based detection triggers falsely when the drone
# tilts forward on the slope before actually leaving the surface.
LIFTOFF_ACC_DELTA      = 1.5   # acc_norm must exceed g + this value [m/s²]
LIFTOFF_CONSEC_MIN     = 12    # consecutive samples needed (IMU ~60 Hz, loop 250 Hz)
LIFTOFF_MIN_THROTTLE_F = 0.85  # only detect after throttle ≥ this × hover throttle

# Safety cutoffs
POST_LIFTOFF_STOP_SEC  = 2.0          # end test this many seconds after liftoff
CRASH_ACC_NORM_THRESH  = float('inf') # disabled — set to 40.0 to re-enable
CRASH_CONSEC_MIN       = 12

# Hover phase (entered after liftoff is confirmed)
HOVER_SEC    = 10.0   # hover this long then kill motors
KP_VZ        = 0.3    # vD [m/s] → thrust correction [normalised]

MAVLINK_CMD_SIM_RESET = 31000
# ─────────────────────────────────────────────────────────────────────────────


def send_motors(conn, u_norm):
    """Send equal normalised [0,1] command to all 4 motors."""
    v = float(u_norm)
    cmds = [v, v, v, v, 0.0, 0.0, 0.0, 0.0]
    conn.mav.set_actuator_control_target_send(
        int(time.time() * 1e6),
        conn.target_system,
        conn.target_component,
        0,
        cmds
    )


def send_attitude_target(conn, p, q, r, thrust_norm):
    """Zero-attitude target: body rates + collective thrust via the sim rate controller."""
    conn.mav.set_attitude_target_send(
        int(time.time() * 1e3) & 0xFFFFFFFF,
        conn.target_system, conn.target_component,
        0x80,                          # type_mask: ignore attitude quaternion
        [1.0, 0.0, 0.0, 0.0],         # quaternion placeholder (ignored)
        float(p), float(q), float(r),
        float(np.clip(thrust_norm, 0.0, 1.0)),
    )


def send_reset(conn):
    conn.mav.command_long_send(
        conn.target_system, conn.target_component,
        MAVLINK_CMD_SIM_RESET,
        0, 0, 0, 0, 0, 0, 0, 0
    )


def arm(conn):
    conn.mav.command_long_send(
        conn.target_system, conn.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 1, 0, 0, 0, 0, 0, 0
    )


def read_imu(shared_data):
    """Return (az, acc_norm, gyro_mag) from latest HIGHRES_IMU, or Nones."""
    imu = shared_data.get('imu_raw')
    if imu is None:
        return None, None, None
    az       = imu['az']
    acc_norm = float(np.sqrt(imu['ax']**2 + imu['ay']**2 + imu['az']**2))
    gyro_mag = float(np.sqrt(imu['gx']**2 + imu['gy']**2 + imu['gz']**2))
    return az, acc_norm, gyro_mag


def main():
    param = load_params()
    T_max = float(param['T_max_motor'])
    m     = float(param['m'])
    g     = float(param['g'])
    T_hover_theory     = m * g / 4.0
    hover_min_throttle = LIFTOFF_MIN_THROTTLE_F * T_hover_theory / T_max

    logger = Logger()

    # ── Connect ─────────────────────────────────────────────────────────────
    print("Connecting…", flush=True)
    conn = mavutil.mavlink_connection(f"udpin:{SIM_IP}:{SIM_PORT}")
    conn.wait_heartbeat()
    print(f"Connected (sys {conn.target_system})", flush=True)

    # ── MAVLink receive thread ───────────────────────────────────────────────
    shared = {}
    import threading
    from mavlink_rx import MAVLinkRX
    rx = MAVLinkRX.create_mavlink_rx(conn, shared, logger)

    # ── Reset + arm ─────────────────────────────────────────────────────────
    print("Resetting sim…", flush=True)
    send_reset(conn)
    time.sleep(2.0)

    print("Press 's' to arm and start…", flush=True)
    while True:
        if msvcrt.kbhit() and msvcrt.getwch().lower() == 's':
            break
        time.sleep(0.05)

    arm(conn)
    time.sleep(0.5)
    logger.reset_flight_data()
    print(f"\nIDLE for {IDLE_SEC:.0f}s to calibrate zacc baseline…", flush=True)
    print(f"  START={START_NORM:.2f}  END={END_NORM:.2f}  over {RAMP_SEC:.0f}s"
          f"   T_max={T_max} N   T_hover_theory={T_hover_theory:.2f} N/motor\n",
          flush=True)

    # ── Calibration buffers ──────────────────────────────────────────────────
    zacc_idle_samples = []
    zacc_baseline     = None

    # ── Data buffers ─────────────────────────────────────────────────────────
    log_t        = []
    log_norm     = []
    log_zacc     = []
    log_acc_norm = []
    log_gyro_mag = []

    blip_dur  = float(param.get('blip_dur_sec',     0.15))
    blip_frac = float(param.get('blip_thrust_frac', 0.80))

    liftoff_norm   = None
    liftoff_t      = None
    liftoff_consec = 0
    crash_consec   = 0
    t_hover_start  = None
    T_hover_norm   = T_hover_theory / T_max

    t_test_start = time.time()
    last_print_t = 0.0
    t_hold_start = 0.0
    t_blip_start = 0.0
    phase        = "IDLE"

    # ── Control loop ─────────────────────────────────────────────────────────
    while True:
        t = time.time() - t_test_start

        # ── Phase transitions ──────────────────────────────────────────────
        if phase == "IDLE":
            u_norm = IDLE_NORM
            if t >= IDLE_SEC:
                if zacc_idle_samples:
                    zacc_baseline = float(np.mean(zacc_idle_samples))
                    print(f"  zacc baseline = {zacc_baseline:.3f} m/s²  "
                          f"(n={len(zacc_idle_samples)} samples)  "
                          f"liftoff trigger: |acc| > {g + LIFTOFF_ACC_DELTA:.2f} m/s²",
                          flush=True)
                else:
                    print("  WARNING: no IMU data received during IDLE — "
                          "liftoff detection disabled", flush=True)
                print(f"  *** BLIP start ({blip_dur:.2f}s @ {blip_frac*100:.0f}%) ***",
                      flush=True)
                t_blip_start = t
                phase = "BLIP"

        elif phase == "BLIP":
            u_norm = blip_frac
            if t - t_blip_start >= blip_dur:
                print("  *** RAMP start ***", flush=True)
                phase = "RAMP"

        elif phase == "RAMP":
            frac   = min(1.0, (t - IDLE_SEC) / RAMP_SEC)
            u_norm = START_NORM + frac * (END_NORM - START_NORM)
            if frac >= 1.0:
                phase        = "HOLD"
                t_hold_start = t

        elif phase == "HOLD":
            u_norm = END_NORM
            if t - t_hold_start >= HOLD_SEC:
                break

        elif phase == "HOVER":
            # Zero body rates + hover thrust with vD feedback from EKF.
            # vD > 0 means climbing in NED (down is positive), so subtract correction.
            mav = shared.get('mav_state')
            vD  = float(mav['vel_ned'][2]) if mav is not None else 0.0
            u_norm = float(np.clip(T_hover_norm - KP_VZ * vD, 0.05, 0.60))
            send_attitude_target(conn, 0.0, 0.0, 0.0, u_norm)
            logger.log_control(int(time.time() * 1000), [u_norm * T_max] * 4)

            az, acc_norm, gyro_mag = read_imu(shared)
            if az is not None:
                log_t.append(t)
                log_norm.append(u_norm)
                log_zacc.append(az)
                log_acc_norm.append(acc_norm)
                log_gyro_mag.append(gyro_mag)
                if t - last_print_t >= 1.0:
                    last_print_t = t
                    print(f"[HOVER t={t:5.1f}s]  "
                          f"T_norm={u_norm:.3f} ({u_norm*100:.1f}%)  "
                          f"vD={vD:+.2f}m/s  "
                          f"zacc={az:.3f}  gyro={gyro_mag:.3f} rad/s",
                          flush=True)

            if t - t_hover_start >= HOVER_SEC:
                print(f"\n[HOVER] {HOVER_SEC:.0f}s complete — killing motors.", flush=True)
                break
            time.sleep(DT)
            continue

        # ── Motor command (all 4 equal) — only for non-HOVER phases ──────
        send_motors(conn, u_norm)
        logger.log_control(int(time.time() * 1000), [u_norm * T_max] * 4)

        # ── IMU read ───────────────────────────────────────────────────────
        az, acc_norm, gyro_mag = read_imu(shared)

        if az is not None:
            if phase == "IDLE" and acc_norm < 20.0:
                zacc_idle_samples.append(az)

            log_t.append(t)
            log_norm.append(u_norm)
            log_zacc.append(az)
            log_acc_norm.append(acc_norm)
            log_gyro_mag.append(gyro_mag)

            # Liftoff detection (RAMP + HOLD only).
            # Uses |acc| rather than az: on the ground |acc| ≈ g regardless of tilt
            # (ground reaction absorbs thrust).  After liftoff, net upward acceleration
            # pushes |acc| above g — attitude-invariant, incline-safe.
            if (liftoff_norm is None
                    and phase not in ("IDLE", "BLIP") and u_norm >= hover_min_throttle):
                if acc_norm > g + LIFTOFF_ACC_DELTA:
                    liftoff_consec += 1
                    if liftoff_consec >= LIFTOFF_CONSEC_MIN:
                        liftoff_norm = u_norm
                        liftoff_t    = t
                        print(f"  *** LIFTOFF detected at t={t:.2f}s  "
                              f"throttle={u_norm:.4f} ({u_norm*100:.1f}%)  "
                              f"|acc|={acc_norm:.3f} m/s² (>{g+LIFTOFF_ACC_DELTA:.2f})  "
                              f"≈ {u_norm*T_max:.2f} N/motor ***",
                              flush=True)
                else:
                    liftoff_consec = 0

            # Transition to hover after POST_LIFTOFF_STOP_SEC
            if (liftoff_norm is not None and phase != "HOVER"
                    and t - liftoff_t >= POST_LIFTOFF_STOP_SEC):
                phase         = "HOVER"
                t_hover_start = t
                print(f"\n[HOVER] Entering hover at t={t:.2f}s  "
                      f"T_hover_norm={T_hover_norm:.3f} ({T_hover_norm*100:.1f}%)",
                      flush=True)

            # Crash detection
            if phase not in ("IDLE", "BLIP") and acc_norm > CRASH_ACC_NORM_THRESH:
                crash_consec += 1
                if crash_consec >= CRASH_CONSEC_MIN:
                    print(f"\n*** CRASH detected at t={t:.2f}s  "
                          f"acc_norm={acc_norm:.1f} m/s² — aborting ***",
                          flush=True)
                    break
            else:
                crash_consec = 0

            # 1 Hz console output
            if t - last_print_t >= 1.0:
                last_print_t = t
                delta_str = (f"  Δz={az-zacc_baseline:+.2f}"
                             if zacc_baseline is not None else "")
                print(f"[{phase:4s}  t={t:5.1f}s]  "
                      f"throttle={u_norm:.4f} ({u_norm*100:.1f}%)  "
                      f"{u_norm*T_max:.2f} N/motor  "
                      f"zacc={az:.3f}{delta_str}  "
                      f"acc_norm={acc_norm:.3f}  "
                      f"gyro={gyro_mag:.3f} rad/s",
                      flush=True)

        time.sleep(DT)

    # Kill motors
    send_motors(conn, 0.0)
    print("\nSweep complete. Killing motors.", flush=True)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n─── Results ────────────────────────────────────────────────────")
    print(f"  zacc baseline                : {zacc_baseline:.3f} m/s²"
          if zacc_baseline is not None else "  zacc baseline: N/A (no IMU)")
    print(f"  T_max_motor (params.yaml)    : {T_max:.1f} N")
    print(f"  Theoretical hover/motor      : {T_hover_theory:.2f} N  "
          f"({T_hover_theory/T_max*100:.1f}% throttle)")
    if liftoff_norm is not None:
        print(f"  Detected liftoff throttle    : {liftoff_norm:.4f} "
              f"({liftoff_norm*100:.1f}%)  → {liftoff_norm*T_max:.2f} N/motor")
        T_max_calibrated = T_hover_theory / liftoff_norm
        print(f"  Calibrated T_max_motor       : {T_max_calibrated:.2f} N  "
              f"← update params.yaml so hover = {liftoff_norm*100:.1f}% throttle")
    else:
        print("  Liftoff NOT detected — lower LIFTOFF_ACC_DELTA or inspect acc_norm log")
    print("────────────────────────────────────────────────────────────────\n")

    # ── Save CSV + Plot ───────────────────────────────────────────────────────
    if log_t:
        out_dir = os.path.join(
            "logs", f"thrust_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(out_dir, exist_ok=True)

        csv_path = os.path.join(out_dir, "thrust_test.csv")
        with open(csv_path, "w") as f:
            f.write("time_s,throttle_norm,throttle_pct,thrust_N_per_motor,"
                    "zacc_ms2,acc_norm_ms2,gyro_mag_rads\n")
            for i in range(len(log_t)):
                f.write(
                    f"{log_t[i]:.4f},{log_norm[i]:.6f},"
                    f"{log_norm[i]*100:.2f},{log_norm[i]*T_max:.4f},"
                    f"{log_zacc[i]:.4f},{log_acc_norm[i]:.4f},{log_gyro_mag[i]:.4f}\n"
                )
        print(f"CSV saved → {csv_path}", flush=True)

        t_arr    = np.array(log_t)
        norm_arr = np.array(log_norm)
        zacc_arr = np.array(log_zacc)
        anorm_arr = np.array(log_acc_norm)

        fig, (ax_u, ax_z) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

        ax_u.plot(t_arr, norm_arr * 100, color="tab:blue", lw=1.0,
                  label="throttle %")
        ax_u.plot(t_arr, norm_arr * T_max, color="tab:orange", lw=1.0,
                  linestyle="--", label="thrust N/motor")
        if liftoff_norm is not None:
            ax_u.axhline(liftoff_norm * 100, color="red", ls=":", lw=1.2,
                         label=f"liftoff {liftoff_norm*100:.1f}%")
            ax_u.axhline(T_hover_theory / T_max * 100, color="green", ls=":",
                         lw=1.2, label=f"theory {T_hover_theory/T_max*100:.1f}%")
        ax_u.set_ylabel("Throttle (%) / Thrust (N/motor)")
        ax_u.set_title(f"Thrust Sweep — T_max_motor={T_max} N, m={m} kg")
        ax_u.legend(loc="upper left")
        ax_u.grid(True, alpha=0.4)

        ax_z.plot(t_arr, zacc_arr,  color="tab:green",  lw=1.0, label="zacc (m/s²)")
        ax_z.plot(t_arr, anorm_arr, color="tab:purple", lw=1.0, ls="--",
                  label="|acc| (m/s²)")
        if zacc_baseline is not None:
            ax_z.axhline(zacc_baseline, color="gray", ls="--", lw=0.8,
                         label=f"zacc baseline {zacc_baseline:.2f}")
        ax_z.axhline(g + LIFTOFF_ACC_DELTA, color="red", ls=":", lw=1.2,
                     label=f"|acc| liftoff trigger ({g+LIFTOFF_ACC_DELTA:.2f} m/s²)")
        if liftoff_t is not None:
            ax_z.axvline(liftoff_t, color="red", ls=":", lw=1.2,
                         label=f"liftoff t={liftoff_t:.1f}s")
        ax_z.set_ylabel("Acceleration (m/s²)")
        ax_z.set_xlabel("Time (s)")
        ax_z.set_title("IMU accelerometer z-axis and magnitude")
        ax_z.legend(loc="lower left")
        ax_z.grid(True, alpha=0.4)

        plt.tight_layout()
        plot_path = os.path.join(out_dir, "thrust_test.png")
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"Plot saved → {plot_path}", flush=True)

    logger.save()
    rx.get_thread_for_join().join(timeout=1.0)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
