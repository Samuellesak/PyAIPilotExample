"""
gate_vision_calib.py
====================
Gate-vision velocity calibration pipeline.

Uses YOLO+PnP measurements of gate 0 (known NED from sim track data) as ground
truth to fit a linear velocity correction model against EKF dead-reckoned velocity
during a constant-speed approach.  Also evaluates YOLO detection reliability with
and without preprocessing (sharpening).

Pipeline
--------
  WAIT        3 s   Static; wait for track_gates_ned to include gate 0
  BLIP        0.15s High-thrust burst (default; set use_ramp_start: true to skip)
  BLIP_OBS    2 s   Level hover; attitude settle post-blip
  — or —
  RAMP        4 s   Linear motor ramp (use_ramp_start: true); no yaw torque spike
  — then —
  HOVER       3 s   Altitude hold; save pos_offset_ned; measure hover thrust
  APPROACH    ≤20s  Velocity P-control toward gate 0; collect PnP vs EKF data
  KILL        —     Motor cutoff

Outputs
-------
  logs/calib_<timestamp>/calib.csv           — raw per-frame data
  logs/calib_<timestamp>/report.txt          — velocity scale + bias + RMSE per axis
  logs/calib_<timestamp>/ekf_vs_pnp_path.png — plan-view EKF path vs PnP positions
  logs/calib_<timestamp>/velocity_fit.png    — vel_ekf vs vel_pnp scatter + regression
  logs/calib_<timestamp>/detection_rate.png  — YOLO detection rate per second
"""

import os
import sys
import time
import msvcrt
from datetime import datetime

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pymavlink import mavutil

from flight_model.dyn import load_params
from comms.mavlink_rx import MAVLinkRX
from vision.vision_rx import VisionRX

# ── Connection ────────────────────────────────────────────────────────────────
SIM_IP   = "127.0.0.1"
SIM_PORT = 14550

# ── Timing ────────────────────────────────────────────────────────────────────
CONTROL_HZ     = 250
DT             = 1.0 / CONTROL_HZ
WAIT_SEC       = 3.0
BLIP_OBS_SEC   = 2.0
HOVER_SEC      = 3.0
APPROACH_MAX_S = 20.0
RAMP_SEC_DEF   = 1.0   # default ramp duration (override via params.yaml ramp_dur_sec)

# ── Approach controller constants ─────────────────────────────────────────────
V_APPROACH   = 3     # cruise speed toward gate 0 [m/s]
K_VEL        = 0.3     # vel error [m/s] → desired tilt [rad]
K_ATT        = 1.0     # tilt error [rad] → rate cmd [rad/s]  (match params.yaml K_att)
MAX_TILT_RAD = 0.26    # ~14°
K_PSI        = 0.8     # yaw P gain [1/s]
Q_PSI_MAX    = 0.20    # max yaw rate for heading-hold [rad/s]

# ── Altitude hold ─────────────────────────────────────────────────────────────
KP_ALT_VD = 0.060   # vD [m/s, NED-down +ve] → T_norm correction
KP_ALT_D  = 0.030   # altitude error [m NED] → T_norm correction
HOVER_ALT_M = 2.0   # target altitude above EKF reset point [m]

MAVLINK_CMD_SIM_RESET = 31000


# ── Helpers (verbatim from flight_sysid) ─────────────────────────────────────

def _quat_to_euler(quat):
    qw, qx, qy, qz = quat
    phi   = np.arctan2(2.0*(qw*qx + qy*qz), 1.0 - 2.0*(qx*qx + qy*qy))
    theta = np.arcsin(np.clip(2.0*(qw*qy - qz*qx), -1.0, 1.0))
    psi   = np.arctan2(2.0*(qw*qz + qx*qy), 1.0 - 2.0*(qy*qy + qz*qz))
    return phi, theta, psi


def _send_motors(conn, u_norm):
    v = float(np.clip(u_norm, 0.0, 1.0))
    conn.mav.set_actuator_control_target_send(
        int(time.time() * 1e6),
        conn.target_system, conn.target_component,
        0, [v, v, v, v, 0.0, 0.0, 0.0, 0.0],
    )


def _send_attitude_target(conn, p, q, r, thrust_norm):
    # Sysid-confirmed sign map (flight_sysid_gt.py):
    #   p: FRD-compatible → send as-is
    #   q: reversed in sim → negate
    #   r: reversed in sim → negate
    conn.mav.set_attitude_target_send(
        int(time.time() * 1e3) & 0xFFFFFFFF,
        conn.target_system, conn.target_component,
        0x80,
        [1.0, 0.0, 0.0, 0.0],
        float(p), float(-q), float(-r),
        float(np.clip(thrust_norm, 0.0, 1.0)),
    )


def _send_reset(conn):
    conn.mav.command_long_send(
        conn.target_system, conn.target_component,
        MAVLINK_CMD_SIM_RESET,
        0, 0, 0, 0, 0, 0, 0, 0,
    )


def _arm(conn):
    conn.mav.command_long_send(
        conn.target_system, conn.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 1, 0, 0, 0, 0, 0, 0,
    )


def _level_rates(shared):
    """Return (p_des, q_des) driving phi/theta toward zero."""
    if shared.get('post_blip_att_reset_done'):
        mav = shared.get('mav_state')
        if mav is None:
            return 0.0, 0.0
        qw, qx, qy, qz = mav['quat']
        theta = float(np.arcsin(np.clip(2.0*(qw*qy - qz*qx), -1.0, 1.0)))
        phi   = float(np.arctan2(2.0*(qw*qx + qy*qz), 1.0 - 2.0*(qx*qx + qy*qy)))
        return float(-K_ATT * phi), float(-K_ATT * theta)
    imu = shared.get('imu_raw')
    if imu is None:
        return 0.0, 0.0
    ax = float(imu.get('ax', 0.0))
    ay = float(imu.get('ay', 0.0))
    az = float(imu.get('az', 0.0))
    acc_norm = float(np.sqrt(ax**2 + ay**2 + az**2))
    if abs(acc_norm - 9.81) > 2.5:
        return 0.0, 0.0
    theta = float(np.arcsin(np.clip(ax / 9.81, -1.0, 1.0)))
    phi   = float(np.arcsin(np.clip(ay / (9.81 * max(np.cos(theta), 0.1)), -1.0, 1.0)))
    return float(-K_ATT * phi), float(-K_ATT * theta)


def _read_state(shared):
    """Return 15-tuple or None."""
    mav = shared.get('mav_state')
    imu = shared.get('imu_raw')
    if mav is None or imu is None:
        return None
    phi, theta, psi = _quat_to_euler(mav['quat'])
    vN, vE, vD = mav['vel_ned']
    pN, pE, pD = mav['pos_ned']
    return (
        np.rad2deg(phi), np.rad2deg(theta), np.rad2deg(psi),
        float(vN), float(vE), float(vD),
        float(imu['ax']), float(imu['ay']), float(imu['az']),
        float(imu['gx']), float(imu['gy']), float(imu['gz']),
        float(pN), float(pE), float(pD),
    )


# ── Trajectory fitting ────────────────────────────────────────────────────────

def _fit_trajectory(log_rows, pnp_rows, out_dir):
    """Fit pos_pnp = pos0 + a*(pos_ekf-pos0) + b*t per NED axis.

    PnP positions are already in local NED (pos_offset_ned was subtracted at save
    time). EKF positions are also in local NED (after hover reset). No conversion
    needed here.
    Returns dict {'N': {'a', 'b', 'rmse_m'}, 'E': ..., 'D': ...} or {} on failure.
    """
    appr = [r for r in log_rows if r['phase'] == 'APPROACH']
    if len(appr) < 5 or len(pnp_rows) < 3:
        return {}

    t_ekf  = np.array([r['t']  for r in appr])
    pN_ekf = np.array([r['pN'] for r in appr])
    pE_ekf = np.array([r['pE'] for r in appr])
    pD_ekf = np.array([r['pD'] for r in appr])

    # PnP already in local-NED (same frame as EKF)
    t_pnp  = np.array([r['t']      for r in pnp_rows])
    pN_pnp = np.array([r['pN_pnp'] for r in pnp_rows])
    pE_pnp = np.array([r['pE_pnp'] for r in pnp_rows])
    pD_pnp = np.array([r['pD_pnp'] for r in pnp_rows])

    mask = (t_pnp >= t_ekf[0]) & (t_pnp <= t_ekf[-1])
    t_pnp, pN_pnp, pE_pnp, pD_pnp = (
        t_pnp[mask], pN_pnp[mask], pE_pnp[mask], pD_pnp[mask])
    if len(t_pnp) < 3:
        return {}

    t0   = t_ekf[0]
    pos0 = np.array([pN_ekf[0], pE_ekf[0], pD_ekf[0]])
    t_rel = t_pnp - t0

    pN_at = np.interp(t_pnp, t_ekf, pN_ekf)
    pE_at = np.interp(t_pnp, t_ekf, pE_ekf)
    pD_at = np.interp(t_pnp, t_ekf, pD_ekf)

    results = {}
    for axis, ekf_at, pnp_arr, p0 in [
        ('N', pN_at, pN_pnp, pos0[0]),
        ('E', pE_at, pE_pnp, pos0[1]),
        ('D', pD_at, pD_pnp, pos0[2]),
    ]:
        d_ekf = ekf_at - p0
        d_pnp = pnp_arr - p0
        ok = np.isfinite(d_ekf) & np.isfinite(d_pnp)
        if ok.sum() < 3:
            results[axis] = {'a': float('nan'), 'b': float('nan'), 'rmse_m': float('nan')}
            continue
        X = np.column_stack([d_ekf[ok], t_rel[ok]])
        try:
            coeff, _, _, _ = np.linalg.lstsq(X, d_pnp[ok], rcond=None)
            a, b = float(coeff[0]), float(coeff[1])
            rmse = float(np.sqrt(np.mean((d_pnp[ok] - (a * d_ekf[ok] + b * t_rel[ok]))**2)))
        except np.linalg.LinAlgError:
            a, b, rmse = float('nan'), float('nan'), float('nan')
        results[axis] = {'a': a, 'b': b, 'rmse_m': rmse}

    if any(np.isnan(results[ax]['a']) for ax in ('N', 'E', 'D')):
        print("  [fit] one or more axes failed — skipping trajectory fit plot", flush=True)
        return results

    # Build corrected trajectory for plotting
    t_all  = t_ekf - t0
    corr_N = pos0[0] + results['N']['a'] * (pN_ekf - pos0[0]) + results['N']['b'] * t_all
    corr_E = pos0[1] + results['E']['a'] * (pE_ekf - pos0[1]) + results['E']['b'] * t_all
    corr_D = pos0[2] + results['D']['a'] * (pD_ekf - pos0[2]) + results['D']['b'] * t_all

    _plot_trajectory_fit(
        t_all, pN_ekf, pE_ekf, pD_ekf,
        corr_N, corr_E, corr_D,
        t_rel, pN_pnp, pE_pnp, pD_pnp,
        results, out_dir,
    )
    return results


def _plot_trajectory_fit(t_all, pN_ekf, pE_ekf, pD_ekf,
                          corr_N, corr_E, corr_D,
                          t_pnp, pN_pnp, pE_pnp, pD_pnp,
                          results, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(
        "Trajectory fit: pos_pnp = pos0 + a·(pos_ekf−pos0) + b·t", fontsize=11)

    for ax, axis, ekf_arr, corr_arr, pnp_arr in [
        (axes[0], 'N', pN_ekf, corr_N, pN_pnp),
        (axes[1], 'E', pE_ekf, corr_E, pE_pnp),
        (axes[2], 'D', pD_ekf, corr_D, pD_pnp),
    ]:
        r = results[axis]
        p0 = ekf_arr[0]
        ax.plot(t_all, ekf_arr - p0, lw=1.2, color='tab:blue', label='EKF (raw)')
        ax.plot(t_all, corr_arr - p0, lw=1.2, ls='--', color='tab:green',
                label=f'Corrected (a={r["a"]:.3f}, b={r["b"]:+.3f})')
        ax.scatter(t_pnp, pnp_arr - p0, s=14, color='tab:orange', zorder=5,
                   label=f'PnP  RMSE={r["rmse_m"]:.2f}m')
        ax.set_xlabel('Time in APPROACH (s)')
        ax.set_ylabel(f'Δ{axis} (m)')
        ax.set_title(f'{axis}-axis')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(out_dir, "trajectory_fit.png")
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Trajectory fit plot → {out}", flush=True)


def _plot_gate_distance(pnp_rows, out_dir):
    if not pnp_rows:
        return
    t0 = pnp_rows[0]['t']
    t_rel = [r['t'] - t0 for r in pnp_rows]
    dist  = [r['dist']   for r in pnp_rows]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t_rel, dist, lw=1.5, color='tab:blue', marker='o', ms=3)
    ax.set_xlabel('Time since first PnP detection (s)')
    ax.set_ylabel('PnP gate distance (m)')
    ax.set_title('Gate distance during APPROACH (PnP)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(out_dir, "gate_distance.png")
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Gate distance plot → {out}", flush=True)


# ── Analysis ──────────────────────────────────────────────────────────────────

def _analyse(log_rows, pnp_rows, pos_offset_ned, gate0_ned_world, out_dir):
    """Fit vel_pnp = a*vel_ekf + b per NED axis; compute position drift."""
    lines = []
    lines.append("── GATE VISION VELOCITY CALIBRATION ────────────────────────────────────")

    if len(pnp_rows) < 3:
        lines.append("  Insufficient PnP frames for regression (need ≥3).")
        _write_report(lines, out_dir)
        return lines

    t_pnp  = np.array([r['t']     for r in pnp_rows])
    pN_pnp = np.array([r['pN_pnp'] for r in pnp_rows])
    pE_pnp = np.array([r['pE_pnp'] for r in pnp_rows])
    pD_pnp = np.array([r['pD_pnp'] for r in pnp_rows])

    # EKF log arrays
    t_ekf  = np.array([r['t']  for r in log_rows if r['phase'] == 'APPROACH'])
    vN_ekf = np.array([r['vN'] for r in log_rows if r['phase'] == 'APPROACH'])
    vE_ekf = np.array([r['vE'] for r in log_rows if r['phase'] == 'APPROACH'])
    vD_ekf = np.array([r['vD'] for r in log_rows if r['phase'] == 'APPROACH'])
    pN_ekf = np.array([r['pN'] for r in log_rows if r['phase'] == 'APPROACH'])
    pE_ekf = np.array([r['pE'] for r in log_rows if r['phase'] == 'APPROACH'])
    pD_ekf = np.array([r['pD'] for r in log_rows if r['phase'] == 'APPROACH'])

    # Differentiate PnP positions → vel_pnp (finite differences between consecutive rows)
    vel_pnp_N, vel_pnp_E, vel_pnp_D, t_vel_pnp = [], [], [], []
    for i in range(1, len(pnp_rows)):
        dt = t_pnp[i] - t_pnp[i-1]
        if 0.01 < dt < 0.3:
            t_mid = (t_pnp[i] + t_pnp[i-1]) / 2.0
            vel_pnp_N.append((pN_pnp[i] - pN_pnp[i-1]) / dt)
            vel_pnp_E.append((pE_pnp[i] - pE_pnp[i-1]) / dt)
            vel_pnp_D.append((pD_pnp[i] - pD_pnp[i-1]) / dt)
            t_vel_pnp.append(t_mid)

    if len(t_vel_pnp) < 2 or len(t_ekf) < 2:
        lines.append("  Insufficient velocity samples for regression.")
        _write_report(lines, out_dir)
        return lines

    t_vel_pnp = np.array(t_vel_pnp)
    vel_pnp_N = np.array(vel_pnp_N)
    vel_pnp_E = np.array(vel_pnp_E)
    vel_pnp_D = np.array(vel_pnp_D)

    # Interpolate EKF velocity onto PnP timestamps
    vN_at_pnp = np.interp(t_vel_pnp, t_ekf, vN_ekf)
    vE_at_pnp = np.interp(t_vel_pnp, t_ekf, vE_ekf)
    vD_at_pnp = np.interp(t_vel_pnp, t_ekf, vD_ekf)

    results = {}
    for axis, v_pnp, v_ekf in [('N', vel_pnp_N, vN_at_pnp),
                                 ('E', vel_pnp_E, vE_at_pnp),
                                 ('D', vel_pnp_D, vD_at_pnp)]:
        A = np.column_stack([v_ekf, np.ones(len(v_ekf))])
        coeff, _, _, _ = np.linalg.lstsq(A, v_pnp, rcond=None)
        scale, bias = float(coeff[0]), float(coeff[1])
        v_pred = scale * v_ekf + bias
        rmse   = float(np.sqrt(np.mean((v_pnp - v_pred)**2)))
        results[axis] = {'scale': scale, 'bias': bias, 'rmse': rmse,
                          'v_pnp': v_pnp, 'v_ekf': v_ekf}
        lines.append(f"  {axis}-axis: scale={scale:.3f}  bias={bias:+.3f} m/s  "
                     f"RMSE={rmse:.3f} m/s")

    # Position drift at end of approach: EKF vs PnP
    if len(t_ekf) > 0 and len(t_pnp) > 0:
        t_end = min(t_ekf[-1], t_pnp[-1])
        pN_ekf_end = float(np.interp(t_end, t_ekf, pN_ekf))
        pE_ekf_end = float(np.interp(t_end, t_ekf, pE_ekf))
        pD_ekf_end = float(np.interp(t_end, t_ekf, pD_ekf))
        pN_pnp_end = float(np.interp(t_end, t_pnp, pN_pnp))
        pE_pnp_end = float(np.interp(t_end, t_pnp, pE_pnp))
        pD_pnp_end = float(np.interp(t_end, t_pnp, pD_pnp))
        dN = pN_ekf_end - pN_pnp_end
        dE = pE_ekf_end - pE_pnp_end
        dD = pD_ekf_end - pD_pnp_end
        lines.append(f"  Position drift at gate: dN={dN:+.3f}m  dE={dE:+.3f}m  "
                     f"dD={dD:+.3f}m")

    lines.append(f"  PnP velocity samples: {len(t_vel_pnp)}  "
                 f"(from {len(pnp_rows)} total PnP detections)")
    lines.append(f"  Gate 0 world NED: {gate0_ned_world}")
    lines.append(f"  pos_offset_ned:   {pos_offset_ned}")

    # Trajectory fit
    traj = _fit_trajectory(log_rows, pnp_rows, out_dir)
    if traj:
        lines.append("  Trajectory fit (pos_pnp = pos0 + a·(pos_ekf−pos0) + b·t):")
        for axis in ('N', 'E', 'D'):
            r = traj[axis]
            lines.append(f"    {axis}-axis: a={r['a']:.3f}  b={r['b']:+.4f} m/s  "
                         f"RMSE={r['rmse_m']:.3f} m")
        lines.append(f"  PnP frames available for trajectory fit: {len(pnp_rows)}")
    else:
        lines.append("  Trajectory fit: insufficient data.")

    lines.append("────────────────────────────────────────────────────────────────────────")

    _write_report(lines, out_dir)
    _plot_ekf_vs_pnp(log_rows, pnp_rows, gate0_ned_world, out_dir)
    _plot_velocity_fit(results, out_dir)
    _plot_detection_rate(log_rows, pnp_rows, out_dir)
    _plot_gate_distance(pnp_rows, out_dir)
    return lines


def _write_report(lines, out_dir):
    path = os.path.join(out_dir, "report.txt")
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"Report → {path}", flush=True)


def _plot_ekf_vs_pnp(log_rows, pnp_rows, gate0_ned_world, out_dir):
    appr = [r for r in log_rows if r['phase'] == 'APPROACH']
    if not appr:
        return
    pN_ekf = np.array([r['pN'] for r in appr])
    pE_ekf = np.array([r['pE'] for r in appr])
    pN_pnp = np.array([r['pN_pnp'] for r in pnp_rows]) if pnp_rows else np.array([])
    pE_pnp = np.array([r['pE_pnp'] for r in pnp_rows]) if pnp_rows else np.array([])

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.plot(pE_ekf, pN_ekf, lw=1.2, color='tab:blue', label='EKF (dead-reckoned)')
    if len(pN_pnp):
        ax.scatter(pE_pnp, pN_pnp, s=20, color='tab:orange', zorder=5, label='PnP positions')
    if gate0_ned_world is not None:
        gN, gE = float(gate0_ned_world[0]), float(gate0_ned_world[1])
        ax.scatter([gE], [gN], s=120, marker='*', color='red', zorder=6, label='Gate 0 (world NED)')
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title("EKF path vs PnP positions — APPROACH phase")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    plt.tight_layout()
    out = os.path.join(out_dir, "ekf_vs_pnp_path.png")
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Path plot → {out}", flush=True)


def _plot_velocity_fit(results, out_dir):
    if not results:
        return
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Velocity calibration: EKF vs PnP (vel_pnp = scale*vel_ekf + bias)", fontsize=11)
    colors = {'N': 'tab:blue', 'E': 'tab:orange', 'D': 'tab:green'}
    for ax, axis in zip(axes, ['N', 'E', 'D']):
        r = results[axis]
        v_e, v_p = r['v_ekf'], r['v_pnp']
        ax.scatter(v_e, v_p, s=12, alpha=0.6, color=colors[axis])
        lim = np.array([min(v_e.min(), v_p.min()) - 0.1, max(v_e.max(), v_p.max()) + 0.1])
        reg_y = r['scale'] * lim + r['bias']
        ax.plot(lim, reg_y, 'k--', lw=1.2,
                label=f"scale={r['scale']:.3f}\nbias={r['bias']:+.3f}\nRMSE={r['rmse']:.3f}")
        ax.plot(lim, lim, 'r:', lw=0.8, alpha=0.5, label='ideal (1:1)')
        ax.set_xlabel(f"vel_ekf_{axis} (m/s)")
        ax.set_ylabel(f"vel_pnp_{axis} (m/s)")
        ax.set_title(f"{axis}-axis")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(out_dir, "velocity_fit.png")
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Velocity fit plot → {out}", flush=True)


def _plot_detection_rate(log_rows, pnp_rows, out_dir):
    appr = [r for r in log_rows if r['phase'] == 'APPROACH']
    if not appr:
        return
    t0 = appr[0]['t']
    t_end = appr[-1]['t']
    bins = np.arange(t0, t_end + 1.0, 1.0)
    if len(bins) < 2:
        return

    total_per_bin = np.zeros(len(bins) - 1)
    det_per_bin   = np.zeros(len(bins) - 1)

    for r in appr:
        idx = min(int(r['t'] - t0), len(bins) - 2)
        total_per_bin[idx] += 1

    for r in pnp_rows:
        idx = min(int(r['t'] - t0), len(bins) - 2)
        det_per_bin[idx] += 1

    det_rate = np.where(total_per_bin > 0, det_per_bin / total_per_bin * 100, 0)
    t_centres = (bins[:-1] + bins[1:]) / 2 - t0

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(t_centres, det_rate, width=0.8, color='tab:orange', alpha=0.8)
    ax.set_ylim(0, 110)
    ax.set_xlabel("Time in APPROACH (s)")
    ax.set_ylabel("PnP detection rate (%)")
    ax.set_title("YOLO detection rate per second during APPROACH")
    ax.axhline(50, color='red', lw=0.8, ls='--', label='50%')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    out = os.path.join(out_dir, "detection_rate.png")
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Detection rate plot → {out}", flush=True)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    param          = load_params()
    T_max          = float(param['T_max_motor'])
    m_yaml         = float(param['m'])
    g              = float(param['g'])


    T_hover_theory = m_yaml * g
    T_hover_norm   = T_hover_theory / (4.0 * T_max)

    out_dir = os.path.join("logs", f"calib_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "calib.csv")

    print("Connecting…", flush=True)
    conn = mavutil.mavlink_connection(f"udpin:{SIM_IP}:{SIM_PORT}")
    conn.wait_heartbeat()
    print(f"Connected (sys {conn.target_system})", flush=True)

    shared = {}
    rx = MAVLinkRX.create_mavlink_rx(conn, shared, logger=None)

    print("Starting VisionRX…", flush=True)
    vision = VisionRX(shared, logger=None)

    print("Resetting sim…", flush=True)
    _send_reset(conn)
    time.sleep(2.0)

    print("\nPress 's' to arm and start calibration pipeline…", flush=True)
    while True:
        if msvcrt.kbhit() and msvcrt.getwch().lower() == 's':
            break
        time.sleep(0.05)

    _arm(conn)
    time.sleep(0.5)

    shared['zupt_enabled'] = True

    log_rows = []
    pnp_rows = []

    def _record(phase, t, p_cmd, q_cmd, r_cmd, T_norm_cmd):
        st = _read_state(shared)
        if st is None:
            return
        phi_d, theta_d, psi_d, vN, vE, vD, ax, ay, az, gx, gy, gz, pN, pE, pD = st
        log_rows.append({
            'phase': phase, 't': t,
            'phi_deg': phi_d, 'theta_deg': theta_d, 'psi_deg': psi_d,
            'vN': vN, 'vE': vE, 'vD': vD,
            'pN': pN, 'pE': pE, 'pD': pD,
            'ax': ax, 'ay': ay, 'az': az,
            'gx': gx, 'gy': gy, 'gz': gz,
            'p_cmd': p_cmd, 'q_cmd': q_cmd, 'r_cmd': r_cmd,
            'T_norm': T_norm_cmd,
        })

    # ── WAIT ─────────────────────────────────────────────────────────────────
    t0     = time.time()
    phase  = "WAIT"
    last_p = t0
    print(f"\n[WAIT] {WAIT_SEC:.1f} s — IMU calibration; collecting gate 0 track data…",
          flush=True)
    _send_motors(conn, 0.0)

    gate0_ned_world = None
    while time.time() - t0 < WAIT_SEC:
        t = time.time() - t0
        _record(phase, t, 0.0, 0.0, 0.0, 0.0)
        gates = shared.get('track_gates_ned', {})
        if gate0_ned_world is None and 0 in gates:
            gate0_ned_world = gates[0]['ned'].copy()
            print(f"  [WAIT] gate 0 NED={gate0_ned_world}  "
                  f"w={gates[0]['width']:.2f}m h={gates[0]['height']:.2f}m", flush=True)
        if time.time() - last_p >= 1.0:
            last_p = time.time()
            st = _read_state(shared)
            if st:
                print(f"  [WAIT t={t:4.1f}s]  theta={st[1]:+.1f}°  "
                      f"gates={list(gates.keys())}", flush=True)
        time.sleep(DT)

    # If track data never arrived, fall back to params.yaml waypoints[1] as gate 0 NED.
    # VisionRX uses the same fallback internally, so PnP drone_ned will be consistent.
    if gate0_ned_world is None:
        _wps = param.get('waypoints')
        if _wps is not None and len(_wps) > 1:
            gate0_ned_world = np.array(_wps[1], dtype=float)
            print(f"  [WAIT] no track data — using params.yaml waypoints[1] as gate 0: "
                  f"{gate0_ned_world}", flush=True)
        else:
            print("  [WAIT] no gate 0 NED (no track data, no waypoints) — "
                  "APPROACH will use PnP bearing only", flush=True)

    # ── HOVER RESET ──────────────────────────────────────────────────────────
    shared['reset_vel_flag'] = True
    shared['zupt_enabled']   = False
    # Use EKF quaternion for leveling (IMU-only path blacks out during thrust)
    shared['post_blip_att_reset_done'] = True
    time.sleep(0.01)

    t_base = time.time()

    # ── HOVER ────────────────────────────────────────────────────────────────
    phase      = "HOVER"
    t_hover    = time.time()
    T_norm_hov = T_hover_norm
    pos_offset_ned = None

    hover_pD = -HOVER_ALT_M   # 2 m above EKF reset point (local NED, D negative = up)
    print(f"\n[HOVER] {HOVER_SEC:.1f} s climbing to {HOVER_ALT_M:.1f} m (hover_pD={hover_pD:.2f})…", flush=True)

    while time.time() - t_hover < HOVER_SEC:
        t   = time.time() - t_base
        _mav_hov = shared.get('mav_state')
        if _mav_hov is not None:
            _vD_hov  = float(_mav_hov['vel_ned'][2])
            _pD_hov  = float(_mav_hov['pos_ned'][2])
        else:
            _vD_hov, _pD_hov = 0.0, hover_pD
        T_norm_hov = float(np.clip(
            T_hover_norm + KP_ALT_VD * _vD_hov + KP_ALT_D * (_pD_hov - hover_pD),
            0.22, 0.38))
        p_lv, q_lv = _level_rates(shared)
        _send_attitude_target(conn, p_lv, q_lv, 0.0, T_norm_hov)
        _record(phase, t, p_lv, q_lv, 0.0, T_norm_hov)

        # Capture pos_offset once EKF has settled
        if pos_offset_ned is None and time.time() - t_hover > HOVER_SEC * 0.5:
            mav = shared.get('mav_state')
            if mav is not None:
                pos_offset_ned = np.array(mav['pos_ned'], dtype=float)
                print(f"  [HOVER] pos_offset_ned={pos_offset_ned}  "
                      f"T_norm={T_norm_hov:.3f}", flush=True)
        time.sleep(DT)

    if pos_offset_ned is None:
        pos_offset_ned = np.zeros(3)

    # Gate 0 in LOCAL NED (EKF frame after hover reset)
    if gate0_ned_world is not None:
        gate0_local = gate0_ned_world - pos_offset_ned
        print(f"  gate0_world={gate0_ned_world}  offset={pos_offset_ned}  "
              f"gate0_local={gate0_local}", flush=True)
    else:
        gate0_local = None
        print(f"  gate0_world=None  offset={pos_offset_ned}  "
              f"gate0_local=None (PnP bearing only)", flush=True)

    # Snap psi reference and target altitude at end of HOVER for APPROACH
    _st = _read_state(shared)
    psi_ref_deg = float(_st[2]) if _st is not None else 0.0
    _mav_end = shared.get('mav_state')
    target_pD = float(_mav_end['pos_ned'][2]) if _mav_end is not None else 0.0
    print(f"  [HOVER] target_pD={target_pD:.3f} m NED", flush=True)

    # ── APPROACH ─────────────────────────────────────────────────────────────
    phase         = "APPROACH"
    t_approach    = time.time()
    last_print    = t_approach
    _diag_printed = False
    print(f"\n[APPROACH] flying toward gate 0 at {V_APPROACH:.1f} m/s "
          f"(max {APPROACH_MAX_S:.0f} s)…", flush=True)

    while time.time() - t_approach < APPROACH_MAX_S:
        t = time.time() - t_base

        # Gate-passed check
        if shared.pop('gate_passed', False):
            print(f"  [APPROACH] COLLISION — gate 0 passed at t={t:.2f}s", flush=True)
            break

        st = _read_state(shared)
        if st is None:
            time.sleep(DT)
            continue

        phi_d, theta_d, psi_d, vN, vE, vD = st[0], st[1], st[2], st[3], st[4], st[5]
        phi_r   = np.deg2rad(phi_d)
        theta_r = np.deg2rad(theta_d)
        psi_r   = np.deg2rad(psi_d)

        # Direction to gate 0 in LOCAL NED
        mav = shared.get('mav_state')
        if mav is not None:
            _pos_D = float(mav['pos_ned'][2])
            _vD    = float(mav['vel_ned'][2])
            T_norm_approach = float(np.clip(
                T_hover_norm + KP_ALT_VD * _vD + KP_ALT_D * (_pos_D - target_pD),
                0.22, 0.38))
        else:
            T_norm_approach = T_hover_norm
        dir_n, dir_e = 1.0, 0.0   # default: straight North
        dist_to_gate = 999.0
        if mav is not None and gate0_local is not None:
            pos_local = np.array(mav['pos_ned'], dtype=float)
            to_gate = gate0_local - pos_local
            dist_to_gate = float(np.linalg.norm(to_gate[:2]))  # horizontal only
            if dist_to_gate < 1.5:
                # Inside gate radius — cut motors, don't try to steer
                print(f"  [APPROACH] at gate (dist={dist_to_gate:.2f}m) — breaking", flush=True)
                break
            elif dist_to_gate < 2.0:
                pass  # keep dir_n=1, dir_e=0: coast straight through
            elif dist_to_gate > 0.1:
                dir_n = to_gate[0] / dist_to_gate
                dir_e = to_gate[1] / dist_to_gate
        elif mav is not None:
            # No gate NED: use PnP bearing from latest detection
            det = shared.get('gate_detection', {})
            if det.get('tvec_cam') is not None:
                _tv = np.asarray(det['tvec_cam'])
                # cam→body: x_b(fwd)=cam_z, y_b(right)=cam_x
                _xb = _tv[2]   # cam z → body x_b (forward)
                _yb = _tv[0]   # cam x → body y_b (right)
                _horiz = np.sqrt(_xb**2 + _yb**2)
                if _horiz > 0.1:
                    _qw, _qx, _qy, _qz = mav['quat']
                    _psi = np.arctan2(2*(_qw*_qz + _qx*_qy),
                                      1 - 2*(_qy*_qy + _qz*_qz))
                    _bear_body = float(np.arctan2(_yb, _xb))   # gate bearing from body nose [rad]
                    _bear_ned  = _psi + _bear_body
                    dir_n = float(np.cos(_bear_ned))
                    dir_e = float(np.sin(_bear_ned))
                    dist_to_gate = float(_horiz)

        # Velocity P-loop → desired tilt.
        # Project NED velocity error onto body x_b/y_b axes (heading compensation).
        # At psi=0° (north-facing) this is identity: e_v_xb = e_vN.
        # Sign: positive e_v_xb (need more forward speed) → negative theta_des (nose-down) → forward thrust.
        e_vN   =  V_APPROACH * dir_n - vN
        e_vE   =  V_APPROACH * dir_e - vE
        cos_psi =  np.cos(psi_r)
        sin_psi =  np.sin(psi_r)
        e_v_xb =  cos_psi * e_vN + sin_psi * e_vE   # body x_b (forward) velocity error
        theta_des = float(np.clip(-K_VEL * e_v_xb, -MAX_TILT_RAD, MAX_TILT_RAD))
        phi_des   = 0.0  # level roll only; lateral correction via heading, not EKF vE

        if not _diag_printed:
            _diag_printed = True
            print(f"  [APPROACH-DIAG] psi={psi_d:.1f}deg  dir_n={dir_n:.2f} dir_e={dir_e:.2f}  "
                  f"vN={vN:.2f} vE={vE:.2f}  e_v_xb={e_v_xb:.3f}  "
                  f"theta_des={theta_des:.3f}rad", flush=True)

        # Attitude P-loop → rate commands
        q_cmd = float( K_ATT * (theta_des - theta_r))
        p_cmd = float( K_ATT * (phi_des   - phi_r))

        # Yaw hold
        e_psi = float(((psi_ref_deg - psi_d) + 180.0) % 360.0 - 180.0)
        r_cmd = float(np.clip(K_PSI * np.deg2rad(e_psi), -Q_PSI_MAX, Q_PSI_MAX))

        _send_attitude_target(conn, p_cmd, q_cmd, r_cmd, T_norm_approach)
        _record(phase, t, p_cmd, q_cmd, r_cmd, T_norm_approach)

        # Collect PnP record from VisionRX
        pnp_sample = shared.pop('_vision_pnp_record', None)
        if pnp_sample is not None:
            _v = pnp_sample.get('vel_ned_pnp')
            pnp_rows.append({
                't':      t,
                'pN_pnp': float(pnp_sample['drone_ned'][0]) - pos_offset_ned[0],
                'pE_pnp': float(pnp_sample['drone_ned'][1]) - pos_offset_ned[1],
                'pD_pnp': float(pnp_sample['drone_ned'][2]) - pos_offset_ned[2],
                'vN_pnp': float(_v[0]) if _v is not None else float('nan'),
                'vE_pnp': float(_v[1]) if _v is not None else float('nan'),
                'vD_pnp': float(_v[2]) if _v is not None else float('nan'),
                'dist':   float(pnp_sample['dist_m']),
                'conf':   float(pnp_sample['conf']),
            })
            if time.time() - last_print >= 2.0:
                last_print = time.time()
                print(f"  [APPROACH t={t:.1f}s]  dist={pnp_sample['dist_m']:.1f}m  "
                      f"conf={pnp_sample['conf']:.2f}  "
                      f"gate_dist={dist_to_gate:.1f}m  "
                      f"vN={vN:.2f} vE={vE:.2f}", flush=True)
        elif time.time() - last_print >= 2.0:
            last_print = time.time()
            print(f"  [APPROACH t={t:.1f}s]  no PnP  gate_dist={dist_to_gate:.1f}m  "
                  f"vN={vN:.2f} vE={vE:.2f}", flush=True)

        time.sleep(DT)

    # ── KILL ─────────────────────────────────────────────────────────────────
    print("\n[KILL] cutting motors…", flush=True)
    _send_motors(conn, 0.0)

    # ── Write CSV ─────────────────────────────────────────────────────────────
    header = ("phase,t,phi_deg,theta_deg,psi_deg,vN,vE,vD,pN,pE,pD,"
              "ax,ay,az,gx,gy,gz,p_cmd,q_cmd,r_cmd,T_norm\n")
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write(header)
        for r in log_rows:
            f.write(
                f"{r['phase']},{r['t']:.4f},"
                f"{r['phi_deg']:.4f},{r['theta_deg']:.4f},{r['psi_deg']:.4f},"
                f"{r['vN']:.4f},{r['vE']:.4f},{r['vD']:.4f},"
                f"{r['pN']:.4f},{r['pE']:.4f},{r['pD']:.4f},"
                f"{r['ax']:.4f},{r['ay']:.4f},{r['az']:.4f},"
                f"{r['gx']:.5f},{r['gy']:.5f},{r['gz']:.5f},"
                f"{r['p_cmd']:.4f},{r['q_cmd']:.4f},{r['r_cmd']:.4f},{r['T_norm']:.4f}\n"
            )
    print(f"CSV → {csv_path}", flush=True)

    # ── Write pnp.csv ─────────────────────────────────────────────────────────
    pnp_csv_path = os.path.join(out_dir, "pnp.csv")
    with open(pnp_csv_path, 'w', encoding='utf-8') as f:
        f.write("t,pN_pnp,pE_pnp,pD_pnp,vN_pnp,vE_pnp,vD_pnp,dist_m,conf\n")
        for r in pnp_rows:
            f.write(
                f"{r['t']:.4f},"
                f"{r['pN_pnp']:.4f},{r['pE_pnp']:.4f},{r['pD_pnp']:.4f},"
                f"{r['vN_pnp']:.4f},{r['vE_pnp']:.4f},{r['vD_pnp']:.4f},"
                f"{r['dist']:.3f},{r['conf']:.3f}\n"
            )
    print(f"PnP CSV → {pnp_csv_path}", flush=True)

    # ── Analysis + plots ──────────────────────────────────────────────────────
    print(f"\nPnP rows collected: {len(pnp_rows)}", flush=True)
    report_lines = _analyse(log_rows, pnp_rows, pos_offset_ned, gate0_ned_world, out_dir)
    for line in report_lines:
        print(line, flush=True)

    # ── Shutdown VisionRX ─────────────────────────────────────────────────────
    recv_thread = vision.get_thread_for_join()
    recv_thread.join(timeout=3.0)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
