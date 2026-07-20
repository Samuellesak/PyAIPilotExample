"""
flight_sysid.py
===============
In-flight rigid-body system identification for the quadcopter simulator.

Root cause of EKF velocity divergence (identified from logs/20260715_174447):
  The launch blip fires 2.86 g of thrust for 0.15 s, causing a >50° pitch
  excursion.  The EKF misses ~9° of nose-UP rotation during recovery, leaving
  a persistent attitude error.  Wrong R_b2n then integrates a false
  v_dot_N ≈ -1.4 m/s² every step, driving EKF velocity to -50 m/s in <10 s.

This script identifies:
  (1) Blip impact: peak body rate, theta excursion, EKF attitude error
  (2) Mass:        m from steady-state hover az  (compare to params.yaml m)
  (3) Sign conv.:  p/q/r direction in SET_ATTITUDE_TARGET vs FRD convention
  (4) Rate tracking: q_meas / q_des ratio (sim rate-controller bandwidth check)

Pipeline
--------
  WAIT        3.5 s   Static calibration (gyro bias, slope attitude)
  BLIP        0.15 s  High-thrust burst — mirrors main.py exactly
  BLIP_OBS    2.0 s   Log post-blip dynamics
  HOVER_TRIM  3.0 s   Level hover — identify mass
  PITCH_PLUS  0.5 s   q_des = +Q_TEST → sign test (FRD: q>0 = nose UP)
  SETTLE      1.0 s
  PITCH_MINUS 0.5 s   q_des = -Q_TEST → confirm sign
  SETTLE      1.0 s
  ROLL_PLUS   0.5 s   p_des = +Q_TEST → roll sign test (FRD: p>0 = roll right)
  SETTLE      1.0 s
  YAW_PLUS    0.5 s   r_des = +Q_TEST → yaw sign test (FRD: r>0 = yaw right)
  SETTLE      1.0 s
  KILL

Outputs
-------
  logs/sysid_<timestamp>/sysid.csv     — raw per-frame data
  logs/sysid_<timestamp>/report.txt    — human-readable findings + suggestions
  Console                              — real-time progress
"""

import os
import sys
import time
import msvcrt
from datetime import datetime

# Force UTF-8 on Windows (default console codec is cp1252, which lacks →, ─, ≈)
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pymavlink import mavutil

from dyn import load_params
from mavlink_rx import MAVLinkRX

# ── Connection ───────────────────────────────────────────────────────────────
SIM_IP   = "127.0.0.1"
SIM_PORT = 14550

# ── Timing ───────────────────────────────────────────────────────────────────
CONTROL_HZ  = 250
DT          = 1.0 / CONTROL_HZ

WAIT_SEC     = 3.5
BLIP_OBS_SEC = 2.0
HOVER_SEC    = 10.0
EXCITE_SEC   = 0.5
SETTLE_SEC   = 1.0

# ── Excitation amplitude ─────────────────────────────────────────────────────
Q_TEST = 0.4          # [rad/s] — small enough to stay controllable

# ── Hover altitude controller ────────────────────────────────────────────────
# Uses IMU body-z specific force (az) instead of EKF vD.
# At any attitude with thrust T along -body_z: az = -T/m (independent of tilt).
# At hover (T = m*g): az = -g.  Deviation from -g → proportional thrust correction.
# Tight clamp prevents runaway if the IMU reading is temporarily unreliable.
KP_AZ  = 0.015        # az deviation [m/s²] → T_norm correction

# ── Attitude hold (outer P loop: angle [rad] → rate cmd [rad/s]) ─────────────
K_ATT        = 3.0    # pitch/roll P gain
K_PSI_SETTLE = 0.8    # yaw heading P gain (rate ratio 1.67 → eff. gain 1.34 s⁻¹, τ≈0.75 s)
Q_PSI_MAX    = 0.20   # max yaw rate for heading-hold settle [rad/s]


# ── Analysis thresholds ───────────────────────────────────────────────────────
ATT_ERROR_WARN_DEG = 3.0    # EKF theta vs acc-implied theta discrepancy [deg]
MASS_ERROR_WARN    = 0.30   # relative mass discrepancy (30 %)
SIGN_THRESH_DEG    = 1.0    # min |Δangle| to call a sign [deg]

MAVLINK_CMD_SIM_RESET = 31000


# ── Helpers ───────────────────────────────────────────────────────────────────

def _quat_to_euler(quat):
    """[qw,qx,qy,qz] → (phi, theta, psi) in radians (ZYX / NED convention)."""
    qw, qx, qy, qz = quat
    phi   = np.arctan2(2.0*(qw*qx + qy*qz), 1.0 - 2.0*(qx*qx + qy*qy))
    theta = np.arcsin(np.clip(2.0*(qw*qy - qz*qx), -1.0, 1.0))
    psi   = np.arctan2(2.0*(qw*qz + qx*qy), 1.0 - 2.0*(qy*qy + qz*qz))
    return phi, theta, psi


def _acc_implied_theta(ax, psi_rad):
    """
    Estimate the pitch angle that is consistent with the observed body-x
    specific force ax, assuming level roll (phi ≈ 0).

    At constant velocity: acc_body = R_n2b @ [0,0,-g].
    For psi=180° (facing South): R_n2b[0,:] = [-cos(theta), 0, sin(-theta)]
    → acc_x = sin(-theta)*(-g) ... let me re-derive:

    R_b2n[0,:] at psi, phi=0: [cos(psi)cos(theta), sin(psi)cos(theta), -sin(theta)]
    R_n2b = R_b2n.T → R_n2b[0,2] = -sin(theta)
    acc_x_body = R_n2b[0,:] @ [0,0,-g] = -sin(theta)*(-g) = g*sin(theta)

    So theta = arcsin(ax / g).
    (This is heading-independent because the North-East components of g_NED are zero.)
    """
    g = 9.81
    return float(np.arcsin(np.clip(ax / g, -1.0, 1.0)))


def _send_motors(conn, u_norm):
    """Send equal normalised thrust to all 4 motors."""
    v = float(np.clip(u_norm, 0.0, 1.0))
    cmds = [v, v, v, v, 0.0, 0.0, 0.0, 0.0]
    conn.mav.set_actuator_control_target_send(
        int(time.time() * 1e6),
        conn.target_system, conn.target_component,
        0,
        cmds,
    )


def _send_attitude_target(conn, p, q, r, thrust_norm):
    """Send body-rate setpoint + collective thrust (type_mask=0x80: ignore quat)."""
    conn.mav.set_attitude_target_send(
        int(time.time() * 1e3) & 0xFFFFFFFF,
        conn.target_system, conn.target_component,
        0x80,
        [1.0, 0.0, 0.0, 0.0],
        float(p), float(q), float(r),
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
    """Return (p_des, q_des) that drive phi/theta toward zero.

    Before the post-blip EKF reset: uses raw IMU acc (gives phi=theta≈0 in
    flight at hover thrust — harmless, drone just floats).
    After the post-blip EKF reset: uses EKF quaternion, which is now accurate
    to within ~3°.  This is necessary because IMU acc at hover thrust always
    reads [0,0,-g] in body frame regardless of physical tilt, so acc-based
    attitude estimation cannot detect roll or pitch angles in flight.
    """
    if shared.get('post_blip_att_reset_done'):
        # EKF is accurate after the reset — use quaternion for real attitude feedback.
        mav = shared.get('mav_state')
        if mav is None:
            return 0.0, 0.0
        phi, theta, _ = _quat_to_euler(mav['quat'])
        return float(-K_ATT * phi), float(-K_ATT * theta)

    # EKF not yet corrected — fall back to acc (returns 0 in flight, which is safe)
    imu = shared.get('imu_raw')
    if imu is None:
        return 0.0, 0.0
    ax = float(imu.get('ax', 0.0))
    ay = float(imu.get('ay', 0.0))
    az = float(imu.get('az', 0.0))
    acc_norm = float(np.sqrt(ax**2 + ay**2 + az**2))
    if abs(acc_norm - 9.81) > 2.5:   # blip thrust — acc is unreliable
        return 0.0, 0.0
    theta = float(np.arcsin(np.clip(ax / 9.81, -1.0, 1.0)))
    phi   = float(np.arcsin(np.clip(ay / (9.81 * max(np.cos(theta), 0.1)), -1.0, 1.0)))
    return float(-K_ATT * phi), float(-K_ATT * theta)


def _read_state(shared):
    """Return (phi_deg, theta_deg, psi_deg, vN, vE, vD, ax, ay, az, gx, gy, gz, pN, pE, pD)
    from latest shared data.  Returns None if no data yet."""
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


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    param      = load_params()
    T_max      = float(param['T_max_motor'])
    m_yaml     = float(param['m'])
    g          = float(param['g'])
    blip_frac  = float(param.get('blip_thrust_frac', 0.35))
    blip_dur   = float(param.get('blip_dur_sec',     0.15))

    T_hover_theory = m_yaml * g          # total hover thrust [N]
    T_hover_norm   = T_hover_theory / (4.0 * T_max)   # normalised

    # ── Output directory ───────────────────────────────────────────────────
    out_dir = os.path.join("logs", f"sysid_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)
    csv_path    = os.path.join(out_dir, "sysid.csv")
    report_path = os.path.join(out_dir, "report.txt")

    # ── Connect ────────────────────────────────────────────────────────────
    print("Connecting…", flush=True)
    conn = mavutil.mavlink_connection(f"udpin:{SIM_IP}:{SIM_PORT}")
    conn.wait_heartbeat()
    print(f"Connected (sys {conn.target_system})", flush=True)

    shared = {}
    rx = MAVLinkRX.create_mavlink_rx(conn, shared, logger=None)

    # ── Reset + arm ────────────────────────────────────────────────────────
    print("Resetting sim…", flush=True)
    _send_reset(conn)
    time.sleep(2.0)

    print("\nPress 's' to arm and start the sysid pipeline…", flush=True)
    while True:
        if msvcrt.kbhit() and msvcrt.getwch().lower() == 's':
            break
        time.sleep(0.05)

    _arm(conn)
    time.sleep(0.5)

    # Disable ZUPT now — the EKF ZUPT is only safe on the ground.
    # The sysid will be airborne after the blip.
    shared['zupt_enabled'] = True   # re-enabled explicitly below

    # ── Data log ───────────────────────────────────────────────────────────
    log_rows = []
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

    # ── Per-phase snapshot buffers ─────────────────────────────────────────
    blip_peak_gy    = 0.0    # peak |gy| during blip+recovery [rad/s]
    blip_min_theta  = 0.0    # most negative theta during blip+recovery [deg]
    blip_max_theta  = 0.0    # most positive theta during blip+recovery [deg]

    # ── WAIT PHASE ─────────────────────────────────────────────────────────
    t0     = time.time()
    phase  = "WAIT"
    last_p = t0
    print(f"\n[WAIT] {WAIT_SEC:.1f} s static calibration…", flush=True)
    _send_motors(conn, 0.0)

    while time.time() - t0 < WAIT_SEC:
        t = time.time() - t0
        _record(phase, t, 0.0, 0.0, 0.0, 0.0)
        if time.time() - last_p >= 1.0:
            last_p = time.time()
            st = _read_state(shared)
            if st:
                print(f"  [WAIT t={t:4.1f}s]  theta={st[1]:+.1f}°  phi={st[0]:+.1f}°  "
                      f"psi={st[2]:+.1f}°", flush=True)
        time.sleep(DT)

    # Snapshot: static tilt (will be used as attitude reference after reset)
    wait_rows = [r for r in log_rows if r['phase'] == 'WAIT']
    if wait_rows:
        theta_static = float(np.mean([r['theta_deg'] for r in wait_rows[-50:]]))
        phi_static   = float(np.mean([r['phi_deg']   for r in wait_rows[-50:]]))
        ax_static    = float(np.mean([r['ax']         for r in wait_rows[-50:]]))
        az_static    = float(np.mean([r['az']         for r in wait_rows[-50:]]))
    else:
        theta_static = phi_static = ax_static = 0.0
        az_static = -9.81
    print(f"  WAIT snapshot: theta={theta_static:+.2f}°  phi={phi_static:+.2f}°  "
          f"ax={ax_static:+.3f}  az={az_static:+.3f}", flush=True)

    # ── HOVER RESET (triggers EKF reset + attitude init from slope acc) ────
    shared['reset_vel_flag'] = True
    shared['zupt_enabled']   = False   # disable ZUPT — we will be airborne
    time.sleep(0.01)                   # give MAVLinkRX one callback cycle

    # ── BLIP PHASE ─────────────────────────────────────────────────────────
    phase   = "BLIP"
    t_blip  = time.time()
    t_base  = t_blip   # new time base after reset
    print(f"\n[BLIP] {blip_frac*100:.0f}% thrust for {blip_dur:.2f}s…", flush=True)

    while time.time() - t_blip < blip_dur:
        t = time.time() - t_base
        _send_motors(conn, blip_frac)
        _record(phase, t, 0.0, 0.0, 0.0, blip_frac)

        st = _read_state(shared)
        if st:
            gy_frd = -st[10]           # gy_sim → FRD: q_body = -gy_sim
            blip_peak_gy   = max(blip_peak_gy,   abs(gy_frd))
            blip_min_theta = min(blip_min_theta, st[1])
            blip_max_theta = max(blip_max_theta, st[1])
        time.sleep(DT)

    # ── BLIP OBSERVATION ───────────────────────────────────────────────────
    phase      = "BLIP_OBS"
    t_obs      = time.time()
    T_obs_norm = T_hover_norm    # command hover thrust to stabilise
    print(f"\n[BLIP_OBS] {BLIP_OBS_SEC:.1f} s observation (commanding hover thrust)…",
          flush=True)

    while time.time() - t_obs < BLIP_OBS_SEC:
        t = time.time() - t_base
        p_lv, q_lv = _level_rates(shared)
        _send_attitude_target(conn, p_lv, q_lv, 0.0, T_obs_norm)
        _record(phase, t, p_lv, q_lv, 0.0, T_obs_norm)

        st = _read_state(shared)
        if st:
            gy_frd = -st[10]
            blip_peak_gy   = max(blip_peak_gy,   abs(gy_frd))
            blip_min_theta = min(blip_min_theta, st[1])
            blip_max_theta = max(blip_max_theta, st[1])

            if time.time() - t_obs >= BLIP_OBS_SEC - 1.0:  # last second
                pass
        time.sleep(DT)

    # Snapshot: post-blip steady state (last 0.5 s of BLIP_OBS)
    obs_rows = [r for r in log_rows
                if r['phase'] == 'BLIP_OBS' and r['t'] >= (time.time() - t_base - 0.5)]
    if not obs_rows:
        obs_rows = [r for r in log_rows if r['phase'] == 'BLIP_OBS'][-25:]

    theta_post_blip_ekf = float(np.mean([r['theta_deg'] for r in obs_rows]))
    ax_post_blip        = float(np.mean([r['ax']         for r in obs_rows]))
    az_post_blip        = float(np.mean([r['az']         for r in obs_rows]))
    psi_post_blip       = float(np.mean([r['psi_deg']    for r in obs_rows]))
    theta_acc_implied   = float(np.rad2deg(_acc_implied_theta(ax_post_blip, np.deg2rad(psi_post_blip))))
    att_error_deg       = theta_acc_implied - theta_post_blip_ekf

    print(f"  Post-blip EKF theta  : {theta_post_blip_ekf:+.2f}°", flush=True)
    print(f"  Acc-implied theta    : {theta_acc_implied:+.2f}°", flush=True)
    print(f"  Attitude error (acc−EKF): {att_error_deg:+.2f}°", flush=True)
    print(f"  Peak |gy| (blip+obs) : {blip_peak_gy:.2f} rad/s", flush=True)
    print(f"  Theta range: [{blip_min_theta:.1f}°, {blip_max_theta:.1f}°]", flush=True)

    # ── HOVER TRIM PHASE ───────────────────────────────────────────────────
    phase     = "HOVER"
    t_hover   = time.time()
    T_norm_hov = T_hover_norm
    print(f"\n[HOVER] {HOVER_SEC:.1f} s hover (T_norm={T_norm_hov:.3f})…", flush=True)

    while time.time() - t_hover < HOVER_SEC:
        t    = time.time() - t_base
        az   = float(shared.get('imu_raw', {}).get('az', -9.81))
        T_norm_hov = float(np.clip(T_hover_norm + KP_AZ * (az + 9.81), 0.22, 0.38))
        p_lv, q_lv = _level_rates(shared)

        _send_attitude_target(conn, p_lv, q_lv, 0.0, T_norm_hov)
        _record(phase, t, p_lv, q_lv, 0.0, T_norm_hov)

        if int((time.time() - t_hover) / 1.0) != int((time.time() - t_hover - DT) / 1.0):
            st = _read_state(shared)
            if st:
                print(f"  [HOVER t={time.time()-t_hover:.1f}s]  "
                      f"theta={st[1]:+.1f}°  az={az:+.2f}m/s²  "
                      f"T_norm={T_norm_hov:.3f}", flush=True)
        time.sleep(DT)

    # Snapshot: mass identification from last 1 s of hover
    hov_rows = [r for r in log_rows
                if r['phase'] == 'HOVER'][-int(1.0/DT):]
    if hov_rows:
        az_hover    = float(np.mean([r['az'] for r in hov_rows]))
        ax_hover    = float(np.mean([r['ax'] for r in hov_rows]))
        T_norm_meas = float(np.mean([r['T_norm'] for r in hov_rows]))
        T_total_meas = T_norm_meas * 4.0 * T_max
        # At level hover: T_total = m*g → m = T_total/g
        m_identified = T_total_meas / g
        theta_hover_ekf   = float(np.mean([r['theta_deg'] for r in hov_rows]))
        theta_hover_acc   = float(np.rad2deg(_acc_implied_theta(ax_hover, np.deg2rad(psi_post_blip))))
        att_error_hover   = theta_hover_acc - theta_hover_ekf
    else:
        m_identified      = m_yaml
        az_hover          = -g
        T_norm_meas       = T_hover_norm
        theta_hover_ekf   = 0.0
        theta_hover_acc   = 0.0
        att_error_hover   = 0.0

    print(f"  Hover az={az_hover:.3f} m/s²  T_norm={T_norm_meas:.3f}", flush=True)
    print(f"  m_identified = {m_identified:.3f} kg  (params.yaml: {m_yaml:.3f} kg)",
          flush=True)

    # ── EXCITATION HELPER ──────────────────────────────────────────────────
    def _excitation(axis, sign, label):
        """
        Fire a rate excitation and return Δangle [deg].
        axis: 'pitch', 'roll', or 'yaw'
        sign: +1 or -1
        """
        st0 = _read_state(shared)
        if st0 is None:
            return None, None
        phi0, theta0, psi0 = st0[0], st0[1], st0[2]

        # Fire excitation
        t_exc = time.time()
        while time.time() - t_exc < EXCITE_SEC:
            t     = time.time() - t_base
            az    = float(shared.get('imu_raw', {}).get('az', -9.81))
            T_exc = float(np.clip(T_hover_norm + KP_AZ * (az + 9.81), 0.22, 0.38))

            p_c = sign * Q_TEST if axis == 'roll'  else 0.0
            q_c = sign * Q_TEST if axis == 'pitch' else 0.0
            r_c = sign * Q_TEST if axis == 'yaw'   else 0.0

            _send_attitude_target(conn, p_c, q_c, r_c, T_exc)
            _record(label, t, p_c, q_c, r_c, T_exc)
            time.sleep(DT)

        # Snapshot angle at end of excitation (mean last 0.2 s)
        exc_rows = [r for r in log_rows
                    if r['phase'] == label][-int(0.2/DT):]
        if not exc_rows:
            return 0.0, 0.0

        if axis == 'pitch':
            ang_end = float(np.mean([r['theta_deg'] for r in exc_rows]))
            delta   = ang_end - theta0
        elif axis == 'roll':
            ang_end = float(np.mean([r['phi_deg'] for r in exc_rows]))
            delta   = ang_end - phi0
        else:   # yaw — wrap difference to [-180°, 180°]
            ang_end = float(np.mean([r['psi_deg'] for r in exc_rows]))
            delta   = float(((ang_end - psi0) + 180.0) % 360.0 - 180.0)

        return delta, ang_end

    def _settle(label, dur=SETTLE_SEC, psi_ref_deg=None):
        """Level the drone after an excitation.

        psi_ref_deg: if given, actively command yaw rate to return to that heading.
        Standard FRD convention applies: r>0 → psi increases. r_cmd = +K_PSI_SETTLE * e_psi_rad.
        """
        t_set = time.time()
        while time.time() - t_set < dur:
            t   = time.time() - t_base
            az  = float(shared.get('imu_raw', {}).get('az', -9.81))
            T_s = float(np.clip(T_hover_norm + KP_AZ * (az + 9.81), 0.22, 0.38))
            p_lv, q_lv = _level_rates(shared)

            r_cmd = 0.0
            if psi_ref_deg is not None:
                mav = shared.get('mav_state')
                if mav is not None:
                    qw, qx, qy, qz = mav['quat']
                    psi_now = float(np.degrees(np.arctan2(
                        2.0*(qw*qz + qx*qy), 1.0 - 2.0*(qy*qy + qz*qz))))
                    e_psi = float(((psi_ref_deg - psi_now) + 180.0) % 360.0 - 180.0)
                    # Standard FRD: +r → psi increases. P ctrl: r_cmd = +K * e_psi.
                    r_cmd = float(np.clip(K_PSI_SETTLE * np.deg2rad(e_psi),
                                         -Q_PSI_MAX, Q_PSI_MAX))
                    if time.time() - t_set < 2 * DT:
                        print(f"  [YAW-SETTLE] psi_ref={psi_ref_deg:.1f}°  "
                              f"psi_now={psi_now:.1f}°  e_psi={e_psi:.2f}°  "
                              f"r_cmd={r_cmd:.3f}", flush=True)

            _send_attitude_target(conn, p_lv, q_lv, r_cmd, T_s)
            _record(label, t, p_lv, q_lv, r_cmd, T_s)
            time.sleep(DT)

    # ── PITCH SIGN TEST ────────────────────────────────────────────────────
    print(f"\n[PITCH+] q_des=+{Q_TEST:.2f} rad/s for {EXCITE_SEC:.1f}s…", flush=True)
    dtheta_plus, theta_end_plus = _excitation('pitch', +1, 'PITCH_PLUS')
    _settle('SETTLE_P1')

    print(f"  Δtheta(q+) = {dtheta_plus:+.2f}°", flush=True)

    print(f"\n[PITCH-] q_des=-{Q_TEST:.2f} rad/s for {EXCITE_SEC:.1f}s…", flush=True)
    dtheta_minus, theta_end_minus = _excitation('pitch', -1, 'PITCH_MINUS')
    _settle('SETTLE_P2')

    print(f"  Δtheta(q-) = {dtheta_minus:+.2f}°", flush=True)

    # ── ROLL SIGN TEST ─────────────────────────────────────────────────────
    print(f"\n[ROLL+] p_des=+{Q_TEST:.2f} rad/s for {EXCITE_SEC:.1f}s…", flush=True)
    dphi_plus, phi_end_plus = _excitation('roll', +1, 'ROLL_PLUS')
    _settle('SETTLE_R1')

    print(f"  Δphi(p+) = {dphi_plus:+.2f}°", flush=True)

    # ── YAW SIGN TEST ──────────────────────────────────────────────────────
    # Capture heading before excitation so the settle can command back to it.
    _st_pre_yaw  = _read_state(shared)
    _psi_ref_deg = _st_pre_yaw[2] if _st_pre_yaw is not None else 0.0
    print(f"\n[YAW+] r_des=+{Q_TEST:.2f} rad/s for {EXCITE_SEC:.1f}s "
          f"(psi_ref={_psi_ref_deg:.1f}°)…", flush=True)
    dpsi_plus, psi_end_plus = _excitation('yaw', +1, 'YAW_PLUS')
    _settle('SETTLE_Y1', dur=2.0, psi_ref_deg=_psi_ref_deg)

    print(f"  Δpsi(r+) = {dpsi_plus:+.2f}°", flush=True)

    # ── KILL ───────────────────────────────────────────────────────────────
    _send_motors(conn, 0.0)
    print("\nMotors killed.", flush=True)
    rx.get_thread_for_join().join(timeout=1.0)

    # ── SAVE CSV ───────────────────────────────────────────────────────────
    header = ("phase,t,phi_deg,theta_deg,psi_deg,vN,vE,vD,pN,pE,pD,"
              "ax,ay,az,gx,gy,gz,p_cmd,q_cmd,r_cmd,T_norm\n")
    with open(csv_path, "w") as f:
        f.write(header)
        for r in log_rows:
            f.write(f"{r['phase']},{r['t']:.4f},"
                    f"{r['phi_deg']:.4f},{r['theta_deg']:.4f},{r['psi_deg']:.4f},"
                    f"{r['vN']:.4f},{r['vE']:.4f},{r['vD']:.4f},"
                    f"{r['pN']:.4f},{r['pE']:.4f},{r['pD']:.4f},"
                    f"{r['ax']:.4f},{r['ay']:.4f},{r['az']:.4f},"
                    f"{r['gx']:.4f},{r['gy']:.4f},{r['gz']:.4f},"
                    f"{r['p_cmd']:.4f},{r['q_cmd']:.4f},{r['r_cmd']:.4f},"
                    f"{r['T_norm']:.4f}\n")
    print(f"CSV → {csv_path}", flush=True)

    # ── ANALYSIS ───────────────────────────────────────────────────────────

    # Sign convention: FRD expects
    #   q > 0 → nose UP → theta increases (+)
    #   p > 0 → roll right → phi increases (+)
    #   r > 0 → yaw right (CW from above) → psi increases (+)
    def _sign_result(delta, threshold=SIGN_THRESH_DEG):
        if abs(delta) < threshold:
            return "INCONCLUSIVE (|Δ| < {:.1f}°)".format(threshold)
        return "CORRECT (+Δ)" if delta > 0 else "REVERSED (-Δ)"

    q_plus_sign  = _sign_result(dtheta_plus)
    q_minus_sign = _sign_result(-dtheta_minus)   # negative q should give negative Δtheta
    p_plus_sign  = _sign_result(dphi_plus)
    r_plus_sign  = _sign_result(dpsi_plus)

    # Rate tracking: for PITCH_PLUS, the commanded q_des=+Q_TEST.
    # Measure the mean q_meas (gy_frd = -gy_sim, in [gx,gy,gz] raw):
    pp_rows = [r for r in log_rows if r['phase'] == 'PITCH_PLUS']
    if pp_rows:
        # gy_sim in imu_raw; gy_frd = -gy_sim
        q_meas_mean = float(np.mean([-r['gy'] for r in pp_rows]))
        rate_ratio  = q_meas_mean / Q_TEST if abs(Q_TEST) > 0.01 else float('nan')
    else:
        q_meas_mean = float('nan')
        rate_ratio  = float('nan')

    # EKF attitude error during hover (cross-check)
    m_error_pct = abs(m_identified - m_yaml) / m_yaml * 100.0

    # ── REPORT ─────────────────────────────────────────────────────────────
    lines = []
    sep   = "=" * 60

    lines += [sep, "FLIGHT SYSID REPORT", sep, ""]
    lines += [f"  params.yaml m       : {m_yaml:.3f} kg"]
    lines += [f"  params.yaml T_max   : {T_max:.2f} N/motor"]
    lines += [f"  T_hover_theory/motor: {T_hover_theory/4:.2f} N  "
              f"({T_hover_norm*100:.1f}% throttle)", ""]

    lines += ["── 1. BLIP CHARACTERISATION ──────────────────────────────"]
    lines += [f"  blip_thrust_frac    : {blip_frac:.2f}"]
    lines += [f"  blip_dur_sec        : {blip_dur:.2f} s"]
    lines += [f"  Peak |gy| (FRD)     : {blip_peak_gy:.2f} rad/s  "
              f"({np.rad2deg(blip_peak_gy):.1f} deg/s)"]
    lines += [f"  Theta excursion     : [{blip_min_theta:.1f}°, {blip_max_theta:.1f}°]"]

    att_warn = " ← SIGNIFICANT" if abs(att_error_deg) > ATT_ERROR_WARN_DEG else " ← OK"
    lines += [f"  Post-blip EKF theta : {theta_post_blip_ekf:+.2f}°"]
    lines += [f"  Acc-implied theta   : {theta_acc_implied:+.2f}°"]
    lines += [f"  EKF attitude error  : {att_error_deg:+.2f}°{att_warn}"]
    lines += [""]

    if abs(att_error_deg) > ATT_ERROR_WARN_DEG:
        lines += ["  → PROBLEM: EKF lost track of attitude during the blip."]
        lines += [f"    The EKF is ~{att_error_deg:+.1f}° wrong after recovery."]
        lines += ["    This causes wrong R_b2n → false v_dot_N → EKF velocity divergence."]
        if blip_peak_gy > 5.0:
            lines += [f"    Peak gy = {blip_peak_gy:.1f} rad/s = very large angular excursion."]
            lines += ["    RECOMMENDATION: reduce blip_thrust_frac or blip_dur_sec."]
            new_frac = round(blip_frac * 0.5, 2)
            lines += [f"    Try: blip_thrust_frac: {new_frac}  (half current)"]
    lines += [""]

    lines += ["── 2. MASS IDENTIFICATION ────────────────────────────────"]
    lines += [f"  Hover T_norm (meas) : {T_norm_meas:.4f}  ({T_norm_meas*100:.1f}%)"]
    lines += [f"  T_total (meas)      : {T_norm_meas*4*T_max:.2f} N"]
    lines += [f"  m identified        : {m_identified:.3f} kg"]
    lines += [f"  m params.yaml       : {m_yaml:.3f} kg"]
    lines += [f"  relative error      : {m_error_pct:.1f} %"]
    lines += [f"  hover az (body)     : {az_hover:.3f} m/s²  (expected ≈ -9.81)"]
    lines += [f"  EKF attitude @ hover: theta={theta_hover_ekf:+.2f}°  "
              f"acc_implied={theta_hover_acc:+.2f}°"]

    m_warn = " ← SIGNIFICANT" if m_error_pct > MASS_ERROR_WARN * 100 else " ← OK"
    lines += [f"  Mass error          : {m_error_pct:.1f}%{m_warn}"]
    if m_error_pct > MASS_ERROR_WARN * 100:
        lines += [f"    RECOMMENDATION: update params.yaml  m: {m_identified:.3f}"]
    lines += [""]

    lines += ["── 3. SET_ATTITUDE_TARGET SIGN CONVENTIONS ───────────────"]
    lines += ["  FRD convention: q>0=nose-UP, p>0=roll-right, r>0=yaw-right(CW)"]
    lines += [""]
    lines += [f"  q=+{Q_TEST:.2f}: Δtheta = {dtheta_plus:+.2f}°  → {q_plus_sign}"]
    lines += [f"  q=-{Q_TEST:.2f}: Δtheta = {dtheta_minus:+.2f}°  → {q_minus_sign}"]
    lines += [f"  p=+{Q_TEST:.2f}: Δphi   = {dphi_plus:+.2f}°  → {p_plus_sign}"]
    lines += [f"  r=+{Q_TEST:.2f}: Δpsi   = {dpsi_plus:+.2f}°  → {r_plus_sign}"]
    lines += [""]

    needs_fix = []
    if "REVERSED" in q_plus_sign or "REVERSED" in q_minus_sign:
        needs_fix.append("q (pitch)")
    if "REVERSED" in p_plus_sign:
        needs_fix.append("p (roll)")
    if "REVERSED" in r_plus_sign:
        needs_fix.append("r (yaw)")

    if needs_fix:
        lines += [f"  SIGN BUG DETECTED: {', '.join(needs_fix)} axes are reversed!"]
        lines += ["  Fix in controller.py _send_attitude_target():"]
        for ax in needs_fix:
            ax_var = ax.split()[0]
            lines += [f"    float(-{ax_var}_des)  instead of  float({ax_var}_des)"]
    else:
        lines += ["  All sign conventions appear CORRECT."]
    lines += [""]

    lines += ["── 4. RATE TRACKING (pitch axis) ─────────────────────────"]
    lines += [f"  q_des  = +{Q_TEST:.2f} rad/s"]
    lines += [f"  q_meas = {q_meas_mean:+.3f} rad/s (mean during PITCH_PLUS)"]
    if not np.isnan(rate_ratio):
        overshoot = (abs(rate_ratio) - 1.0) * 100.0
        tracking = ("TRACKING WELL" if abs(overshoot) < 30
                    else "SIGNIFICANT OVERSHOOT" if overshoot > 30
                    else "UNDERSHOOT")
        lines += [f"  q_meas/q_des = {rate_ratio:+.2f}  ({overshoot:+.0f}% error) → {tracking}"]
        if abs(overshoot) > 50:
            new_katt = round(float(param.get('K_att', 6.0)) / abs(rate_ratio), 2)
            lines += [f"  RECOMMENDATION: reduce K_att in params.yaml to compensate"]
            lines += [f"    Current K_att={param.get('K_att', 6.0)}, rate_ratio={rate_ratio:+.2f}"]
            lines += [f"    Try: K_att: {new_katt}  (= current / rate_ratio)"]
    lines += [""]

    lines += ["── 5. ROOT CAUSE SUMMARY ─────────────────────────────────"]
    if abs(att_error_deg) > ATT_ERROR_WARN_DEG:
        ev = abs(att_error_deg)
        v_error_rate = float(g * np.sin(np.deg2rad(ev)))
        lines += [f"  EKF attitude error {att_error_deg:+.1f}° → false v_dot_N ≈ "
                  f"{v_error_rate:.2f} m/s² → velocity diverges"]
        lines += [f"  Over 10 s: Δv ≈ {v_error_rate*10:.1f} m/s (explains reported ~50 m/s)"]
        lines += [""]
        lines += ["  FIX: lower acc-gate threshold so blip thrust is excluded"]
        lines += ["    mavlink_rx.py: abs(acc_norm - 9.81) < 2.0  (was 3.0)"]
        lines += ["    This blocks the acc update while thrust > m*g + 2*m = ~11.8 m/s²"]
        lines += ["    but re-enables it within ~50 ms after the blip ends as the"]
        lines += ["    motor decays.  The extended window (+0.50 s) is also required."]
    else:
        lines += ["  No significant attitude tracking error detected."]
        lines += ["  EKF velocity divergence may be from a different source."]

    lines += ["", sep, ""]

    report = "\n".join(lines)
    print("\n" + report, flush=True)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report → {report_path}", flush=True)
    print(f"CSV    → {csv_path}", flush=True)

    _plot_ekf_sysid(log_rows, out_dir)


def _plot_ekf_sysid(log_rows, out_dir):
    """Two-panel plot: EKF position and velocity vs time with phase shading."""
    if not log_rows:
        return

    t   = np.array([r['t']  for r in log_rows])
    pN  = np.array([r['pN'] for r in log_rows])
    pE  = np.array([r['pE'] for r in log_rows])
    pD  = np.array([r['pD'] for r in log_rows])
    vN  = np.array([r['vN'] for r in log_rows])
    vE  = np.array([r['vE'] for r in log_rows])
    vD  = np.array([r['vD'] for r in log_rows])
    phases = [r['phase'] for r in log_rows]

    t -= t[0]   # relative time from first sample

    _PC = {
        'WAIT':       '#bbbbbb',
        'BLIP':       '#ff4444',
        'BLIP_OBS':   '#ff9922',
        'HOVER':      '#66cc66',
    }

    def _phase_color(ph):
        if ph in _PC:
            return _PC[ph]
        if ph.startswith('SETTLE'):
            return '#88ccff'
        if 'PITCH_PLUS' in ph:
            return '#4488ff'
        if 'PITCH_MINUS' in ph:
            return '#2255cc'
        if 'ROLL_PLUS' in ph:
            return '#aa44ff'
        if 'YAW' in ph and '+' in ph:
            return '#cc8800'
        if 'YAW' in ph and '-' in ph:
            return '#997700'
        return '#eeeeee'

    def _shade_phases(ax):
        seen = {}
        i = 0
        while i < len(phases):
            ph = phases[i]
            j  = i + 1
            while j < len(phases) and phases[j] == ph:
                j += 1
            t_end = t[j - 1] if j < len(t) else t[-1]
            color = _phase_color(ph)
            ax.axvspan(t[i], t_end, alpha=0.20, color=color, lw=0)
            if ph not in seen:
                seen[ph] = color
            i = j
        return [mpatches.Patch(facecolor=c, alpha=0.6, label=ph)
                for ph, c in seen.items()]

    fig, (ax_pos, ax_vel) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    fig.suptitle("Sysid — EKF Position & Velocity", fontsize=13)

    # ── Position ──────────────────────────────────────────────────────────────
    phase_patches = _shade_phases(ax_pos)
    ln_pN,  = ax_pos.plot(t,  pN,  lw=1.0, color='tab:blue',   label='pN (m)')
    ln_pE,  = ax_pos.plot(t,  pE,  lw=1.0, color='tab:orange', label='pE (m)')
    ln_alt, = ax_pos.plot(t, -pD,  lw=1.0, color='tab:green',  label='alt = -pD (m)')
    ax_pos.axhline(0, color='k', lw=0.5, ls='--')
    ax_pos.set_ylabel("Position (m)")
    ax_pos.set_title("EKF NED Position (dead-reckoned from hover entry)")
    ax_pos.legend(handles=[ln_pN, ln_pE, ln_alt] + phase_patches,
                  loc='upper right', fontsize=7, ncol=2)
    ax_pos.grid(True, alpha=0.3)

    # ── Velocity ──────────────────────────────────────────────────────────────
    phase_patches_v = _shade_phases(ax_vel)
    ln_vN, = ax_vel.plot(t, vN, lw=1.0, color='tab:blue',   label='vN (m/s)')
    ln_vE, = ax_vel.plot(t, vE, lw=1.0, color='tab:orange', label='vE (m/s)')
    ln_vD, = ax_vel.plot(t, vD, lw=1.0, color='tab:green',  label='vD (m/s)')
    ax_vel.axhline(0, color='k', lw=0.5, ls='--')
    ax_vel.set_ylabel("Velocity (m/s)")
    ax_vel.set_xlabel("Time (s)")
    ax_vel.set_title("EKF NED Velocity")
    ax_vel.legend(handles=[ln_vN, ln_vE, ln_vD] + phase_patches_v,
                  loc='upper right', fontsize=7, ncol=2)
    ax_vel.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(out_dir, "ekf_pos_vel.png")
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"EKF plot → {out}", flush=True)


if __name__ == "__main__":
    main()
