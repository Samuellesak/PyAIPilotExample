#!/usr/bin/env python3
"""
flight_sysid_gt.py  — Ground-truth system identification for quadcopter sim.

For each rotation axis (p, q, r) this script empirically identifies:
  1. Sign convention of SET_ATTITUDE_TARGET vs FRD body-rate convention
  2. Inner rate-loop gain:  |omega_measured / omega_commanded|

Uses the same setup_components() init as main.py and requires
  ground_truth_mode: true  in params.yaml.

Phase sequence:
  WAIT -> TAKEOFF -> SIGN_PITCH -> SETTLE -> SIGN_ROLL -> SETTLE ->
  SIGN_YAW -> SETTLE -> LEVEL -> HOVER ->
  GAIN_P_POS -> SETTLE -> GAIN_P_NEG -> SETTLE ->
  GAIN_R_POS -> SETTLE -> GAIN_Y_POS -> SETTLE -> KILL

Run (from the repo root):
  python -m sysid.flight_sysid_gt
  Press 's' to arm and start.
"""

import csv
import time
import msvcrt
from pathlib import Path

import numpy as np

from flight_model.dyn import load_params
from setup import setup_components

# ── Tuning constants ─────────────────────────────────────────────────────────

ALT_TARGET_M    = 3.5     # [m]   hover target altitude
TAKEOFF_TIMEOUT = 15.0    # [s]   abort if altitude not reached in time
WAIT_SEC        = 3.5     # [s]   motors-off ground-truth stabilisation window
SIGN_SEC        = 0.5     # [s]   sign-identification excitation duration
SETTLE_SEC      = 1.2     # [s]   settle between excitations (zero rate cmd)
LEVEL_SEC       = 2.5     # [s]   levelling with identified signs
HOVER_SEC       = 3.0     # [s]   stable hover before gain tests
GAIN_SEC        = 0.5     # [s]   gain excitation duration
GAIN_SETTLE_SEC = 1.2     # [s]   settle between gain excitations

Q_SIGN = 0.40   # [rad/s]  excitation amplitude for sign identification
Q_GAIN = 0.50   # [rad/s]  excitation amplitude for gain identification
K_ATT  = 3.0    # [rad/s/rad]  attitude P gain used in levelling

KP_ALT = 0.12   # altitude P gain
KD_ALT = 0.15   # altitude D gain
T_MIN  = 0.20   # normalised thrust lower bound
T_MAX  = 0.50   # normalised thrust upper bound

TILT_LIMIT_DEG   = 55.0   # [deg]    abort if |phi| or |theta| exceeds this
SPIN_LIMIT_RADPS = 8.0    # [rad/s]  abort if any body rate exceeds this

CONTROL_HZ = 100
DT         = 1.0 / CONTROL_HZ

SIM_IP   = "127.0.0.1"
SIM_PORT = 14550


# ── Helpers ──────────────────────────────────────────────────────────────────

def _quat_to_euler(q):
    """ZYX Euler angles (phi, theta, psi) in radians from unit quaternion [qw,qx,qy,qz]."""
    qw, qx, qy, qz = q
    phi   = np.arctan2(2.0*(qw*qx + qy*qz), 1.0 - 2.0*(qx**2 + qy**2))
    theta = np.arcsin(np.clip(2.0*(qw*qy - qz*qx), -1.0, 1.0))
    psi   = np.arctan2(2.0*(qw*qz + qx*qy), 1.0 - 2.0*(qy**2 + qz**2))
    return float(phi), float(theta), float(psi)


def _send_raw(conn, p, q, r, thrust_norm):
    """Send SET_ATTITUDE_TARGET with body rates as-is (no sign correction).

    All sign correction is the caller's responsibility so that the test
    script never hides the sim's actual convention.
    """
    conn.mav.set_attitude_target_send(
        int(time.time() * 1e3) & 0xFFFFFFFF,
        conn.target_system, conn.target_component,
        0x80,                           # type_mask: ignore attitude, use rates
        [1.0, 0.0, 0.0, 0.0],          # dummy quaternion
        float(p), float(q), float(r),
        float(np.clip(thrust_norm, 0.0, 1.0)),
    )


def _send_levelled(conn, mav, sign_p, sign_q, thrust_norm):
    """Level the drone using identified axis signs and FRD P-controller."""
    phi, theta, _ = _quat_to_euler(mav['quat'])
    p_raw = sign_p * (-K_ATT * phi)
    q_raw = sign_q * (-K_ATT * theta)
    _send_raw(conn, p_raw, q_raw, 0.0, thrust_norm)
    return p_raw, q_raw


def _kill(conn):
    """Cut all motors via direct actuator control (bypasses rate loop)."""
    conn.mav.set_actuator_control_target_send(
        int(time.time() * 1e6),
        conn.target_system, conn.target_component,
        0, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )


def _alt_hold(shared, T_hover_norm, pD_ref):
    """PD altitude controller using GT pos/vel. Returns normalised thrust."""
    mav = shared.get('mav_state')
    if mav is None:
        return T_hover_norm
    pD = float(mav['pos_ned'][2])
    vD = float(mav['vel_ned'][2])
    return float(np.clip(
        T_hover_norm + KP_ALT * (pD - pD_ref) + KD_ALT * vD,
        T_MIN, T_MAX,
    ))


def _gt(shared):
    """Return GT state tuple or None if not yet available.

    Returns (phi, theta, psi, pos_ned, vel_ned, att_rates)
    where att_rates = (rollspeed, pitchspeed, yawspeed) from ATTITUDE msg.
    """
    mav = shared.get('mav_state')
    if mav is None or mav.get('quat') is None:
        return None
    phi, theta, psi = _quat_to_euler(mav['quat'])
    att = shared.get('mavlink', {}).get('latest', {}).get('ATTITUDE')
    if att is not None:
        att_rates = (att['rollspeed'], att['pitchspeed'], att['yawspeed'])
    else:
        r = mav.get('rates', (0.0, 0.0, 0.0))
        att_rates = (float(r[0]), float(r[1]), float(r[2]))
    return phi, theta, psi, mav['pos_ned'], mav['vel_ned'], att_rates


def _abort_check(phi, theta, att_rates, phase):
    if abs(np.degrees(phi)) > TILT_LIMIT_DEG:
        return f'|phi|={np.degrees(phi):.1f}° > {TILT_LIMIT_DEG}°'
    if abs(np.degrees(theta)) > TILT_LIMIT_DEG:
        return f'|theta|={np.degrees(theta):.1f}° > {TILT_LIMIT_DEG}°'
    for i, r in enumerate(att_rates):
        if abs(r) > SPIN_LIMIT_RADPS:
            return f'|rate[{i}]|={abs(r):.2f} > {SPIN_LIMIT_RADPS} rad/s'
    return None


# ── CSV logging ──────────────────────────────────────────────────────────────

_LOG_FIELDS = [
    'phase', 't_wall', 't_phase',
    'phi_deg', 'theta_deg', 'psi_deg',
    'vN', 'vE', 'vD', 'pN', 'pE', 'pD',
    'rollspeed', 'pitchspeed', 'yawspeed',
    'gx_imu', 'gy_imu', 'gz_imu',
    'p_raw', 'q_raw', 'r_raw', 'T_norm',
]

_log_rows = []


def _log_row(phase, t_phase, phi, theta, psi, pos, vel, att_rates, imu,
             p_raw, q_raw, r_raw, T_norm):
    _log_rows.append({
        'phase':      phase,
        't_wall':     round(time.time(), 4),
        't_phase':    round(t_phase, 4),
        'phi_deg':    round(np.degrees(phi), 4),
        'theta_deg':  round(np.degrees(theta), 4),
        'psi_deg':    round(np.degrees(psi), 4),
        'vN':         round(float(vel[0]), 4),
        'vE':         round(float(vel[1]), 4),
        'vD':         round(float(vel[2]), 4),
        'pN':         round(float(pos[0]), 4),
        'pE':         round(float(pos[1]), 4),
        'pD':         round(float(pos[2]), 4),
        'rollspeed':  round(float(att_rates[0]), 5),
        'pitchspeed': round(float(att_rates[1]), 5),
        'yawspeed':   round(float(att_rates[2]), 5),
        'gx_imu':     round(float(imu.get('gx', 0.0)), 5) if imu else 0.0,
        'gy_imu':     round(float(imu.get('gy', 0.0)), 5) if imu else 0.0,
        'gz_imu':     round(float(imu.get('gz', 0.0)), 5) if imu else 0.0,
        'p_raw':      round(float(p_raw), 5),
        'q_raw':      round(float(q_raw), 5),
        'r_raw':      round(float(r_raw), 5),
        'T_norm':     round(float(T_norm), 5),
    })


def _save_csv():
    if not _log_rows:
        return
    out_dir  = Path('logs') / ('sysid_gt_' + time.strftime('%Y%m%d_%H%M%S'))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'sysid_gt.csv'
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=_LOG_FIELDS)
        w.writeheader()
        w.writerows(_log_rows)
    print(f'[sysid_gt] CSV -> {out_path}', flush=True)


# ── Post-flight analysis ─────────────────────────────────────────────────────

def _analyze(sign_p, sign_q, sign_r, T_hover_est, param):
    m_p     = float(param.get('m', 5.405))
    T_max   = float(param.get('T_max_motor', 49.9))
    g_acc   = float(param.get('g', 9.81))
    m_est   = T_hover_est * 4.0 * T_max / g_acc

    def phase_rows(name):
        rows = [r for r in _log_rows if r['phase'] == name]
        skip = max(1, len(rows) // 10)
        return rows[skip:]

    def angle_dot(rows, key):
        """Mean rate of change of a logged angle (deg/s), wrap-safe for yaw."""
        if len(rows) < 2:
            return 0.0
        angles = np.radians([r[key] for r in rows])
        if key == 'psi_deg':
            diffs = np.angle(np.exp(1j * np.diff(angles)))
        else:
            diffs = np.diff(angles)
        return float(np.degrees(np.mean(diffs) / DT))

    def mean_col(rows, key):
        return float(np.mean([r[key] for r in rows])) if rows else float('nan')

    # ── Sign confirmation from SIGN_* phases ──────────────────────────────
    sp = phase_rows('SIGN_PITCH')
    sr = phase_rows('SIGN_ROLL')
    sy = phase_rows('SIGN_YAW')

    tdot = angle_dot(sp, 'theta_deg')
    pdot = angle_dot(sr, 'phi_deg')
    rdot = angle_dot(sy, 'psi_deg')

    conf_q = int(np.sign(tdot)) if sp and abs(tdot) > 0.5 else sign_q
    conf_p = int(np.sign(pdot)) if sr and abs(pdot) > 0.5 else sign_p
    conf_r = int(np.sign(rdot)) if sy and abs(rdot) > 0.5 else sign_r

    pitch_sp = mean_col(sp, 'pitchspeed')
    roll_sp  = mean_col(sr, 'rollspeed')
    yaw_sp   = mean_col(sy, 'yawspeed')

    # ── Gain from GAIN_* phases ──────────────────────────────────────────
    def axis_gain(pos_phase, neg_phase, rate_col):
        rp = phase_rows(pos_phase)
        rn = phase_rows(neg_phase)
        vals = []
        if rp:
            vals.append(abs(mean_col(rp, rate_col)))
        if rn:
            vals.append(abs(mean_col(rn, rate_col)))
        return float(np.mean(vals)) / Q_GAIN if vals else float('nan')

    k_pitch = axis_gain('GAIN_P_POS', 'GAIN_P_NEG', 'pitchspeed')
    k_roll  = axis_gain('GAIN_R_POS', 'GAIN_R_POS', 'rollspeed')
    k_yaw   = axis_gain('GAIN_Y_POS', 'GAIN_Y_POS', 'yawspeed')

    # ── Print report ──────────────────────────────────────────────────────
    def sign_str(s):
        return '+1  FRD-compatible (no correction needed)' if s > 0 \
               else '-1  REVERSED (negate before sending)'

    def k_str(k):
        return f'{k:.3f}' if not np.isnan(k) else 'n/a (phase not reached)'

    def fix_str(s, ax):
        return f'  {ax}: negate   (sign={s:+d})' if s < 0 \
               else f'  {ax}: no change (sign={s:+d})'

    print()
    print('=' * 60)
    print('  SYSID_GT RESULTS')
    print('=' * 60)
    print(f'  {"Axis":<8} {"Sign result":<45} {"Rate gain |meas/cmd|"}')
    print(f'  {"pitch (q)":<8} {sign_str(conf_q):<45} {k_str(k_pitch)}')
    print(f'  {"roll  (p)":<8} {sign_str(conf_p):<45} {k_str(k_roll)}')
    print(f'  {"yaw   (r)":<8} {sign_str(conf_r):<45} {k_str(k_yaw)}')
    print()
    print(f'  ATTITUDE pitchspeed at q_raw=+{Q_SIGN}: {pitch_sp:.4f} rad/s')
    print(f'  ATTITUDE rollspeed  at p_raw=+{Q_SIGN}: {roll_sp:.4f} rad/s')
    print(f'  ATTITUDE yawspeed   at r_raw=+{Q_SIGN}: {yaw_sp:.4f} rad/s')
    print()
    print(f'  Estimated mass:     {m_est:.3f} kg  (params.yaml: {m_p:.3f} kg)')
    print(f'  Hover thrust frac:  {T_hover_est:.4f}')
    print()
    print('  Recommended _send_attitude_target fix:')
    print(fix_str(conf_p, 'p'))
    print(fix_str(conf_q, 'q'))
    print(fix_str(conf_r, 'r'))
    print('=' * 60)
    print()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    param = load_params('params.yaml')

    system_boot_ms = int(time.time() * 1000)
    shared         = {}
    components     = setup_components(shared, system_boot_ms, SIM_IP, SIM_PORT)
    conn           = components['sim_conn']
    mavlink_rx     = components['mavlink_rx']
    controller     = components['controller']
    ts_loop        = components['ts_loop']
    vision_rx      = components['vision_rx']
    logger         = components.get('logger')

    m         = float(param.get('m', 5.405))
    g_acc     = float(param.get('g', 9.81))
    T_max_mot = float(param.get('T_max_motor', 49.9))
    T_hover   = m * g_acc / (4.0 * T_max_mot)   # running estimate, updated in HOVER
    pD_target = -ALT_TARGET_M

    print('Press "s" to arm and start sysid_gt...', flush=True)
    while True:
        if msvcrt.kbhit() and msvcrt.getwch().lower() == 's':
            break
        time.sleep(0.05)

    print('Resetting sim...', flush=True)
    controller.send_sim_reset_command()
    time.sleep(2.0)
    mavlink_rx.request_ground_truth_streams(rate_hz=50)

    print('Arming...', flush=True)
    controller.arm()
    time.sleep(1.0)
    if logger is not None:
        logger.reset_flight_data()

    # Identified signs; set to +1 (no correction) until empirically determined.
    sign_p = sign_q = sign_r = 1

    # Per-sign-phase sample buffers: list of (angle_deg, att_rate_rad_s)
    buf_pitch = []   # (theta_deg, pitchspeed)
    buf_roll  = []   # (phi_deg,   rollspeed)
    buf_yaw   = []   # (psi_deg,   yawspeed)

    aborted = False
    phase   = 'WAIT'
    t_phase = time.time()

    def elapsed():
        return time.time() - t_phase

    def next_phase(name):
        nonlocal phase, t_phase
        phase   = name
        t_phase = time.time()
        print(f'[sysid_gt] -> {name}', flush=True)

    # ── Control loop ─────────────────────────────────────────────────────────
    try:
        while not aborted:
            t0   = time.time()
            state = _gt(shared)

            if state is None:
                time.sleep(DT)
                continue

            phi, theta, psi, pos, vel, att_rates = state
            imu = shared.get('imu_raw')
            mav = shared['mav_state']

            # Abort guard — skipped in WAIT and KILL (motors off / just cut)
            if phase not in ('WAIT', 'KILL'):
                reason = _abort_check(phi, theta, att_rates, phase)
                if reason:
                    print(f'[sysid_gt] ABORT in {phase}: {reason}', flush=True)
                    aborted = True
                    _kill(conn)
                    break

            p_raw = q_raw = r_raw = 0.0
            T = T_hover

            # ── WAIT ─────────────────────────────────────────────────────────
            if phase == 'WAIT':
                conn.mav.set_actuator_control_target_send(
                    int(time.time() * 1e6),
                    conn.target_system, conn.target_component,
                    0, [0.0]*8,
                )
                if elapsed() >= WAIT_SEC:
                    next_phase('TAKEOFF')
                time.sleep(max(0.0, DT - (time.time() - t0)))
                continue   # skip logging during WAIT

            # ── TAKEOFF: zero angular-rate cmd + altitude hold ───────────────
            elif phase == 'TAKEOFF':
                T = _alt_hold(shared, T_hover, pD_target)
                _send_raw(conn, 0.0, 0.0, 0.0, T)
                if float(pos[2]) < pD_target + 0.15:
                    next_phase('SIGN_PITCH')
                elif elapsed() > TAKEOFF_TIMEOUT:
                    print('[sysid_gt] TAKEOFF timeout — abort', flush=True)
                    aborted = True
                    _kill(conn)
                    break

            # ── SIGN_PITCH: command +q_raw, measure theta_dot ───────────────
            elif phase == 'SIGN_PITCH':
                T = _alt_hold(shared, T_hover, pD_target)
                q_raw = Q_SIGN
                _send_raw(conn, 0.0, q_raw, 0.0, T)
                buf_pitch.append((np.degrees(theta), att_rates[1]))
                if elapsed() >= SIGN_SEC:
                    skip = max(1, len(buf_pitch) // 10)
                    buf  = buf_pitch[skip:]
                    if len(buf) >= 2:
                        tdot   = float(np.mean(np.diff([b[0] for b in buf])) / DT)
                        ps_avg = float(np.mean([b[1] for b in buf]))
                        sign_q = int(np.sign(tdot)) if abs(tdot) > 0.5 else 1
                        print(f'[sysid_gt] SIGN_PITCH  theta_dot={tdot:+.1f} deg/s '
                              f'pitchspeed={ps_avg:+.3f} rad/s  '
                              f'sign_q={sign_q:+d}', flush=True)
                    next_phase('SETTLE_SP')

            elif phase == 'SETTLE_SP':
                T = _alt_hold(shared, T_hover, pD_target)
                _send_raw(conn, 0.0, 0.0, 0.0, T)
                if elapsed() >= SETTLE_SEC:
                    next_phase('SIGN_ROLL')

            # ── SIGN_ROLL: command +p_raw, measure phi_dot ──────────────────
            elif phase == 'SIGN_ROLL':
                T = _alt_hold(shared, T_hover, pD_target)
                p_raw = Q_SIGN
                _send_raw(conn, p_raw, 0.0, 0.0, T)
                buf_roll.append((np.degrees(phi), att_rates[0]))
                if elapsed() >= SIGN_SEC:
                    skip = max(1, len(buf_roll) // 10)
                    buf  = buf_roll[skip:]
                    if len(buf) >= 2:
                        pdot   = float(np.mean(np.diff([b[0] for b in buf])) / DT)
                        rs_avg = float(np.mean([b[1] for b in buf]))
                        sign_p = int(np.sign(pdot)) if abs(pdot) > 0.5 else 1
                        print(f'[sysid_gt] SIGN_ROLL   phi_dot={pdot:+.1f} deg/s '
                              f'rollspeed={rs_avg:+.3f} rad/s  '
                              f'sign_p={sign_p:+d}', flush=True)
                    next_phase('SETTLE_SR')

            elif phase == 'SETTLE_SR':
                T = _alt_hold(shared, T_hover, pD_target)
                _send_raw(conn, 0.0, 0.0, 0.0, T)
                if elapsed() >= SETTLE_SEC:
                    next_phase('SIGN_YAW')

            # ── SIGN_YAW: command +r_raw, measure psi_dot ───────────────────
            elif phase == 'SIGN_YAW':
                T = _alt_hold(shared, T_hover, pD_target)
                r_raw = Q_SIGN
                _send_raw(conn, 0.0, 0.0, r_raw, T)
                buf_yaw.append((psi, att_rates[2]))   # store rad for wrap-safe diff
                if elapsed() >= SIGN_SEC:
                    skip = max(1, len(buf_yaw) // 10)
                    buf  = buf_yaw[skip:]
                    if len(buf) >= 2:
                        psis = np.array([b[0] for b in buf])
                        # wrap-safe difference: map to [-pi, pi]
                        rdot   = float(np.degrees(
                            np.mean(np.angle(np.exp(1j * np.diff(psis)))) / DT))
                        ys_avg = float(np.mean([b[1] for b in buf]))
                        sign_r = int(np.sign(rdot)) if abs(rdot) > 0.5 else 1
                        print(f'[sysid_gt] SIGN_YAW    psi_dot={rdot:+.1f} deg/s '
                              f'yawspeed={ys_avg:+.3f} rad/s  '
                              f'sign_r={sign_r:+d}', flush=True)
                    next_phase('SETTLE_SY')

            elif phase == 'SETTLE_SY':
                T = _alt_hold(shared, T_hover, pD_target)
                _send_raw(conn, 0.0, 0.0, 0.0, T)
                if elapsed() >= SETTLE_SEC:
                    next_phase('LEVEL')

            # ── LEVEL: apply identified signs to drive phi/theta -> 0 ───────
            elif phase == 'LEVEL':
                T = _alt_hold(shared, T_hover, pD_target)
                p_raw, q_raw = _send_levelled(conn, mav, sign_p, sign_q, T)
                if elapsed() >= LEVEL_SEC:
                    next_phase('HOVER')

            # ── HOVER: stable hover; update T_hover running estimate ─────────
            elif phase == 'HOVER':
                T = _alt_hold(shared, T_hover, pD_target)
                T_hover = 0.99 * T_hover + 0.01 * T   # slow IIR update
                p_raw, q_raw = _send_levelled(conn, mav, sign_p, sign_q, T)
                if elapsed() >= HOVER_SEC:
                    next_phase('GAIN_P_POS')

            # ── GAIN_P_POS/NEG: pitch rate gain ─────────────────────────────
            elif phase == 'GAIN_P_POS':
                T = _alt_hold(shared, T_hover, pD_target)
                q_raw = sign_q * Q_GAIN
                _send_raw(conn, 0.0, q_raw, 0.0, T)
                if elapsed() >= GAIN_SEC:
                    next_phase('SETTLE_GP')

            elif phase == 'SETTLE_GP':
                T = _alt_hold(shared, T_hover, pD_target)
                p_raw, q_raw = _send_levelled(conn, mav, sign_p, sign_q, T)
                if elapsed() >= GAIN_SETTLE_SEC:
                    next_phase('GAIN_P_NEG')

            elif phase == 'GAIN_P_NEG':
                T = _alt_hold(shared, T_hover, pD_target)
                q_raw = sign_q * (-Q_GAIN)
                _send_raw(conn, 0.0, q_raw, 0.0, T)
                if elapsed() >= GAIN_SEC:
                    next_phase('SETTLE_GN')

            elif phase == 'SETTLE_GN':
                T = _alt_hold(shared, T_hover, pD_target)
                p_raw, q_raw = _send_levelled(conn, mav, sign_p, sign_q, T)
                if elapsed() >= GAIN_SETTLE_SEC:
                    next_phase('GAIN_R_POS')

            # ── GAIN_R_POS: roll rate gain ───────────────────────────────────
            elif phase == 'GAIN_R_POS':
                T = _alt_hold(shared, T_hover, pD_target)
                p_raw = sign_p * Q_GAIN
                _send_raw(conn, p_raw, 0.0, 0.0, T)
                if elapsed() >= GAIN_SEC:
                    next_phase('SETTLE_GR')

            elif phase == 'SETTLE_GR':
                T = _alt_hold(shared, T_hover, pD_target)
                p_raw, q_raw = _send_levelled(conn, mav, sign_p, sign_q, T)
                if elapsed() >= GAIN_SETTLE_SEC:
                    next_phase('GAIN_Y_POS')

            # ── GAIN_Y_POS: yaw rate gain ─────────────────────────────────────
            elif phase == 'GAIN_Y_POS':
                T = _alt_hold(shared, T_hover, pD_target)
                r_raw = sign_r * Q_GAIN
                _send_raw(conn, 0.0, 0.0, r_raw, T)
                if elapsed() >= GAIN_SEC:
                    next_phase('SETTLE_GY')

            elif phase == 'SETTLE_GY':
                T = _alt_hold(shared, T_hover, pD_target)
                p_raw, q_raw = _send_levelled(conn, mav, sign_p, sign_q, T)
                if elapsed() >= GAIN_SETTLE_SEC:
                    next_phase('KILL')

            # ── KILL ─────────────────────────────────────────────────────────
            elif phase == 'KILL':
                _kill(conn)
                print('[sysid_gt] Motors killed.', flush=True)
                time.sleep(0.3)
                break

            _log_row(phase, elapsed(), phi, theta, psi, pos, vel,
                     att_rates, imu, p_raw, q_raw, r_raw, T)

            sleep_t = DT - (time.time() - t0)
            if sleep_t > 0:
                time.sleep(sleep_t)

    except KeyboardInterrupt:
        print('\n[sysid_gt] Interrupted — killing motors', flush=True)
        _kill(conn)

    finally:
        _analyze(sign_p, sign_q, sign_r, T_hover, param)
        _save_csv()
        if logger is not None:
            logger.save()
        for _c in [ts_loop, mavlink_rx, vision_rx]:
            _t = _c.get_thread_for_join()
            if _t is not None:
                _t.join(timeout=1.0)
        print('[sysid_gt] Done.', flush=True)


if __name__ == '__main__':
    main()
