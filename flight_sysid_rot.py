#!/usr/bin/env python3
"""
flight_sysid_rot.py  —  In-flight rotational system identification.

Identifies [Dw_xy, Dw_z, kappa, m_motor] by minimising one-step angular-rate
prediction error of dyn.py's rotational equation against ground-truth body
rates, using actual per-motor thrust from ACTUATOR_OUTPUT_STATUS (not the
commanded rate-loop setpoint, which the sim's internal controller may not
track exactly).

Total mass m is held FIXED at its params.yaml value (already confirmed
correct from fit_shadow.py's translational fit) — this is what makes m_motor
identifiable through Ixx/Izz, unlike sysid.py's ground-rig fit which frees m
instead and fixes m_motor.

    Ixx = Iyy = 2*m_motor*L^2 + m_frame*L^2/6      m_frame = m - 4*m_motor
    Izz       = 4*m_motor*L^2 + m_frame*L^2/3

    tau_x = d*(FL+BL-FR-BR)     d = L/sqrt(2)   (dyn.py mixer convention;
    tau_y = d*(FL+FR-BL-BR)      ACTUATOR_OUTPUT_STATUS is already in the
    tau_z = kappa*(FL+BR-FR-BL)  sim-native [FL,FR,BL,BR] order — no reorder)

    omega_dot = I_inv @ (tau - Dw*|omega|*omega - omega x (I @ omega))

Uses the same setup_components() init as main.py / flight_sysid_gt.py and
requires ground_truth_mode: true in params.yaml. Body-rate sign convention
(rollspeed/pitchspeed/yawspeed from ATTITUDE, commanded via SET_ATTITUDE_TARGET
through _send_raw) follows the FRD convention already validated empirically by
flight_sysid_gt.py — no extra negation applied here.

Phase sequence:
  WAIT -> TAKEOFF -> HOVER ->
  ROLL_POS -> ROLL_NEG -> SETTLE_R ->
  PITCH_POS -> PITCH_NEG -> SETTLE_P ->
  YAW_POS -> YAW_NEG -> SETTLE_Y -> KILL

Run:
  python flight_sysid_rot.py            # fly + fit
  python flight_sysid_rot.py --fit-only logs/sysid_rot_<ts>/sysid_rot.csv
  Press 's' to arm and start.
"""

import csv
import sys
import time
import msvcrt
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from dyn import load_params
from setup import setup_components

# ── Tuning constants ─────────────────────────────────────────────────────────

ALT_TARGET_M    = 3.5     # [m]   hover target altitude
TAKEOFF_TIMEOUT = 15.0    # [s]   abort if altitude not reached in time
WAIT_SEC        = 3.5     # [s]   motors-off ground-truth stabilisation window
HOVER_SEC       = 2.5     # [s]   settle before excitation begins
EXC_SEC         = 0.7     # [s]   excitation duration per leg
SETTLE_SEC      = 1.2     # [s]   settle between excitations (level flight)

Q_EXC = 0.5     # [rad/s]  excitation amplitude

KP_ALT = 0.12   # altitude P gain
KD_ALT = 0.15   # altitude D gain
T_MIN  = 0.20   # normalised thrust lower bound
T_MAX  = 0.50   # normalised thrust upper bound
K_ATT  = 3.0    # attitude P gain used while levelling in SETTLE phases

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
    """Send SET_ATTITUDE_TARGET with body rates as-is (no sign correction)."""
    conn.mav.set_attitude_target_send(
        int(time.time() * 1e3) & 0xFFFFFFFF,
        conn.target_system, conn.target_component,
        0x80,                           # type_mask: ignore attitude, use rates
        [1.0, 0.0, 0.0, 0.0],
        float(p), float(q), float(r),
        float(np.clip(thrust_norm, 0.0, 1.0)),
    )


def _send_levelled(conn, mav, thrust_norm):
    """Level the drone with a simple FRD attitude P-controller."""
    phi, theta, _ = _quat_to_euler(mav['quat'])
    p_raw = -K_ATT * phi
    q_raw = -K_ATT * theta
    _send_raw(conn, p_raw, q_raw, 0.0, thrust_norm)
    return p_raw, q_raw


def _kill(conn):
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
    """Return (phi, theta, psi, pos_ned, vel_ned, att_rates) or None if not ready.

    att_rates = (rollspeed, pitchspeed, yawspeed) from the ATTITUDE message —
    ground truth, not IMU-derived, and in the same FRD convention already
    validated by flight_sysid_gt.py's SIGN_* phases.
    """
    mav = shared.get('mav_state')
    if mav is None or mav.get('quat') is None:
        return None
    phi, theta, psi = _quat_to_euler(mav['quat'])
    att = shared.get('mavlink', {}).get('latest', {}).get('ATTITUDE')
    if att is None:
        return None
    att_rates = (att['rollspeed'], att['pitchspeed'], att['yawspeed'])
    return phi, theta, psi, mav['pos_ned'], mav['vel_ned'], att_rates


def _motors(shared):
    """Return the 4 per-motor 0-1 commands [FL,FR,BL,BR] or None if not ready."""
    act = shared.get('mavlink', {}).get('latest', {}).get('ACTUATOR_OUTPUT_STATUS')
    if act is None:
        return None
    return [float(v) for v in act['actuator'][:4]]


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
    'act_fl', 'act_fr', 'act_bl', 'act_br',
    'p_raw', 'q_raw', 'r_raw', 'T_norm',
]

_log_rows = []


def _log_row(phase, t_phase, phi, theta, psi, pos, vel, att_rates, motors,
             p_raw, q_raw, r_raw, T_norm):
    m = motors if motors is not None else [float('nan')] * 4
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
        'act_fl':     round(m[0], 5),
        'act_fr':     round(m[1], 5),
        'act_bl':     round(m[2], 5),
        'act_br':     round(m[3], 5),
        'p_raw':      round(float(p_raw), 5),
        'q_raw':      round(float(q_raw), 5),
        'r_raw':      round(float(r_raw), 5),
        'T_norm':     round(float(T_norm), 5),
    })


def _save_csv():
    if not _log_rows:
        return None
    out_dir  = Path('logs') / ('sysid_rot_' + time.strftime('%Y%m%d_%H%M%S'))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'sysid_rot.csv'
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=_LOG_FIELDS)
        w.writeheader()
        w.writerows(_log_rows)
    print(f'[sysid_rot] CSV -> {out_path}', flush=True)
    return out_dir


# ── Physics helpers (dyn.py's own convention — sim-native [FL,FR,BL,BR]) ────

def _inertia(m_fixed, m_motor, L):
    m_frame = m_fixed - 4.0 * m_motor
    Ixx = 2.0 * m_motor * L**2 + m_frame * L**2 / 6.0
    Izz = 4.0 * m_motor * L**2 + m_frame * L**2 / 3.0
    return Ixx, Izz


def _mixer_torque(u_motors_n, kappa, L):
    """u_motors_n: (N,4) per-motor thrust [N] in sim order [FL,FR,BL,BR]."""
    d = L / np.sqrt(2.0)
    FL, FR, BL, BR = u_motors_n[:, 0], u_motors_n[:, 1], u_motors_n[:, 2], u_motors_n[:, 3]
    return np.stack([
        d * (FL + BL - FR - BR),        # tau_x
        d * (FL + FR - BL - BR),        # tau_y
        kappa * (FL + BR - FR - BL),    # tau_z
    ], axis=1)


def cost_fn(theta, omega, u_motors_n, dt_arr, m_fixed, L):
    """One-step angular-rate prediction MSE. theta = [Dw_xy, Dw_z, kappa, m_motor]."""
    Dw_xy, Dw_z, kappa, m_motor = theta
    if m_motor <= 0 or any(v < 0 for v in theta):
        return 1e9
    Ixx, Izz = _inertia(m_fixed, m_motor, L)
    if Ixx <= 0 or Izz <= 0:
        return 1e9

    I_diag     = np.array([Ixx, Ixx, Izz])
    I_inv_diag = np.array([1.0/Ixx, 1.0/Ixx, 1.0/Izz])
    Dw = np.array([Dw_xy, Dw_xy, Dw_z])

    om  = omega[:-1]
    u   = u_motors_n[:-1]
    dt  = dt_arr[:, None]

    tau       = _mixer_torque(u, kappa, L)
    Io        = om * I_diag
    cross_oIo = np.cross(om, Io)
    damp      = Dw * np.abs(om) * om
    omega_dot = (tau - damp - cross_oIo) * I_inv_diag
    omega_pred = om + omega_dot * dt
    e_omega    = omega_pred - omega[1:]

    return float(np.sum(e_omega**2) / len(dt_arr))


def _simulate_omega(omega0, u_motors_n, dt_arr, theta, m_fixed, L):
    """Forward-integrate omega for plotting (not used by the optimiser)."""
    Dw_xy, Dw_z, kappa, m_motor = theta
    Ixx, Izz = _inertia(m_fixed, m_motor, L)
    I_diag     = np.array([Ixx, Ixx, Izz])
    I_inv_diag = np.array([1.0/Ixx, 1.0/Ixx, 1.0/Izz])
    Dw = np.array([Dw_xy, Dw_xy, Dw_z])

    N = len(u_motors_n)
    om_sim = np.zeros((N, 3))
    om_sim[0] = omega0
    for k in range(N - 1):
        tau_k  = _mixer_torque(u_motors_n[k:k+1], kappa, L)[0]
        Io_k   = om_sim[k] * I_diag
        damp_k = Dw * np.abs(om_sim[k]) * om_sim[k]
        od_k   = (tau_k - damp_k - np.cross(om_sim[k], Io_k)) * I_inv_diag
        om_sim[k+1] = om_sim[k] + od_k * dt_arr[k]
    return om_sim


# ── Fit from logged rows ──────────────────────────────────────────────────────

def _arrays_from_rows(rows):
    """Extract fit-ready arrays from logged rows (WAIT/TAKEOFF/KILL excluded
    by the caller). Drops rows with missing motor/rate data."""
    keep = [r for r in rows
            if r['phase'] not in ('WAIT', 'TAKEOFF', 'KILL')
            and not any(np.isnan(float(r[k])) for k in ('act_fl', 'act_fr', 'act_bl', 'act_br'))]
    t_wall = np.array([r['t_wall'] for r in keep], dtype=float)
    omega  = np.array([[r['rollspeed'], r['pitchspeed'], r['yawspeed']] for r in keep], dtype=float)
    act    = np.array([[r['act_fl'], r['act_fr'], r['act_bl'], r['act_br']] for r in keep], dtype=float)
    dt_arr = np.clip(np.diff(t_wall), 5e-4, 0.05)
    return t_wall, omega, act, dt_arr


def fit_rotational(rows, param):
    m_fixed = float(param['m'])
    L       = float(param['L'])
    T_max   = float(param['T_max_motor'])

    t_wall, omega, act, dt_arr = _arrays_from_rows(rows)
    n = len(dt_arr)
    if n < 20:
        raise ValueError(f'Too few usable samples ({n}) for a rotational fit.')

    u_motors_n = act * T_max   # per-motor thrust [N], sim-native [FL,FR,BL,BR]

    Dw_nom = param['Dw'].diagonal()
    m_motor_nom = float(param.get('m_motor', 0.050))
    theta0 = np.array([Dw_nom[0], Dw_nom[2], float(param['kappa']), m_motor_nom])

    J0 = cost_fn(theta0, omega, u_motors_n, dt_arr, m_fixed, L)
    print(f'\n[sysid_rot] samples={n}  nominal theta={theta0}  nominal cost={J0:.6f}',
          flush=True)

    m_motor_hi = min(0.15, m_fixed / 4.0 * 0.95)
    bounds = [
        (1e-5, 0.5),          # Dw_xy
        (1e-5, 0.5),          # Dw_z
        (1e-4, 0.5),          # kappa
        (0.005, m_motor_hi),  # m_motor  (m held fixed — identifiable via Ixx/Izz)
    ]

    n_calls = [0]
    def cb(theta):
        n_calls[0] += 1
        if n_calls[0] % 20 == 0:
            J = cost_fn(theta, omega, u_motors_n, dt_arr, m_fixed, L)
            print(f'  iter {n_calls[0]:4d}  cost={J:.6f}  m_motor={theta[3]:.4f}  '
                  f'kappa={theta[2]:.5f}  Dw_xy={theta[0]:.5f}  Dw_z={theta[1]:.5f}',
                  flush=True)

    print('[sysid_rot] optimising...', flush=True)
    result = minimize(
        cost_fn, theta0,
        args=(omega, u_motors_n, dt_arr, m_fixed, L),
        method='L-BFGS-B',
        bounds=bounds,
        callback=cb,
        options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8},
    )
    theta_opt = result.x
    J_opt     = result.fun
    print(f'[sysid_rot] optimised cost: {J_opt:.6f}  '
          f'({100*(J0-J_opt)/max(J0,1e-9):+.1f}%)', flush=True)

    om_nom = _simulate_omega(omega[0], u_motors_n, dt_arr, theta0, m_fixed, L)
    om_opt = _simulate_omega(omega[0], u_motors_n, dt_arr, theta_opt, m_fixed, L)

    return theta0, theta_opt, t_wall - t_wall[0], omega, om_nom, om_opt, u_motors_n


def plot_and_report(out_dir, theta0, theta_opt, t_s, omega, om_nom, om_opt, m_fixed):
    labels = ['p  [rad/s]', 'q  [rad/s]', 'r  [rad/s]']
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle('flight_sysid_rot: measured vs model body rates', fontsize=13)
    for i, (ax, lbl) in enumerate(zip(axes, labels)):
        ax.plot(t_s, omega[:, i],  'k',   lw=1.3, label='GT (ATTITUDE)')
        ax.plot(t_s, om_nom[:, i], 'b--', lw=1.0, label='nominal model')
        ax.plot(t_s, om_opt[:, i], 'r-',  lw=1.0, label='optimised model')
        ax.set_ylabel(lbl, fontsize=9)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('time [s]')
    plt.tight_layout()
    out_path = (out_dir / 'sysid_rot_comparison.png') if out_dir is not None \
               else Path('sysid_rot_comparison.png')
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f'[sysid_rot] plot -> {out_path}', flush=True)

    names = ['Dw_xy', 'Dw_z', 'kappa', 'm_motor']
    print(f"\n{'Parameter':<12} {'nominal':>12} {'optimised':>12} {'delta%':>8}")
    for name, nom, opt in zip(names, theta0, theta_opt):
        pct = (opt - nom) / nom * 100 if nom != 0 else float('nan')
        print(f'  {name:<10} {nom:>12.5f} {opt:>12.5f}  {pct:>+8.1f}%')
    print(f"\nm (fixed)      : {m_fixed:.4f} kg")
    print(f"m_motor        : {theta_opt[3]:.4f} kg  (was {theta0[3]:.4f})")
    print('\nSuggested params.yaml updates:')
    print(f"  m_motor: {theta_opt[3]:.4f}")
    print(f"  kappa:   {theta_opt[2]:.6f}")
    print(f"  Dw: [{theta_opt[0]:.6f}, {theta_opt[0]:.6f}, {theta_opt[1]:.6f}]")


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
    T_hover   = m * g_acc / (4.0 * T_max_mot)
    pD_target = -ALT_TARGET_M

    print('Press "s" to arm and start sysid_rot...', flush=True)
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

    aborted = False
    phase   = 'WAIT'
    t_phase = time.time()

    def elapsed():
        return time.time() - t_phase

    def next_phase(name):
        nonlocal phase, t_phase
        phase   = name
        t_phase = time.time()
        print(f'[sysid_rot] -> {name}', flush=True)

    try:
        while not aborted:
            t0    = time.time()
            state = _gt(shared)
            if state is None:
                time.sleep(DT)
                continue
            phi, theta, psi, pos, vel, att_rates = state
            motors = _motors(shared)
            mav    = shared['mav_state']

            if phase not in ('WAIT', 'KILL'):
                reason = _abort_check(phi, theta, att_rates, phase)
                if reason:
                    print(f'[sysid_rot] ABORT in {phase}: {reason}', flush=True)
                    aborted = True
                    _kill(conn)
                    break

            p_raw = q_raw = r_raw = 0.0
            T = T_hover

            if phase == 'WAIT':
                conn.mav.set_actuator_control_target_send(
                    int(time.time() * 1e6),
                    conn.target_system, conn.target_component,
                    0, [0.0]*8,
                )
                if elapsed() >= WAIT_SEC:
                    next_phase('TAKEOFF')
                time.sleep(max(0.0, DT - (time.time() - t0)))
                continue

            elif phase == 'TAKEOFF':
                T = _alt_hold(shared, T_hover, pD_target)
                _send_raw(conn, 0.0, 0.0, 0.0, T)
                if float(pos[2]) < pD_target + 0.15:
                    next_phase('HOVER')
                elif elapsed() > TAKEOFF_TIMEOUT:
                    print('[sysid_rot] TAKEOFF timeout — abort', flush=True)
                    aborted = True
                    _kill(conn)
                    break

            elif phase == 'HOVER':
                T = _alt_hold(shared, T_hover, pD_target)
                T_hover = 0.99 * T_hover + 0.01 * T
                p_raw, q_raw = _send_levelled(conn, mav, T)
                if elapsed() >= HOVER_SEC:
                    next_phase('ROLL_POS')

            elif phase == 'ROLL_POS':
                T = _alt_hold(shared, T_hover, pD_target)
                p_raw = Q_EXC
                _send_raw(conn, p_raw, 0.0, 0.0, T)
                if elapsed() >= EXC_SEC:
                    next_phase('ROLL_NEG')

            elif phase == 'ROLL_NEG':
                T = _alt_hold(shared, T_hover, pD_target)
                p_raw = -Q_EXC
                _send_raw(conn, p_raw, 0.0, 0.0, T)
                if elapsed() >= EXC_SEC:
                    next_phase('SETTLE_R')

            elif phase == 'SETTLE_R':
                T = _alt_hold(shared, T_hover, pD_target)
                p_raw, q_raw = _send_levelled(conn, mav, T)
                if elapsed() >= SETTLE_SEC:
                    next_phase('PITCH_POS')

            elif phase == 'PITCH_POS':
                T = _alt_hold(shared, T_hover, pD_target)
                q_raw = Q_EXC
                _send_raw(conn, 0.0, q_raw, 0.0, T)
                if elapsed() >= EXC_SEC:
                    next_phase('PITCH_NEG')

            elif phase == 'PITCH_NEG':
                T = _alt_hold(shared, T_hover, pD_target)
                q_raw = -Q_EXC
                _send_raw(conn, 0.0, q_raw, 0.0, T)
                if elapsed() >= EXC_SEC:
                    next_phase('SETTLE_P')

            elif phase == 'SETTLE_P':
                T = _alt_hold(shared, T_hover, pD_target)
                p_raw, q_raw = _send_levelled(conn, mav, T)
                if elapsed() >= SETTLE_SEC:
                    next_phase('YAW_POS')

            elif phase == 'YAW_POS':
                T = _alt_hold(shared, T_hover, pD_target)
                r_raw = Q_EXC
                _send_raw(conn, 0.0, 0.0, r_raw, T)
                if elapsed() >= EXC_SEC:
                    next_phase('YAW_NEG')

            elif phase == 'YAW_NEG':
                T = _alt_hold(shared, T_hover, pD_target)
                r_raw = -Q_EXC
                _send_raw(conn, 0.0, 0.0, r_raw, T)
                if elapsed() >= EXC_SEC:
                    next_phase('SETTLE_Y')

            elif phase == 'SETTLE_Y':
                T = _alt_hold(shared, T_hover, pD_target)
                p_raw, q_raw = _send_levelled(conn, mav, T)
                if elapsed() >= SETTLE_SEC:
                    next_phase('KILL')

            elif phase == 'KILL':
                _kill(conn)
                print('[sysid_rot] Motors killed.', flush=True)
                time.sleep(0.3)
                break

            _log_row(phase, elapsed(), phi, theta, psi, pos, vel,
                     att_rates, motors, p_raw, q_raw, r_raw, T)

            sleep_t = DT - (time.time() - t0)
            if sleep_t > 0:
                time.sleep(sleep_t)

    except KeyboardInterrupt:
        print('\n[sysid_rot] Interrupted — killing motors', flush=True)
        _kill(conn)

    finally:
        out_dir = _save_csv()
        if _log_rows:
            try:
                theta0, theta_opt, t_s, omega, om_nom, om_opt, _ = \
                    fit_rotational(_log_rows, param)
                plot_and_report(out_dir, theta0, theta_opt, t_s, omega,
                                 om_nom, om_opt, float(param['m']))
            except ValueError as e:
                print(f'[sysid_rot] fit skipped: {e}', flush=True)
        if logger is not None:
            logger.save()
        for _c in [ts_loop, mavlink_rx, vision_rx]:
            _t = _c.get_thread_for_join()
            if _t is not None:
                _t.join(timeout=1.0)
        print('[sysid_rot] Done.', flush=True)


def _fit_only(csv_path):
    param = load_params('params.yaml')
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    theta0, theta_opt, t_s, omega, om_nom, om_opt, _ = fit_rotational(rows, param)
    plot_and_report(Path(csv_path).parent, theta0, theta_opt, t_s, omega,
                     om_nom, om_opt, float(param['m']))


if __name__ == '__main__':
    if '--fit-only' in sys.argv:
        idx = sys.argv.index('--fit-only')
        if idx + 1 >= len(sys.argv):
            print('Usage: python flight_sysid_rot.py --fit-only <path/to/sysid_rot.csv>')
            sys.exit(1)
        _fit_only(sys.argv[idx + 1])
    else:
        main()
