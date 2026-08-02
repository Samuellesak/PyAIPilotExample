#!/usr/bin/env python3
"""
flight_sysid_vel.py  —  Velocity dynamics identification for quadcopter sim.

Identifies the first-order tilt-to-velocity transfer function per NED axis.

Tilt-to-velocity coupling depends on initial_yaw_deg (ψ₀) from params.yaml:

    dv_N/dt ≈  g·cos(ψ₀)·sin(θ)  −  v_N/τ_v
    dv_E/dt ≈  g·cos(ψ₀)·sin(φ)  −  v_E/τ_v

    For ψ₀ = 180° (params.yaml default):  cos(ψ₀) = −1
        positive θ → SOUTH accel  (a_drive_N = −g·sin θ)
        positive φ → WEST  accel  (a_drive_E = −g·sin φ)

Tilt direction sign is derived from sign(cos(ψ₀)) so the script works for any heading:
    TILT_N_POS: θ = sign(cos ψ₀) · THETA_VEL_DEG  → positive v_N
    TILT_N_NEG: θ = −sign(cos ψ₀) · THETA_VEL_DEG → negative v_N
    TILT_E_POS: φ = sign(cos ψ₀) · THETA_VEL_DEG  → positive v_E

Parameters identified:
    τ_v   [s]         velocity time constant   (= m / Dv)
    Dv    [N·s/m]     linear drag coefficient
    v_ss  [m/s]       steady-state velocity at THETA_VEL_DEG tilt

From these the script computes critically-damped velocity PI gains for params.yaml.

Phase sequence:
    WAIT → TAKEOFF → LEVEL → HOVER →
    TILT_N_POS → DECEL_N1 →
    TILT_N_NEG → DECEL_N2 →
    TILT_E_POS → DECEL_E1 →
    KILL → analysis

Sign corrections (empirically confirmed by flight_sysid_gt.py):
    p  →  as-is (FRD-compatible)
    q  →  negated before sending to sim
    r  →  negated before sending to sim

Requires: ground_truth_mode: true  in params.yaml.

Run (from the repo root):
    python -m sysid.flight_sysid_vel
    Press 's' to arm and start.
"""

import csv
import time
import msvcrt
from pathlib import Path

import numpy as np

from flight_model.dyn import load_params
from setup import setup_components

# ── Timing ───────────────────────────────────────────────────────────────────

WAIT_SEC        = 3.5    # [s]  GT stabilisation on ground
TAKEOFF_TIMEOUT = 15.0   # [s]  abort if altitude not reached
LEVEL_SEC       = 3.0    # [s]  attitude levelling after TAKEOFF
HOVER_SEC       = 15.0    # [s]  stable hover + v→0 before velocity tests
TILT_SEC        = 7.0    # [s]  hold tilt  (needs >> τ_v to capture steady-state)
DECEL_SEC       = 30.0   # [s]  return level + wait for velocity to decay (max timeout)
DECEL_V_THRESH  = 0.20   # [m/s]  early-exit DECEL when |v| drops below this
HOVER_V_THRESH  = 0.20   # [m/s]  max horizontal speed accepted before tilt tests

# ── Excitation ────────────────────────────────────────────────────────────────

THETA_VEL_DEG = 10.0    # [deg]  tilt angle for all velocity tests
ALT_TARGET_M  = 3.5     # [m]   hover altitude

# ── Control ───────────────────────────────────────────────────────────────────

K_ATT  = 3.0    # [rad/s/rad]  attitude P gain
KP_ALT = 0.12
KD_ALT = 0.15
T_MIN  = 0.20
T_MAX  = 0.55   # slightly higher than hover to allow tilt compensation

TILT_LIMIT_DEG   = 55.0
SPIN_LIMIT_RADPS = 8.0
CONTROL_HZ       = 100
DT               = 1.0 / CONTROL_HZ

# ── Confirmed sign corrections (flight_sysid_gt.py) ──────────────────────────
#   sign_p = +1  →  p_sim = +p_frd   (FRD-compatible)
#   sign_q = -1  →  q_sim = -q_frd   (reversed in sim)
#   sign_r = -1  →  r_sim = -r_frd   (reversed in sim)

# ── Gain design target ────────────────────────────────────────────────────────

T_SETTLE_DES = 3.0   # [s]  desired velocity loop settling time (critically damped)

SIM_IP   = "127.0.0.1"
SIM_PORT = 14550


# ── Core helpers ─────────────────────────────────────────────────────────────

def _quat_to_euler(q):
    qw, qx, qy, qz = q
    phi   = np.arctan2(2.0*(qw*qx + qy*qz), 1.0 - 2.0*(qx**2 + qy**2))
    theta = np.arcsin(np.clip(2.0*(qw*qy - qz*qx), -1.0, 1.0))
    psi   = np.arctan2(2.0*(qw*qz + qx*qy), 1.0 - 2.0*(qy**2 + qz**2))
    return float(phi), float(theta), float(psi)


def _send_to_sim(conn, p_sim, q_sim, r_sim, thrust_norm):
    """Send body-rate setpoint directly to sim (sign corrections already applied)."""
    conn.mav.set_attitude_target_send(
        int(time.time() * 1e3) & 0xFFFFFFFF,
        conn.target_system, conn.target_component,
        0x80,
        [1.0, 0.0, 0.0, 0.0],
        float(p_sim), float(q_sim), float(r_sim),
        float(np.clip(thrust_norm, 0.0, 1.0)),
    )


def _kill(conn):
    conn.mav.set_actuator_control_target_send(
        int(time.time() * 1e6),
        conn.target_system, conn.target_component,
        0, [0.0] * 8,
    )


def _level(mav):
    """Compute sim-frame rate commands to drive phi,theta → 0.

    Returns (p_sim, q_sim, phi, theta).
    Applies sign corrections: p→as-is, q→negated.
    """
    phi, theta, _ = _quat_to_euler(mav['quat'])
    # FRD P-control:  p_frd = -K*(phi-0),  q_frd = -K*(theta-0)
    # Sim convention: p_sim = +p_frd,       q_sim = -q_frd
    p_sim = +1.0 * (-K_ATT * phi)
    q_sim = -1.0 * (-K_ATT * theta)    # = K_ATT * theta  (sign flip for pitch)
    return p_sim, q_sim, phi, theta


def _tilt_hold(mav, phi_target, theta_target):
    """Compute sim-frame rate commands to hold a specific tilt angle.

    Returns (p_sim, q_sim, phi, theta).
    """
    phi, theta, _ = _quat_to_euler(mav['quat'])
    # FRD P-control toward target:
    p_frd = -K_ATT * (phi   - phi_target)
    q_frd = -K_ATT * (theta - theta_target)
    # Apply sign corrections:
    p_sim = +1.0 * p_frd
    q_sim = -1.0 * q_frd
    return p_sim, q_sim, phi, theta


def _alt_hold_thrust(shared, T_hover, pD_ref, phi, theta):
    """PD altitude hold with tilt-compensation thrust boost."""
    mav = shared.get('mav_state')
    if mav is None:
        return T_hover
    pD = float(mav['pos_ned'][2])
    vD = float(mav['vel_ned'][2])
    T_pd = T_hover + KP_ALT * (pD - pD_ref) + KD_ALT * vD
    cos_tilt = max(np.cos(phi) * np.cos(theta), 0.5)
    return float(np.clip(T_pd / cos_tilt, T_MIN, T_MAX))


def _gt(shared):
    """Return (phi, theta, psi, pos_ned, vel_ned, att_rates) or None."""
    mav = shared.get('mav_state')
    if mav is None or mav.get('quat') is None:
        return None
    phi, theta, psi = _quat_to_euler(mav['quat'])
    att = shared.get('mavlink', {}).get('latest', {}).get('ATTITUDE')
    att_rates = (att['rollspeed'], att['pitchspeed'], att['yawspeed']) \
                if att else (0.0, 0.0, 0.0)
    return phi, theta, psi, mav['pos_ned'], mav['vel_ned'], att_rates


def _abort(phi, theta, att_rates):
    if abs(np.degrees(phi)) > TILT_LIMIT_DEG:
        return f'|phi|={np.degrees(phi):.1f}°'
    if abs(np.degrees(theta)) > TILT_LIMIT_DEG:
        return f'|theta|={np.degrees(theta):.1f}°'
    for i, r in enumerate(att_rates):
        if abs(r) > SPIN_LIMIT_RADPS:
            return f'|rate[{i}]|={abs(r):.2f} rad/s'
    return None


# ── Logging ───────────────────────────────────────────────────────────────────

_LOG_FIELDS = [
    'phase', 't_wall', 't_phase',
    'phi_deg', 'theta_deg', 'psi_deg',
    'vN', 'vE', 'vD', 'pN', 'pE', 'pD',
    'rollspeed', 'pitchspeed', 'yawspeed',
    'p_sim', 'q_sim', 'r_sim', 'T_norm',
]
_log_rows = []


def _log_row(phase, t_phase, phi, theta, psi, pos, vel, att_rates,
             p_sim, q_sim, r_sim, T):
    _log_rows.append({
        'phase':      phase,
        't_wall':     round(time.time(), 4),
        't_phase':    round(t_phase, 4),
        'phi_deg':    round(np.degrees(phi), 4),
        'theta_deg':  round(np.degrees(theta), 4),
        'psi_deg':    round(np.degrees(psi), 4),
        'vN':         round(float(vel[0]), 5),
        'vE':         round(float(vel[1]), 5),
        'vD':         round(float(vel[2]), 5),
        'pN':         round(float(pos[0]), 4),
        'pE':         round(float(pos[1]), 4),
        'pD':         round(float(pos[2]), 4),
        'rollspeed':  round(float(att_rates[0]), 5),
        'pitchspeed': round(float(att_rates[1]), 5),
        'yawspeed':   round(float(att_rates[2]), 5),
        'p_sim':      round(float(p_sim), 5),
        'q_sim':      round(float(q_sim), 5),
        'r_sim':      round(float(r_sim), 5),
        'T_norm':     round(float(T), 5),
    })


def _save_csv():
    if not _log_rows:
        return
    out_dir = Path('logs') / ('sysid_vel_' + time.strftime('%Y%m%d_%H%M%S'))
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / 'sysid_vel.csv'
    with open(p, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=_LOG_FIELDS)
        w.writeheader()
        w.writerows(_log_rows)
    print(f'[sysid_vel] CSV  ->  {p}', flush=True)


# ── Model fitting ─────────────────────────────────────────────────────────────

def _fit_tau(t_arr, v_arr, a_drive):
    """Fit  v(t) = v_ss*(1-exp(-t/τ)) + v0*exp(-t/τ)  to velocity data.

    a_drive [m/s²] = effective horizontal acceleration = -g*sin(angle_mean).
    Sign convention: North-facing drone, θ>0 → South → a_drive_N = -g*sin(θ).
    Returns (tau_v, v_ss, v0, rms_fit) or (None,)*4 on failure.
    """
    if len(t_arr) < 5 or abs(a_drive) < 0.01:
        return None, None, None, None

    t = t_arr - t_arr[0]

    # Rough steady-state estimate from last quarter
    n_ss  = max(3, len(v_arr) // 4)
    v_ss0 = float(np.mean(v_arr[-n_ss:]))
    if abs(v_ss0) < 0.05:
        return None, None, None, None
    tau0  = abs(v_ss0 / a_drive)

    def _model(t, tau_v, v0):
        tau_v = abs(tau_v)
        v_ss  = a_drive * tau_v
        return v_ss * (1.0 - np.exp(-t / tau_v)) + v0 * np.exp(-t / tau_v)

    try:
        from scipy.optimize import curve_fit
        (tau_v, v0), _ = curve_fit(
            _model, t, v_arr,
            p0=[tau0, float(v_arr[0])],
            bounds=([0.05, -15.0], [60.0, 15.0]),
            maxfev=20000,
        )
    except Exception:
        tau_v = tau0
        v0    = float(v_arr[0])

    tau_v = abs(tau_v)
    v_ss  = a_drive * tau_v
    v_pred = _model(t, tau_v, v0)
    rms   = float(np.sqrt(np.mean((v_arr - v_pred) ** 2)))
    return float(tau_v), float(v_ss), float(v0), rms


def _pi_gains(tau_v, g=9.81, T_settle=T_SETTLE_DES):
    """Compute velocity PI gains for critically-damped closed-loop response.

    Plant (North-facing drone): dv_N/dt = −g·sin(θ) − v_N/τ_v ≈ −g·θ − v_N/τ_v
    With the sign convention in the controller (v_err → θ_des, positive θ_des
    commands South acceleration), the effective gain is −g, so the gain formula
    uses |g| and the controller negates the desired tilt internally.
    PI output: θ_des = Kp·e_v + Ki·∫e_v dt.

    Closed-loop characteristic equation:
        s²  +  (g·Kp + 1/τ_v)·s  +  g·Ki  =  0
    Desired poles at  s = −ω_n  (double):
        ω_n  = 2 / T_settle              (critically damped)
        Kp   = (2·ω_n − 1/τ_v) / g
        Ki   = ω_n² / g
    """
    omega_n = 2.0 / T_settle
    Kp = (2.0 * omega_n - 1.0 / tau_v) / g
    Ki = omega_n ** 2 / g
    Kp = max(0.005, Kp)   # clamp: avoid negative when τ_v is large
    return float(Kp), float(Ki)


# ── Terminal analysis ─────────────────────────────────────────────────────────

def _analyze(m, psi_0, g=9.81):
    cos_psi = float(np.cos(psi_0))
    print()
    print('=' * 64)
    print('  SYSID_VEL  —  VELOCITY DYNAMICS IDENTIFICATION RESULTS')
    print(f'  ψ₀ = {np.degrees(psi_0):.1f}°  cos(ψ₀) = {cos_psi:+.4f}  '
          f'a_drive = g·sin(angle)·cos(ψ₀)', flush=True)
    print('=' * 64)

    SKIP = max(1, int(0.5 / DT))   # drop first 0.5 s (attitude transient)

    def extract(phase_name, vel_col, angle_col):
        rows = [r for r in _log_rows if r['phase'] == phase_name][SKIP:]
        if len(rows) < 5:
            return None, None, None
        t     = np.array([r['t_phase'] for r in rows])
        vel   = np.array([r[vel_col]   for r in rows])
        angle = np.radians([r[angle_col] for r in rows])
        return t, vel, angle

    tau_results = {}

    for phase, vel_col, angle_col, label in [
        ('TILT_N_POS', 'vN', 'theta_deg', 'N+  (pitch +)'),
        ('TILT_N_NEG', 'vN', 'theta_deg', 'N−  (pitch −)'),
        ('TILT_E_POS', 'vE', 'phi_deg',   'E+  (roll  +)'),
    ]:
        t_arr, v_arr, angle_arr = extract(phase, vel_col, angle_col)
        if t_arr is None:
            print(f'\n  {label}: no data collected')
            continue

        angle_mean = float(np.mean(angle_arr))
        # dv/dt = g·cos(ψ₀)·sin(angle) − v/τ_v
        # For ψ₀=180°: cos(ψ₀)=−1 → a_drive = −g·sin(angle).
        # Tilt targets were chosen as tilt_dir·θ_test so a_drive > 0 for N+/E+.
        a_drive    = g * np.sin(angle_mean) * cos_psi
        tau_v, v_ss, v0, rms = _fit_tau(t_arr, v_arr, a_drive)

        print(f'\n  {label}')
        print(f'    mean angle = {np.degrees(angle_mean):+.2f}°   '
              f'a_drive = {a_drive:+.4f} m/s²   '
              f'v_end = {float(v_arr[-1]):+.3f} m/s')

        if tau_v is None:
            print('    fit failed (too little velocity response — increase THETA_VEL_DEG '
                  'or TILT_SEC)')
            continue

        Dv = m / tau_v
        print(f'    τ_v   = {tau_v:.3f} s         '
              f'Dv = {Dv:.4f} N·s/m   (m/τ_v = {m:.3f}/{tau_v:.3f})')
        print(f'    v_ss  = {v_ss:+.3f} m/s    '
              f'v0 = {v0:+.3f} m/s    fit RMS = {rms:.4f} m/s')

        tau_results[phase] = tau_v

    # Aggregate and compute gains
    print()
    print('  ── Aggregate ──────────────────────────────────────────────')

    tau_N_list = [tau_results[k] for k in ('TILT_N_POS', 'TILT_N_NEG') if k in tau_results]
    tau_E_list = [tau_results[k] for k in ('TILT_E_POS',)              if k in tau_results]

    def _print_axis(label, tau_list, kp_name, ki_name):
        if not tau_list:
            return None, None, None
        tau_v = float(np.mean(tau_list))
        Dv    = m / tau_v
        Kp, Ki = _pi_gains(tau_v, g, T_SETTLE_DES)
        print(f'  {label}  τ_v = {tau_v:.3f} s   Dv = {Dv:.4f} N·s/m')
        print(f'         → {kp_name}: {Kp:.6f}   {ki_name}: {Ki:.6f}  '
              f'(T_settle = {T_SETTLE_DES:.1f} s, critically damped)')
        return tau_v, Kp, Ki

    tau_vN, Kp_vN, Ki_vN = _print_axis('North', tau_N_list, 'Kp_vN', 'Ki_vN')
    tau_vE, Kp_vE, Ki_vE = _print_axis('East ', tau_E_list, 'Kp_vE', 'Ki_vE')

    # If East was not measured, assume same as North
    if tau_vE is None and tau_vN is not None:
        tau_vE, Kp_vE, Ki_vE = tau_vN, Kp_vN, Ki_vN
        print(f'  East  τ_v assumed = τ_vN → Kp_vE = Kp_vN, Ki_vE = Ki_vN')

    print()
    print('  ── Recommended params.yaml update ────────────────────────')
    if Kp_vN is not None:
        print(f'  Kp_vN:  {Kp_vN:.6f}')
        print(f'  Ki_vN:  {Ki_vN:.6f}')
    if Kp_vE is not None:
        print(f'  Kp_vE:  {Kp_vE:.6f}')
        print(f'  Ki_vE:  {Ki_vE:.6f}')

    print()
    print('  ── Cascade structure reminder ─────────────────────────────')
    print('  v_err →[Kp + Ki/s]→ θ_des →[K_att]→ q_sim →[sim ×2.4]→ θ')
    print(f'  dv/dt ≈ g·cos(ψ₀)·θ − v/τ_v   (ψ₀={np.degrees(psi_0):.0f}°, '
          f'cos(ψ₀)={cos_psi:+.3f})')
    print(f'  Rate loop inner gain ≈ 2.4 is absorbed into K_att in params.yaml.')
    print('=' * 64)
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    param     = load_params('params.yaml')
    m         = float(param.get('m', 5.405))
    g_acc     = float(param.get('g', 9.81))
    T_max_mot = float(param.get('T_max_motor', 49.9))
    T_hover   = m * g_acc / (4.0 * T_max_mot)
    pD_target = -ALT_TARGET_M
    theta_test = np.radians(THETA_VEL_DEG)

    # Tilt-to-velocity sign from initial yaw.
    # dv_N/dt = g*cos(psi_0)*sin(theta) → sign of coupling = sign(cos(psi_0)).
    # For psi_0=180°: cos=-1 → positive theta → South accel → tilt_dir=-1.
    psi_0    = np.radians(float(param.get('initial_yaw_deg', 0.0)))
    tilt_dir = int(np.sign(np.cos(psi_0)))   # -1 for 180°, +1 for 0°
    if tilt_dir == 0:
        tilt_dir = 1
    print(f'[sysid_vel] initial_yaw_deg={np.degrees(psi_0):.1f}°  '
          f'tilt_dir={tilt_dir:+d}  '
          f'(cos(ψ₀)={np.cos(psi_0):+.3f})', flush=True)

    system_boot_ms = int(time.time() * 1000)
    shared         = {}
    components     = setup_components(shared, system_boot_ms, SIM_IP, SIM_PORT)
    conn      = components['sim_conn']
    mav_rx    = components['mavlink_rx']
    ctrl      = components['controller']
    ts_loop   = components['ts_loop']
    vis_rx    = components['vision_rx']
    logger    = components.get('logger')

    print('Press "s" to arm and start sysid_vel ...', flush=True)
    while True:
        if msvcrt.kbhit() and msvcrt.getwch().lower() == 's':
            break
        time.sleep(0.05)

    print('Resetting sim ...', flush=True)
    ctrl.send_sim_reset_command()
    time.sleep(2.0)
    mav_rx.request_ground_truth_streams(rate_hz=50)

    print('Arming ...', flush=True)
    ctrl.arm()
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
        print(f'[sysid_vel]  →  {name}', flush=True)

    try:
        while not aborted:
            t0    = time.time()
            state = _gt(shared)

            if state is None:
                time.sleep(DT)
                continue

            phi, theta, psi, pos, vel, att_rates = state
            mav = shared['mav_state']

            # Abort guard (skip in WAIT / KILL)
            if phase not in ('WAIT', 'KILL'):
                reason = _abort(phi, theta, att_rates)
                if reason:
                    print(f'[sysid_vel] ABORT  phase={phase}  {reason}', flush=True)
                    aborted = True
                    _kill(conn)
                    break

            p_sim = q_sim = r_sim = 0.0
            T = T_hover

            # ── WAIT ─────────────────────────────────────────────────────────
            if phase == 'WAIT':
                conn.mav.set_actuator_control_target_send(
                    int(time.time() * 1e6),
                    conn.target_system, conn.target_component,
                    0, [0.0] * 8,
                )
                if elapsed() >= WAIT_SEC:
                    next_phase('TAKEOFF')
                time.sleep(max(0.0, DT - (time.time() - t0)))
                continue   # no log during WAIT

            # ── TAKEOFF: zero angular-rate + altitude hold ───────────────────
            elif phase == 'TAKEOFF':
                T = _alt_hold_thrust(shared, T_hover, pD_target, phi, theta)
                _send_to_sim(conn, 0.0, 0.0, 0.0, T)
                if float(pos[2]) < pD_target + 0.15:
                    next_phase('LEVEL')
                elif elapsed() > TAKEOFF_TIMEOUT:
                    print('[sysid_vel] TAKEOFF timeout — aborting', flush=True)
                    aborted = True; _kill(conn); break

            # ── LEVEL ────────────────────────────────────────────────────────
            elif phase == 'LEVEL':
                p_sim, q_sim, phi, theta = _level(mav)
                T = _alt_hold_thrust(shared, T_hover, pD_target, phi, theta)
                _send_to_sim(conn, p_sim, q_sim, 0.0, T)
                if elapsed() >= LEVEL_SEC:
                    next_phase('HOVER')

            # ── HOVER: wait until roughly level and slow ──────────────────────
            elif phase == 'HOVER':
                p_sim, q_sim, phi, theta = _level(mav)
                T = _alt_hold_thrust(shared, T_hover, pD_target, phi, theta)
                T_hover = 0.99 * T_hover + 0.01 * T   # slow running hover estimate
                _send_to_sim(conn, p_sim, q_sim, 0.0, T)
                v_horiz = float(np.hypot(vel[0], vel[1]))
                if elapsed() >= HOVER_SEC:
                    if v_horiz < HOVER_V_THRESH:
                        next_phase('TILT_N_POS')
                    elif elapsed() > HOVER_SEC + 15.0:
                        print(f'[sysid_vel] warn: residual |v|={v_horiz:.2f} m/s, '
                              f'starting anyway after {elapsed():.1f}s', flush=True)
                        next_phase('TILT_N_POS')

            # ── TILT_N_POS: θ = tilt_dir*θ_test → positive v_N ─────────────
            elif phase == 'TILT_N_POS':
                p_sim, q_sim, phi, theta = _tilt_hold(mav, 0.0, tilt_dir * theta_test)
                T = _alt_hold_thrust(shared, T_hover, pD_target, phi, theta)
                _send_to_sim(conn, p_sim, q_sim, 0.0, T)
                if elapsed() >= TILT_SEC:
                    print(f'[sysid_vel]  TILT_N_POS end  '
                          f'vN={float(vel[0]):+.3f} m/s  '
                          f'θ={np.degrees(theta):+.1f}°', flush=True)
                    next_phase('DECEL_N1')

            # ── DECEL_N1: level + let v_N decay ─────────────────────────────
            elif phase == 'DECEL_N1':
                p_sim, q_sim, phi, theta = _level(mav)
                T = _alt_hold_thrust(shared, T_hover, pD_target, phi, theta)
                _send_to_sim(conn, p_sim, q_sim, 0.0, T)
                vN_now = float(vel[0])
                if elapsed() >= 3.0 and abs(vN_now) < DECEL_V_THRESH:
                    print(f'[sysid_vel]  DECEL_N1 settled: vN={vN_now:+.3f} m/s '
                          f'at {elapsed():.1f}s', flush=True)
                    next_phase('TILT_N_NEG')
                elif elapsed() >= DECEL_SEC:
                    if abs(vN_now) > 0.5:
                        print(f'[sysid_vel]  warn: vN still {vN_now:+.2f} m/s '
                              f'after DECEL_N1', flush=True)
                    next_phase('TILT_N_NEG')

            # ── TILT_N_NEG: θ = −tilt_dir*θ_test → negative v_N ────────────
            elif phase == 'TILT_N_NEG':
                p_sim, q_sim, phi, theta = _tilt_hold(mav, 0.0, -tilt_dir * theta_test)
                T = _alt_hold_thrust(shared, T_hover, pD_target, phi, theta)
                _send_to_sim(conn, p_sim, q_sim, 0.0, T)
                if elapsed() >= TILT_SEC:
                    print(f'[sysid_vel]  TILT_N_NEG end  '
                          f'vN={float(vel[0]):+.3f} m/s  '
                          f'θ={np.degrees(theta):+.1f}°', flush=True)
                    next_phase('DECEL_N2')

            # ── DECEL_N2 ─────────────────────────────────────────────────────
            elif phase == 'DECEL_N2':
                p_sim, q_sim, phi, theta = _level(mav)
                T = _alt_hold_thrust(shared, T_hover, pD_target, phi, theta)
                _send_to_sim(conn, p_sim, q_sim, 0.0, T)
                vN_now = float(vel[0])
                if elapsed() >= 3.0 and abs(vN_now) < DECEL_V_THRESH:
                    print(f'[sysid_vel]  DECEL_N2 settled: vN={vN_now:+.3f} m/s '
                          f'at {elapsed():.1f}s', flush=True)
                    next_phase('TILT_E_POS')
                elif elapsed() >= DECEL_SEC:
                    next_phase('TILT_E_POS')

            # ── TILT_E_POS: φ = tilt_dir*θ_test → positive v_E ─────────────
            elif phase == 'TILT_E_POS':
                p_sim, q_sim, phi, theta = _tilt_hold(mav, tilt_dir * theta_test, 0.0)
                T = _alt_hold_thrust(shared, T_hover, pD_target, phi, theta)
                _send_to_sim(conn, p_sim, q_sim, 0.0, T)
                if elapsed() >= TILT_SEC:
                    print(f'[sysid_vel]  TILT_E_POS end  '
                          f'vE={float(vel[1]):+.3f} m/s  '
                          f'φ={np.degrees(phi):+.1f}°', flush=True)
                    next_phase('DECEL_E1')

            # ── DECEL_E1 ─────────────────────────────────────────────────────
            elif phase == 'DECEL_E1':
                p_sim, q_sim, phi, theta = _level(mav)
                T = _alt_hold_thrust(shared, T_hover, pD_target, phi, theta)
                _send_to_sim(conn, p_sim, q_sim, 0.0, T)
                vE_now = float(vel[1])
                if elapsed() >= 3.0 and abs(vE_now) < DECEL_V_THRESH:
                    print(f'[sysid_vel]  DECEL_E1 settled: vE={vE_now:+.3f} m/s '
                          f'at {elapsed():.1f}s', flush=True)
                    next_phase('KILL')
                elif elapsed() >= DECEL_SEC:
                    next_phase('KILL')

            # ── KILL ─────────────────────────────────────────────────────────
            elif phase == 'KILL':
                _kill(conn)
                print('[sysid_vel] Motors killed.', flush=True)
                time.sleep(0.3)
                break

            _log_row(phase, elapsed(), phi, theta, psi, pos, vel,
                     att_rates, p_sim, q_sim, r_sim, T)

            sleep_t = DT - (time.time() - t0)
            if sleep_t > 0:
                time.sleep(sleep_t)

    except KeyboardInterrupt:
        print('\n[sysid_vel] KeyboardInterrupt — killing motors', flush=True)
        _kill(conn)

    finally:
        _analyze(m, psi_0, g_acc)
        _save_csv()
        if logger is not None:
            logger.save()
        for _c in [ts_loop, mav_rx, vis_rx]:
            _t = _c.get_thread_for_join()
            if _t is not None:
                _t.join(timeout=1.0)
        print('[sysid_vel] Done.', flush=True)


if __name__ == '__main__':
    main()
