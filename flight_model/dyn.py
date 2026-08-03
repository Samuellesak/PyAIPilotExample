import numpy as np
import yaml

from flight_model import rotations


def load_params(yaml_path="params.yaml"):
    """
    Load vehicle parameters from YAML and derive the inertia tensor.

    Inertia model (DJI X config, motors at 45°, 4 uniform arms):
        d       = L / sqrt(2)          (motor offset projected onto each body axis)

        Motor term: each motor sits at distance L from CoM (d²+d² = L²).
            I_motors_xx = 4 * m_motor * d²  = 2 * m_motor * L²
            I_motors_zz = 4 * m_motor * L²

        Frame term: 4 uniform rods of length L at 45° to body axes.
            Each arm at angle 45°: I_arm_xx = m_arm * L² * sin²(45°) / 3
                                             = m_arm * L² / 6
            Summed over 4 arms (m_arm = m_frame / 4):
            I_frame_xx = m_frame * L² / 6
            I_frame_zz = m_frame * L² / 3    (polar: I = m_arm * L²/3 per arm)

        Ixx = Iyy = 2 * m_motor * L² + m_frame * L² / 6
        Izz       = 4 * m_motor * L² + m_frame * L² / 3

    I_inv is precomputed so the ODE does not invert I on every call.
    """
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    p = {
        'm':     float(raw['m']),
        'g':     float(raw['g']),
        'L':     float(raw['L']),
        'kappa': float(raw['kappa']),
        'C_M_q': float(raw['C_M_q']),
        'Dv':    np.diag(raw['Dv']),
        'Dw':    np.diag(raw['Dw']),
    }

    m_motor = float(raw['m_motor'])
    L       = p['L']
    m_frame = p['m'] - 4.0 * m_motor

    if m_frame < 0.0:
        raise ValueError(
            f"4 * m_motor = {4*m_motor:.3f} kg exceeds total mass m = {p['m']} kg."
        )

    Ixx = 2.0 * m_motor * L**2  +  m_frame * L**2 / 6.0
    Iyy = Ixx
    Izz = 4.0 * m_motor * L**2  +  m_frame * L**2 / 3.0

    p['I']     = np.diag([Ixx, Iyy, Izz])
    p['I_inv'] = np.linalg.inv(p['I'])

    # ── Carrot tracker ─────────────────────────────────────────────────
    p['waypoints']       = np.array(raw['waypoints'], dtype=float)  # (M, 3)
    p['v_ref']           = float(raw['v_ref'])
    p['lookahead_gain']  = float(raw['lookahead_gain'])
    p['T_max_motor']     = float(raw['T_max_motor'])
    p['tau_motor']       = float(raw.get('tau_motor', 0.05))

    # Controller selection and gains
    p['controller_type'] = int(raw.get('controller_type', 1))
    p['rate_bandwidth'] = float(raw.get('rate_bandwidth', 12.0))
    p['Kp_vN']          = float(raw.get('Kp_vN', raw.get('Kp_vel', 0.1)))
    p['Ki_vN']          = float(raw.get('Ki_vN', raw.get('Ki_vel', 0.0)))
    p['Kp_vE']          = float(raw.get('Kp_vE', raw.get('Kp_vel', 0.1)))
    p['Ki_vE']          = float(raw.get('Ki_vE', raw.get('Ki_vel', 0.0)))
    p['Kp_vD']          = float(raw.get('Kp_vD', raw.get('Kp_vz',  0.3)))
    p['Ki_vD']          = float(raw.get('Ki_vD', raw.get('Ki_vz',  0.05)))
    p['K_att']          = float(raw.get('K_att',    3.0))
    p['K_psi']          = float(raw.get('K_psi',    1.0))
    p['Ki_psi']         = float(raw.get('Ki_psi',   0.200))
    p['MAX_TILT_DEG']   = float(raw.get('MAX_TILT_DEG', 25.0))
    p['m_motor']        = float(raw.get('m_motor',  0.050))
    p['K_rate_roll_override']  = float(raw.get('K_rate_roll_override',  0.0))
    p['K_rate_pitch_override'] = float(raw.get('K_rate_pitch_override', 0.0))
    p['hover_only']            = bool(raw.get('hover_only', False))
    p['logging']               = int(raw.get('logging', 1))
    p['initial_yaw_deg']       = float(raw.get('initial_yaw_deg', 0.0))

    # LQI cost weights (optional — only present when lqi.py is used)
    if 'Q_vel' in raw:
        Q_vel     = list(raw['Q_vel'])
        Q_quat    = list(raw['Q_quat'])
        Q_rates   = list(raw['Q_rates'])
        Q_int_vel = list(raw['Q_int_vel'])
        Q_int_psi = float(raw['Q_int_psi'])
        Q_diag    = Q_vel + Q_quat + Q_rates + Q_int_vel + [Q_int_psi]
        p['Q_aug'] = np.diag(Q_diag)
        p['R_cost'] = np.eye(4) * float(raw['R_cost'])

    # Pass through any YAML keys not already explicitly extracted above
    # (e.g. vision params, logging flags, blip tuning).
    # Physics keys already in p keep their transformed numpy forms.
    for k, v in raw.items():
        if k not in p:
            p[k] = v

    return p


# Rotation matrix R such that v_body = R @ v_NED — see rotations.py
# (centralized; kept as _quat_to_R here since ekf_shadow.py, imu_ekf.py,
# linearize.py, and lqi.py all import this name from dyn.py).
_quat_to_R = rotations.quat_to_R_ned2body


def _omega_matrix(omega):
    """
    4×4 skew matrix Omega(omega) for quaternion kinematics.
    qdot = 0.5 * Omega @ q   (NED-to-body passive rotation, omega in body frame)
    """
    p, q, r = omega
    return np.array([
        [ 0, -p, -q, -r],
        [ p,  0,  r, -q],
        [ q, -r,  0,  p],
        [ r,  q, -p,  0],
    ])


def dyn(x, u, param):
    """
    6DOF rigid-body dynamics for a quadcopter (DJI X config).

    State x (13,):
        x[0:3]   position NED               [m]
        x[3:6]   body velocity  [u, v, w]   [m/s]
        x[6:10]  quaternion     [qw,qx,qy,qz]  NED-to-body, unit
        x[10:13] body rates     [p, q, r]    [rad/s]

    Input u (4,):  motor thrusts [FL, FR, BL, BR]  [N]
        Matches sim actuator order [0=FL, 1=FR, 2=BL, 3=BR].
        Body frame FRD: x=forward, y=right, z=down.
        Motor positions (d = L/sqrt(2)):
            FL (+d, -d),  FR (+d, +d),  BL (-d, -d),  BR (-d, +d)
        Torques:
            tau_x = d*(FL+BL - FR-BR)     roll:  left>right  → +roll (right side down)
            tau_y = d*(FL+FR - BL-BR)     pitch: front>back  → +pitch (nose up)
            tau_z = k*(FL+BR - FR-BL)     yaw:   FL/BR pair CW if k>0

    Initial condition for level flight: x[6:10] = [1, 0, 0, 0].

    Returns xdot (13,).
    """
    vel_b   = x[3:6]
    quat    = x[6:10]
    omega_b = x[10:13]

    m     = param['m']
    g     = param['g']
    I     = param['I']
    I_inv = param['I_inv']
    l     = param['L']
    kappa = param['kappa']
    Dv    = param['Dv']
    Dw    = param['Dw']
    C_M_q = param['C_M_q']
    C_M_p = C_M_q

    # ── Mixer — sim native order [FL, FR, BL, BR] ─────────────────────
    d = l / np.sqrt(2.0)
    FL, FR, BL, BR = u[0], u[1], u[2], u[3]

    T_total = FL + FR + BL + BR
    tau_x   = d     * (FL + BL - FR - BR)   # roll:  left  > right → nose right side down
    tau_y   = d     * (FL + FR - BL - BR)   # pitch: front > back  → nose up
    tau_z   = kappa * (FL + BR - FR - BL)   # yaw:   FL/BR (CW pair) > FR/BL → +yaw

    F_b   = np.array([0.0, 0.0, -T_total])
    tau_b = np.array([tau_x, tau_y, tau_z])

    # ── Rotation (NED to body): v_body = R @ v_NED ───────────────────
    R   = _quat_to_R(quat)
    g_b = R @ np.array([0.0, 0.0, g])

    # ── Aerodynamic drag ─────────────────────────────────────────────
    F_drag = -Dv @ (np.abs(vel_b)   * vel_b)
    M_drag = -Dw @ (np.abs(omega_b) * omega_b)

    # ── Aerodynamic moments from translational velocity ───────────────
    M_aero = np.array([
        -C_M_p * vel_b[1],   # roll  from sideways velocity
         C_M_q * vel_b[0],   # pitch from forward  velocity
         0.0,
    ])

    # ── Translational dynamics (Newton in body frame) ─────────────────
    vel_dot_b = (F_b + m * g_b + F_drag) / m - np.cross(omega_b, vel_b)

    # ── Rotational dynamics (Euler's equation) ────────────────────────
    omega_dot_b = I_inv @ (tau_b + M_drag + M_aero - np.cross(omega_b, I @ omega_b))

    # ── Position kinematics (body vel → NED) ─────────────────────────
    pos_dot_n = R.T @ vel_b

    # ── Quaternion kinematics with normalization feedback ─────────────
    # qdot = 0.5 * Omega(omega) @ q
    # Baumgarte term keeps |q| = 1 under numerical drift: -k*(|q|^2 - 1)*q
    quat_dot  = 0.5 * (_omega_matrix(omega_b) @ quat)
    quat_dot -= 0.5 * (quat @ quat - 1.0) * quat

    return np.concatenate([pos_dot_n, vel_dot_b, quat_dot, omega_dot_b])


def dyn_lag(x, u_cmd, param):
    """
    6DOF dynamics with first-order motor lag.

    Extends dyn() by treating commanded thrust u_cmd as the controller output
    and tracking actual motor thrust as four additional state variables.

    State x (17,):
        x[0:13]  — same layout as dyn()
        x[13:17] — actual motor thrusts [FL, FR, BL, BR]  [N]

    Input u_cmd (4,):  commanded motor thrusts [FL, FR, BL, BR] [N]  (sim native order)

    Motor lag ODE (first-order low-pass per motor):
        dT_i/dt = (u_cmd_i - T_i) / tau_motor

    At trim: T_i == u_cmd_i  →  dT_i/dt = 0.

    Returns xdot (17,).

    Note
    ----
    lqi.py designs gains against the 13-state dyn() (B_r non-zero, CARE solvable).
    Including motor states in the CARE would require a state observer for T_actual
    (not available from ODOMETRY).  Use dyn_lag() for simulation / validation only.
    """
    T_actual  = x[13:17]
    tau_m     = param['tau_motor']

    mech_dot  = dyn(x[:13], T_actual, param)
    T_dot     = (u_cmd - T_actual) / tau_m

    return np.concatenate([mech_dot, T_dot])
