#!/usr/bin/env python3
"""
sysid.py — Quadcopter system identification via predefined input sequences.

Sends predefined motor commands to the sim, records HIGHRES_IMU responses,
then optimises {Dv, Dw, kappa, m} by minimising one-step prediction error
of the dyn.py rotational + translational model.

Usage:
    python sysid.py            # collect new data and fit
    python sysid.py --fit-only # fit from existing sysid_data.npy
    python sysid.py --plot-only # plot comparison without fitting

Motor ordering (dyn.py / controller.py convention):
    u[0]=T1 (BR), u[1]=T2 (BL), u[2]=T3 (FL), u[3]=T4 (FR)
Sim expects [FL, FR, BL, BR] = [T3, T4, T2, T1] — reordering applied in send.
"""

import argparse
import time
import threading
import numpy as np
import matplotlib
matplotlib.use('Agg')          # non-interactive backend — no window, no blocking
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from pymavlink import mavutil

from ekf import QuadEKF
from dyn import load_params

# ─── Configuration ────────────────────────────────────────────────────────────
SIM_IP    = "127.0.0.1"
SIM_PORT  = 14550
DATA_FILE = "sysid_data.npy"
DT_CMD    = 1.0 / 250          # motor command rate [s]

_PARAM = load_params()
T_MAX  = _PARAM['T_max_motor']
M_NOM  = _PARAM['m']
G      = _PARAM['g']
T_HOVER_NORM = (M_NOM * G / 4.0) / T_MAX   # per-motor hover normalised

_U0 = 0.05*1.0
_US = 0.10*1.0
_DU = 0.06*1.0

# Each entry: (duration_s, [T1_BR_norm, T2_BL_norm, T3_FL_norm, T4_FR_norm])
SEQUENCES = [
    # Settle: EKF converges to slope tilt
    (2.0,  [_U0]*4),

    # Roll excitation (+tau_x then -tau_x)
    (1.2,  [_US-_DU, _US+_DU, _US+_DU, _US-_DU]),
    (0.6,  [_US]*4),
    (1.2,  [_US+_DU, _US-_DU, _US-_DU, _US+_DU]),
    (0.6,  [_US]*4),

    # Pitch excitation (+tau_y then -tau_y)
    (1.2,  [_US+_DU, _US+_DU, _US-_DU, _US-_DU]),
    (0.6,  [_US]*4),
    (1.2,  [_US-_DU, _US-_DU, _US+_DU, _US+_DU]),
    (0.6,  [_US]*4),

    # Yaw excitation (+tau_z then -tau_z)
    (1.2,  [_US+_DU, _US-_DU, _US+_DU, _US-_DU]),
    (0.6,  [_US]*4),
    (1.2,  [_US-_DU, _US+_DU, _US-_DU, _US+_DU]),
    (0.6,  [_US]*4),

    # Wind-down (no collective sweep — drone on ground has no attitude control;
    # use thrust_test.py separately to calibrate T_max / m from liftoff thrust)
    (1.0,  [_U0]*4),
]


# ─── Data collector ───────────────────────────────────────────────────────────

class DataCollector:
    def __init__(self, sim_conn):
        self._conn       = sim_conn
        self._ekf        = QuadEKF()
        self._lock       = threading.Lock()
        self._samples    = []
        self._last_imu_t = None
        self._is_running = True
        self._u_now      = np.zeros(4)

    def set_command(self, u_norm):
        with self._lock:
            self._u_now[:] = u_norm

    def receive_loop(self):
        while self._is_running:
            try:
                msg = self._conn.recv_match(blocking=False)
            except Exception:
                time.sleep(0.001)
                continue
            if msg is None:
                time.sleep(0.0005)
                continue
            if msg.get_type() == "HIGHRES_IMU":
                self._on_imu(msg)

    def _on_imu(self, msg):
        ax, ay, az = msg.xacc, msg.yacc, msg.zacc
        gx, gy, gz = msg.xgyro, msg.ygyro, msg.zgyro
        t_us = msg.time_usec
        now  = time.time()

        gyro = np.array([-gx, -gy, -gz])   # all three axes sign-flipped vs FRD (matches mavlink_rx.py)
        acc  = np.array([ax, ay, az])

        dt = (now - self._last_imu_t) if self._last_imu_t is not None else 0.004
        dt = float(np.clip(dt, 0.0005, 0.05))
        self._last_imu_t = now

        self._ekf.predict(gyro, acc, dt)
        self._ekf.update_accel(acc)
        self._ekf.update_zupt(gyro, acc)

        with self._lock:
            u = self._u_now.copy()

        self._samples.append({
            't_us':    t_us,
            'gyro':    gyro.copy(),
            'acc':     acc.copy(),
            'quat':    self._ekf.quat.copy(),
            'vel_ned': self._ekf.vel_ned.copy(),
            'u_norm':  u.copy(),
        })

    def stop(self):
        self._is_running = False

    def get_samples(self):
        return list(self._samples)


# ─── MAVLink motor send ───────────────────────────────────────────────────────

def send_motors_norm(conn, u_norm):
    """Reorders [T1=BR,T2=BL,T3=FL,T4=FR] -> sim [FL,FR,BL,BR]."""
    u = [float(x) for x in u_norm]
    cmds = [u[2], u[3], u[1], u[0]] + [0.0] * 8
    conn.mav.set_actuator_control_target_send(
        int(time.time() * 1e6),
        conn.target_system, conn.target_component,
        0, cmds,
    )


# ─── Array conversion ─────────────────────────────────────────────────────────

def arrays_from_samples(samples):
    t_us    = np.array([s['t_us']    for s in samples], dtype=float)
    gyro    = np.array([s['gyro']    for s in samples])   # (N,3)
    acc     = np.array([s['acc']     for s in samples])   # (N,3)
    quat    = np.array([s['quat']    for s in samples])   # (N,4)
    vel_ned = np.array([s['vel_ned'] for s in samples])   # (N,3)
    u_thr   = np.array([s['u_norm']  for s in samples]) * T_MAX  # (N,4) [N]
    dt_arr  = np.clip(np.diff(t_us) * 1e-6, 5e-4, 0.05)          # (N-1,)
    return t_us, gyro, acc, quat, vel_ned, u_thr, dt_arr


# ─── Vectorised physics helpers ───────────────────────────────────────────────

def _inertia(m):
    m_motor = _PARAM['m_motor']
    L       = _PARAM['L']
    m_frame = m - 4.0 * m_motor
    Ixx = 2.0 * m_motor * L**2 + m_frame * L**2 / 6.0
    Izz = 4.0 * m_motor * L**2 + m_frame * L**2 / 3.0
    return Ixx, Izz


def _torques(u_thr, kappa):
    """Vectorised mixer: returns (N,3) tau array."""
    d  = _PARAM['L'] / np.sqrt(2.0)
    T1, T2, T3, T4 = u_thr[:,0], u_thr[:,1], u_thr[:,2], u_thr[:,3]
    return np.stack([
        d * (-T1 + T2 + T3 - T4),       # tau_x: right-down positive
        d * (-T1 - T2 + T3 + T4),       # tau_y: nose-up positive (FIXED: was +T1+T2-T3-T4)
        kappa * (T1 - T2 + T3 - T4),    # tau_z: CW positive
    ], axis=1)                                             # (N,3)


def _rotate_vec_n2b(quat, v_ned):
    """
    Vectorised v_body = R_n2b(q) @ v_ned for (N,4) quat and (N,3) v_ned.
    Returns (N,3).
    """
    qw = quat[:,0]; qx = quat[:,1]; qy = quat[:,2]; qz = quat[:,3]
    vx = v_ned[:,0]; vy = v_ned[:,1]; vz = v_ned[:,2]
    bx = (1-2*(qy**2+qz**2))*vx + 2*(qx*qy+qw*qz)*vy + 2*(qx*qz-qw*qy)*vz
    by = 2*(qx*qy-qw*qz)*vx + (1-2*(qx**2+qz**2))*vy + 2*(qy*qz+qw*qx)*vz
    bz = 2*(qx*qz+qw*qy)*vx + 2*(qy*qz-qw*qx)*vy + (1-2*(qx**2+qy**2))*vz
    return np.stack([bx, by, bz], axis=1)                 # (N,3)



# ─── Cost function (fully vectorised — no Python loop) ───────────────────────

def cost_fn(theta, gyro, acc, quat, vel_ned, u_thr, dt_arr):
    """
    One-step prediction MSE: angular rates only (N-1 samples).
    theta = [Dw_xy, Dw_z, kappa, m]
    Dv and acc cost excluded — drone is ground-constrained during this sysid.
    """
    Dw_xy, Dw_z, kappa, m = theta
    if m <= 0 or any(v < 0 for v in theta):
        return 1e9

    Ixx, Izz = _inertia(m)
    if Ixx <= 0 or Izz <= 0:
        return 1e9

    I_inv_diag = np.array([1.0/Ixx, 1.0/Ixx, 1.0/Izz])
    I_diag     = np.array([Ixx,     Ixx,     Izz    ])
    Dw = np.array([Dw_xy, Dw_xy, Dw_z])

    # Slice to [:-1] for prediction, [1:] for target
    omega = gyro[:-1]      # (N-1, 3)
    q     = quat[:-1]
    vn    = vel_ned[:-1]
    u     = u_thr[:-1]
    dt    = dt_arr[:, None]   # (N-1, 1) for broadcasting

    # ── Angular rate one-step prediction ─────────────────────────────────
    tau       = _torques(u, kappa)                           # (N-1, 3)
    Io        = omega * I_diag                               # (N-1, 3)
    cross_oIo = np.cross(omega, Io)                          # (N-1, 3)
    damp      = Dw * np.abs(omega) * omega                   # (N-1, 3)
    omega_dot = (tau - damp - cross_oIo) * I_inv_diag        # (N-1, 3)
    omega_pred = omega + omega_dot * dt                       # (N-1, 3)
    e_omega   = omega_pred - gyro[1:]                        # (N-1, 3)

    # Accelerometer cost omitted: drone is on the ground, contact force corrupts
    # the specific-force equation.  Gyro is unaffected by contact (pure linear
    # force, no torque), so Dw_xy is identifiable from gyro alone.
    # Use thrust_test.py to identify m from liftoff thrust; Dv / kappa need
    # in-flight data and are not fit here.
    n = len(dt_arr)
    return np.sum(e_omega**2) / n


# ─── Forward simulation for plotting ─────────────────────────────────────────

def simulate_dyn(gyro, acc, quat, vel_ned, u_thr, dt_arr, theta):
    """Integrate angular rates forward using the model; acc computed per-step."""
    Dw_xy, Dw_z, kappa, m = theta
    Dv_xy, Dv_z = 0.0, 0.0   # not identified on ground; zero for plotting
    Ixx, Izz    = _inertia(m)
    I_inv_diag  = np.array([1.0/Ixx, 1.0/Ixx, 1.0/Izz])
    I_diag      = np.array([Ixx, Ixx, Izz])
    Dw = np.array([Dw_xy, Dw_xy, Dw_z])
    Dv = np.array([Dv_xy, Dv_xy, Dv_z])

    N = len(gyro)
    omega_sim = np.zeros((N, 3))
    acc_sim   = np.zeros((N, 3))
    omega_sim[0] = gyro[0]

    for k in range(N - 1):
        dt      = dt_arr[k]
        omega_k = omega_sim[k]
        u_k     = u_thr[k]
        q_k     = quat[k:k+1]     # (1,4) for vectorised helpers
        vn_k    = vel_ned[k:k+1]  # (1,3)

        tau_k     = _torques(u_k[None, :], kappa)[0]
        Io_k      = omega_k * I_diag
        damp_k    = Dw * np.abs(omega_k) * omega_k
        od_k      = (tau_k - damp_k - np.cross(omega_k, Io_k)) * I_inv_diag
        omega_sim[k+1] = omega_k + od_k * dt

        vb_k   = _rotate_vec_n2b(q_k, vn_k)[0]
        Tt     = u_k.sum()
        Fd_k   = -Dv * np.abs(vb_k) * vb_k
        cross_k = np.cross(omega_k, vb_k)
        a_k    = np.array([
            Fd_k[0] / m - cross_k[0],
            Fd_k[1] / m - cross_k[1],
            (-Tt + Fd_k[2]) / m - cross_k[2],
        ])
        acc_sim[k] = a_k

    acc_sim[-1] = acc_sim[-2]
    return omega_sim, acc_sim


# ─── Plotting (saves PNG; does NOT block) ────────────────────────────────────

def plot_comparison(t_s, gyro, acc, omega_nom, acc_nom, omega_opt, acc_opt,
                    u_thr, theta_nom, theta_opt):
    labels_omega = ['p  [rad/s]', 'q  [rad/s]', 'r  [rad/s]']
    labels_acc   = ['ax [m/s²]',  'ay [m/s²]',  'az [m/s²]' ]

    fig, axes = plt.subplots(6, 1, figsize=(14, 14), sharex=True)
    fig.suptitle('System ID: SIM vs dyn.py', fontsize=13)

    for i, (ax_p, lbl) in enumerate(zip(axes[:3], labels_omega)):
        ax_p.plot(t_s, gyro[:, i],       'k',   lw=1.5, label='SIM (IMU)')
        ax_p.plot(t_s, omega_nom[:, i],  'b--', lw=1.0, label='dyn nominal')
        ax_p.plot(t_s, omega_opt[:, i],  'r-',  lw=1.0, label='dyn optimised')
        ax_p.set_ylabel(lbl, fontsize=8)
        ax_p.legend(fontsize=7, loc='upper right')
        ax_p.grid(True, alpha=0.3)

    for i, (ax_p, lbl) in enumerate(zip(axes[3:], labels_acc)):
        ax_p.plot(t_s, acc[:, i],        'k',   lw=1.5, label='SIM (IMU)')
        ax_p.plot(t_s, acc_nom[:, i],    'b--', lw=1.0, label='dyn nominal')
        ax_p.plot(t_s, acc_opt[:, i],    'r-',  lw=1.0, label='dyn optimised')
        ax_p.set_ylabel(lbl, fontsize=8)
        ax_p.legend(fontsize=7, loc='upper right')
        ax_p.grid(True, alpha=0.3)

    axes[-1].set_xlabel('time [s]')
    plt.tight_layout()
    fig.savefig('sysid_comparison.png', dpi=120)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(14, 3))
    T_total_norm = u_thr.sum(axis=1) / (4.0 * T_MAX)
    ax2.plot(t_s, T_total_norm, 'k', lw=1.2)
    ax2.axhline(T_HOVER_NORM, color='b', ls='--', lw=1.0, label=f'hover ({T_HOVER_NORM:.3f})')
    ax2.set_ylabel('collective norm'); ax2.set_xlabel('time [s]')
    ax2.set_title('Motor input (total, normalised)')
    ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    fig2.savefig('sysid_input.png', dpi=120)
    plt.close(fig2)

    print("Saved: sysid_comparison.png  sysid_input.png", flush=True)

    print(f"\n{'Parameter':<12} {'nominal':>12} {'optimised':>12} {'delta%':>8}")
    names = ['Dw_xy', 'Dw_z', 'kappa', 'm']
    for name, nom, opt in zip(names, theta_nom, theta_opt):
        pct = (opt - nom) / nom * 100 if nom != 0 else float('nan')
        print(f"  {name:<10} {nom:>12.5f} {opt:>12.5f}  {pct:>+8.1f}%")


# ─── Data collection ──────────────────────────────────────────────────────────

def collect_data():
    print("Connecting to sim...", flush=True)
    conn = mavutil.mavlink_connection(f'udpin:{SIM_IP}:{SIM_PORT}')
    conn.wait_heartbeat()
    print(f"Connected (sys={conn.target_system})", flush=True)

    collector = DataCollector(conn)
    rx_thread = threading.Thread(target=collector.receive_loop, daemon=True)
    rx_thread.start()

    print("Arming...", flush=True)
    conn.mav.command_long_send(
        conn.target_system, conn.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 1, 0, 0, 0, 0, 0, 0)
    time.sleep(1.0)

    # ESC initialisation burst
    t0 = time.time()
    while time.time() - t0 < 1.0:
        send_motors_norm(conn, [_U0]*4)
        time.sleep(DT_CMD)

    total_dur = sum(d for d, _ in SEQUENCES)
    print(f"Running {len(SEQUENCES)} sequences ({total_dur:.1f} s total)...", flush=True)

    seq_t0 = time.time()
    for idx, (duration, u_norm) in enumerate(SEQUENCES):
        t_end = time.time() + duration
        collector.set_command(u_norm)
        while time.time() < t_end:
            send_motors_norm(conn, u_norm)
            time.sleep(DT_CMD)
        elapsed = time.time() - seq_t0
        print(f"  seg {idx+1:2d}/{len(SEQUENCES)}  "
              f"u=[{','.join(f'{v:.3f}' for v in u_norm)}]  t={elapsed:.1f}s", flush=True)

    conn.mav.command_long_send(
        conn.target_system, conn.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 0, 0, 0, 0, 0, 0, 0)

    collector.stop()
    time.sleep(0.1)   # let the rx thread drain its last message
    try:
        conn.close()
    except Exception:
        pass

    samples = collector.get_samples()
    print(f"Collected {len(samples)} IMU samples", flush=True)
    return samples


# ─── Fitting ──────────────────────────────────────────────────────────────────

def fit_params(samples):
    t_us, gyro, acc, quat, vel_ned, u_thr, dt_arr = arrays_from_samples(samples)
    t_s = (t_us - t_us[0]) * 1e-6

    Dw_nom = _PARAM['Dw'].diagonal()
    theta0 = np.array([Dw_nom[0], Dw_nom[2], _PARAM['kappa'], _PARAM['m']])

    print(f"\nNominal theta: {theta0}", flush=True)
    J0 = cost_fn(theta0, gyro, acc, quat, vel_ned, u_thr, dt_arr)
    print(f"Nominal cost:  {J0:.6f}", flush=True)

    bounds = [
        (1e-5, 0.5),    # Dw_xy
        (1e-5, 0.5),    # Dw_z
        (1e-4, 0.5),    # kappa  (likely hits lower bound — yaw constrained on ground)
        (1.0,  50.0),   # m
    ]

    n_calls = [0]
    def cb(theta):
        n_calls[0] += 1
        if n_calls[0] % 20 == 0:
            J = cost_fn(theta, gyro, acc, quat, vel_ned, u_thr, dt_arr)
            print(f"  iter {n_calls[0]:4d}  cost={J:.6f}  "
                  f"m={theta[3]:.2f}  kappa={theta[2]:.5f}  "
                  f"Dw_xy={theta[0]:.5f}", flush=True)

    print("Optimising...", flush=True)
    result = minimize(
        cost_fn, theta0,
        args=(gyro, acc, quat, vel_ned, u_thr, dt_arr),
        method='L-BFGS-B',
        bounds=bounds,
        callback=cb,
        options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8},
    )
    theta_opt = result.x
    J_opt     = result.fun
    print(f"\nOptimised cost: {J_opt:.6f}  ({100*(J0-J_opt)/max(J0,1e-9):+.1f}%)", flush=True)

    omega_nom, acc_nom = simulate_dyn(gyro, acc, quat, vel_ned, u_thr, dt_arr, theta0)
    omega_opt, acc_opt = simulate_dyn(gyro, acc, quat, vel_ned, u_thr, dt_arr, theta_opt)
    plot_comparison(t_s, gyro, acc, omega_nom, acc_nom, omega_opt, acc_opt,
                    u_thr, theta0, theta_opt)

    Dw_xy_o, Dw_z_o, kappa_o, m_o = theta_opt
    print("\n=== Suggested params.yaml updates ===")
    print(f"m:     {m_o:.4f}   # verify with thrust_test.py liftoff calibration")
    print(f"Dw: [{Dw_xy_o:.6f}, {Dw_xy_o:.6f}, {Dw_z_o:.6f}]")
    print(f"# kappa={kappa_o:.6f} — likely unreliable (yaw constrained on ground)")
    print(f"# Dv: needs in-flight sysid (velocity not observable on ground)")
    return theta_opt, result


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fit-only',  action='store_true')
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args()

    if args.fit_only or args.plot_only:
        print(f"Loading {DATA_FILE}...", flush=True)
        samples = list(np.load(DATA_FILE, allow_pickle=True))
    else:
        import msvcrt
        print("Press 's' to start experiment...", flush=True)
        while True:
            if msvcrt.kbhit() and msvcrt.getwch().lower() == 's':
                break
            time.sleep(0.05)
        samples = collect_data()
        np.save(DATA_FILE, np.array(samples, dtype=object))
        print(f"Saved {DATA_FILE}", flush=True)

    if args.plot_only:
        t_us, gyro, acc, quat, vel_ned, u_thr, dt_arr = arrays_from_samples(samples)
        t_s = (t_us - t_us[0]) * 1e-6
        Dw_nom = _PARAM['Dw'].diagonal()
        theta_nom = np.array([Dw_nom[0], Dw_nom[2], _PARAM['kappa'], _PARAM['m']])
        omega_nom, acc_nom = simulate_dyn(gyro, acc, quat, vel_ned, u_thr, dt_arr, theta_nom)
        plot_comparison(t_s, gyro, acc, omega_nom, acc_nom, omega_nom, acc_nom,
                        u_thr, theta_nom, theta_nom)
        return

    fit_params(samples)
    print("Done.", flush=True)


if __name__ == '__main__':
    main()
