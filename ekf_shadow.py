"""
ekf_shadow.py  —  Offline EKF replay: integration vs model_predict vs PnP vision,
compared against simulator ground truth, with a fitting regime for the
IMU/model_predict blend weights (sigma_acc_imu, sigma_acc_model).

Why offline replay instead of a live comparison
------------------------------------------------
params.yaml's ground_truth_mode: true overwrites the live EKF's state with GT
every tick (imu_ekf.py, end of on_imu_msg) — the safe, working flight-control
configuration. That means ekf.csv's logged trajectory is really "last tick's GT,
propagated one predict+accel+zupt step forward," not a genuine multi-second
free-running estimate, and vision fusion is skipped entirely while GT mode is on.
So this script reconstructs an *independent* QuadEKF instance offline, from data
logged during an ordinary, unchanged, safe flight, and replays it tick-by-tick in
four configurations:

    integration_only : predict + accel + zupt only (raw IMU)
    +model            : + model_predict (physics) blend into the predict accel
    +vision            : + PnP vision position/velocity/yaw fixes
    full                : both of the above

Since the replay is a brand-new EKF instance, it doesn't need imu_ekf.py's
R_align/pos_offset_ned frame-alignment machinery — it's seeded directly from
raw (unrotated) ground truth at the hover-reset instant, so GT and replay share
a frame by construction.

Data required (added to ekf.csv / vision_fix.csv this session)
-----------------------------------------------------------------
ekf.csv       : gt_pN/E/D, gt_vN/E/D, gt_qw/x/y/z, T_total_N, actuator_sum,
                hover_reset_done, wait_phase_done  (logged every tick, independent
                of ground_truth_mode/use_model_predict/model_predict_shadow).
vision_fix.csv: raw shared['_vision_ekf_update'] fixes (pos/vel/yaw + sigma/gate),
                logged whenever vision_rx.py produces one — independent of
                ground_truth_mode too (that flag only gates whether imu_ekf.py
                *consumes* the fix, not whether vision_rx.py produces it).

Usage
-----
    python ekf_shadow.py                          # auto-find latest logs/*/ekf.csv
    python ekf_shadow.py logs/<session>            # explicit session dir
    python ekf_shadow.py logs/<session> --write    # write fitted sigmas to params.yaml
    python ekf_shadow.py --fit-config +model       # fit against a config other than 'full'
    python ekf_shadow.py --fit-stride 10           # subsample the fit residual (default 5)
    python ekf_shadow.py --fit-mode gap            # fit only inter-vision-fix drift (see below)

--fit-mode full (default): residual is every tick's pos/vel error, stride-
    subsampled. On a vision-dense flight this buries the blend's signal — most
    ticks sit right after a correction with near-zero error, so the sum-of-
    squares barely responds to sigma_acc_imu/model (seen in practice: RMS
    changed ~2% while sigma swung 10x/6x to its bounds — a flat landscape, not
    a real preference).
--fit-mode gap: residual is one sample per inter-vision-fix gap (the error
    right before the NEXT correction arrives) — isolates exactly how much the
    blend lets the estimate wander between corrections, which is the actual
    quantity you're trying to tune. Requires --fit-config full or +vision
    (gap boundaries are defined by applied vision corrections).
"""

import os
import sys
import csv
import glob
from collections import deque

import numpy as np
from scipy.optimize import least_squares
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml

from ekf import QuadEKF
from dyn import load_params, _quat_to_R


# ── 1. Locate + load session ──────────────────────────────────────────────────

def find_latest_session():
    candidates = glob.glob('logs/*/ekf.csv')
    if not candidates:
        raise FileNotFoundError(
            'No ekf.csv found under logs/. Run a flight first (logging: 1, the '
            'default) — ekf_shadow.py needs the gt_*/T_total_N/actuator_sum/'
            'hover_reset_done/wait_phase_done columns added this session.')
    return os.path.dirname(max(candidates, key=os.path.getmtime))


def _load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline='') as f:
        rows = []
        for row in csv.DictReader(f):
            try:
                rows.append({k: float(v) for k, v in row.items()})
            except ValueError:
                continue
        return rows


CONFIGS = {
    'integration_only': dict(use_model=False, use_vision=False),
    '+model':           dict(use_model=True,  use_vision=False),
    '+vision':          dict(use_model=False, use_vision=True),
    'full':             dict(use_model=True,  use_vision=True),
}
CONFIG_COLORS = {
    'integration_only': '#999999',
    '+model':           'tab:blue',
    '+vision':          'tab:orange',
    'full':             'tab:red',
}


# ── 2. Replay engine ───────────────────────────────────────────────────────────

def run_replay(sigma_acc_imu, sigma_acc_model, use_model, use_vision,
               t_wall, t0_wall, dt_arr, gyro, acc, T_total,
               ekf0_pos, ekf0_vel, ekf0_quat, gyro_bias0,
               param, raw_params, vis_events, gt_pos0):
    """Replay one config over the logged flight. Returns a dict of (N,) / (N,3) arrays."""
    N = len(t_wall)
    ekf = QuadEKF(sigma_bias_z_proc=float(raw_params.get('sigma_bias_z_proc', 5e-6)))
    ekf.x[0:3]   = ekf0_pos
    ekf.x[3:6]   = ekf0_vel
    ekf.x[6:10]  = ekf0_quat
    ekf.x[10:13] = gyro_bias0

    Dv         = param['Dv']
    m          = float(param['m'])
    tau_motor  = float(param['tau_motor'])
    accel_bias = np.array(raw_params.get('accel_bias', [0.0, 0.0, 0.0]), dtype=float)

    accel_settle_sec  = float(raw_params.get('accel_settle_sec', 0.50))
    vision_settle_sec = float(raw_params.get('vision_settle_sec', 1.50))
    yaw_ramp_sec      = float(raw_params.get('vision_yaw_confidence_ramp_sec', 2.5))
    yaw_sigma_initial = float(raw_params.get('vision_yaw_sigma_initial', 3.0))
    vision_latency_s  = float(raw_params.get('vision_latency_s', 0.0))

    T_actual = None   # motor-lag filter state (mirrors imu_ekf.py's self._T_actual)

    pos  = np.empty((N, 3)); vel = np.empty((N, 3)); quat = np.empty((N, 4))
    acc_applied  = np.zeros(N); acc_innov  = np.zeros(N)
    zupt_applied = np.zeros(N); zupt_innov = np.zeros(N)
    vis_pos_applied = np.zeros(N); vis_pos_innov = np.zeros(N)
    vis_vel_applied = np.zeros(N); vis_vel_innov = np.zeros(N)
    vis_yaw_applied = np.zeros(N); vis_yaw_innov = np.zeros(N)

    pos[0] = ekf.x[0:3]; vel[0] = ekf.x[3:6]; quat[0] = ekf.x[6:10]

    state_buf = deque(maxlen=200)   # (wall_t, pos) — vision latency compensation
    state_buf.append((t_wall[0], ekf.x[0:3].copy()))

    vis_ptr = 0
    n_vis = len(vis_events)

    for i in range(1, N):
        dt = dt_arr[i - 1]
        g = gyro[i]; a = acc[i]

        if use_model and not np.isnan(T_total[i]):
            T_cmd = T_total[i]
            T_actual = T_cmd if T_actual is None else \
                T_actual + (T_cmd - T_actual) * np.clip(dt / max(tau_motor, 1e-6), 0.0, 1.0)
            R_nb = _quat_to_R(ekf.x[6:10])
            v_b = R_nb @ ekf.x[3:6]
            F_drag = -Dv @ (np.abs(v_b) * v_b)
            acc_model = (np.array([0.0, 0.0, -T_actual]) + F_drag) / m + accel_bias
            s2_imu, s2_mdl = sigma_acc_imu ** 2, sigma_acc_model ** 2
            w_imu = s2_mdl / (s2_imu + s2_mdl)
            acc_predict = w_imu * a + (1.0 - w_imu) * acc_model
        else:
            acc_predict = a

        ekf.predict(g, acc_predict, dt)
        state_buf.append((t_wall[i], ekf.x[0:3].copy()))

        if (t_wall[i] - t0_wall) < accel_settle_sec:
            _ap, _ai = ekf.update_accel(a)
        else:
            _ap, _ai = False, 0.0
        acc_applied[i], acc_innov[i] = float(_ap), _ai
        # ZUPT is intentionally NOT replayed: every real flight script in this
        # codebase sets shared['zupt_enabled'] = False once airborne (ZUPT's
        # "stationary" assumption is only valid on the ground) — and this replay
        # starts exactly at hover_reset_done, i.e. already airborne. Calling it
        # unconditionally here would fire it whenever velocity dips under its
        # 4 m/s gate mid-flight and yank the estimate toward zero, which is
        # exactly why live flights turn it off. (zupt_applied/zupt_innov stay
        # zero — the responsibility plot's "zupt" channel will correctly show
        # no events for an airborne replay.)

        if use_vision:
            while vis_ptr < n_vis and vis_events[vis_ptr]['wall_t'] <= t_wall[i]:
                vr = vis_events[vis_ptr]
                vis_ptr += 1
                if (t_wall[i] - t0_wall) <= vision_settle_sec:
                    continue
                if int(vr['has_pos']):
                    local_pos = np.array([vr['pos_N'], vr['pos_E'], vr['pos_D']]) - gt_pos0
                    if vision_latency_s > 0 and len(state_buf) >= 2:
                        t_img = vr['wall_t'] - vision_latency_s
                        pos_hist = state_buf[0][1]
                        for bt, bp in state_buf:
                            if bt >= t_img:
                                pos_hist = bp
                                break
                        local_pos = local_pos + (ekf.x[0:3] - pos_hist)
                    ap, ai = ekf.update_position(local_pos, sigma_pos=vr['sigma_pos'], gate_dist=vr['gate'])
                    vis_pos_applied[i], vis_pos_innov[i] = float(ap), ai
                if int(vr['has_vel']):
                    ap, ai = ekf.update_velocity(
                        [vr['vel_N'], vr['vel_E'], vr['vel_D']],
                        sigma_vel=vr['sigma_vel'], gate_dist=vr['vel_gate'])
                    vis_vel_applied[i], vis_vel_innov[i] = float(ap), ai
                if int(vr['has_yaw']):
                    t_since_settle = (t_wall[i] - t0_wall) - vision_settle_sec
                    ramp = np.clip(t_since_settle / max(yaw_ramp_sec, 1e-6), 0.0, 1.0)
                    sigma_yaw_eff = yaw_sigma_initial + (vr['sigma_yaw'] - yaw_sigma_initial) * ramp
                    ap, ai = ekf.update_yaw(vr['yaw_ned'], sigma_yaw=sigma_yaw_eff,
                                            gate_dist=vr.get('yaw_gate', 1.0))
                    vis_yaw_applied[i], vis_yaw_innov[i] = float(ap), ai

        pos[i] = ekf.x[0:3]; vel[i] = ekf.x[3:6]; quat[i] = ekf.x[6:10]

    return dict(pos=pos, vel=vel, quat=quat,
                acc_applied=acc_applied, acc_innov=acc_innov,
                zupt_applied=zupt_applied, zupt_innov=zupt_innov,
                vis_pos_applied=vis_pos_applied, vis_pos_innov=vis_pos_innov,
                vis_vel_applied=vis_vel_applied, vis_vel_innov=vis_vel_innov,
                vis_yaw_applied=vis_yaw_applied, vis_yaw_innov=vis_yaw_innov)


# ── 3. Data preparation ────────────────────────────────────────────────────────

def prepare(session_dir):
    rows = _load_csv(os.path.join(session_dir, 'ekf.csv'))
    if not rows:
        raise ValueError(f'{session_dir}/ekf.csv is empty or missing.')
    if 'hover_reset_done' not in rows[0]:
        raise ValueError(
            f'{session_dir}/ekf.csv predates this session\'s logging changes '
            '(no hover_reset_done column) — re-fly to generate a compatible log.')

    t0_idx = next((i for i, r in enumerate(rows) if int(r['hover_reset_done']) == 1), None)
    if t0_idx is None:
        raise ValueError('hover_reset_done never reaches 1 in this log — flight '
                          'never left WAIT, nothing to replay.')

    wait_rows = [r for r in rows[:t0_idx] if int(r['wait_phase_done']) == 0]
    if len(wait_rows) > 50:
        gyro_bias0 = np.clip(
            [np.mean([r['gx'] for r in wait_rows]),
             np.mean([r['gy'] for r in wait_rows]),
             np.mean([r['gz'] for r in wait_rows])],
            -0.05, 0.05)
    else:
        gyro_bias0 = np.zeros(3)

    r0 = rows[t0_idx]
    gt_pos0   = np.array([r0['gt_pN'], r0['gt_pE'], r0['gt_pD']])
    ekf0_pos  = np.zeros(3)
    ekf0_vel  = np.array([r0['gt_vN'], r0['gt_vE'], r0['gt_vD']])
    ekf0_quat = np.array([r0['gt_qw'], r0['gt_qx'], r0['gt_qy'], r0['gt_qz']])

    flight = rows[t0_idx:]
    t_us   = np.array([r['t_us'] for r in flight])
    t_wall = np.array([r['wall_t'] for r in flight])
    dt_arr = np.clip(np.diff(t_us) / 1e6, 0.0005, 0.05)
    gyro   = np.array([[r['gx'], r['gy'], r['gz']] for r in flight])
    acc    = np.array([[r['ax'], r['ay'], r['az']] for r in flight])
    T_total  = np.array([r['T_total_N'] for r in flight])
    gt_pos = np.array([[r['gt_pN'], r['gt_pE'], r['gt_pD']] for r in flight]) - gt_pos0
    gt_vel = np.array([[r['gt_vN'], r['gt_vE'], r['gt_vD']] for r in flight])

    vis_rows = _load_csv(os.path.join(session_dir, 'vision_fix.csv'))
    if not vis_rows:
        print('vision_fix.csv missing/empty — +vision and full configs will '
              'degrade to their non-vision counterparts.')
    t0_wall = t_wall[0]
    vis_events = sorted([v for v in vis_rows if v['wall_t'] >= t0_wall],
                        key=lambda v: v['wall_t'])

    return dict(t_wall=t_wall, t0_wall=t0_wall, dt_arr=dt_arr, gyro=gyro, acc=acc,
                T_total=T_total, ekf0_pos=ekf0_pos, ekf0_vel=ekf0_vel,
                ekf0_quat=ekf0_quat, gyro_bias0=gyro_bias0, gt_pos0=gt_pos0,
                gt_pos=gt_pos, gt_vel=gt_vel, vis_events=vis_events,
                n_vis_total=len(vis_rows))


# ── 4. Fitting regime ──────────────────────────────────────────────────────────

def _gap_residual(out, gt_pos, gt_vel):
    """Residual over inter-vision-fix gaps only: for each gap between two
    consecutive applied position/velocity corrections, take the error at the
    tick right before the NEXT correction arrives (the worst drift the blend
    accumulated before vision bailed it out) — one sample per gap, rather than
    one per tick.

    This exists because fitting against every tick of a vision-dense flight
    (see 'full'-config fit_stride residual below) buries the blend's signal:
    right after each of the ~700+ corrections a typical flight gets, pos_err
    is tiny, and those many near-zero ticks dominate the sum-of-squares over
    the few ticks that actually reveal how much the blend let the estimate
    wander. Concentrating on one worst-case-per-gap sample isolates exactly
    the quantity the blend controls.
    """
    applied = (out['vis_pos_applied'] > 0.5) | (out['vis_vel_applied'] > 0.5)
    fix_idxs = np.where(applied)[0]
    if len(fix_idxs) < 2:
        return np.array([])
    resid = []
    for i in range(len(fix_idxs) - 1):
        gap_end = fix_idxs[i + 1] - 1
        if gap_end <= fix_idxs[i]:
            continue   # back-to-back corrections, no gap to measure
        resid.append(out['pos'][gap_end] - gt_pos[gap_end])
        resid.append(out['vel'][gap_end] - gt_vel[gap_end])
    return np.concatenate(resid) if resid else np.array([])


def fit_sigmas(data, param, raw_params, fit_config, fit_stride, fit_mode='full'):
    cfg = CONFIGS[fit_config]
    if fit_mode == 'gap' and not cfg['use_vision']:
        raise ValueError(
            f'--fit-mode gap requires a vision-including --fit-config '
            f'(+vision or full), got "{fit_config}" — gap boundaries are '
            f'defined by applied vision corrections.')

    def residual(p):
        sigma_acc_imu, sigma_acc_model = p
        out = run_replay(sigma_acc_imu, sigma_acc_model,
                          cfg['use_model'], cfg['use_vision'],
                          data['t_wall'], data['t0_wall'], data['dt_arr'],
                          data['gyro'], data['acc'], data['T_total'],
                          data['ekf0_pos'], data['ekf0_vel'], data['ekf0_quat'],
                          data['gyro_bias0'], param, raw_params,
                          data['vis_events'], data['gt_pos0'])
        if fit_mode == 'gap':
            r = _gap_residual(out, data['gt_pos'], data['gt_vel'])
            if r.size == 0:
                raise ValueError('No inter-fix gaps found (need >= 2 applied '
                                 'vision corrections) — cannot use --fit-mode gap.')
            return r
        pos_err = out['pos'] - data['gt_pos']
        vel_err = out['vel'] - data['gt_vel']
        return np.concatenate([pos_err[::fit_stride].ravel(),
                                vel_err[::fit_stride].ravel()])

    p0 = [float(raw_params.get('sigma_acc_imu', 0.5)),
          float(raw_params.get('sigma_acc_model', 0.3))]
    bounds = ([0.05, 0.05], [5.0, 5.0])

    r0 = residual(p0)
    _stride_note = f'stride={fit_stride}' if fit_mode == 'full' else f'{len(r0)//6} gaps'
    print(f'\nFitting sigma_acc_imu/sigma_acc_model against "{fit_config}" config '
          f'(mode={fit_mode}, {_stride_note})')
    print(f'Initial RMS error: {np.sqrt(np.mean(r0 ** 2)):.4f} (m and m/s mixed)')

    result = least_squares(residual, p0, bounds=bounds, method='trf',
                           loss='soft_l1', f_scale=1.0, verbose=1)
    rms_after = np.sqrt(np.mean(result.fun ** 2))
    print(f'Final   RMS error: {rms_after:.4f}')

    sigma_acc_imu_opt, sigma_acc_model_opt = result.x
    print(f'  sigma_acc_imu   = {sigma_acc_imu_opt:.4f}  (was {p0[0]:.4f})')
    print(f'  sigma_acc_model = {sigma_acc_model_opt:.4f}  (was {p0[1]:.4f})')

    pinned = []
    for name, val, lo, hi in zip(['sigma_acc_imu', 'sigma_acc_model'], result.x,
                                  bounds[0], bounds[1]):
        span = hi - lo
        if val - lo < 0.01 * span or hi - val < 0.01 * span:
            pinned.append(name)
    if pinned:
        print(f'  WARNING: pinned at bound (underdetermined by this data, not a '
              f'converged estimate): {", ".join(pinned)}')

    return sigma_acc_imu_opt, sigma_acc_model_opt, pinned


# ── 5. Visualization ────────────────────────────────────────────────────────────

def plot_trajectories(out_dir, t, data, results):
    fig, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=True)
    labels = ['pN [m]', 'pE [m]', 'altitude = -pD [m]', '|pos err| [m]']
    gt_p = data['gt_pos']
    for name, out in results.items():
        c = CONFIG_COLORS[name]
        p = out['pos']
        axes[0].plot(t, p[:, 0], color=c, lw=1.0, label=name)
        axes[1].plot(t, p[:, 1], color=c, lw=1.0, label=name)
        axes[2].plot(t, -p[:, 2], color=c, lw=1.0, label=name)
        err = np.linalg.norm(p - gt_p, axis=1)
        axes[3].plot(t, err, color=c, lw=1.0, label=name)
    axes[0].plot(t, gt_p[:, 0], 'k--', lw=1.3, label='GT')
    axes[1].plot(t, gt_p[:, 1], 'k--', lw=1.3, label='GT')
    axes[2].plot(t, -gt_p[:, 2], 'k--', lw=1.3, label='GT')
    for ax, lbl in zip(axes, labels):
        ax.set_ylabel(lbl, fontsize=9)
        ax.grid(True, lw=0.3)
    axes[0].legend(fontsize=7, loc='upper right', ncol=5)
    axes[-1].set_xlabel('time [s]')
    fig.suptitle('ekf_shadow: position vs ground truth, by config', fontsize=12)
    fig.tight_layout()
    path = os.path.join(out_dir, 'ekf_shadow_trajectories.png')
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f'Trajectories plot -> {path}')


def plot_velocity(out_dir, t, data, results):
    fig, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=True)
    labels = ['vN [m/s]', 'vE [m/s]', 'vD [m/s]', '|vel err| [m/s]']
    gt_v = data['gt_vel']
    for name, out in results.items():
        c = CONFIG_COLORS[name]
        v = out['vel']
        axes[0].plot(t, v[:, 0], color=c, lw=1.0, label=name)
        axes[1].plot(t, v[:, 1], color=c, lw=1.0, label=name)
        axes[2].plot(t, v[:, 2], color=c, lw=1.0, label=name)
        err = np.linalg.norm(v - gt_v, axis=1)
        axes[3].plot(t, err, color=c, lw=1.0, label=name)
    axes[0].plot(t, gt_v[:, 0], 'k--', lw=1.3, label='GT')
    axes[1].plot(t, gt_v[:, 1], 'k--', lw=1.3, label='GT')
    axes[2].plot(t, gt_v[:, 2], 'k--', lw=1.3, label='GT')
    for ax, lbl in zip(axes, labels):
        ax.set_ylabel(lbl, fontsize=9)
        ax.grid(True, lw=0.3)
    axes[0].legend(fontsize=7, loc='upper right', ncol=5)
    axes[-1].set_xlabel('time [s]')
    fig.suptitle('ekf_shadow: velocity vs ground truth, by config', fontsize=12)
    fig.tight_layout()
    path = os.path.join(out_dir, 'ekf_shadow_velocity.png')
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f'Velocity plot -> {path}')


def plot_responsibility(out_dir, t, data, full_out):
    pos_err = np.linalg.norm(full_out['pos'] - data['gt_pos'], axis=1)

    channels = [
        ('accel',      full_out['acc_applied'],     full_out['acc_innov']),
        ('zupt',       full_out['zupt_applied'],     full_out['zupt_innov']),
        ('vision_pos', full_out['vis_pos_applied'],  full_out['vis_pos_innov']),
        ('vision_vel', full_out['vis_vel_applied'],  full_out['vis_vel_innov']),
        ('vision_yaw', full_out['vis_yaw_applied'],  full_out['vis_yaw_innov']),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                             gridspec_kw={'height_ratios': [1, 1.3]})

    axes[0].plot(t, pos_err, color='tab:red', lw=1.0)
    axes[0].set_ylabel('|pos err|\n[m] (full)', fontsize=9)
    axes[0].grid(True, lw=0.3)

    ax = axes[1]
    for row_i, (name, applied, innov) in enumerate(channels):
        mask = applied > 0.5
        if not np.any(mask):
            continue
        innov_m = innov[mask]
        span = max(float(np.max(innov_m)), 1e-6)
        sizes = 10 + 60 * (innov_m / span)
        ax.scatter(t[mask], np.full(int(mask.sum()), row_i),
                   s=sizes, alpha=0.6, color=f'C{row_i}')
    ax.set_yticks(range(len(channels)))
    ax.set_yticklabels([c[0] for c in channels])
    ax.set_xlabel('time [s]')
    ax.set_ylabel('correction applied\n(marker size = |innovation|)', fontsize=9)
    ax.grid(True, lw=0.3, axis='x')

    fig.suptitle('ekf_shadow: what is correcting the "full" estimate, over time',
                 fontsize=12)
    fig.tight_layout()
    path = os.path.join(out_dir, 'ekf_shadow_responsibility.png')
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f'Responsibility plot -> {path}')


# ── 6. Entry point ──────────────────────────────────────────────────────────────

def main():
    args = list(sys.argv[1:])
    do_write = '--write' in args
    fit_config = 'full'
    fit_stride = 5
    fit_mode = 'full'
    session_arg = None

    i = 0
    while i < len(args):
        if args[i] == '--fit-config':
            fit_config = args[i + 1]; i += 2; continue
        if args[i] == '--fit-stride':
            fit_stride = int(args[i + 1]); i += 2; continue
        if args[i] == '--fit-mode':
            fit_mode = args[i + 1]; i += 2; continue
        if args[i] == '--write':
            i += 1; continue
        if not args[i].startswith('--'):
            session_arg = args[i]; i += 1; continue
        i += 1

    if fit_config not in CONFIGS:
        raise ValueError(f'--fit-config must be one of {list(CONFIGS)}')
    if fit_mode not in ('full', 'gap'):
        raise ValueError('--fit-mode must be "full" or "gap"')

    session_dir = session_arg or find_latest_session()
    if session_dir.endswith('.csv'):
        session_dir = os.path.dirname(session_dir)
    print(f'Session: {session_dir}')

    data = prepare(session_dir)
    print(f'Replay window: {len(data["t_wall"])} ticks, '
          f'{data["t_wall"][-1] - data["t0_wall"]:.1f} s, '
          f'{len(data["vis_events"])} vision fixes in window '
          f'({data["n_vis_total"]} total in log)')

    param = load_params('params.yaml')
    with open('params.yaml') as f:
        raw_params = yaml.safe_load(f)

    sigma_acc_imu_opt, sigma_acc_model_opt, pinned = fit_sigmas(
        data, param, raw_params, fit_config, fit_stride, fit_mode)

    print('\nRunning all 4 configs for comparison plots...')
    results = {}
    for name, cfg in CONFIGS.items():
        results[name] = run_replay(
            sigma_acc_imu_opt, sigma_acc_model_opt,
            cfg['use_model'], cfg['use_vision'],
            data['t_wall'], data['t0_wall'], data['dt_arr'],
            data['gyro'], data['acc'], data['T_total'],
            data['ekf0_pos'], data['ekf0_vel'], data['ekf0_quat'],
            data['gyro_bias0'], param, raw_params,
            data['vis_events'], data['gt_pos0'])

    t = data['t_wall'] - data['t0_wall']
    plot_trajectories(session_dir, t, data, results)
    plot_velocity(session_dir, t, data, results)
    plot_responsibility(session_dir, t, data, results['full'])

    if not do_write:
        ans = input('\nWrite fitted sigma_acc_imu/sigma_acc_model to params.yaml? '
                    '[y/N] ').strip().lower()
        do_write = (ans == 'y')

    if do_write:
        if pinned:
            print(f'Refusing to write — pinned at bound ({", ".join(pinned)}), '
                  f'not a converged estimate.')
        else:
            raw_params['sigma_acc_imu']   = round(float(sigma_acc_imu_opt), 4)
            raw_params['sigma_acc_model'] = round(float(sigma_acc_model_opt), 4)
            with open('params.yaml', 'w') as f:
                yaml.dump(raw_params, f, default_flow_style=None, sort_keys=False,
                          allow_unicode=True)
            print('params.yaml updated (sigma_acc_imu, sigma_acc_model).')
    else:
        print('params.yaml NOT modified.')


if __name__ == '__main__':
    main()
