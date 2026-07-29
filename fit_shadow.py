"""
fit_shadow.py  —  Identify translational physics parameters from a shadow log.

Fits [Dv_xy, Dv_z, T_max_motor, tau_motor, ba_x, ba_y, ba_z] by minimising the
residual between the physics model and the raw IMU accelerometer captured in
model_predict_shadow.csv. Body velocity is taken from the sim's ground truth
(gt_vb_x/y/z) wherever available, falling back to the EKF-derived vb_x/y/z
otherwise. Dv_x/Dv_y are tied to a single Dv_xy by default (a symmetric
X-frame should have equal x/y drag; pass --free-dv to fit them independently
if the log has real lateral excitation). Total mass m is held fixed at its
params.yaml value by default (pass --free-mass to fit it jointly instead —
see the note below on why that's degenerate on low-excitation data).

Model (body frame, FRD, specific force):
    acc_model  = ([0, 0, -T_actual] + F_drag) / m + ba
    F_drag     = -diag(Dv) @ (|v_b| * v_b)
    T_cmd      = actuator_sum * T_max_motor
    T_actual   = T_cmd passed through a first-order lag with time-constant tau_motor
    ba         = constant accelerometer bias [ba_x, ba_y, ba_z]

tau_motor is applied by imu_ekf.py's live model_predict path (not just this
offline fit). ba is opt-in only — imu_ekf.py reads an optional accel_bias key
from params.yaml (defaults to zero), written only via --write-bias, since a
fitted bias is only trustworthy when it didn't pin at its bound (see the
WARNING this script prints for any pinned parameter).

Non-observable from translational data (need rotational sysid):
    m_motor  → Ixx/Iyy/Izz via geometry
    kappa    → yaw torque coefficient
    Dw       → rotational drag
    (see flight_sysid_rot.py)

Usage:
    python fit_shadow.py                        # auto-find latest shadow CSV
    python fit_shadow.py path/to/shadow.csv
    python fit_shadow.py path/to/shadow.csv --write         # write Dv/m/tau_motor/T_max_motor
    python fit_shadow.py path/to/shadow.csv --write --write-bias   # also write accel_bias
    python fit_shadow.py path/to/shadow.csv --free-mass  # fit m instead of holding it fixed
    python fit_shadow.py path/to/shadow.csv --free-dv    # fit Dv_x/Dv_y independently
"""

import sys
import csv
import os
import glob

import numpy as np
from scipy.optimize import least_squares
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml


# ── 1. Locate shadow CSV ──────────────────────────────────────────────────────

def find_latest_shadow():
    candidates = glob.glob('logs/*/model_predict_shadow.csv')
    if not candidates:
        raise FileNotFoundError(
            'No model_predict_shadow.csv found under logs/. '
            'Run main.py with model_predict_shadow: true first.')
    return max(candidates, key=os.path.getmtime)


csv_path = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('--') \
           else find_latest_shadow()
print(f'Loading: {csv_path}')
out_dir = os.path.dirname(csv_path)


# ── 2. Load and filter ────────────────────────────────────────────────────────

rows = []
with open(csv_path, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            rows.append({k: float(v) for k, v in row.items()})
        except ValueError:
            continue

if len(rows) < 10:
    raise ValueError(f'Too few rows ({len(rows)}) — need at least 10.')

# Filter: only rows where motors are meaningfully running
T_total_raw = np.array([r['T_total_N']    for r in rows])
mask        = T_total_raw > 10.0
rows_f      = [r for r, m in zip(rows, mask) if m]
N           = len(rows_f)
print(f'Total rows: {len(rows)}  →  after T>10N filter: {N}')
if N < 5:
    raise ValueError('Too few rows after filtering.')

t          = np.array([r['t_wall_s']     for r in rows_f]) - rows_f[0]['t_wall_s']
T_total    = np.array([r['T_total_N']    for r in rows_f])
vel_b      = np.column_stack([
                 [r['vb_x'] for r in rows_f],
                 [r['vb_y'] for r in rows_f],
                 [r['vb_z'] for r in rows_f]])
acc_imu    = np.column_stack([
                 [r['acc_imu_x'] for r in rows_f],
                 [r['acc_imu_y'] for r in rows_f],
                 [r['acc_imu_z'] for r in rows_f]])

# Ground-truth body velocity (independent of the EKF's own state estimate) —
# prefer this over vel_b wherever the sim's GT mavlink messages were available.
if 'gt_vb_x' in rows_f[0]:
    gt_vb = np.column_stack([
                 [r['gt_vb_x'] for r in rows_f],
                 [r['gt_vb_y'] for r in rows_f],
                 [r['gt_vb_z'] for r in rows_f]])
    gt_valid = ~np.any(np.isnan(gt_vb), axis=1)
    n_gt = int(np.sum(gt_valid))
    vel_b = np.where(gt_valid[:, None], gt_vb, vel_b)
    print(f'Ground-truth body velocity available for {n_gt}/{N} rows — '
          f'used GT where present, EKF-derived vb elsewhere.')
else:
    print('gt_vb_x column not found (old log) — using EKF-derived vb_x/y/z only.')

# actuator_sum: raw 0-1 motor fraction sum (logged separately for T_max fitting)
has_act_sum = 'actuator_sum' in rows_f[0]
if has_act_sum:
    act_sum = np.array([r['actuator_sum'] for r in rows_f])
    print('actuator_sum column found — fitting T_max_motor as free parameter.')
else:
    act_sum = None
    print('actuator_sum column missing (old log) — T_max_motor fixed to T_total/4 avg.')


# ── 3. Residual function ──────────────────────────────────────────────────────

# Nominal motor lag time-constant from params.yaml — used as the fit's initial
# guess for tau_motor (dyn.dyn_lag defines this ODE; the shadow model and
# fit_shadow.py previously ignored it and treated commanded thrust as instant).
with open('params.yaml') as f:
    _params_raw = yaml.safe_load(f)
_tau_motor_nom = float(_params_raw.get('tau_motor', 0.05))


def apply_motor_lag(T_cmd, t_arr, tau):
    """Causal first-order lag: dT_actual/dt = (T_cmd - T_actual) / tau.

    Filters the commanded thrust signal using the true (irregular) sample
    spacing in t_arr, so the model compares against what the motors could
    actually have produced rather than treating T_cmd as instantaneous.
    """
    T_act = np.empty_like(T_cmd)
    T_act[0] = T_cmd[0]
    for i in range(1, len(T_cmd)):
        alpha = np.clip((t_arr[i] - t_arr[i - 1]) / max(tau, 1e-6), 0.0, 1.0)
        T_act[i] = T_act[i - 1] + (T_cmd[i] - T_act[i - 1]) * alpha
    return T_act


def model_acc(Dv, m, T_total_vec, bias):
    """Body-frame specific force predicted by the physics model.

    bias: constant accelerometer offset [ba_x, ba_y, ba_z] — real
    accelerometers rarely read exactly zero-bias; without this term any
    persistent offset gets incorrectly absorbed into Dv/m.
    """
    F_drag  = -(Dv * np.abs(vel_b) * vel_b)          # (N, 3) element-wise
    thrust  = np.column_stack([np.zeros(N), np.zeros(N), -T_total_vec])
    return (thrust + F_drag) / m + bias                # (N, 3)


# Hold total mass fixed at its params.yaml value by default. m, T_max_motor
# and ba_z all appear only in the z-axis hover equation (acc_z ≈ -T_total/m +
# ba_z), so with m free they're only identifiable up to a joint manifold — on
# data with little velocity excitation (e.g. mostly hover) this shows up as
# ba_z pinning at its bound while m/T_max_motor drift to implausible values.
# Since mass is externally confirmed correct, fixing it removes that
# degeneracy. Pass --free-mass to restore the old behaviour (m as a free
# parameter) if you specifically want to re-derive mass here.
fix_mass = '--free-mass' not in sys.argv
m_fixed_val = float(_params_raw['m'])
if fix_mass:
    print(f'Mass held fixed at params.yaml value: {m_fixed_val:.4f} kg  '
          '(pass --free-mass to fit it instead)')
else:
    print('--free-mass: fitting m jointly (may be degenerate with T_max_motor/ba_z '
          'on low-excitation data)')

# Tie Dv_x = Dv_y by default (single Dv_xy parameter, mirroring the Dw_xy/Dw_z
# convention already used for rotational drag elsewhere in this codebase).
# For a symmetric X-frame the two really should be the same physical drag
# coefficient; fitting them independently only makes sense with genuine
# lateral (y) excitation in the log. Without it, Dv_y is barely constrained
# by the data (contributes ~0 to the residual when v_y stays small) and the
# optimiser can park it on an arbitrary large value, as seen when vb_y's std
# is much smaller than vb_x's. Pass --free-dv to fit Dv_x/Dv_y independently.
tie_dv = '--free-dv' not in sys.argv
if tie_dv:
    _vy_std, _vx_std = float(np.std(vel_b[:, 1])), float(np.std(vel_b[:, 0]))
    print(f'Dv_x/Dv_y tied (shared Dv_xy)  [vb_x std={_vx_std:.3f}  '
          f'vb_y std={_vy_std:.3f} m/s]  (pass --free-dv to fit independently)')
else:
    print('--free-dv: fitting Dv_x/Dv_y independently')

# Free-parameter spec: (name, p0, lo, hi). p0 is seeded from the CURRENT
# params.yaml values (not hardcoded literals) so "model (before)" in the
# residual plot actually reflects what's live right now, and so re-running
# against a new log after a --write starts from where the last fit left off
# rather than always restarting from the same fixed guess.
_Dv_nom = [float(v) for v in _params_raw.get('Dv', [1.2, 1.2, 1.2])]
if tie_dv:
    _spec = [('Dv_xy', (_Dv_nom[0] + _Dv_nom[1]) / 2.0, 0.0, 20.0),
             ('Dv_z', _Dv_nom[2], 0.0, 20.0)]
else:
    _spec = [('Dv_x', _Dv_nom[0], 0.0, 20.0), ('Dv_y', _Dv_nom[1], 0.0, 20.0),
              ('Dv_z', _Dv_nom[2], 0.0, 20.0)]
if not fix_mass:
    _spec.append(('m', float(_params_raw.get('m', 5.405)), 1.0, 15.0))
if has_act_sum:
    _spec.append(('T_max_motor', float(_params_raw.get('T_max_motor', 49.9)), 5.0, 200.0))
_spec.append(('tau_motor', _tau_motor_nom, 0.005, 0.3))
# Bias bounds are kept to a physically-realistic sensor-bias range (typical
# MEMS accel bias is well under 0.1 m/s²). Wider bounds let ba_z become a
# dumping ground for unrelated model error whenever T_max_motor is only
# weakly identifiable (e.g. a log with little thrust dynamic range once
# transients are excluded) — T_max_motor/ba_z then trade off almost losslessly
# and ba_z drifts to whatever bound is given rather than converging. Keeping
# it tight forces that mismatch to show up as visible residual/warnings
# instead of being silently absorbed.
_BA_BOUND = 0.15
_spec += [('ba_x', 0.0, -_BA_BOUND, _BA_BOUND), ('ba_y', 0.0, -_BA_BOUND, _BA_BOUND),
          ('ba_z', 0.0, -_BA_BOUND, _BA_BOUND)]

param_names = [s[0] for s in _spec]
p0          = [s[1] for s in _spec]
bounds      = ([s[2] for s in _spec], [s[3] for s in _spec])


def _unpack(p):
    d     = dict(zip(param_names, p))
    Dv    = np.array([d['Dv_xy'], d['Dv_xy'], d['Dv_z']]) if tie_dv \
            else np.array([d['Dv_x'], d['Dv_y'], d['Dv_z']])
    m     = d['m'] if not fix_mass else m_fixed_val
    T_max = d['T_max_motor'] if has_act_sum else None
    tau   = d['tau_motor']
    bias  = np.array([d['ba_x'], d['ba_y'], d['ba_z']])
    return Dv, m, T_max, tau, bias


def residual(p):
    Dv, m, T_max, tau, bias = _unpack(p)
    T_cmd = (act_sum * T_max) if has_act_sum else T_total
    T_fit = apply_motor_lag(T_cmd, t, tau)
    return (model_acc(Dv, m, T_fit, bias) - acc_imu).ravel()


# ── 4. Optimise ───────────────────────────────────────────────────────────────

r0     = residual(p0)
rms_before = float(np.sqrt(np.mean(r0**2)))
print(f'\nInitial RMS residual: {rms_before:.4f} m/s²')

# Robust loss: real flight logs contain brief high-g transients (gate
# collisions, aggressive maneuvers) that this steady-flight |v|v drag model
# was never meant to capture. Under the default L2 loss those few spike
# samples (10-20 m/s² vs a normal ~0.5 m/s² residual) dominate the cost and
# drag tau_motor/ba_z toward whatever partially blunts the spikes rather than
# fitting the well-behaved majority of the data. soft_l1 downweights
# residuals beyond f_scale so those outliers stop steering the fit.
F_SCALE = 1.0   # [m/s²] residual magnitude beyond which outliers are downweighted
result = least_squares(residual, p0, bounds=bounds, method='trf',
                       loss='soft_l1', f_scale=F_SCALE,
                       ftol=1e-9, xtol=1e-9, verbose=1)

rms_after = float(np.sqrt(np.mean(result.fun**2)))
print(f'Final   RMS residual: {rms_after:.4f} m/s²')
print(f'Improvement: {100*(1-rms_after/rms_before):.1f}%\n')

for name, val in zip(param_names, result.x):
    print(f'  {name:14s} = {val:.6f}')

# A fitted value sitting on its bound is not a converged estimate — it means
# this parameter is (at best) underdetermined by this log's data, most often
# because it trades off against another parameter (e.g. T_max_motor vs ba_z
# when the log has little thrust dynamic range) rather than being pinned by
# a genuine physical limit. Flag it instead of letting it look like a result.
_pinned = []
for name, val, lo, hi in zip(param_names, result.x, bounds[0], bounds[1]):
    span = hi - lo
    if span > 0 and (val - lo < 0.01 * span or hi - val < 0.01 * span):
        _pinned.append(name)
if _pinned:
    print(f'  WARNING: pinned at bound (underdetermined by this data, not a '
          f'converged estimate): {", ".join(_pinned)}')

Dv_opt, m_opt, T_max_opt, tau_opt, ba_opt = _unpack(result.x)
Dv_before, m_before, T_max_before, tau_before, ba_before = _unpack(p0)

T_cmd_before = (act_sum * T_max_before) if has_act_sum else T_total
T_cmd_after  = (act_sum * T_max_opt)    if has_act_sum else T_total
T_before = apply_motor_lag(T_cmd_before, t, tau_before)
T_after  = apply_motor_lag(T_cmd_after, t, tau_opt)

print(f'\nMotor lag tau_motor: {_tau_motor_nom:.4f}s (nominal) → {tau_opt:.4f}s (fit)')
print(f'Accel bias [ba_x,ba_y,ba_z] = [{ba_opt[0]:+.4f}, {ba_opt[1]:+.4f}, {ba_opt[2]:+.4f}] m/s²  '
      '(diagnostic only — not written to params.yaml)')

print('\nNote: m_motor, kappa, Dw affect rotational dynamics only — not fitted here.')
print('      Use a rotational sysid experiment (rate step-responses) to identify those.')


# ── 5. 2-D sweep: isotropic Dv vs m ──────────────────────────────────────────

print('\nRunning 2D sweep (Dv_scalar vs m) …')

Dv_grid = np.linspace(0.0, min(float(Dv_opt[0]) * 5 + 0.5, 10.0), 50)
m_grid  = np.linspace(1.0, 12.0, 50)
mse_grid = np.empty((len(m_grid), len(Dv_grid)))

T_sweep = T_after   # thrust profile fixed at the optimum tau_motor/T_max
for i, m_i in enumerate(m_grid):
    for j, dv_j in enumerate(Dv_grid):
        r = (model_acc(np.array([dv_j, dv_j, dv_j]), m_i, T_sweep, ba_opt) - acc_imu).ravel()
        mse_grid[i, j] = np.mean(r**2)

fig, ax = plt.subplots(figsize=(8, 6))
cf = ax.contourf(Dv_grid, m_grid, np.log10(mse_grid + 1e-9), levels=40, cmap='viridis')
fig.colorbar(cf, ax=ax, label='log10 MSE [(m/s²)²]')
ax.set_xlabel('Dv_scalar [N·s²/m²]')
ax.set_ylabel('m [kg]')
ax.set_title('Loss landscape — isotropic Dv vs mass\n(T_max fixed at optimum)')
# Mark optimum (using mean of Dv_x/y/z)
_dv_mean = float(np.mean(Dv_opt))
ax.plot(_dv_mean, m_opt, 'r*', markersize=14, label=f'opt ({_dv_mean:.3f}, {m_opt:.3f})')
ax.legend()
sweep_path = os.path.join(out_dir, 'fit_sweep.png')
fig.savefig(sweep_path, dpi=120)
plt.close(fig)
print(f'Sweep plot saved → {sweep_path}')


# ── 6. Residual time-series plot ─────────────────────────────────────────────
# Shows the raw IMU (ground truth for this fit) against the model prediction
# before/after optimisation on each axis — not just the abstract error — plus
# a GT-vs-model NED velocity panel, so a mismatch's *character* (e.g. only
# during the blip, or a constant offset) is visible, not just its magnitude.

acc_model_before = model_acc(Dv_before, m_before, T_before, ba_before)
acc_model_after  = model_acc(Dv_opt, m_opt, T_after, ba_opt)
r_before = acc_model_before - acc_imu
r_after  = acc_model_after - acc_imu

fig, axes = plt.subplots(5, 1, figsize=(12, 13), sharex=True)
fig.suptitle('Model fit: IMU vs model acceleration, before vs after optimisation',
             fontsize=12)
labels = ['x (fwd)', 'y (right)', 'z (down)']
colors_a = ['tab:blue', 'tab:green', 'tab:red']

for i in range(3):
    axes[i].plot(t, acc_imu[:, i],          color='k',        lw=1.1, label='IMU (raw)')
    axes[i].plot(t, acc_model_before[:, i], color='#bbbbbb',  lw=0.9, ls='--', label='model (before)')
    axes[i].plot(t, acc_model_after[:, i],  color=colors_a[i], lw=1.0, label='model (after)')
    axes[i].set_ylabel(f'acc_{labels[i]}\n[m/s²]', fontsize=8)
    axes[i].legend(fontsize=7, loc='upper right')
    axes[i].grid(True, lw=0.3)

err_norm_before = np.linalg.norm(r_before, axis=1)
err_norm_after  = np.linalg.norm(r_after, axis=1)
axes[3].plot(t, err_norm_before, color='#bbbbbb', lw=0.8, label='before')
axes[3].plot(t, err_norm_after,  color='tab:orange', lw=0.9, label='after')
axes[3].set_ylabel('|err| [m/s²]', fontsize=8)
axes[3].legend(fontsize=7, loc='upper right')
axes[3].grid(True, lw=0.3)

# GT (or EKF-tracked, under ground_truth_mode) NED velocity vs the shadow
# log's own forward-integrated model velocity — same columns/labelling
# imu_ekf.py's own shadow plot uses.
vel_N = np.array([r['vel_N'] for r in rows_f])
vel_E = np.array([r['vel_E'] for r in rows_f])
vel_D = np.array([r['vel_D'] for r in rows_f])
vm_N  = np.array([r['v_model_N'] for r in rows_f])
vm_E  = np.array([r['v_model_E'] for r in rows_f])
vm_D  = np.array([r['v_model_D'] for r in rows_f])
axes[4].plot(t, vel_N, color='tab:blue',   lw=1.0, label='GT vN')
axes[4].plot(t, vel_E, color='tab:orange', lw=1.0, label='GT vE')
axes[4].plot(t, vel_D, color='tab:green',  lw=1.0, label='GT vD')
axes[4].plot(t, vm_N, color='tab:blue',   lw=1.0, ls='--', alpha=0.7, label='model vN')
axes[4].plot(t, vm_E, color='tab:orange', lw=1.0, ls='--', alpha=0.7, label='model vE')
axes[4].plot(t, vm_D, color='tab:green',  lw=1.0, ls='--', alpha=0.7, label='model vD')
axes[4].set_ylabel('vel NED\n[m/s]', fontsize=8)
axes[4].set_xlabel('time [s]')
axes[4].legend(fontsize=6, loc='upper right', ncol=2)
axes[4].grid(True, lw=0.3)

fig.tight_layout()
resid_path = os.path.join(out_dir, 'fit_residual.png')
fig.savefig(resid_path, dpi=120)
plt.close(fig)
print(f'Residual plot saved → {resid_path}')


# ── 7. Optionally write to params.yaml ───────────────────────────────────────

do_write = '--write' in sys.argv
if not do_write:
    ans = input('\nWrite optimal params to params.yaml? [y/N] ').strip().lower()
    do_write = (ans == 'y')

if do_write:
    _params_raw['Dv']         = [round(float(v), 6) for v in Dv_opt]
    _params_raw['m']          = round(float(m_opt), 6)
    _params_raw['tau_motor']  = round(float(tau_opt), 4)
    if T_max_opt is not None:
        _params_raw['T_max_motor'] = round(float(T_max_opt), 4)
    _wrote = 'Dv, m, tau_motor, T_max_motor'

    # accel_bias is opt-in only (--write-bias): imu_ekf.py's live model_predict
    # now applies params.yaml's accel_bias if present, but a fitted bias is
    # only trustworthy when it didn't pin at its bound (see the WARNING above)
    # — writing it unconditionally on every --write would silently feed a
    # possibly-degenerate estimate into the live EKF fusion path.
    if '--write-bias' in sys.argv:
        if 'ba_z' in _pinned or 'ba_x' in _pinned or 'ba_y' in _pinned:
            print('--write-bias requested but a bias component is pinned at its '
                  'bound (see WARNING above) — refusing to write accel_bias.')
        else:
            _params_raw['accel_bias'] = [round(float(v), 4) for v in ba_opt]
            _wrote += ', accel_bias'

    with open('params.yaml', 'w') as f:
        yaml.dump(_params_raw, f, default_flow_style=None, sort_keys=False, allow_unicode=True)
    print(f'params.yaml updated ({_wrote}).')
    if '--write-bias' not in sys.argv:
        print('accel_bias NOT written (pass --write-bias to opt in once you trust '
              'the estimate — it is unpinned/converged, see WARNING above).')
else:
    print('params.yaml NOT modified.')
