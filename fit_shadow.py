"""
fit_shadow.py  —  Identify translational physics parameters from a shadow log.

Fits [Dv_x, Dv_y, Dv_z, m, T_max_motor] by minimising the residual between
the physics model and the raw IMU accelerometer captured in model_predict_shadow.csv.

Model (body frame, FRD, specific force):
    acc_model = ([0, 0, -T_total] + F_drag) / m
    F_drag    = -diag(Dv) @ (|v_b| * v_b)
    T_total   = actuator_sum * T_max_motor

Non-observable from translational data (need rotational sysid):
    m_motor  → Ixx/Iyy/Izz via geometry
    kappa    → yaw torque coefficient
    Dw       → rotational drag

Usage:
    python fit_shadow.py                        # auto-find latest shadow CSV
    python fit_shadow.py path/to/shadow.csv
    python fit_shadow.py path/to/shadow.csv --write   # update params.yaml directly
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

# actuator_sum: raw 0-1 motor fraction sum (logged separately for T_max fitting)
has_act_sum = 'actuator_sum' in rows_f[0]
if has_act_sum:
    act_sum = np.array([r['actuator_sum'] for r in rows_f])
    print('actuator_sum column found — fitting T_max_motor as free parameter.')
else:
    act_sum = None
    print('actuator_sum column missing (old log) — T_max_motor fixed to T_total/4 avg.')


# ── 3. Residual function ──────────────────────────────────────────────────────

def model_acc(Dv, m, T_total_vec):
    """Body-frame specific force predicted by the physics model."""
    F_drag  = -(Dv * np.abs(vel_b) * vel_b)          # (N, 3) element-wise
    thrust  = np.column_stack([np.zeros(N), np.zeros(N), -T_total_vec])
    return (thrust + F_drag) / m                       # (N, 3)


if has_act_sum:
    # Free params: [Dv_x, Dv_y, Dv_z, m, T_max_motor]
    def residual(p):
        Dv_x, Dv_y, Dv_z, m, T_max = p
        T_fit = act_sum * T_max
        return (model_acc(np.array([Dv_x, Dv_y, Dv_z]), m, T_fit) - acc_imu).ravel()

    p0     = [1.2,  1.2,  1.2,  5.405, 49.9]
    bounds = ([0.0,  0.0,  0.0,  1.0,   5.0],
              [20.0, 20.0, 20.0, 15.0, 200.0])
    param_names = ['Dv_x', 'Dv_y', 'Dv_z', 'm', 'T_max_motor']
else:
    # Free params: [Dv_x, Dv_y, Dv_z, m]  (T_total from CSV is already in Newtons)
    def residual(p):
        Dv_x, Dv_y, Dv_z, m = p
        return (model_acc(np.array([Dv_x, Dv_y, Dv_z]), m, T_total) - acc_imu).ravel()

    p0     = [1.2,  1.2,  1.2,  5.405]
    bounds = ([0.0,  0.0,  0.0,  1.0],
              [20.0, 20.0, 20.0, 15.0])
    param_names = ['Dv_x', 'Dv_y', 'Dv_z', 'm']


# ── 4. Optimise ───────────────────────────────────────────────────────────────

r0     = residual(p0)
rms_before = float(np.sqrt(np.mean(r0**2)))
print(f'\nInitial RMS residual: {rms_before:.4f} m/s²')

result = least_squares(residual, p0, bounds=bounds, method='trf',
                       ftol=1e-9, xtol=1e-9, verbose=1)

rms_after = float(np.sqrt(np.mean(result.fun**2)))
print(f'Final   RMS residual: {rms_after:.4f} m/s²')
print(f'Improvement: {100*(1-rms_after/rms_before):.1f}%\n')

for name, val in zip(param_names, result.x):
    print(f'  {name:14s} = {val:.6f}')

Dv_opt   = result.x[:3]
m_opt    = result.x[3]
T_max_opt = result.x[4] if has_act_sum else None

print('\nNote: m_motor, kappa, Dw affect rotational dynamics only — not fitted here.')
print('      Use a rotational sysid experiment (rate step-responses) to identify those.')


# ── 5. 2-D sweep: isotropic Dv vs m ──────────────────────────────────────────

print('\nRunning 2D sweep (Dv_scalar vs m) …')

Dv_grid = np.linspace(0.0, min(float(Dv_opt[0]) * 5 + 0.5, 10.0), 50)
m_grid  = np.linspace(1.0, 12.0, 50)
mse_grid = np.empty((len(m_grid), len(Dv_grid)))

T_sweep = (act_sum * T_max_opt) if has_act_sum else T_total
for i, m_i in enumerate(m_grid):
    for j, dv_j in enumerate(Dv_grid):
        r = (model_acc(np.array([dv_j, dv_j, dv_j]), m_i, T_sweep) - acc_imu).ravel()
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

r_before = (model_acc(np.array(p0[:3]), p0[3],
                       act_sum * p0[4] if has_act_sum else T_total) - acc_imu)
r_after  = (model_acc(Dv_opt, m_opt,
                       act_sum * T_max_opt if has_act_sum else T_total) - acc_imu)

fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
fig.suptitle('Model-fit residual: before vs after optimisation', fontsize=12)
labels = ['x (fwd)', 'y (right)', 'z (down)']
colors_b = ['#aec6cf', '#aed6aec', '#cfaed6']   # muted
colors_a = ['tab:blue', 'tab:green', 'tab:red']

for i in range(3):
    axes[i].plot(t, r_before[:, i], color='#bbbbbb', lw=0.8, label='before')
    axes[i].plot(t, r_after[:, i],  color=colors_a[i], lw=0.9, label='after')
    axes[i].axhline(0, color='k', lw=0.5, ls='--')
    axes[i].set_ylabel(f'err_{labels[i]}\n[m/s²]', fontsize=8)
    axes[i].legend(fontsize=7, loc='upper right')
    axes[i].grid(True, lw=0.3)

err_norm_before = np.linalg.norm(r_before, axis=1)
err_norm_after  = np.linalg.norm(r_after, axis=1)
axes[3].plot(t, err_norm_before, color='#bbbbbb', lw=0.8, label='before')
axes[3].plot(t, err_norm_after,  color='tab:orange', lw=0.9, label='after')
axes[3].set_ylabel('|err| [m/s²]', fontsize=8)
axes[3].set_xlabel('time [s]')
axes[3].legend(fontsize=7, loc='upper right')
axes[3].grid(True, lw=0.3)

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
    with open('params.yaml') as f:
        raw = yaml.safe_load(f)
    raw['Dv'] = [round(float(v), 6) for v in Dv_opt]
    raw['m']  = round(float(m_opt), 6)
    if T_max_opt is not None:
        raw['T_max_motor'] = round(float(T_max_opt), 4)
    with open('params.yaml', 'w') as f:
        yaml.dump(raw, f, default_flow_style=None, sort_keys=False, allow_unicode=True)
    print('params.yaml updated.')
else:
    print('params.yaml NOT modified.')
