"""
plot_calib.py
=============
Offline plotter for gate_vision_calib.py flight logs.

Usage (from the repo root)
-----
  python -m sysid.plot_calib                      # plots most recent logs/calib_* directory
  python -m sysid.plot_calib logs/calib_20260720  # plots specific directory

Outputs
-------
  <dir>/overview.png   — 5-panel flight overview with phase bands
"""

import os
import sys
import glob

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Phase colour map ──────────────────────────────────────────────────────────
PHASE_COLORS = {
    'WAIT':     '#d0e8ff',
    'RAMP':     '#ffe0a0',
    'BLIP':     '#ffd0d0',
    'BLIP_OBS': '#ffe8c0',
    'HOVER':    '#d0ffd8',
    'APPROACH': '#f0d0ff',
    'KILL':     '#e0e0e0',
}


def _find_latest_calib():
    dirs = sorted(glob.glob('logs/calib_*'))
    if not dirs:
        raise FileNotFoundError("No logs/calib_* directories found.")
    return dirs[-1]


def _load_csv(path):
    """Load a CSV with a header row into a dict of float arrays."""
    with open(path, encoding='utf-8') as f:
        header = f.readline().strip().split(',')
        rows = [line.strip().split(',') for line in f if line.strip()]
    if not rows:
        return {}
    d = {col: [] for col in header}
    for row in rows:
        for col, val in zip(header, row):
            try:
                d[col].append(float(val))
            except ValueError:
                d[col].append(float('nan'))
    return {k: np.array(v) for k, v in d.items()}


def _phase_bands(ax, phases, t, y_min=None, y_max=None):
    """Draw shaded background bands for each flight phase."""
    if y_min is None:
        y_min, y_max = ax.get_ylim()
    current = phases[0]
    t_start = t[0]
    for i in range(1, len(phases)):
        if phases[i] != current or i == len(phases) - 1:
            t_end = t[i]
            col = PHASE_COLORS.get(current, '#f8f8f8')
            ax.axvspan(t_start, t_end, color=col, alpha=0.35, zorder=0)
            current = phases[i]
            t_start = t[i]
    # legend patches
    seen = dict.fromkeys(phases)
    patches = [mpatches.Patch(color=PHASE_COLORS.get(p, '#f8f8f8'),
                               alpha=0.6, label=p) for p in seen]
    return patches


def plot(log_dir):
    calib_path = os.path.join(log_dir, 'calib.csv')
    pnp_path   = os.path.join(log_dir, 'pnp.csv')

    if not os.path.exists(calib_path):
        raise FileNotFoundError(f"No calib.csv in {log_dir}")

    d   = _load_csv(calib_path)
    pnp = _load_csv(pnp_path) if os.path.exists(pnp_path) else {}
    has_pnp = bool(pnp) and len(pnp.get('t', [])) > 0

    t      = d['t']
    phases = []
    with open(calib_path, encoding='utf-8') as f:
        f.readline()
        for line in f:
            if line.strip():
                phases.append(line.split(',')[0])
    phases = np.array(phases)

    t0 = t[0]
    t  = t - t0
    if has_pnp:
        t_pnp = pnp['t'] - t0

    # ── Figure: 5 panels ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(5, 1, figsize=(14, 18), sharex=True)
    fig.suptitle(f"Calibration flight overview — {os.path.basename(log_dir)}",
                 fontsize=13, y=0.995)

    # ── Panel 0: Attitude ────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(t, d['phi_deg'],   lw=0.9, label='roll φ (deg)')
    ax.plot(t, d['theta_deg'], lw=0.9, label='pitch θ (deg)')
    ax.plot(t, d['psi_deg'],   lw=0.9, label='yaw ψ (deg)')
    ax.axhline(0, color='k', lw=0.4, ls=':')
    ax.set_ylabel('Angle (deg)')
    ax.set_title('Attitude')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    _phase_bands(ax, phases, t)

    # ── Panel 1: EKF velocity + PnP velocity ────────────────────────────────
    ax = axes[1]
    ax.plot(t, d['vN'], lw=0.9, color='tab:blue',   label='vN EKF')
    ax.plot(t, d['vE'], lw=0.9, color='tab:orange', label='vE EKF')
    ax.plot(t, d['vD'], lw=0.9, color='tab:green',  label='vD EKF')
    if has_pnp:
        vN_ok = ~np.isnan(pnp.get('vN_pnp', np.full(len(t_pnp), np.nan)))
        if vN_ok.any():
            ax.scatter(t_pnp[vN_ok], pnp['vN_pnp'][vN_ok],
                       s=14, color='tab:blue', marker='^', zorder=5, label='vN PnP')
            ax.scatter(t_pnp[vN_ok], pnp['vE_pnp'][vN_ok],
                       s=14, color='tab:orange', marker='^', zorder=5, label='vE PnP')
    ax.axhline(0, color='k', lw=0.4, ls=':')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('EKF Velocity (NED)' + (' + PnP velocity (▲)' if has_pnp else ''))
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    _phase_bands(ax, phases, t)

    # ── Panel 2: Altitude (−pD) ──────────────────────────────────────────────
    ax = axes[2]
    alt_ekf = -d['pD']
    ax.plot(t, alt_ekf, lw=0.9, color='tab:blue', label='EKF altitude')
    if has_pnp:
        alt_pnp = -pnp.get('pD_pnp', np.full(len(t_pnp), np.nan))
        ax.scatter(t_pnp, alt_pnp, s=14, color='tab:orange', zorder=5, label='PnP altitude')
        # gate distance on right axis
        ax2 = ax.twinx()
        ax2.plot(t_pnp, pnp['dist_m'], lw=0.8, color='gray', ls='--', label='gate dist (m)')
        ax2.set_ylabel('Gate distance (m)', color='gray')
        ax2.tick_params(axis='y', colors='gray')
        ax2.legend(loc='upper right', fontsize=8)
    # Draw hover target altitude line (median altitude during HOVER)
    hover_mask = phases == 'HOVER'
    if hover_mask.any():
        target_alt = float(np.median(alt_ekf[hover_mask]))
        ax.axhline(target_alt, color='tab:green', lw=1.0, ls='--',
                   label=f'hover alt={target_alt:.2f}m')
    ax.set_ylabel('Altitude (m)')
    ax.set_title('Altitude — EKF vs PnP' if has_pnp else 'Altitude (EKF)')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    _phase_bands(ax, phases, t)

    # ── Panel 3: Rate commands ────────────────────────────────────────────────
    ax = axes[3]
    ax.plot(t, d['p_cmd'], lw=0.8, label='p_cmd (roll)')
    ax.plot(t, d['q_cmd'], lw=0.8, label='q_cmd (pitch)')
    ax.plot(t, d['r_cmd'], lw=0.8, label='r_cmd (yaw)')
    ax.axhline(0, color='k', lw=0.4, ls=':')
    ax.set_ylabel('Rate cmd (rad/s)')
    ax.set_title('Body-rate commands')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    _phase_bands(ax, phases, t)

    # ── Panel 4: Thrust ───────────────────────────────────────────────────────
    ax = axes[4]
    ax.plot(t, d['T_norm'], lw=0.9, color='tab:red', label='T_norm')
    hover_mask = phases == 'HOVER'
    if hover_mask.any():
        t_hover_norm = float(np.median(d['T_norm'][hover_mask]))
        ax.axhline(t_hover_norm, color='k', lw=0.8, ls='--',
                   label=f'hover T={t_hover_norm:.3f}')
    ax.set_ylabel('Collective thrust (norm)')
    ax.set_xlabel('Time (s)')
    ax.set_title('Collective thrust')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    _phase_bands(ax, phases, t)

    # ── Phase legend on last panel ─────────────────────────────────────────────
    seen = dict.fromkeys(phases)
    patches = [mpatches.Patch(color=PHASE_COLORS.get(p, '#f8f8f8'),
                               alpha=0.6, label=p) for p in seen]
    axes[4].legend(handles=patches + axes[4].get_legend_handles_labels()[0],
                   loc='lower right', fontsize=7, ncol=2)

    plt.tight_layout(rect=[0, 0, 1, 0.993])
    out = os.path.join(log_dir, 'overview.png')
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved -> {out}")

    # ── Plan view ─────────────────────────────────────────────────────────────
    appr_mask = phases == 'APPROACH'
    if appr_mask.any():
        fig2, ax = plt.subplots(figsize=(8, 8))
        ax.plot(d['pE'][appr_mask], d['pN'][appr_mask],
                lw=1.2, color='tab:blue', label='EKF path (APPROACH)')
        if has_pnp:
            ax.scatter(pnp['pE_pnp'], pnp['pN_pnp'],
                       s=20, color='tab:orange', zorder=5, label='PnP positions')
        ax.set_xlabel('East (m)')
        ax.set_ylabel('North (m)')
        ax.set_title('Plan view — APPROACH phase')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        plt.tight_layout()
        out2 = os.path.join(log_dir, 'plan_view.png')
        plt.savefig(out2, dpi=150)
        plt.close(fig2)
        print(f"Saved -> {out2}")


if __name__ == '__main__':
    log_dir = sys.argv[1] if len(sys.argv) > 1 else _find_latest_calib()
    print(f"Plotting {log_dir} …")
    plot(log_dir)
