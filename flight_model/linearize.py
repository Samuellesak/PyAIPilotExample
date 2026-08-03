"""
Numerically linearize the quadrotor dynamics at steady level forward-flight
trim conditions for a range of airspeeds.

Trim assumptions
----------------
  - Pure pitch (phi = psi = 0), omega_b = 0, constant altitude.
  - All four motors produce equal thrust.
  - Forward NED velocity = [V, 0, 0].

Output: linearization.npz
  speeds_kmh : (N,)        speed set-points [km/h]
  A          : (N, 13, 13) state Jacobian   df/dx at each trim point
  B          : (N, 13,  4) input Jacobian   df/du at each trim point
  x_trim     : (N, 13)     trim state vector
  u_trim     : (N,  4)     trim motor thrusts [N]

State layout (13,):
  [0:3]  position NED             [m]         (arbitrary at trim)
  [3:6]  body velocity [u, v, w]  [m/s]
  [6:10] quaternion [qw,qx,qy,qz] NED-to-body (unit)
  [10:13] body rates [p, q, r]    [rad/s]

Usage (from the repo root)
-----
  python -m flight_model.linearize

Load results
------------
  data = np.load("linearization.npz")
  A_at_0kmh   = data['A'][0]      # shape (13,13)
  B_at_100kmh = data['B'][5]      # shape (13, 4)
"""

import numpy as np
from scipy.optimize import fsolve

from flight_model.dyn import load_params, dyn, _quat_to_R

# ── Configuration ─────────────────────────────────────────────────────────────
SPEEDS_KMH = [0, 20, 40, 60, 80, 100, 120, 140]
PARAMS_YAML = "params.yaml"
OUTPUT_FILE = "linearization.npz"

EPS = 1e-6   # central-difference step size


# ── Numerical Jacobian ────────────────────────────────────────────────────────

def _jacobian(f, x0):
    """Central-difference Jacobian of f : R^n → R^m evaluated at x0."""
    f0 = f(x0)
    n  = len(x0)
    m  = len(f0)
    J  = np.zeros((m, n))
    for i in range(n):
        dx      = np.zeros(n)
        dx[i]   = EPS
        J[:, i] = (f(x0 + dx) - f(x0 - dx)) / (2.0 * EPS)
    return J


# ── Trim Solver ───────────────────────────────────────────────────────────────

def find_trim(V_ms, param):
    """
    Find (theta_trim [rad], T_per_motor [N]) for level flight at V_ms m/s.

    Solves the body-frame x and z force balance with omega_b = 0:
        0 = F_thrust + m*g_body + F_drag   (x and z components)

    Returns
    -------
    theta : float   pitch angle [rad]  (negative = nose-down / forward)
    T     : float   thrust per motor   [N]
    """
    m  = param['m']
    g  = param['g']
    Dv = param['Dv']

    def residual(vars):
        theta, T_each = vars
        q     = np.array([np.cos(theta / 2.0), 0.0, np.sin(theta / 2.0), 0.0])
        R     = _quat_to_R(q)
        vel_b = R @ np.array([V_ms, 0.0, 0.0])

        F_b    = np.array([0.0, 0.0, -4.0 * T_each])
        g_b    = R @ np.array([0.0, 0.0, g])
        F_drag = -Dv @ (np.abs(vel_b) * vel_b)

        acc = (F_b + m * g_b + F_drag) / m   # omega=0 → no cross term
        return [acc[0], acc[2]]               # x and z must vanish

    # Initial guess: linear pitch scaling + hover thrust
    theta0 = -np.deg2rad(2.0 * V_ms / (20.0 / 3.6))
    T0     = m * g / 4.0

    sol, _, ier, msg = fsolve(residual, [theta0, T0], full_output=True)
    if ier != 1:
        print(f"  WARNING: trim solver did not converge — {msg.strip()}")

    return float(sol[0]), float(sol[1])


def build_trim_vectors(V_ms, theta, T_each):
    """Assemble full 13-element state and 4-element input at trim."""
    q     = np.array([np.cos(theta / 2.0), 0.0, np.sin(theta / 2.0), 0.0])
    R     = _quat_to_R(q)
    vel_b = R @ np.array([V_ms, 0.0, 0.0])

    x_trim = np.zeros(13)
    x_trim[3:6]  = vel_b   # body velocity
    x_trim[6:10] = q       # quaternion (position x[0:3] and rates x[10:13] stay 0)

    u_trim = np.full(4, T_each)
    return x_trim, u_trim


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    param     = load_params(PARAMS_YAML)
    speeds_ms = [v / 3.6 for v in SPEEDS_KMH]

    all_A, all_B, all_x, all_u = [], [], [], []

    T_max = float(param['T_max_motor'])
    print(f"{'Speed':>10}  {'theta [deg]':>12}  {'T/motor [N]':>12}  "
          f"{'trim%':>7}  {'head_up[N]':>11}  {'head_dn[N]':>11}  {'residual':>10}")
    print("-" * 90)

    for V_kmh, V_ms in zip(SPEEDS_KMH, speeds_ms):
        theta, T_each   = find_trim(V_ms, param)
        x_trim, u_trim  = build_trim_vectors(V_ms, theta, T_each)

        # Trim quality: check body accelerations are zero
        xdot    = dyn(x_trim, u_trim, param)
        residual = np.max(np.abs(np.concatenate([xdot[3:6], xdot[10:13]])))

        head_up   = T_max - T_each
        head_down = T_each
        trim_pct  = 100.0 * T_each / T_max

        flag = "  *** T > T_max ***" if T_each > T_max else ""
        print(f"{V_kmh:>8} km/h  {np.rad2deg(theta):>+11.3f}°  "
              f"{T_each:>11.4f} N  {trim_pct:>6.1f}%  "
              f"{head_up:>9.3f} N  {head_down:>9.3f} N  {residual:>10.2e}{flag}")

        A = _jacobian(lambda x, u=u_trim: dyn(x, u, param), x_trim)
        B = _jacobian(lambda u, x=x_trim: dyn(x, u, param), u_trim)

        all_A.append(A)
        all_B.append(B)
        all_x.append(x_trim)
        all_u.append(u_trim)

    A_arr = np.array(all_A)
    B_arr = np.array(all_B)

    np.savez(
        OUTPUT_FILE,
        speeds_kmh = np.array(SPEEDS_KMH, dtype=float),
        A          = A_arr,
        B          = B_arr,
        x_trim     = np.array(all_x),
        u_trim     = np.array(all_u),
    )

    print(f"\nSaved '{OUTPUT_FILE}'")
    print(f"  A : {A_arr.shape}   (n_speeds × 13 × 13)")
    print(f"  B : {B_arr.shape}   (n_speeds × 13 ×  4)")

    # Print inferred inertia for reference
    I = param['I']
    print(f"\nInertia tensor [kg·m²]:")
    print(f"  Ixx = {I[0,0]:.5f}   Iyy = {I[1,1]:.5f}   Izz = {I[2,2]:.5f}")


if __name__ == "__main__":
    main()
