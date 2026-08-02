"""
lqi.py

Designs LQR + Integral (LQI) controllers for NED velocity and yaw tracking.

Why 14 states, not 17
---------------------
The full state is 13 (pos+vel+quat+rates). For a velocity controller we drop
the 3 position states from the LQR problem: NED velocity and yaw depend only
on [vel_b, quat, rates], so position columns in C_out are zero and including
them would create redundant integrators that cause the CARE to fail.

Reduced state z (10,) = x[3:13]:
  [0:3]  body velocity [u,v,w]     [m/s]
  [3:7]  quaternion [qw,qx,qy,qz]  NED-to-body
  [7:10] body rates  [p,q,r]       [rad/s]

Augmented state (14,) = [z (10), xi (4)]
  xi[0:3]  integral of NED velocity error  [m]
  xi[3]    integral of yaw error           [rad*s]

Control law (per timestep):
  delta_z = z_meas - z_trim          # (10,)
  xi     += dt * (y_meas - y_ref)    # y = [v_NED(3), psi(1)]
  delta_u = -K @ np.concatenate([delta_z, xi])
  u       = np.clip(u_trim + delta_u, 0, T_max)

Output: lqi_gains.npz
  labels  : operating-point names
  v_ned   : (N,3)    trim NED velocities [m/s]
  psi_ref : (N,)     trim yaw [rad]
  K       : (N,4,14) LQI gains
  z_trim  : (N,10)   trim reduced state
  u_trim  : (N,4)    trim motor thrusts [N]
  C_out   : (N,4,10) output matrices (reduced state)
  Q_aug   : (14,14)
  R_cost  : (4,4)
"""

import numpy as np
from scipy.linalg import solve_continuous_are
from scipy.optimize import fsolve

from flight_model.dyn import load_params, dyn, _quat_to_R

PARAMS_YAML = "params.yaml"
OUTPUT_FILE = "lqi_gains.npz"
EPS_JAC     = 1e-6


# ── Quaternion helpers ────────────────────────────────────────────────────────

def euler_to_quat(phi, theta, psi):
    """ZYX Euler angles -> quaternion [qw,qx,qy,qz] (NED-to-body)."""
    cp, sp = np.cos(phi/2),   np.sin(phi/2)
    ct, st = np.cos(theta/2), np.sin(theta/2)
    cy, sy = np.cos(psi/2),   np.sin(psi/2)
    return np.array([
        cp*ct*cy + sp*st*sy,
        sp*ct*cy - cp*st*sy,
        cp*st*cy + sp*ct*sy,
        cp*ct*sy - sp*st*cy,
    ])


def quat_to_yaw(q):
    """Yaw angle [rad] from quaternion [qw,qx,qy,qz] (NED-to-body)."""
    qw, qx, qy, qz = q
    return np.arctan2(2.0*(qw*qz + qx*qy), 1.0 - 2.0*(qy**2 + qz**2))


# ── 3D Trim Solver ────────────────────────────────────────────────────────────

def find_trim_3d(v_ned, psi_ref, param):
    """
    Solve for (phi, theta, T_per_motor) for level flight at NED velocity
    v_ned = [Vx, Vy, Vz] with heading psi_ref [rad].

    Solves 3-component body-frame force balance (omega_b = 0):
        0 = F_thrust/m + g_body + F_drag/m

    Returns (phi [rad], theta [rad], T_each [N]).
    """
    m  = param['m']
    g  = param['g']
    Dv = param['Dv']
    Vx, Vy, Vz = float(v_ned[0]), float(v_ned[1]), float(v_ned[2])

    def residual(vars):
        phi, theta, T_each = vars
        q     = euler_to_quat(phi, theta, psi_ref)
        R     = _quat_to_R(q)
        vel_b = R @ np.array([Vx, Vy, Vz])

        F_b    = np.array([0.0, 0.0, -4.0 * T_each])
        g_b    = R @ np.array([0.0, 0.0, g])
        F_drag = -Dv @ (np.abs(vel_b) * vel_b)

        return ((F_b + m * g_b + F_drag) / m).tolist()

    theta0 = -np.arctan(Vx / max(g, 0.1))
    phi0   =  np.arctan(Vy / max(g, 0.1))
    T0     = m * g / 4.0

    sol, _, ier, msg = fsolve(residual, [phi0, theta0, T0], full_output=True)
    if ier != 1:
        print(f"  WARNING: trim solver did not converge: {msg.strip()}")

    return float(sol[0]), float(sol[1]), float(sol[2])


def build_trim_state(v_ned, psi_ref, phi, theta, T_each):
    """
    Assemble full 13-element state, reduced 10-element state, and 4-element
    input at trim.
    """
    q     = euler_to_quat(phi, theta, psi_ref)
    R     = _quat_to_R(q)
    vel_b = R @ np.asarray(v_ned, dtype=float)

    x_trim = np.zeros(13)
    x_trim[3:6]  = vel_b
    x_trim[6:10] = q

    z_trim = x_trim[3:13]           # reduced state (drop position)
    u_trim = np.full(4, T_each)
    return x_trim, z_trim, u_trim


# ── Jacobians ─────────────────────────────────────────────────────────────────

def _jacobian(f, x0):
    """Central-difference Jacobian of f : R^n -> R^m at x0."""
    f0   = f(x0)
    n, m = len(x0), len(f0)
    J    = np.zeros((m, n))
    for i in range(n):
        d = np.zeros(n); d[i] = EPS_JAC
        J[:, i] = (f(x0 + d) - f(x0 - d)) / (2.0 * EPS_JAC)
    return J


def compute_output_matrix(x_trim):
    """
    Build C_out (4x10) mapping reduced state delta_z -> [delta_v_NED(3), delta_psi].
    Position columns are zero (NED velocity and yaw don't depend on position).
    """
    def outputs(x):
        vel_b = x[3:6]
        q     = x[6:10] / np.linalg.norm(x[6:10])
        v_ned = _quat_to_R(q).T @ vel_b
        psi   = quat_to_yaw(q)
        return np.append(v_ned, psi)

    C_full = _jacobian(outputs, x_trim)   # (4, 13)
    return C_full[:, 3:13]                # (4, 10) - drop position columns


# ── LQR / LQI ─────────────────────────────────────────────────────────────────

def _lqr(A, B, Q, R_cost):
    """Continuous LQR via CARE. Returns K such that u = -K @ x."""
    P = solve_continuous_are(A, B, Q, R_cost)
    return np.linalg.inv(R_cost) @ B.T @ P


def build_lqi_gain(A_full, B_full, C_out_r, Q_aug, R_cost):
    """
    Build 14-state augmented system from reduced (10-state) dynamics and
    solve LQI.

    Reduced state z = x[3:13] (drops position — those columns are zero in
    C_out anyway, and keeping them creates degenerate zero-eigenvalue modes
    that make the CARE unsolvable).

    Augmented:
      A_aug = [[A_r,     0    ],     B_aug = [[B_r],
               [C_out_r, 0    ]]              [0  ]]
    where A_r = A_full[3:13, 3:13], B_r = B_full[3:13, :]

    Returns K (4 x 14): delta_u = -K @ [delta_z; xi]
    """
    A_r = A_full[3:13, 3:13]    # (10, 10)
    B_r = B_full[3:13, :]       # (10,  4)
    n, p = A_r.shape[0], C_out_r.shape[0]   # 10, 4

    A_aug = np.block([[A_r,     np.zeros((n, p))],
                      [C_out_r, np.zeros((p, p))]])
    B_aug = np.block([[B_r              ],
                      [np.zeros((p, 4)) ]])

    return _lqr(A_aug, B_aug, Q_aug, R_cost)   # (4, 14)


def closed_loop_eigs(A_full, B_full, C_out_r, K):
    """Eigenvalues of the closed-loop 14-state augmented system."""
    A_r = A_full[3:13, 3:13]
    B_r = B_full[3:13, :]
    n, p = A_r.shape[0], C_out_r.shape[0]

    A_aug = np.block([[A_r,     np.zeros((n, p))],
                      [C_out_r, np.zeros((p, p))]])
    B_aug = np.block([[B_r], [np.zeros((p, 4))]])

    return np.linalg.eigvals(A_aug - B_aug @ K)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    param = load_params(PARAMS_YAML)

    # ── Cost weights — loaded from params.yaml ───────────────────────
    Q_aug  = param['Q_aug']
    R_cost = param['R_cost']

    # ── Operating points (v_ned [m/s], psi_ref [rad], label) ─────────
    operating_points = [
        ([  0,  0,  0],  0.0,      "hover"        ),
        ([  0,  0,  0], -np.pi,    "hover_south"  ),   # for south-facing launch
        ([  5,  0,  0],  0.0, "5m/s_fwd"     ),
        ([ 10,  0,  0],  0.0, "10m/s_fwd"    ),
        ([ 20,  0,  0],  0.0, "20m/s_fwd"    ),
        ([  0,  5,  0],  0.0, "5m/s_lat"     ),
        ([  0,  0, -2],  0.0, "2m/s_climb"   ),
        ([  0,  0,  2],  0.0, "2m/s_descent" ),
        ([ 10,  5,  0],  0.0, "10fwd_5lat"   ),
    ]

    all_labels, all_vneds, all_psis          = [], [], []
    all_K, all_z, all_u, all_C, all_A, all_B = [], [], [], [], [], []

    print(f"\n{'Label':<16} {'phi':>8} {'theta':>8} {'T/mot':>8} "
          f"{'trim_res':>10} {'max_Re_eig':>12} {'stable':>7}")
    print("-" * 78)

    for v_ned, psi_ref, label in operating_points:
        phi, theta, T        = find_trim_3d(v_ned, psi_ref, param)
        x_t, z_t, u_t        = build_trim_state(v_ned, psi_ref, phi, theta, T)

        xdot = dyn(x_t, u_t, param)
        res  = np.max(np.abs(np.concatenate([xdot[3:6], xdot[10:13]])))

        A = _jacobian(lambda x, u=u_t: dyn(x, u, param), x_t)
        B = _jacobian(lambda u, x=x_t: dyn(x, u, param), u_t)

        C_out_r = compute_output_matrix(x_t)
        K       = build_lqi_gain(A, B, C_out_r, Q_aug, R_cost)

        eigs   = closed_loop_eigs(A, B, C_out_r, K)
        max_re = np.max(eigs.real)
        ok     = "OK" if max_re < 0 else "UNSTABLE"

        print(f"{label:<16} {np.rad2deg(phi):>+7.2f}  {np.rad2deg(theta):>+7.2f}  "
              f"{T:>7.3f}  {res:>10.2e}  {max_re:>+11.4f}  {ok:>7}")

        all_labels.append(label);  all_vneds.append(v_ned)
        all_psis.append(psi_ref);  all_K.append(K)
        all_z.append(z_t);         all_u.append(u_t)
        all_C.append(C_out_r);     all_A.append(A);  all_B.append(B)

    np.savez(
        OUTPUT_FILE,
        labels  = np.array(all_labels),
        v_ned   = np.array(all_vneds,  dtype=float),
        psi_ref = np.array(all_psis,   dtype=float),
        K       = np.array(all_K),       # (N, 4, 14)
        z_trim  = np.array(all_z),       # (N, 10)
        u_trim  = np.array(all_u),       # (N,  4)
        C_out   = np.array(all_C),       # (N, 4, 10)
        A       = np.array(all_A),
        B       = np.array(all_B),
        Q_aug   = Q_aug,
        R_cost  = R_cost,
    )

    print(f"\nSaved '{OUTPUT_FILE}'")
    K_arr = np.array(all_K)
    print(f"  K shape : {K_arr.shape}  (n_points x 4 motors x 14 states)")
    print(f"\n  K[:10]  multiplies  delta_z = z_meas - z_trim  (reduced state)")
    print(f"  K[10:]  multiplies  xi  (4 integrators)")
    print(f"\nRuntime loop:")
    print(f"  delta_z  = x_meas[3:13] - z_trim")
    print(f"  xi      += dt * (C_out @ x_meas[3:13] - y_ref)")
    print(f"  delta_u  = -K @ np.concatenate([delta_z, xi])")
    print(f"  u        = np.clip(u_trim + delta_u, 0, T_max)")


if __name__ == "__main__":
    main()
