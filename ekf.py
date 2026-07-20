"""
ekf.py — 13-state Extended Kalman Filter for quadcopter IMU-only estimation.

State  x = [pN, pE, pD, vN, vE, vD, qw, qx, qy, qz, bgx, bgy, bgz]
            0   1   2   3   4   5   6   7   8   9   10  11   12

Process model  (predict step, driven by raw IMU at ~250 Hz):
  p_dot  = v_ned                                             position integration
  v_dot  = R_b2n(q) @ acc_raw  +  [0, 0, g]                NED specific-force + gravity
  q_dot  = 0.5 * Omega(gyro_raw − bias) @ q                 quaternion kinematics
  bg_dot = 0                                                 gyro-bias random walk via Q

Measurement models (update steps):
  Gravity alignment   z = acc_raw   ↔   z_pred = R_n2b(q) @ [0, 0, −g]
      Skipped when |acc_norm − g| > threshold  (too much linear acceleration)
  ZUPT (zero-velocity update)   z = 0   ↔   z_pred = v_ned
      Skipped unless gyro_mag < 0.3 rad/s  AND  |acc_norm − g| < 1.5 m/s²

Convention: FRD body frame / NED world frame (same as dyn.py / mavlink_rx.py).
At rest level: az ≈ −9.81 m/s², gravity in NED = [0, 0, +9.81].

Position is estimated by dead-reckoning (integrating velocity from IMU).
Without GPS/barometer corrections it drifts, but provides a useful reference
for the carrot tracker and altitude hold over the time scale of a flight.
"""

import numpy as np


class QuadEKF:
    _IP  = slice(0, 3)    # position NED       [m]
    _IV  = slice(3, 6)    # velocity NED        [m/s]
    _IQ  = slice(6, 10)   # quaternion [qw qx qy qz]
    _IBG = slice(10, 13)  # gyro bias           [rad/s]
    N    = 13

    def __init__(self,
                 sigma_acc_proc  = 0.3,   # m/s²  / sqrt(Hz) — accel process noise
                 sigma_gyro_proc = 0.02,  # rad/s / sqrt(Hz) — gyro  process noise
                 sigma_bias_proc = 5e-5,  # rad/s²/ sqrt(Hz) — bias  random walk
                 sigma_acc_meas  = 0.5,   # m/s²               — acc measurement noise
                 sigma_zupt      = 0.05,  # m/s                — ZUPT velocity noise
                 g               = 9.81,
                 ):
        self._sa   = float(sigma_acc_proc)
        self._sg   = float(sigma_gyro_proc)
        self._sb   = float(sigma_bias_proc)
        self._R_a  = float(sigma_acc_meas) ** 2 * np.eye(3)
        self._R_z  = float(sigma_zupt)     ** 2 * np.eye(3)
        self._g    = float(g)

        self.x = np.zeros(self.N)
        self.x[6] = 1.0                   # qw = 1 (identity attitude)
        self.P = np.diag(
            [1.0**2] * 3 +               # position: large initial uncertainty
            [0.5**2] * 3 +               # velocity
            [0.1**2] * 4 +               # quaternion
            [0.05**2] * 3                # gyro bias
        ).astype(float)
        self.P[12, 12] = 0.0             # bgz unobservable (no yaw sensor) — freeze at 0

    # ── Public API ────────────────────────────────────────────────────────

    def predict(self, gyro_raw, acc_raw, dt):
        """Prediction step using raw IMU measurements."""
        p  = self.x[self._IP]
        v  = self.x[self._IV]
        q  = self.x[self._IQ]
        bg = self.x[self._IBG]
        g  = self._g

        omega = gyro_raw - bg                              # bias-corrected rate

        # Propagate state (first-order Euler)
        R_b2n = _rot_b2n(q)
        a_ned = R_b2n @ acc_raw + np.array([0., 0., g])
        v_new = v + a_ned * dt
        p_new = p + v * dt                                 # integrate position from velocity

        q_new = q + 0.5 * dt * _omega_mat(omega) @ q
        nrm = np.linalg.norm(q_new)
        q_new = q_new / nrm if nrm > 1e-9 else q.copy()

        self.x[self._IP]  = p_new
        self.x[self._IV]  = v_new
        self.x[self._IQ]  = q_new
        # bias unchanged: self.x[self._IBG] stays

        # Propagate covariance  P ← (I + F·dt) P (I + F·dt)ᵀ + Q
        F  = _jacobian_F(q, omega, acc_raw)
        Fd = np.eye(self.N) + F * dt
        Q  = _build_Q(q, dt, self._sa, self._sg, self._sb)
        self.P = Fd @ self.P @ Fd.T + Q

    def update_accel(self, acc_raw, threshold=2.0):
        """
        Gravity-alignment measurement update.
        Returns (applied: bool, innov_norm: float).
        Skipped when |acc_norm − g| > threshold to avoid contaminating
        the estimate with large linear accelerations.
        """
        acc_norm = float(np.linalg.norm(acc_raw))
        if abs(acc_norm - self._g) > threshold:
            return False, 0.0

        q = self.x[self._IQ]
        g = self._g

        # z_pred = R_n2b @ [0, 0, −g]  (expected specific force when stationary)
        z_pred = _rot_b2n(q).T @ np.array([0., 0., -g])
        innov  = acc_raw - z_pred                         # innovation (3,)

        H      = np.zeros((3, self.N))
        H[:, 6:10] = _H_acc_quat(q, g)                   # quaternion block at cols 6:10

        self._apply_update(innov, H, self._R_a)
        return True, float(np.linalg.norm(innov))

    def update_zupt(self, gyro_raw, acc_raw,
                    gyro_threshold=0.3, acc_threshold=1.5):
        """
        Zero-velocity update: treat v = 0 as a measurement when stationary.
        Returns (applied: bool, innov_norm: float).
        Fires only when the drone appears stationary (low gyro + gravity-aligned acc
        AND the EKF's own velocity estimate is near zero).
        """
        gyro_mag = float(np.linalg.norm(gyro_raw))
        acc_norm = float(np.linalg.norm(acc_raw))
        vel_mag  = float(np.linalg.norm(self.x[self._IV]))
        if gyro_mag > gyro_threshold or abs(acc_norm - self._g) > acc_threshold or vel_mag > 4.0:
            return False, 0.0

        innov = -self.x[self._IV]                        # z_meas − z_pred = 0 − v
        H     = np.zeros((3, self.N))
        H[0, 3] = H[1, 4] = H[2, 5] = 1.0              # measures v_ned at indices 3:6

        self._apply_update(innov, H, self._R_z)
        return True, float(np.linalg.norm(innov))

    def update_position(self, pos_ned_meas, sigma_pos=0.5, gate_dist=10.0):
        """
        NED position measurement from visual PnP against a known gate landmark.
        Rejects innovations larger than gate_dist metres (outlier/bad PnP).
        Returns (applied: bool, innov_norm: float).
        """
        innov = np.asarray(pos_ned_meas, dtype=float) - self.x[self._IP]
        innov_norm = float(np.linalg.norm(innov))
        if innov_norm > gate_dist:
            return False, innov_norm

        H       = np.zeros((3, self.N))
        H[0, 0] = H[1, 1] = H[2, 2] = 1.0   # measures position states 0:3
        R       = (sigma_pos ** 2) * np.eye(3)

        self._apply_update(innov, H, R)
        self.x[self._IQ] /= np.linalg.norm(self.x[self._IQ])
        return True, innov_norm

    def update_velocity(self, vel_ned_meas, sigma_vel=1.0, gate_dist=5.0):
        """
        NED velocity measurement derived from finite-differencing consecutive PnP positions.
        Rejects innovations larger than gate_dist m/s (outlier/dropped frame).
        Returns (applied: bool, innov_norm: float).
        """
        innov = np.asarray(vel_ned_meas, dtype=float) - self.x[self._IV]
        innov_norm = float(np.linalg.norm(innov))
        if innov_norm > gate_dist:
            return False, innov_norm

        H       = np.zeros((3, self.N))
        H[0, 3] = H[1, 4] = H[2, 5] = 1.0   # measures velocity states 3:6
        R       = (sigma_vel ** 2) * np.eye(3)

        self._apply_update(innov, H, R)
        return True, innov_norm

    def update_yaw(self, yaw_meas, sigma_yaw=0.1, gate_dist=np.pi):
        """
        NED yaw measurement [rad] from PnP + known gate orientation.
        Uses the linearised Jacobian of yaw(q) w.r.t. the quaternion state.
        Wraps innovation to [-π, π].  Rejects |innov| > gate_dist.
        Returns (applied: bool, innov_abs: float).
        """
        qw, qx, qy, qz = self.x[self._IQ]
        f = 2.0 * (qw * qz + qx * qy)
        g = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw_est = float(np.arctan2(f, g))

        innov = float(yaw_meas) - yaw_est
        innov = (innov + np.pi) % (2.0 * np.pi) - np.pi   # wrap to [-π, π]
        if abs(innov) > gate_dist:
            return False, abs(innov)

        denom = f * f + g * g + 1e-12
        H = np.zeros((1, self.N))
        H[0, 6] = 2.0 * qz * g / denom
        H[0, 7] = 2.0 * qy * g / denom
        H[0, 8] = (2.0 * qx * g + 4.0 * qy * f) / denom
        H[0, 9] = (2.0 * qw * g + 4.0 * qz * f) / denom

        R = np.array([[sigma_yaw ** 2]])
        self._apply_update(np.array([innov]), H, R)
        self.x[self._IQ] /= np.linalg.norm(self.x[self._IQ])
        return True, abs(innov)

    def reset(self):
        """Reset to identity attitude, zero velocity, zero bias, zero position."""
        self.x[:] = 0.
        self.x[6] = 1.0                                  # qw = 1
        self.P[:] = 0.
        np.fill_diagonal(self.P,
            [1.0**2] * 3 + [0.5**2] * 3 + [0.1**2] * 4 + [0.05**2] * 3)
        self.P[12, 12] = 0.0                             # bgz: frozen (unobservable)

    def set_yaw(self, psi):
        """
        Inject a ground-truth yaw angle [rad] into the quaternion while
        preserving the current roll and pitch estimates.

        Decompose current quat into roll (φ) and pitch (θ), then rebuild
        from (φ, θ, psi) using the ZYX Tait-Bryan convention:
            q = q_z(psi) * q_y(theta) * q_x(phi)
        """
        qw, qx, qy, qz = self.x[self._IQ]
        # Extract roll and pitch from current EKF quaternion
        phi   = np.arctan2(2.0*(qw*qx + qy*qz), 1.0 - 2.0*(qx*qx + qy*qy))
        theta = np.arcsin(np.clip(2.0*(qw*qy - qz*qx), -1.0, 1.0))
        # Rebuild quaternion with corrected yaw
        cp, sp = np.cos(phi/2),   np.sin(phi/2)
        ct, st = np.cos(theta/2), np.sin(theta/2)
        cy, sy = np.cos(psi/2),   np.sin(psi/2)
        self.x[6]  = cy*ct*cp + sy*st*sp   # qw
        self.x[7]  = cy*ct*sp - sy*st*cp   # qx
        self.x[8]  = cy*st*cp + sy*ct*sp   # qy
        self.x[9]  = sy*ct*cp - cy*st*sp   # qz
        # Normalise to guard against floating-point drift
        self.x[self._IQ] /= np.linalg.norm(self.x[self._IQ])

    def set_attitude(self, phi, theta, psi):
        """Rebuild quaternion from explicit roll/pitch/yaw (ZYX Tait-Bryan)."""
        cp, sp = np.cos(phi/2),   np.sin(phi/2)
        ct, st = np.cos(theta/2), np.sin(theta/2)
        cy, sy = np.cos(psi/2),   np.sin(psi/2)
        self.x[6]  = cy*ct*cp + sy*st*sp   # qw
        self.x[7]  = cy*ct*sp - sy*st*cp   # qx
        self.x[8]  = cy*st*cp + sy*ct*sp   # qy
        self.x[9]  = sy*ct*cp - cy*st*sp   # qz
        self.x[self._IQ] /= np.linalg.norm(self.x[self._IQ])

    def set_roll(self, phi):
        """
        Inject a roll angle [rad] into the quaternion while preserving
        the current pitch and yaw estimates.

        Decompose current quat into pitch (θ) and yaw (ψ), then rebuild
        from (phi, θ, ψ) using the ZYX Tait-Bryan convention.
        """
        qw, qx, qy, qz = self.x[self._IQ]
        theta = np.arcsin(np.clip(2.0*(qw*qy - qz*qx), -1.0, 1.0))
        psi   = np.arctan2(2.0*(qw*qz + qx*qy), 1.0 - 2.0*(qy*qy + qz*qz))
        cp, sp = np.cos(phi/2),   np.sin(phi/2)
        ct, st = np.cos(theta/2), np.sin(theta/2)
        cy, sy = np.cos(psi/2),   np.sin(psi/2)
        self.x[6]  = cy*ct*cp + sy*st*sp   # qw
        self.x[7]  = cy*ct*sp - sy*st*cp   # qx
        self.x[8]  = cy*st*cp + sy*ct*sp   # qy
        self.x[9]  = sy*ct*cp - cy*st*sp   # qz
        self.x[self._IQ] /= np.linalg.norm(self.x[self._IQ])

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def pos_ned(self):
        return self.x[self._IP].copy()

    @property
    def quat(self):
        return self.x[self._IQ].copy()

    @property
    def vel_ned(self):
        return self.x[self._IV].copy()

    @property
    def gyro_bias(self):
        return self.x[self._IBG].copy()

    # ── Internal helpers ──────────────────────────────────────────────────

    def _apply_update(self, innov, H, R_meas):
        """Generic EKF measurement update (Joseph form for numerical stability)."""
        S  = H @ self.P @ H.T + R_meas                  # innovation covariance
        K  = self.P @ H.T @ np.linalg.solve(S, np.eye(S.shape[0])).T  # Kalman gain (N×m)
        dx = K @ innov

        self.x[self._IP]  += dx[self._IP]
        self.x[self._IV]  += dx[self._IV]
        self.x[self._IQ]  += dx[self._IQ]
        self.x[self._IBG] += dx[self._IBG]

        # Renormalise quaternion after additive correction
        nrm = np.linalg.norm(self.x[self._IQ])
        if nrm > 1e-9:
            self.x[self._IQ] /= nrm

        # Joseph form:  P ← (I−KH) P (I−KH)ᵀ + K R Kᵀ
        IKH    = np.eye(self.N) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R_meas @ K.T


# ── Module-level pure helpers ─────────────────────────────────────────────

def _rot_b2n(q):
    """R_body_to_NED: v_NED = R @ v_body.  q = [qw, qx, qy, qz]."""
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*(qy*qy + qz*qz),  2*(qx*qy - qw*qz),  2*(qx*qz + qw*qy)],
        [    2*(qx*qy + qw*qz),  1 - 2*(qx*qx + qz*qz),  2*(qy*qz - qw*qx)],
        [    2*(qx*qz - qw*qy),  2*(qy*qz + qw*qx),  1 - 2*(qx*qx + qy*qy)],
    ])


def _omega_mat(omega):
    """4×4 Omega matrix: q_dot = 0.5 * Omega(omega) @ q.
    Matches dyn._omega_matrix (NED-to-body passive rotation, omega in body)."""
    p, q, r = omega
    return np.array([
        [ 0, -p, -q, -r],
        [ p,  0,  r, -q],
        [ q, -r,  0,  p],
        [ r,  q, -p,  0],
    ])


def _jacobian_F(q, omega, acc):
    """
    Continuous-time Jacobian  F = ∂f/∂x  (13×13).

    Non-zero blocks:
      F[0:3,  3:6]   = I₃                               position ← velocity
      F[3:6,  6:10]  = ∂(R_b2n(q) @ acc)/∂q            velocity ← quaternion
      F[6:10, 6:10]  = 0.5 * Omega(omega)               quaternion ← quaternion
      F[6:10, 10:13] = −0.5 * Xi(q)                     quaternion ← gyro bias
    """
    qw, qx, qy, qz = q
    ax, ay, az = acc
    p, qr, r   = omega                                   # corrected rates

    F = np.zeros((13, 13))

    # position ← velocity
    F[0:3, 3:6] = np.eye(3)

    # velocity ← quaternion  (same 3×4 block as the old 10-state EKF)
    F[3:6, 6:10] = np.array([
        # ∂vN_dot / ∂[qw, qx, qy, qz]
        [-2*qz*ay + 2*qy*az,
          2*qy*ay + 2*qz*az,
         -4*qy*ax + 2*qx*ay + 2*qw*az,
         -4*qz*ax - 2*qw*ay + 2*qx*az],
        # ∂vE_dot / ∂[qw, qx, qy, qz]
        [ 2*qz*ax - 2*qx*az,
          2*qy*ax - 4*qx*ay - 2*qw*az,
          2*qx*ax + 2*qz*az,
          2*qw*ax - 4*qz*ay + 2*qy*az],
        # ∂vD_dot / ∂[qw, qx, qy, qz]
        [-2*qy*ax + 2*qx*ay,
          2*qz*ax + 2*qw*ay - 4*qx*az,
         -2*qw*ax + 2*qz*ay - 4*qy*az,
          2*qx*ax + 2*qy*ay],
    ])

    # quaternion ← quaternion
    F[6:10, 6:10] = 0.5 * _omega_mat(omega)

    # quaternion ← gyro bias  (Xi matrix)
    F[6:10, 10:13] = -0.5 * np.array([
        [-qx, -qy, -qz],
        [ qw, -qz,  qy],
        [ qz,  qw, -qx],
        [-qy,  qx,  qw],
    ])

    return F


def _build_Q(q, dt, sa, sg, sb):
    """
    Discrete process noise covariance  Q (13×13).
    Position has no direct noise; uncertainty grows via the F[0:3,3:6] = I block.
    Uses the identity  Xi @ Xiᵀ = I₄ − q qᵀ  for unit quaternions.
    """
    Q = np.zeros((13, 13))
    # velocity
    Q[3:6, 3:6]     = dt * sa**2 * np.eye(3)
    # quaternion: projects onto tangent space of unit-sphere
    Q[6:10, 6:10]   = dt * sg**2 * 0.25 * (np.eye(4) - np.outer(q, q))
    # gyro bias bgx, bgy observable via roll/pitch gravity alignment
    Q[10:12, 10:12] = dt * sb**2 * np.eye(2)
    # bgz (index 12) NOT observable: frozen, no random walk added
    return Q


def _H_acc_quat(q, g):
    """
    Measurement Jacobian (3×4): ∂z_pred/∂q
    where z_pred = R_n2b(q) @ [0, 0, −g]
                 = −g · R_b2n[2, :]          (third row of R_b2n, scaled by −g)

    z_pred[0] = −g · 2(qx qz − qw qy)
    z_pred[1] = −g · 2(qy qz + qw qx)
    z_pred[2] = −g · (1 − 2qx² − 2qy²)
    """
    qw, qx, qy, qz = q
    return -g * np.array([
        [-2*qy,  2*qz, -2*qw,  2*qx],
        [ 2*qx,  2*qw,  2*qz,  2*qy],
        [    0, -4*qx, -4*qy,     0],
    ])
