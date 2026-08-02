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

import rotations


class QuadEKF:
    _IP  = slice(0, 3)    # position NED       [m]
    _IV  = slice(3, 6)    # velocity NED        [m/s]
    _IQ  = slice(6, 10)   # quaternion [qw qx qy qz]
    _IBG = slice(10, 13)  # gyro bias           [rad/s]
    N    = 13

    def __init__(self,
                 sigma_acc_proc    = 0.3,    # m/s²  / sqrt(Hz) — accel process noise
                 sigma_gyro_proc   = 0.02,   # rad/s / sqrt(Hz) — gyro  process noise
                 sigma_bias_proc   = 5e-5,   # rad/s²/ sqrt(Hz) — bias  random walk (bgx, bgy)
                 sigma_bias_z_proc = 5e-6,   # rad/s²/ sqrt(Hz) — bgz random walk; 0 = frozen
                 sigma_acc_meas    = 0.5,    # m/s²               — acc measurement noise
                 sigma_zupt        = 0.05,   # m/s                — ZUPT velocity noise
                 g                 = 9.81,
                 ):
        self._sa   = float(sigma_acc_proc)
        self._sg   = float(sigma_gyro_proc)
        self._sb   = float(sigma_bias_proc)
        self._sb_z = float(sigma_bias_z_proc)
        self._R_a  = float(sigma_acc_meas) ** 2 * np.eye(3)
        self._R_z  = float(sigma_zupt)     ** 2 * np.eye(3)
        self._g    = float(g)

        self.x = np.zeros(self.N)
        self.x[6] = 1.0                   # qw = 1 (identity attitude)
        self.P = np.diag(
            [1.0**2] * 3 +               # position: large initial uncertainty
            [0.5**2] * 3 +               # velocity
            [0.1**2] * 4 +               # quaternion
            [0.05**2] * 3                # gyro bias (bgx, bgy, bgz)
        ).astype(float)
        if self._sb_z == 0.0:
            self.P[12, 12] = 0.0         # bgz frozen (no yaw sensor)

        # Adaptive gate state (position/velocity/yaw updates only — see
        # _effective_gate). A gate rejection is meant as an outlier safety
        # net, but if the FILTER (not the measurement) has drifted past
        # gate_dist, every subsequent — even perfectly correct — measurement
        # also looks like an outlier, and a fixed gate becomes a one-way door
        # with no way back. Tracking a per-channel consecutive-rejection
        # streak and widening the effective gate once it runs long enough
        # lets a persistently-diverged filter accept a large correction and
        # snap back, rather than staying locked out for the rest of the flight.
        self._pos_reject_streak = 0
        self._vel_reject_streak = 0
        self._yaw_reject_streak = 0

        # Raw-measurement-to-measurement speed check for update_position (see
        # its docstring). Tracks the immediately preceding PnP position
        # measurement (regardless of whether it was applied), so it never
        # free-runs/drifts the way an open-loop dead-reckoned reference would
        # — position is a double integral of acceleration and drifts far too
        # fast for that approach (tried and reverted: rejected 76% of fixes
        # on a real log and made overall error much worse). Comparing
        # consecutive raw measurements directly avoids that entirely.
        self._last_pos_meas   = None
        self._last_pos_meas_t = None

        # Vision-immune shadow velocity — see predict()'s note and
        # update_velocity's IMU-consistency check for why this exists. Only
        # resynced to the (vision-corrected) state after _V_IMU_REF_RESYNC_SEC
        # has elapsed since the last resync (see update_velocity) — NOT on
        # every accepted correction. Resyncing every time would let this
        # reference get dragged along by the same slow, self-consistent
        # sequence of small corrections it exists to catch, one tick behind
        # the state it's supposed to be independently checking (found via a
        # synthetic "consensus attack" test: with per-update resync, a run of
        # individually-small-but-cumulatively-wrong velocity fixes was
        # accepted 15/15 times regardless of this check — identical to not
        # having it at all).
        self._v_imu_ref     = self.x[self._IV].copy()
        self._v_imu_ref_age = 0.0

        # Same idea, for yaw: a gyro-only shadow quaternion, checked in
        # update_yaw alongside the state gate. Needed because a PnP 180°-flip-
        # type yaw outlier is discrete and self-consistent once accepted (the
        # flipped detection keeps looking flipped), so it can ride the same
        # streak-widening path that let the velocity consensus-attack through.
        self._q_imu_ref       = self.x[self._IQ].copy()
        self._q_imu_ref_age   = 0.0

    _V_IMU_REF_RESYNC_SEC = 3.0  # min time between _v_imu_ref resyncs from vision

    _GATE_WIDEN_AFTER = 10    # consecutive rejections before the gate starts widening
    _GATE_WIDEN_STEP  = 0.5  # gate-multiples added per rejection beyond that
    _GATE_WIDEN_MAX   = 15.0 # cap on the widening multiplier

    def _effective_gate(self, gate_dist, streak):
        """Widen gate_dist after a long run of consecutive rejections (see
        __init__'s note on the adaptive-gate state)."""
        extra = streak - self._GATE_WIDEN_AFTER
        if extra <= 0:
            return gate_dist
        mult = min(1.0 + self._GATE_WIDEN_STEP * extra, self._GATE_WIDEN_MAX)
        return gate_dist * mult

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

        # Vision-immune shadow velocity: integrates the same IMU(+model)
        # specific force as the real state but is never touched by a vision
        # update. update_velocity gates incoming PnP velocity against this
        # reference in addition to self.x[self._IV] — the real state's
        # velocity can co-drift with a run of self-consistent-but-wrong PnP
        # fixes (e.g. tracking a false gate structure visible through the
        # true one's beams during a transit, which has its own smooth
        # relative motion), each individually inside gate_dist of the
        # *previous*, already-nudged state. _v_imu_ref can't be dragged that
        # way, so it stays an honest, independent witness to what IMU+model
        # alone predicts.
        self._v_imu_ref     = self._v_imu_ref + a_ned * dt
        self._v_imu_ref_age = self._v_imu_ref_age + dt

        # Gyro-only shadow quaternion for update_yaw's IMU-consistency check
        # (see __init__'s note) — same kinematics as the real quaternion
        # above, using the same bias-corrected rate, but never touched by a
        # vision yaw update.
        q_ref_new = self._q_imu_ref + 0.5 * dt * _omega_mat(omega) @ self._q_imu_ref
        nrm_ref = np.linalg.norm(q_ref_new)
        self._q_imu_ref     = q_ref_new / nrm_ref if nrm_ref > 1e-9 else self._q_imu_ref
        self._q_imu_ref_age = self._q_imu_ref_age + dt

        # Propagate covariance  P ← (I + F·dt) P (I + F·dt)ᵀ + Q
        F  = _jacobian_F(q, omega, acc_raw)
        Fd = np.eye(self.N) + F * dt
        Q  = _build_Q(q, dt, self._sa, self._sg, self._sb, self._sb_z)
        self.P = Fd @ self.P @ Fd.T + Q

    def update_accel(self, acc_raw, threshold=2.0):
        """
        Gravity-alignment measurement update.
        Returns (applied: bool, innov_norm: float).
        Skipped when |acc_norm − g| > threshold to avoid contaminating
        the estimate with large linear accelerations.
        """
        if not np.all(np.isfinite(acc_raw)):
            return False, 0.0
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

    def update_attitude(self, roll_meas, pitch_meas, sigma_att=0.15, gate_dist=5.0):
        """
        Roll/pitch measurement from PnP (see vision_rx.py's gate-orientation
        solve, which yields a full R_b2n — update_yaw already consumes its
        yaw component; this consumes the roll/pitch half of the same matrix).

        Reformulated as a synthetic gravity-vector measurement so it can
        reuse _H_acc_quat's existing Jacobian rather than deriving a new one:
        the body-frame gravity direction implied by (roll_meas, pitch_meas)
        alone (yaw=0; gravity direction in body frame is yaw-invariant) is
        compared against R_b2n(q).T @ [0,0,-g] from the CURRENT full state,
        whose own yaw doesn't affect the comparison — same mechanism as
        update_accel, just with the "measurement" coming from vision instead
        of the accelerometer. This is vision's only source of *ongoing*
        roll/pitch correction that doesn't go silent during real linear
        acceleration the way update_accel must (see its docstring) — PnP
        doesn't care whether the vehicle is accelerating, only how the gate
        looks.
        No adaptive gate-widening here (unlike update_position/velocity/yaw)
        — kept as simple as update_accel's fixed threshold for this first
        version; add streak-widening later if a real lockout shows up.

        IMU-consistency check against the vision-immune gyro-only shadow
        quaternion (_q_imu_ref) — identical pattern to update_yaw's imu_innov
        check, added for the same reason: gating only against the live state
        (self.x) can't catch a persistent, gradually-compounding roll/pitch
        bias, since each small update drags the state and the next update is
        then judged against that already-nudged value, looking "consistent"
        the whole way. Confirmed in a flight log: a real ~10-20° attitude/yaw
        error developed and persisted for the rest of an approach following a
        gate handoff — small enough per-tick that neither this gate nor
        update_yaw's own state-based gate ever rejected an individual step.
        _q_imu_ref is gyro-only and resynced to vision only after a cooldown
        (see update_yaw/__init__), so it can't be dragged the same way over a
        single approach.
        Returns (applied: bool, innov_norm: float).
        """
        if not (np.isfinite(roll_meas) and np.isfinite(pitch_meas)):
            return False, float('nan')

        g = self._g
        # ZYX quaternion from (roll, pitch, yaw=0) — same convention as set_attitude.
        q_synth = rotations.euler_to_quat(roll_meas, pitch_meas, 0.0)
        z_meas  = _rot_b2n(q_synth).T @ np.array([0., 0., -g])

        z_pred_imu = _rot_b2n(self._q_imu_ref).T @ np.array([0., 0., -g])
        imu_innov_norm = float(np.linalg.norm(z_meas - z_pred_imu))
        if imu_innov_norm > gate_dist:
            return False, imu_innov_norm

        q = self.x[self._IQ]
        z_pred = _rot_b2n(q).T @ np.array([0., 0., -g])
        innov  = z_meas - z_pred
        innov_norm = float(np.linalg.norm(innov))
        if innov_norm > gate_dist:
            return False, innov_norm

        H = np.zeros((3, self.N))
        H[:, 6:10] = _H_acc_quat(q, g)
        R = (sigma_att ** 2) * np.eye(3)

        self._apply_update(innov, H, R)
        # Resync only after a cooldown, not on every acceptance — see
        # update_yaw's identical pattern and __init__'s note.
        if self._q_imu_ref_age >= self._V_IMU_REF_RESYNC_SEC:
            self._q_imu_ref     = self.x[self._IQ].copy()
            self._q_imu_ref_age = 0.0
        return True, innov_norm

    def update_zupt(self, gyro_raw, acc_raw,
                    gyro_threshold=0.3, acc_threshold=1.5):
        """
        Zero-velocity update: treat v = 0 as a measurement when stationary.
        Returns (applied: bool, innov_norm: float).
        Fires only when the drone appears stationary (low gyro + gravity-aligned acc
        AND the EKF's own velocity estimate is near zero).
        """
        if not np.all(np.isfinite(gyro_raw)) or not np.all(np.isfinite(acc_raw)):
            return False, 0.0
        gyro_mag = float(np.linalg.norm(gyro_raw))
        acc_norm = float(np.linalg.norm(acc_raw))
        vel_mag  = float(np.linalg.norm(self.x[self._IV]))
        if gyro_mag > gyro_threshold or abs(acc_norm - self._g) > acc_threshold or vel_mag > 4.0:
            return False, 0.0

        innov = -self.x[self._IV]                        # z_meas − z_pred = 0 − v
        H     = np.zeros((3, self.N))
        H[0, 3] = H[1, 4] = H[2, 5] = 1.0              # measures v_ned at indices 3:6

        self._apply_update(innov, H, self._R_z)
        # ZUPT's own gating (near-zero gyro/accel/velocity) is a reliable,
        # vision-independent truth signal, so it's safe to always resync here
        # (unlike update_velocity's cooldown-gated resync below).
        self._v_imu_ref     = self.x[self._IV].copy()
        self._v_imu_ref_age = 0.0
        return True, float(np.linalg.norm(innov))

    def update_position(self, pos_ned_meas, sigma_pos=0.5, gate_dist=10.0,
                         t=None, max_speed=15.0):
        """
        NED position measurement from visual PnP against a known gate landmark.
        Rejects innovations larger than gate_dist metres (outlier/bad PnP), but
        the effective gate widens after a long run of consecutive rejections
        (see __init__'s note) so a diverged filter can eventually recover.

        Also rejects a measurement whose implied speed from the immediately
        preceding raw measurement exceeds max_speed [m/s] (only checked when
        `t` — the measurement's wall-clock time — is given). Confirmed in a
        flight log: a run of self-consistent-but-wrong PnP position fixes
        smoothly ramped ~3.5m over ~0.26s (implying 18-42 m/s, well above the
        vehicle's observed ~12 m/s top speed), each step individually small
        enough relative to the *already-dragged* state to slip past the gate
        above — the same consensus-attack mechanism already fixed for
        velocity/yaw. Deliberately NOT an open-loop dead-reckoned reference
        like _v_imu_ref/_q_imu_ref: position is a double integral of
        acceleration and drifts far too fast for that (tried and reverted —
        rejected 76% of fixes on a real log and made things much worse).
        Comparing consecutive raw measurements avoids any free-running drift
        entirely, at the cost of only catching the initial jump into a false
        lock rather than a sustained one — sufficient here because once the
        jump is blocked, the state never moves, so subsequent self-consistent
        bad measurements still fail the ordinary gate above.
        Returns (applied: bool, innov_norm: float).
        """
        pos_ned_meas = np.asarray(pos_ned_meas, dtype=float)
        if not np.all(np.isfinite(pos_ned_meas)):
            # A non-finite measurement (e.g. a degenerate PnP solve) must never
            # reach the gate check below: `nan > gate_dist` is False in numpy,
            # so a NaN innovation would silently pass the gate and inject NaN
            # straight into self.x via _apply_update, corrupting the filter
            # permanently. Reject outright and don't count it toward the
            # reject streak — it's not a "far away" measurement, just garbage.
            return False, float('nan')

        if t is not None and self._last_pos_meas_t is not None:
            dt_meas = t - self._last_pos_meas_t
            if dt_meas > 1e-3:
                jump = float(np.linalg.norm(pos_ned_meas - self._last_pos_meas))
                if jump / dt_meas > max_speed:
                    self._last_pos_meas   = pos_ned_meas.copy()
                    self._last_pos_meas_t = t
                    # Deliberately does NOT touch _pos_reject_streak: a
                    # physically-implausible jump is evidence the MEASUREMENT
                    # is garbage, not that the filter has diverged, so it must
                    # not feed the widening logic below (confirmed by testing:
                    # letting it increment the streak eventually widened
                    # eff_gate enough to admit a later, equally-bad point
                    # anyway — the same class of bug as update_velocity/
                    # update_yaw's imu-ref checks, just reached differently).
                    return False, jump
        self._last_pos_meas   = pos_ned_meas.copy()
        self._last_pos_meas_t = t

        innov = pos_ned_meas - self.x[self._IP]
        innov_norm = float(np.linalg.norm(innov))
        eff_gate = self._effective_gate(gate_dist, self._pos_reject_streak)
        if innov_norm > eff_gate:
            self._pos_reject_streak += 1
            return False, innov_norm
        self._pos_reject_streak = 0

        H       = np.zeros((3, self.N))
        H[0, 0] = H[1, 1] = H[2, 2] = 1.0   # measures position states 0:3
        R       = (sigma_pos ** 2) * np.eye(3)

        self._apply_update(innov, H, R)
        self.x[self._IQ] /= np.linalg.norm(self.x[self._IQ])
        return True, innov_norm

    def update_velocity(self, vel_ned_meas, sigma_vel=1.0, gate_dist=5.0):
        """
        NED velocity measurement derived from finite-differencing consecutive PnP positions.
        Rejects innovations larger than gate_dist m/s (outlier/dropped frame), but
        the effective gate widens after a long run of consecutive rejections
        (see __init__'s note) so a diverged filter can eventually recover.
        Returns (applied: bool, innov_norm: float).
        """
        vel_ned_meas = np.asarray(vel_ned_meas, dtype=float)
        if not np.all(np.isfinite(vel_ned_meas)):
            return False, float('nan')   # see update_position's note on why this guard exists

        eff_gate = self._effective_gate(gate_dist, self._vel_reject_streak)

        # IMU-consistency check against the vision-immune reference (see
        # predict()'s note). Runs first and independently of the state-based
        # gate below: a false-target lock that smoothly tracks its own
        # relative motion across several frames can pass the state gate one
        # small step at a time (each within eff_gate of the *previous*,
        # already-nudged state) while still being far from what pure
        # IMU+model integration says velocity should be. This check catches
        # that accumulated drift even when each individual step looked fine.
        # Deliberately gated on the *unwidened* gate_dist, not eff_gate: this
        # reference can't be dragged by a streak of bad vision fixes the way
        # the state can, so there's no "filter has genuinely diverged, let it
        # back in" case to accommodate here — gating it on eff_gate would let
        # exactly the outliers this check exists to catch ride through once
        # the streak (from whatever cause) widens past their size, which is
        # what happened in practice (confirmed in a flight log: a 78° yaw
        # innovation — the analogous bug in update_yaw — was accepted at
        # streak=12, eff_gate=114.6°, same root cause).
        imu_innov_norm = float(np.linalg.norm(vel_ned_meas - self._v_imu_ref))
        if imu_innov_norm > gate_dist:
            self._vel_reject_streak += 1
            return False, imu_innov_norm

        innov = vel_ned_meas - self.x[self._IV]
        innov_norm = float(np.linalg.norm(innov))
        if innov_norm > eff_gate:
            self._vel_reject_streak += 1
            return False, innov_norm
        self._vel_reject_streak = 0

        H       = np.zeros((3, self.N))
        H[0, 3] = H[1, 4] = H[2, 5] = 1.0   # measures velocity states 3:6
        R       = (sigma_vel ** 2) * np.eye(3)

        self._apply_update(innov, H, R)
        # Resync _v_imu_ref only after a cooldown, not on every acceptance —
        # see __init__'s note. This bounds _v_imu_ref's worst-case drift to
        # whatever pure IMU+model integration accumulates over
        # _V_IMU_REF_RESYNC_SEC (preventing a permanent lockout from genuine
        # long-run IMU drift) while keeping it immune to attack over any
        # shorter window, which is where the failure mode this check guards
        # against actually plays out.
        if self._v_imu_ref_age >= self._V_IMU_REF_RESYNC_SEC:
            self._v_imu_ref     = self.x[self._IV].copy()
            self._v_imu_ref_age = 0.0
        return True, innov_norm

    def update_yaw(self, yaw_meas, sigma_yaw=0.1, gate_dist=np.pi):
        """
        NED yaw measurement [rad] from PnP + known gate orientation.
        Uses the linearised Jacobian of yaw(q) w.r.t. the quaternion state.
        Wraps innovation to [-π, π].  Rejects |innov| > gate_dist, widening
        after a long run of consecutive rejections (see __init__'s note).
        Returns (applied: bool, innov_abs: float).
        """
        if not np.isfinite(yaw_meas):
            return False, float('nan')   # see update_position's note on why this guard exists

        eff_gate = self._effective_gate(gate_dist, self._yaw_reject_streak)

        # IMU-consistency check against the vision-immune gyro-only shadow
        # quaternion (see __init__'s note and predict()'s _q_imu_ref update).
        # Catches a discrete, self-consistent yaw outlier (e.g. a PnP 180°
        # flip) that the state gate's streak-widening would otherwise admit —
        # confirmed in a flight log: 50-80° yaw innovations were being
        # accepted via widening, each one immediately dragging the estimate
        # and looking "consistent" to the next flipped detection. Gated on the
        # unwidened gate_dist, not eff_gate — see update_velocity's identical
        # note on why this reference must not honor streak-widening.
        imu_innov = float(yaw_meas) - _yaw_of_quat(self._q_imu_ref)
        imu_innov = (imu_innov + np.pi) % (2.0 * np.pi) - np.pi
        if abs(imu_innov) > gate_dist:
            self._yaw_reject_streak += 1
            return False, abs(imu_innov)

        qw, qx, qy, qz = self.x[self._IQ]
        f = 2.0 * (qw * qz + qx * qy)
        g = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw_est = float(np.arctan2(f, g))

        innov = float(yaw_meas) - yaw_est
        innov = (innov + np.pi) % (2.0 * np.pi) - np.pi   # wrap to [-π, π]
        if abs(innov) > eff_gate:
            self._yaw_reject_streak += 1
            return False, abs(innov)
        self._yaw_reject_streak = 0

        denom = f * f + g * g + 1e-12
        H = np.zeros((1, self.N))
        H[0, 6] = 2.0 * qz * g / denom
        H[0, 7] = 2.0 * qy * g / denom
        H[0, 8] = (2.0 * qx * g + 4.0 * qy * f) / denom
        H[0, 9] = (2.0 * qw * g + 4.0 * qz * f) / denom

        R = np.array([[sigma_yaw ** 2]])
        self._apply_update(np.array([innov]), H, R)
        self.x[self._IQ] /= np.linalg.norm(self.x[self._IQ])
        # Resync only after a cooldown, not on every acceptance — see
        # update_velocity's identical pattern and __init__'s note.
        if self._q_imu_ref_age >= self._V_IMU_REF_RESYNC_SEC:
            self._q_imu_ref     = self.x[self._IQ].copy()
            self._q_imu_ref_age = 0.0
        return True, abs(innov)

    def reset(self):
        """Reset to identity attitude, zero velocity, zero bias, zero position."""
        self.x[:] = 0.
        self.x[6] = 1.0                                  # qw = 1
        self.P[:] = 0.
        np.fill_diagonal(self.P,
            [1.0**2] * 3 + [0.5**2] * 3 + [0.1**2] * 4 + [0.05**2] * 3)
        if self._sb_z == 0.0:
            self.P[12, 12] = 0.0                         # bgz: frozen
        self._pos_reject_streak = 0
        self._vel_reject_streak = 0
        self._yaw_reject_streak = 0
        self._last_pos_meas   = None
        self._last_pos_meas_t = None
        self._v_imu_ref     = self.x[self._IV].copy()
        self._v_imu_ref_age = 0.0
        self._q_imu_ref     = self.x[self._IQ].copy()
        self._q_imu_ref_age = 0.0

    def set_yaw(self, psi):
        """
        Inject a ground-truth yaw angle [rad] into the quaternion while
        preserving the current roll and pitch estimates.

        Decompose current quat into roll (φ) and pitch (θ), then rebuild
        from (φ, θ, psi) using the ZYX Tait-Bryan convention:
            q = q_z(psi) * q_y(theta) * q_x(phi)
        """
        # Extract roll and pitch from current EKF quaternion, rebuild with
        # corrected yaw — see rotations.py for the shared euler<->quat formulas.
        phi, theta, _ = rotations.quat_to_euler(self.x[self._IQ])
        self.x[6:10] = rotations.euler_to_quat(phi, theta, psi)
        # Normalise to guard against floating-point drift
        self.x[self._IQ] /= np.linalg.norm(self.x[self._IQ])
        self._q_imu_ref     = self.x[self._IQ].copy()
        self._q_imu_ref_age = 0.0

    def set_attitude(self, phi, theta, psi):
        """Rebuild quaternion from explicit roll/pitch/yaw (ZYX Tait-Bryan)."""
        self.x[6:10] = rotations.euler_to_quat(phi, theta, psi)
        self.x[self._IQ] /= np.linalg.norm(self.x[self._IQ])
        self._q_imu_ref     = self.x[self._IQ].copy()
        self._q_imu_ref_age = 0.0

    def set_roll(self, phi):
        """
        Inject a roll angle [rad] into the quaternion while preserving
        the current pitch and yaw estimates.

        Decompose current quat into pitch (θ) and yaw (ψ), then rebuild
        from (phi, θ, ψ) using the ZYX Tait-Bryan convention.
        """
        _, theta, psi = rotations.quat_to_euler(self.x[self._IQ])
        self.x[6:10] = rotations.euler_to_quat(phi, theta, psi)
        self.x[self._IQ] /= np.linalg.norm(self.x[self._IQ])
        self._q_imu_ref     = self.x[self._IQ].copy()
        self._q_imu_ref_age = 0.0

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

# Extract NED yaw [rad] from q = [qw,qx,qy,qz]. Shared by update_yaw and its
# IMU-consistency check so both use the identical formula — see rotations.py.
_yaw_of_quat = rotations.quat_to_yaw

# R_body_to_NED: v_NED = R @ v_body.  q = [qw, qx, qy, qz]. — see rotations.py
_rot_b2n = rotations.quat_to_R_body2ned


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


def _build_Q(q, dt, sa, sg, sb, sb_z=0.0):
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
    # bgz: small random walk lets filter slowly estimate yaw bias when yaw is
    # observed via PnP or velocity-heading updates; sb_z=0 keeps it fully frozen
    Q[12, 12] = dt * sb_z**2
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
