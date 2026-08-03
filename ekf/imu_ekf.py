"""
imu_ekf.py
==========
IMU-driven EKF: gyro/acc calibration, hover-entry zeroing, model-aided predict,
and vision corrections.

Extracted from the monolithic mavlink_rx.py so mavlink_rx stays a clean
MAVLink transport layer with no physics knowledge.

Usage
-----
    from flight_model.dyn import load_params
    from comms.mavlink_rx import MAVLinkRX
    from ekf.imu_ekf import IMUEKFHandler

    param  = load_params()
    shared = {}
    rx     = MAVLinkRX.create_mavlink_rx(conn, shared, logger=logger)
    ekf_h  = IMUEKFHandler(shared, param, logger=logger)
    ekf_h.register(rx)          # runs inline in the MAVLinkRX thread at 250 Hz

Shared-data keys written
------------------------
    shared['mav_state']                 — EKF pos / vel / quat / rates
    shared['pos_offset_ned']            — world-NED position at hover-entry reset
    shared['imu_raw']                   — raw IMU sample (legacy key, read by scripts)
    shared['_imu_rates']                — FRD-corrected gyro rates (legacy key)
"""

import time
from collections import deque

import numpy as np

from ekf.ekf import QuadEKF
from flight_model.dyn import _quat_to_R as _dyn_quat_to_R
from flight_model import rotations


def _quat_mult(q1, q2):
    """Hamilton product q1 ⊗ q2, both [qw, qx, qy, qz]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def _quat_to_euler_deg(q):
    """ZYX Tait-Bryan (roll, pitch, yaw) in degrees from q = [qw, qx, qy, qz].
    See rotations.py (quat_to_euler) for the shared radians formula."""
    roll, pitch, yaw = rotations.quat_to_euler(q)
    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)


class IMUEKFHandler:
    """
    Drives a QuadEKF from HIGHRES_IMU messages forwarded by MAVLinkRX.

    Runs in the MAVLinkRX receive thread via a callback — no extra thread.
    """

    def __init__(self, shared_data, param, logger=None):
        self._data   = shared_data
        self._param  = param
        self._logger = logger

        self._ekf        = QuadEKF(
            sigma_bias_z_proc=float(param.get('sigma_bias_z_proc', 5e-6)),
        )
        self._last_imu_t = None
        # See on_imu_msg's predict-dt clamp for why this needs to be bigger
        # than the nominal ~250Hz period once the actual IMU delivery rate
        # is lower — confirmed via flight-log wall_t deltas (~61Hz + jitter
        # up to ~62ms in a real run). Not tied to any single assumed rate.
        self._imu_dt_max_s = float(param.get('imu_predict_dt_max_s', 0.15))

        # World-NED position at hover-entry reset (EKF zeroes at that point).
        # Subtracted from every vision position update so EKF and PnP share a frame.
        self._pos_offset_ned = np.zeros(3)

        # Gyro + acc accumulation during WAIT (drone static on slope).
        # Gyro mean → bias injected at reset; acc mean → attitude init.
        self._wait_gyro_sum  = np.zeros(3)
        self._wait_gyro_cnt  = 0
        self._wait_acc_sum   = np.zeros(3)
        self._wait_acc_cnt   = 0
        self._hover_reset_done = False

        # Initial EKF yaw from params.yaml
        _init_yaw_deg = float(param.get('initial_yaw_deg', 0.0))
        self._init_yaw_rad = np.deg2rad(_init_yaw_deg)
        if _init_yaw_deg != 0.0:
            self._ekf.set_yaw(self._init_yaw_rad)
            print(f'[IMUEKFHandler] EKF yaw pre-set: {_init_yaw_deg:.1f}°', flush=True)

        self._hover_reset_t            = None

        # Item 3: rolling RMS buffer for adaptive motor-model sigma (250 samples ≈ 1 s)
        self._acc_model_errs = deque(maxlen=250)

        # Item 4: EKF position history for PnP latency compensation [(wall_t, pos), ...]
        self._state_buf = deque(maxlen=200)

        # Item 2: last gate id whose position has already been injected as an EKF fix
        self._last_gate_id_processed = None

        # Ground truth mode: sim NED position at hover-entry reset (frame origin)
        # and yaw-alignment rotation to map sim NED → EKF frame.
        self._sim_ned_offset = np.zeros(3)
        self._R_align = np.eye(3)                           # identity until hover reset
        self._q_align = np.array([1.0, 0.0, 0.0, 0.0])    # identity quaternion

        # Shadow model-predict log: runs the physics acc computation alongside the
        # EKF (regardless of use_model_predict) and writes acc_model vs acc_imu to CSV.
        self._mp_shadow_enabled = bool(param.get('model_predict_shadow', False))
        self._mp_shadow_tick    = 0
        self._mp_shadow_writer  = None
        self._mp_shadow_file    = None
        self._mp_shadow_path    = None
        self._v_model_ned       = None   # integrated model velocity [m/s] in NED

        # Motor first-order lag + accelerometer bias — both identified offline
        # by fit_shadow.py. tau_motor is a real physical param (already in
        # params.yaml); accel_bias is diagnostic/session-specific (fit_shadow.py
        # never auto-writes it — only via its --write-bias flag) so it defaults
        # to zero unless the user explicitly opts a trusted fit into params.yaml.
        self._tau_motor  = float(param.get('tau_motor', 0.05))
        self._accel_bias = np.array(param.get('accel_bias', [0.0, 0.0, 0.0]), dtype=float)
        self._T_actual   = None   # lag-filtered thrust state [N], None until first tick
        if self._mp_shadow_enabled:
            import csv, os
            _dir = (logger.session_dir
                    if logger is not None and hasattr(logger, 'session_dir')
                    else 'logs')
            os.makedirs(_dir, exist_ok=True)
            self._mp_shadow_path   = os.path.join(_dir, 'model_predict_shadow.csv')
            self._mp_shadow_file   = open(self._mp_shadow_path, 'w', newline='', buffering=1)
            self._mp_shadow_writer = csv.writer(self._mp_shadow_file)
            self._mp_shadow_writer.writerow([
                't_wall_s', 'T_total_N', 'actuator_sum',
                'act_fl', 'act_fr', 'act_bl', 'act_br',
                'vb_x', 'vb_y', 'vb_z',
                'gt_vb_x', 'gt_vb_y', 'gt_vb_z',
                'acc_imu_x', 'acc_imu_y', 'acc_imu_z',
                'acc_model_x', 'acc_model_y', 'acc_model_z',
                'err_x', 'err_y', 'err_z', 'err_norm',
                'vel_N', 'vel_E', 'vel_D',
                'v_model_N', 'v_model_E', 'v_model_D',
            ])
            print(f'[IMUEKFHandler] model_predict_shadow → {self._mp_shadow_path}', flush=True)

        # Shadow log: live EKF estimate vs ground truth, error computed every
        # tick. Most meaningful with ground_truth_mode: false (a genuinely
        # free-running EKF) — in ground_truth_mode: true this would just show
        # ~0 error, since self._ekf.x IS gt every tick by construction. Reuses
        # the same raw-GT read + R_align/sim_ned_offset/q_align transform as
        # the ground-truth-override block below, so the comparison is in the
        # EKF's own (hover-reset-zeroed) frame — comparing against raw
        # unaligned GT would show a frame offset, not real estimation error.
        self._gt_err_shadow_enabled = bool(param.get('ekf_gt_error_shadow', False))
        self._gt_err_shadow_tick    = 0
        self._gt_err_shadow_writer  = None
        self._gt_err_shadow_file    = None
        self._gt_err_shadow_path    = None
        if self._gt_err_shadow_enabled:
            import csv, os
            _dir = (logger.session_dir
                    if logger is not None and hasattr(logger, 'session_dir')
                    else 'logs')
            os.makedirs(_dir, exist_ok=True)
            self._gt_err_shadow_path   = os.path.join(_dir, 'ekf_gt_error_shadow.csv')
            self._gt_err_shadow_file   = open(self._gt_err_shadow_path, 'w', newline='', buffering=1)
            self._gt_err_shadow_writer = csv.writer(self._gt_err_shadow_file)
            self._gt_err_shadow_writer.writerow([
                't_wall_s',
                'pN', 'pE', 'pD', 'gt_pN', 'gt_pE', 'gt_pD',
                'pos_err_N', 'pos_err_E', 'pos_err_D', 'pos_err_norm',
                'vN', 'vE', 'vD', 'gt_vN', 'gt_vE', 'gt_vD',
                'vel_err_N', 'vel_err_E', 'vel_err_D', 'vel_err_norm',
                'roll_deg', 'pitch_deg', 'yaw_deg',
                'roll_gt_deg', 'pitch_gt_deg', 'yaw_gt_deg',
                'roll_err_deg', 'pitch_err_deg', 'yaw_err_deg',
            ])
            print(f'[IMUEKFHandler] ekf_gt_error_shadow → {self._gt_err_shadow_path}', flush=True)

    def register(self, rx):
        """Attach this handler as the HIGHRES_IMU callback on a MAVLinkRX instance."""
        rx._on_imu_cb = self.on_imu_msg

    # ------------------------------------------------------------------

    def on_imu_msg(self, msg):
        """Called from the MAVLinkRX thread on every HIGHRES_IMU message (~250 Hz)."""
        ax, ay, az = msg.xacc, msg.yacc, msg.zacc
        gx, gy, gz = msg.xgyro, msg.ygyro, msg.zgyro
        t_us = msg.time_usec

        # ── Hover-entry EKF reset ─────────────────────────────────────────────
        if self._data.get('reset_vel_flag'):
            self._data['reset_vel_flag'] = False
            self._hover_reset_done       = True

            # Save world-frame position (vision was locked on during WAIT)
            self._pos_offset_ned = self._ekf.x[0:3].copy()
            self._data['pos_offset_ned'] = self._pos_offset_ned.copy()

            self._ekf.reset()

            # Re-inject calibrated gyro bias (clamped to ±0.05 rad/s).
            # Values above the limit indicate contamination from motion during WAIT
            # and must be rejected to prevent the EKF from diverging at lift-off.
            _MAX_GYRO_BIAS = 0.05
            if self._wait_gyro_cnt > 50:
                _gyro_bias = self._wait_gyro_sum / self._wait_gyro_cnt
                if np.any(np.abs(_gyro_bias) > _MAX_GYRO_BIAS):
                    print(f'[IMUEKFHandler] gyro bias {_gyro_bias} exceeds '
                          f'±{_MAX_GYRO_BIAS} rad/s — zeroed', flush=True)
                    _gyro_bias = np.zeros(3)
                self._ekf.x[10:13] = _gyro_bias
                # Pin bias states: freeze the WAIT-phase calibration so forward-
                # flight acc updates cannot corrupt the yaw via the bias cross-term.
                self._ekf.P[10:13, :] = 0.0
                self._ekf.P[:, 10:13] = 0.0
            else:
                _gyro_bias = np.zeros(3)

            # Set yaw only — NOT roll/pitch from the static acc average anymore.
            # That average is captured only over the WAIT window (see the
            # wait_phase_done gate above), i.e. before the controller's pre-blip
            # levelling phase runs. self._ekf.reset() just above already zeroed
            # the quaternion to identity (phi=theta=0), which is the right prior
            # now: the drone has been actively levelled since WAIT ended, so the
            # WAIT-only tilt reading is stale and re-applying it here would
            # overwrite the just-achieved level attitude with the old ~17° slope
            # tilt — confirmed in a flight log as an exact-looking snap-back
            # (theta recovering to ~-4° during levelling, then jumping back to
            # ~-18° right at this reset). set_yaw() preserves the current
            # (post-reset-identity, i.e. level) roll/pitch and only sets yaw.
            self._ekf.set_yaw(self._init_yaw_rad)
            if self._wait_acc_cnt > 50:
                _acc_avg = self._wait_acc_sum / self._wait_acc_cnt
                _g       = 9.81
                _theta = float(np.arcsin(np.clip(_acc_avg[0] / _g, -1.0, 1.0)))
                _phi   = float(np.arcsin(
                    np.clip(_acc_avg[1] / (_g * max(np.cos(_theta), 0.1)), -1.0, 1.0)))
                print(f'[IMUEKFHandler] WAIT static tilt (diagnostic only, not applied): '
                      f'phi={np.rad2deg(_phi):.1f}°  theta={np.rad2deg(_theta):.1f}°  '
                      f'psi={np.rad2deg(self._init_yaw_rad):.1f}°  '
                      f'(n={self._wait_acc_cnt})', flush=True)

            # ── Ground truth frame: position offset ───────────────────────────
            _latest_gt = self._data.get('mavlink', {}).get('latest', {})
            _lpn_reset = _latest_gt.get('LOCAL_POSITION_NED')
            if _lpn_reset is not None:
                self._sim_ned_offset = np.array(_lpn_reset['pos_ned_m'], dtype=float)
                print(f'[IMUEKFHandler] sim NED offset at reset: {self._sim_ned_offset}',
                      flush=True)
            else:
                self._sim_ned_offset = np.zeros(3)
                print('[IMUEKFHandler] LOCAL_POSITION_NED not available at reset '
                      '— sim NED offset zeroed', flush=True)

            # ── Ground truth frame: yaw alignment ─────────────────────────────
            # If the sim spawns the drone at a yaw different from initial_yaw_deg,
            # rotate the sim NED frame so pos/vel/attitude match the EKF convention.
            # delta_psi = expected_yaw - actual_sim_yaw  (rotation applied to all GT)
            _att_reset = _latest_gt.get('ATTITUDE')
            if _att_reset is not None:
                _psi_sim   = float(_att_reset['yaw_rad'])
                _delta_psi = self._init_yaw_rad - _psi_sim
                self._q_align = np.array([np.cos(_delta_psi / 2), 0.0, 0.0,
                                          np.sin(_delta_psi / 2)])
                self._R_align = rotations.quat_to_R_body2ned(self._q_align)
                if abs(np.rad2deg(_delta_psi)) > 1.0:
                    print(f'[IMUEKFHandler] GT frame yaw correction: '
                          f'sim={np.rad2deg(_psi_sim):.1f}°  '
                          f'expected={np.rad2deg(self._init_yaw_rad):.1f}°  '
                          f'delta={np.rad2deg(_delta_psi):.1f}°', flush=True)
                else:
                    print(f'[IMUEKFHandler] GT frame aligned '
                          f'(delta={np.rad2deg(_delta_psi):.2f}°)', flush=True)
            else:
                self._R_align = np.eye(3)
                self._q_align = np.array([1.0, 0.0, 0.0, 0.0])
                print('[IMUEKFHandler] ATTITUDE not available at reset '
                      '— GT frame not rotated', flush=True)

            print(f'[IMUEKFHandler] EKF reset; '
                  f'pos_offset={self._pos_offset_ned}  '
                  f'gyro_bias={_gyro_bias} (n={self._wait_gyro_cnt})', flush=True)
            self._hover_reset_t = time.time()

        # ── Sim sensor → FRD conversion ───────────────────────────────────────
        from flight_model.sim_convention import sim_to_frd_gyro
        gyro = sim_to_frd_gyro(gx, gy, gz)
        acc  = np.array([ax, ay, az])

        # ── WAIT-phase accumulation ───────────────────────────────────────────
        # Gated on wait_phase_done (set by controller.py the moment WAIT ends),
        # not hover_reset_done: the controller now runs a pre-blip levelling
        # phase between WAIT and the actual hover-entry reset, and that phase
        # involves real, intentional rotation — averaging it in here would
        # corrupt both the gyro-bias calibration and the static-tilt reading,
        # which are only meaningful over the truly-static WAIT window.
        if not self._data.get('wait_phase_done', False):
            self._wait_gyro_sum += gyro
            self._wait_gyro_cnt += 1
            self._wait_acc_sum  += acc
            self._wait_acc_cnt  += 1
            # Expose running average so controller can display ground tilt (sysid).
            self._data['wait_acc_avg'] = self._wait_acc_sum / max(self._wait_acc_cnt, 1)
            # Pin EKF position to zero so pos_offset_ned stays correct at reset
            if self._data.get('_vision_ekf_update') is None:
                self._ekf.x[0:3] = 0.0

        # ── IMU spike gate ────────────────────────────────────────────────────
        # Physical max ≈ 41.9 m/s² (4 × 57.53 N / 5.49 kg at full throttle).
        # Spikes above 42 m/s² are sim artefacts; replace with the exact gravity
        # vector in body frame so R_b2n @ acc_fallback + g_ned = 0 (velocity holds).
        if np.linalg.norm(acc) > 42.0:
            qw, qx, qy, qz = self._ekf.x[6:10]
            _g = 9.81
            acc = np.array([
                2.0 * (qx * qz - qw * qy) * (-_g),
                2.0 * (qy * qz + qw * qx) * (-_g),
                (1.0 - 2.0 * qx * qx - 2.0 * qy * qy) * (-_g),
            ])

        # Legacy shared keys (read by flight scripts and the old logger API)
        self._data['_imu_rates'] = gyro
        self._data['imu_raw'] = {
            'ax': float(ax), 'ay': float(ay), 'az': float(az),
            'gx': float(gx), 'gy': float(gy), 'gz': float(gz),
            't_us': t_us,
        }

        now = time.time()
        dt  = (now - self._last_imu_t) if self._last_imu_t is not None else 0.004
        # Ceiling was 0.05s, sized for a ~250Hz nominal IMU rate — confirmed
        # in a flight log the sim can now deliver HIGHRES_IMU at ~61Hz with
        # jitter up to ~62ms, so that ceiling was clipping (and silently
        # discarding) the tail of ordinary gaps, not just genuine dropouts.
        # Raised to comfortably cover that jitter; still bounded so a real,
        # much longer dropout doesn't get integrated as one enormous step.
        dt  = float(np.clip(dt, 0.0005, self._imu_dt_max_s))
        self._last_imu_t = now

        # ── Model-aided EKF predict ───────────────────────────────────────────
        # Compute physics-derived specific force whenever motors are known.
        # acc_model is used by EKF (use_model_predict) and/or shadow log.
        _use_mp  = self._param.get('use_model_predict', False)
        _use_shd = self._mp_shadow_enabled
        acc_model = None
        if _use_mp or _use_shd:
            _act = (self._data.get('mavlink', {})
                              .get('latest', {})
                              .get('ACTUATOR_OUTPUT_STATUS'))
            if _act is not None:
                _T_max   = float(self._param.get('T_max_motor', 49.9))
                _T_total = float(np.sum(_act['actuator'][:4])) * _T_max
                if _T_total > 1.0:
                    # First-order motor lag: commanded thrust != actual thrust
                    # (dyn.dyn_lag's ODE, applied here to the live model_predict
                    # path — previously only the offline fit_shadow.py model
                    # accounted for this, leaving this path assuming instant
                    # thrust response).
                    if self._T_actual is None:
                        self._T_actual = _T_total
                    else:
                        _alpha = float(np.clip(dt / max(self._tau_motor, 1e-6), 0.0, 1.0))
                        self._T_actual += (_T_total - self._T_actual) * _alpha

                    _R_nb      = _dyn_quat_to_R(self._ekf.x[6:10])
                    _vel_b     = _R_nb @ self._ekf.x[3:6]
                    _F_drag    = -self._param['Dv'] @ (np.abs(_vel_b) * _vel_b)
                    acc_model  = ((np.array([0.0, 0.0, -self._T_actual]) + _F_drag)
                                  / float(self._param['m'])) + self._accel_bias
                    # Shadow log: compare model vs raw IMU (every 10th tick ≈ 25 Hz)
                    if _use_shd and self._hover_reset_done and self._mp_shadow_writer is not None:
                        # Integrate model velocity in NED at every IMU tick for accuracy.
                        # acc_model is body-frame specific force; add gravity to get true NED accel.
                        # Guard: only integrate when thrust is near hover level — avoids corrupting
                        # v_model during the blip (T>>hover) and the motor-ramp/low-thrust transient
                        # (T<<hover) where the model would predict near-freefall.
                        _T_hover = float(self._param['m']) * float(self._param.get('g', 9.81))
                        _g_ned = np.array([0.0, 0.0, float(self._param.get('g', 9.81))])
                        _acc_model_ned = _R_nb.T @ acc_model + _g_ned
                        if self._v_model_ned is None:
                            if _T_total > 0.5 * _T_hover:
                                # Only initialise once motors are meaningfully loaded
                                self._v_model_ned = self._ekf.x[3:6].copy()
                        elif 0.3 * _T_hover < _T_total < 2.5 * _T_hover:
                            # Integrate only within a sensible thrust band; outside this band
                            # (blip peak or near-idle) resync to GT to prevent drift accumulation
                            self._v_model_ned = self._v_model_ned + _acc_model_ned * dt
                        else:
                            self._v_model_ned = self._ekf.x[3:6].copy()
                        self._mp_shadow_tick += 1
                        if self._mp_shadow_tick % 10 == 0:
                            _err = acc_model - acc
                            _act_sum = float(np.sum(_act['actuator'][:4]))
                            _act4 = [float(v) for v in _act['actuator'][:4]]

                            # Ground-truth body velocity, computed independently of
                            # self._ekf.x directly from the raw sim mavlink messages
                            # (ATTITUDE + LOCAL_POSITION_NED are both in the sim's
                            # native NED frame, so no R_align/pos_offset correction
                            # is needed — velocity and the rotation to body frame
                            # are frame-offset-invariant).
                            _latest_gt_row = self._data.get('mavlink', {}).get('latest', {})
                            _att_gt_row    = _latest_gt_row.get('ATTITUDE')
                            _lpn_gt_row    = _latest_gt_row.get('LOCAL_POSITION_NED')
                            if _att_gt_row is not None and _lpn_gt_row is not None:
                                _gt_vb = (_dyn_quat_to_R(_att_gt_row['quat'])
                                          @ np.array(_lpn_gt_row['vel_ned_mps'], dtype=float))
                                _gt_vb_str = [f'{_gt_vb[0]:.3f}', f'{_gt_vb[1]:.3f}', f'{_gt_vb[2]:.3f}']
                            else:
                                _gt_vb_str = ['nan', 'nan', 'nan']

                            self._mp_shadow_writer.writerow([
                                f'{now:.4f}',
                                f'{_T_total:.2f}',
                                f'{_act_sum:.4f}',
                                f'{_act4[0]:.4f}', f'{_act4[1]:.4f}', f'{_act4[2]:.4f}', f'{_act4[3]:.4f}',
                                f'{_vel_b[0]:.3f}', f'{_vel_b[1]:.3f}', f'{_vel_b[2]:.3f}',
                                *_gt_vb_str,
                                f'{acc[0]:.4f}', f'{acc[1]:.4f}', f'{acc[2]:.4f}',
                                f'{acc_model[0]:.4f}', f'{acc_model[1]:.4f}', f'{acc_model[2]:.4f}',
                                f'{_err[0]:.4f}', f'{_err[1]:.4f}', f'{_err[2]:.4f}',
                                f'{float(np.linalg.norm(_err)):.4f}',
                                f'{self._ekf.x[3]:.3f}', f'{self._ekf.x[4]:.3f}', f'{self._ekf.x[5]:.3f}',
                                f'{self._v_model_ned[0]:.3f}', f'{self._v_model_ned[1]:.3f}', f'{self._v_model_ned[2]:.3f}',
                            ])

        if _use_mp and acc_model is not None:
            # Sigma-weighted fusion: minimum-variance blend of IMU and physics model.
            # w_imu = σ²_model / (σ²_imu + σ²_model); both contribute every tick.
            # update_accel() and update_zupt() always use raw acc — unaffected by blend.
            _s2_imu = float(self._param.get('sigma_acc_imu', 0.5)) ** 2
            # Item 3: adaptive sigma_acc_model from rolling RMS of model-vs-IMU error
            if len(self._acc_model_errs) >= 10:
                _sig_mdl = float(np.clip(
                    np.sqrt(np.mean(np.array(self._acc_model_errs) ** 2)), 0.1, 3.0))
            else:
                _sig_mdl = float(self._param.get('sigma_acc_model', 0.3))
            _s2_model = _sig_mdl ** 2
            _w_imu    = _s2_model / (_s2_imu + _s2_model)
            acc_predict = _w_imu * acc + (1.0 - _w_imu) * acc_model
            self._acc_model_errs.append(float(np.linalg.norm(acc_model - acc)))
        else:
            acc_predict = acc
        self._ekf.predict(gyro, acc_predict, dt)
        # Item 4: record EKF position after predict for latency compensation
        self._state_buf.append((now, self._ekf.x[0:3].copy()))
        _acc_norm_upd = float(np.linalg.norm(acc))  # raw IMU for update gate

        # No more attitude freeze or backup post-blip re-init: both existed only
        # to protect against the liftoff blip's motor-ramp transient, which no
        # longer exists (see controller.py PHASE 1.5 / hover-entry). Firing a
        # one-shot accel-derived attitude reset here would actually be actively
        # harmful now — the drone typically has real, sustained acceleration
        # within the first ~0.5s of cascade control, and treating that as a pure
        # tilt error is exactly the bug fixed by keeping accel_settle_sec short.
        # update_accel() below already has the same protection via that window.

        # ── Accelerometer attitude update ─────────────────────────────────────
        # accel_settle_sec: short post-reset window during which the gravity-
        # alignment update is trusted. Kept short (default 0.50s) — too long and
        # this update starts firing during genuine flight acceleration,
        # misreading it as attitude error and (via velocity-attitude covariance)
        # corrupting velocity too. Deliberately separate from vision_settle_sec
        # below: roll/pitch (what this update corrects) and yaw (what that one
        # protects) settle on different timescales post-reset.
        #
        # Briefly tried making this run continuously (motivated by a flight
        # where a frozen/stale gyro reading drove roll/pitch to diverge
        # unboundedly with zero ongoing correction — see update_attitude
        # below, added the same session, for the actual fix to that gap) —
        # reverted after confirming on a real log that the |acc_norm-g|<2.0
        # threshold does NOT reliably protect against this vehicle's real
        # flight profile: a genuine -39° GT pitch dive at t=2.02s had
        # accnorm=9.79 (well inside the 2.0 threshold), so update_accel fired
        # and dragged the estimate from the true -39° toward level, landing
        # at -17° — wrong by 22° from a single "protected" update, repeating
        # every time the vehicle maneuvers aggressively (128 of 245 ticks in
        # one 4s window). The banked-turn/dive norm-cancellation risk this
        # window was originally added to avoid is real and frequent for a
        # racing profile, not occasional — restored the short window.
        # update_attitude (PnP-based) was tried as the ongoing roll/pitch
        # correction instead, since it doesn't look at acceleration — but it
        # has now also been disabled (below) after causing a crash via a
        # different failure mode (planar-target PnP pose ambiguity). Currently
        # there is no ongoing roll/pitch correction between accel_settle_sec
        # windows; only the short post-reset accel window and update_zupt.
        _accel_settle_window = (
            not self._hover_reset_done
            or (self._hover_reset_t is not None
                and now - self._hover_reset_t
                    < self._param.get('accel_settle_sec', 0.50))
        )
        if abs(_acc_norm_upd - 9.81) < 2.0 and _accel_settle_window:
            acc_applied, acc_innov = self._ekf.update_accel(acc)
        else:
            acc_applied, acc_innov = False, 0.0

        if self._data.get('zupt_enabled', True):
            zupt_applied, zupt_innov = self._ekf.update_zupt(gyro, acc)
        else:
            zupt_applied, zupt_innov = False, 0.0

        # ── Vision position / velocity / yaw correction ───────────────────────
        # Suppress vision EKF updates in GT mode: ekf.x is overwritten by GT anyway,
        # so update_velocity/update_yaw only shrink P without improving x, leaving P
        # inconsistently small when GT is later disabled.
        # Also suppress until yaw has actually settled: during WAIT the gyro bias
        # hasn't been calibrated yet (confirmed: psi wandering 140-148° during WAIT
        # vs. the correct ~180°), and yaw has no fast-converging correction post-reset
        # (no magnetometer; gyro bias for yaw is frozen) so it settles more slowly
        # than roll/pitch — confirmed directly: at t=0.55s post-reset (just past the
        # old shared 0.5s window) psi was still 15° off target, producing a ~9m PnP
        # position error that the EKF absorbed *confidently* (sig_pE shrank while
        # trusting the bad update), a lasting bias the carrot tracker then chased
        # with sustained wrong-direction roll. hover_reset_done alone flips True
        # immediately at reset and doesn't cover that settling tail, so gate on
        # elapsed time since reset instead, via its own vision_settle_sec (longer
        # than accel_settle_sec — see that param's note for why they're separate).
        # ekf_vision_gate is a second, adaptive backstop against whatever residual
        # error remains after this window. Matches the velocity-heading yaw update
        # and gate-pass position fix elsewhere in this file, which use
        # hover_reset_done directly because they have no such settling tail.
        vis = self._data.pop('_vision_ekf_update', None)
        _vision_settled = (
            self._hover_reset_done
            and self._hover_reset_t is not None
            and now - self._hover_reset_t
                > self._param.get('vision_settle_sec', 1.50)
        )
        if (vis is not None and _vision_settled
                and not self._param.get('ground_truth_mode', False)):
            # Single overall trust knob: divides every vision-update sigma
            # below before it reaches the Kalman update, without touching any
            # gate (gates decide what's an outlier; this only affects how much
            # a within-gate fix moves the state). 1.0 = today's calibrated
            # behaviour. See params.yaml's vision_authority comment.
            _vision_authority = float(np.clip(
                self._param.get('vision_authority', 1.0), 0.05, 1.0))
            if vis.get('pos_ned') is not None:
                # PnP is world-frame; EKF is local-frame (zeroed at hover entry).
                local_pos = np.asarray(vis['pos_ned']) - self._pos_offset_ned
                # Item 4: latency compensation — shift measurement forward by dead-reckoning.
                # The image was captured vision_latency_s ago; the EKF has since moved.
                # Adjusting local_pos by (current_pos - historical_pos) removes this offset.
                _vis_t = vis.get('wall_t')
                _lat   = float(self._param.get('vision_latency_s', 0.0))
                if _vis_t is not None and _lat > 0.0 and len(self._state_buf) >= 2:
                    _t_img = _vis_t - _lat
                    _pos_hist = self._state_buf[0][1]   # fallback: oldest entry
                    for _bt, _bp in self._state_buf:
                        if _bt >= _t_img:
                            _pos_hist = _bp
                            break
                    local_pos = local_pos + (self._ekf.x[0:3] - _pos_hist)
                self._ekf.update_position(
                    local_pos, sigma_pos=vis['sigma_pos'] / _vision_authority, gate_dist=vis['gate'],
                    t=_vis_t, max_speed=self._param.get('ekf_vision_pos_max_speed', 15.0))
            if vis.get('vel_ned') is not None:
                self._ekf.update_velocity(
                    vis['vel_ned'], sigma_vel=vis['sigma_vel'] / _vision_authority,
                    gate_dist=vis['vel_gate'])
            if vis.get('yaw_ned') is not None:
                # Confidence ramp: vision_settle_sec is a hard on/off cliff, so
                # whatever the first trusted yaw sample happens to be (or the
                # circular-mean of the first few, in vision_rx.py) gets applied at
                # full nominal confidence (sigma_yaw=0.1 rad) immediately. Confirmed
                # in a flight log: hover_reset_t=3.71s, vision_settle_sec=1.5s ->
                # settle at 5.21s; psi jumped ~9.5deg at t=5.28s (the very next
                # update), which the tight sigma let through and the controller
                # then chased into a real, growing roll/E-position oscillation.
                # This is a smaller instance of the same class of bug fixed
                # earlier this session (LK-bridge, PnP dual-solution flips, frame
                # duplication, single-frame yaw outliers) — smoothing/hysteresis
                # fixes there only catch transient or single-frame errors, not
                # ordinary PnP scatter that happens to land in the first update.
                # Ramp sigma_yaw down from a heavily-distrusted initial value to
                # the nominal one over vision_yaw_confidence_ramp_sec so the first
                # updates after the cliff can only nudge the estimate, not snap it.
                _yaw_ramp_sec  = float(self._param.get('vision_yaw_confidence_ramp_sec', 1.0))
                _sigma_yaw_max = float(self._param.get('vision_yaw_sigma_initial', 0.5))
                _t_since_settle = (now - self._hover_reset_t
                                    - self._param.get('vision_settle_sec', 1.50))
                _ramp_frac = float(np.clip(_t_since_settle / max(_yaw_ramp_sec, 1e-6),
                                            0.0, 1.0))
                _sigma_yaw_base = vis['sigma_yaw'] / _vision_authority
                _sigma_yaw_eff = _sigma_yaw_max + (_sigma_yaw_base - _sigma_yaw_max) * _ramp_frac
                self._ekf.update_yaw(
                    vis['yaw_ned'], sigma_yaw=_sigma_yaw_eff,
                    gate_dist=vis.get('yaw_gate', 1.0))
            # RE-ENABLED: was DISABLED 2026-07-29 after a real crash into gate
            # 1's bottom beam — the gate's 4 corners are coplanar, and
            # solvePnP on a near-planar target has a well-known pose ambiguity
            # (two rotations reproject almost equally well) that gets worse
            # the more head-on the view, i.e. exactly during final approach.
            # roll_ned swung across the full [-pi, +pi] range (std=2.5 rad)
            # tick-to-tick while GT roll sat near 0; the fixed gate_dist=5.0
            # didn't reject it since each self-consistent-but-wrong rotation
            # only produced a moderate gravity-vector innovation, dragging EKF
            # roll to a ~24 deg error within ~1s and diverging into the beam.
            # Since then vision_rx.py's _pnp_gate gained a position-consistency
            # tie-break (an independent anchor a single bad frame can't have
            # already poisoned) that roll/pitch/yaw all share via the same
            # disambiguated rotation. Before flying this live with
            # ground_truth_mode: false, re-validate offline against logged
            # flights (ideally the original crash log) via
            # `ekf_shadow.py --use-attitude` — see its docstring — and use the
            # fitted ekf_vision_att_sigma/ekf_vision_att_gate rather than
            # flying on unvalidated defaults.
            if vis.get('roll_ned') is not None and vis.get('pitch_ned') is not None:
                self._ekf.update_attitude(
                    vis['roll_ned'], vis['pitch_ned'],
                    sigma_att=self._param.get('ekf_vision_att_sigma', 0.15) / _vision_authority,
                    gate_dist=self._param.get('ekf_vision_att_gate', 5.0))

        # ── Item 1: Velocity-heading yaw update ───────────────────────────────
        # At high horizontal speed, velocity direction ≈ nose heading.
        # This gives a continuous weak yaw constraint between gate sightings.
        if (self._hover_reset_done
                and not self._param.get('ground_truth_mode', False)):
            _v_ne  = self._ekf.x[3:5]
            _v_mag = float(np.linalg.norm(_v_ne))
            if _v_mag > float(self._param.get('vel_heading_min_mps', 4.0)):
                _psi_vel = float(np.arctan2(_v_ne[1], _v_ne[0]))
                self._ekf.update_yaw(
                    _psi_vel,
                    sigma_yaw=float(self._param.get('sigma_vel_heading', 0.5)),
                    gate_dist=float(self._param.get('gate_vel_heading', 1.0)))

        # ── Item 2: Gate-pass position fix ────────────────────────────────────
        # On COLLISION the controller writes 'last_gate_id' to shared data.
        # The gate world-NED position is a known landmark → apply as an EKF fix.
        if (self._hover_reset_done
                and not self._param.get('ground_truth_mode', False)):
            _gid = self._data.get('last_gate_id')
            if _gid is not None and _gid != self._last_gate_id_processed:
                _gates_ned = self._data.get('track_gates_ned', {})
                if _gid in _gates_ned:
                    _gate_world = np.asarray(_gates_ned[_gid], dtype=float)
                    _gate_local = (self._R_align @ (_gate_world - self._sim_ned_offset)
                                   - self._pos_offset_ned)
                    _sigma_gp = float(self._param.get('sigma_gate_pass_pos', 0.5))
                    self._ekf.update_position(
                        _gate_local, sigma_pos=_sigma_gp, gate_dist=5.0)
                    print(f'[IMUEKFHandler] gate-pass fix gate={_gid} '
                          f'local_pos={_gate_local}', flush=True)
                self._last_gate_id_processed = _gid

        # ── EKF log ───────────────────────────────────────────────────────────
        if self._logger is not None:
            # Diagnostic snapshot for ekf_shadow.py's offline replay: raw
            # (unrotated) GT and thrust, logged every tick regardless of
            # ground_truth_mode/use_model_predict/model_predict_shadow so a
            # normal flight always has enough data for an offline EKF replay.
            # Deliberately duplicates a few lines from the use_model_predict/
            # model_predict_shadow block above rather than reusing it, so this
            # addition can't affect that block's gating or the live shadow log.
            _latest_diag = self._data.get('mavlink', {}).get('latest', {})
            _att_diag = _latest_diag.get('ATTITUDE')
            _lpn_diag = _latest_diag.get('LOCAL_POSITION_NED')
            if _att_diag is not None and _lpn_diag is not None:
                _gt_pos_diag  = np.array(_lpn_diag['pos_ned_m'], dtype=float)
                _gt_vel_diag  = np.array(_lpn_diag['vel_ned_mps'], dtype=float)
                _gt_quat_diag = np.array(_att_diag['quat'], dtype=float)
            else:
                _gt_pos_diag  = np.full(3, np.nan)
                _gt_vel_diag  = np.full(3, np.nan)
                _gt_quat_diag = np.full(4, np.nan)

            _act_diag = _latest_diag.get('ACTUATOR_OUTPUT_STATUS')
            if _act_diag is not None:
                _act_sum_diag = float(np.sum(_act_diag['actuator'][:4]))
                _T_total_diag = _act_sum_diag * float(self._param.get('T_max_motor', 49.9))
            else:
                _act_sum_diag = float('nan')
                _T_total_diag = float('nan')

            try:
                self._logger.log_ekf(
                    t_us, now,
                    self._ekf.x.copy(), np.diag(self._ekf.P).copy(),
                    acc_applied, acc_innov,
                    zupt_applied, zupt_innov,
                    gyro, acc,
                    gt_pos=_gt_pos_diag, gt_vel=_gt_vel_diag, gt_quat=_gt_quat_diag,
                    T_total_N=_T_total_diag, actuator_sum=_act_sum_diag,
                    hover_reset_done=self._hover_reset_done,
                    wait_phase_done=bool(self._data.get('wait_phase_done', False)),
                )
            except Exception:
                pass

        # ── EKF-vs-GT error shadow log ──────────────────────────────────────────
        # Independent of the EKF-log block above (fetches its own raw GT read)
        # so it works even when self._logger is None — that block's
        # _att_diag/_lpn_diag/_gt_*_diag locals are scoped inside `if
        # self._logger is not None:` and aren't available out here.
        if self._gt_err_shadow_enabled and self._hover_reset_done and self._gt_err_shadow_writer is not None:
            _latest_ge = self._data.get('mavlink', {}).get('latest', {})
            _att_ge = _latest_ge.get('ATTITUDE')
            _lpn_ge = _latest_ge.get('LOCAL_POSITION_NED')
        else:
            _att_ge = _lpn_ge = None
        if _att_ge is not None and _lpn_ge is not None:
            self._gt_err_shadow_tick += 1
            if self._gt_err_shadow_tick % 10 == 0:
                # Align raw GT into the EKF's own hover-reset-zeroed frame —
                # same transform as the ground-truth-override block below.
                # Comparing self._ekf.x against RAW (unaligned) GT would show
                # a frame offset baked in at hover-reset, not real estimation
                # error.
                _gt_pos_ge  = np.array(_lpn_ge['pos_ned_m'], dtype=float)
                _gt_vel_ge  = np.array(_lpn_ge['vel_ned_mps'], dtype=float)
                _gt_quat_ge = np.array(_att_ge['quat'], dtype=float)
                gt_pos = self._R_align @ (_gt_pos_ge - self._sim_ned_offset)
                gt_vel = self._R_align @ _gt_vel_ge
                gt_quat = _quat_mult(_gt_quat_ge, self._q_align)
                _qn = float(np.linalg.norm(gt_quat))
                if _qn > 1e-9:
                    gt_quat = gt_quat / _qn

                pos = self._ekf.x[0:3]
                vel = self._ekf.x[3:6]
                quat = self._ekf.x[6:10]
                pos_err = pos - gt_pos
                vel_err = vel - gt_vel
                roll, pitch, yaw = _quat_to_euler_deg(quat)
                roll_gt, pitch_gt, yaw_gt = _quat_to_euler_deg(gt_quat)
                roll_err  = (roll  - roll_gt  + 180) % 360 - 180
                pitch_err = (pitch - pitch_gt + 180) % 360 - 180
                yaw_err   = (yaw   - yaw_gt   + 180) % 360 - 180

                self._gt_err_shadow_writer.writerow([
                    f'{now:.4f}',
                    f'{pos[0]:.4f}', f'{pos[1]:.4f}', f'{pos[2]:.4f}',
                    f'{gt_pos[0]:.4f}', f'{gt_pos[1]:.4f}', f'{gt_pos[2]:.4f}',
                    f'{pos_err[0]:.4f}', f'{pos_err[1]:.4f}', f'{pos_err[2]:.4f}',
                    f'{float(np.linalg.norm(pos_err)):.4f}',
                    f'{vel[0]:.4f}', f'{vel[1]:.4f}', f'{vel[2]:.4f}',
                    f'{gt_vel[0]:.4f}', f'{gt_vel[1]:.4f}', f'{gt_vel[2]:.4f}',
                    f'{vel_err[0]:.4f}', f'{vel_err[1]:.4f}', f'{vel_err[2]:.4f}',
                    f'{float(np.linalg.norm(vel_err)):.4f}',
                    f'{roll:.3f}', f'{pitch:.3f}', f'{yaw:.3f}',
                    f'{roll_gt:.3f}', f'{pitch_gt:.3f}', f'{yaw_gt:.3f}',
                    f'{roll_err:.3f}', f'{pitch_err:.3f}', f'{yaw_err:.3f}',
                ])

        # ── Publish EKF state to shared dict ──────────────────────────────────
        self._data['mav_state'] = {
            'pos_ned':  self._ekf.pos_ned,
            'vel_body': np.zeros(3),
            'vel_ned':  self._ekf.vel_ned,
            'quat':     self._ekf.quat,
            'rates':    gyro,
            'wall_t':   now,
            'sim_t_us': t_us,
        }

        # ── Ground truth override ──────────────────────────────────────────────
        # Replace pos, vel, attitude, and rates with sim ground truth when enabled.
        # Injects into EKF state too so model-aided predict stays consistent.
        if self._param.get('ground_truth_mode', False):
            _latest = self._data.get('mavlink', {}).get('latest', {})
            _lpn = _latest.get('LOCAL_POSITION_NED')
            _att = _latest.get('ATTITUDE')
            if _lpn is not None:
                _gt_pos = self._R_align @ (
                    np.array(_lpn['pos_ned_m'], dtype=float) - self._sim_ned_offset)
                _gt_vel = self._R_align @ np.array(_lpn['vel_ned_mps'], dtype=float)
                self._ekf.x[0:3] = _gt_pos
                self._ekf.x[3:6] = _gt_vel
                self._data['mav_state']['pos_ned'] = _gt_pos
                self._data['mav_state']['vel_ned'] = _gt_vel
            if _att is not None:
                # Rotate attitude into EKF frame: q' = q_att ⊗ q_align
                # Passive convention: q1⊗q2 ↔ R1@R2; we need R_att @ R_align^T.
                # q_align = [cos(Δψ/2), 0, 0, sin(Δψ/2)] ↔ R_align^T via _rot_from_quat,
                # so q_att ⊗ q_align ↔ R_att @ R_align^T = R_{N'→B}. ✓
                # Body rates are in body frame — unaffected by NED frame rotation.
                _gt_quat = _quat_mult(_att['quat'], self._q_align)
                _qn = float(np.linalg.norm(_gt_quat))
                if _qn > 1e-9:
                    _gt_quat /= _qn
                _gt_rates = np.array([_att['rollspeed'], _att['pitchspeed'],
                                      _att['yawspeed']], dtype=float)
                self._ekf.x[6:10] = _gt_quat
                self._data['mav_state']['quat']  = _gt_quat
                self._data['mav_state']['rates'] = _gt_rates

    # ------------------------------------------------------------------
    def save(self):
        """Close shadow CSVs and generate comparison plot PNGs."""
        self._save_model_predict_shadow()
        self._save_gt_error_shadow()

    def _save_model_predict_shadow(self):
        if self._mp_shadow_file is None:
            return
        # Null out writer first — on_imu_msg checks writer is not None before writing,
        # so after this assignment the MAVLink thread will skip any further writes.
        _writer = self._mp_shadow_writer
        _file   = self._mp_shadow_file
        self._mp_shadow_writer = None
        self._mp_shadow_file   = None
        _file.flush()
        _file.close()

        if self._mp_shadow_path is None:
            return

        import csv, os
        import numpy as np

        rows = []
        try:
            with open(self._mp_shadow_path, newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append({k: float(v) for k, v in row.items()})
        except Exception as e:
            print(f'[IMUEKFHandler] shadow plot: could not read CSV — {e}', flush=True)
            return

        if len(rows) < 2:
            print('[IMUEKFHandler] shadow plot: not enough rows to plot', flush=True)
            return

        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            print('[IMUEKFHandler] shadow plot: matplotlib not available', flush=True)
            return

        t   = np.array([r['t_wall_s']    for r in rows])
        t  -= t[0]

        acc_imu_x   = np.array([r['acc_imu_x']   for r in rows])
        acc_imu_y   = np.array([r['acc_imu_y']   for r in rows])
        acc_imu_z   = np.array([r['acc_imu_z']   for r in rows])
        acc_mdl_x   = np.array([r['acc_model_x'] for r in rows])
        acc_mdl_y   = np.array([r['acc_model_y'] for r in rows])
        acc_mdl_z   = np.array([r['acc_model_z'] for r in rows])
        err_norm    = np.array([r['err_norm']     for r in rows])
        T_total     = np.array([r['T_total_N']    for r in rows])

        fig, axes = plt.subplots(6, 1, figsize=(12, 16), sharex=True)
        fig.suptitle('Model-predict shadow: IMU vs model acceleration', fontsize=13)

        for ax, imu, mdl, label in [
            (axes[0], acc_imu_x, acc_mdl_x, 'acc_x (body fwd)  [m/s²]'),
            (axes[1], acc_imu_y, acc_mdl_y, 'acc_y (body right) [m/s²]'),
            (axes[2], acc_imu_z, acc_mdl_z, 'acc_z (body down)  [m/s²]'),
        ]:
            ax.plot(t, imu, label='IMU',   lw=1.0, alpha=0.8)
            ax.plot(t, mdl, label='model', lw=1.0, alpha=0.8, linestyle='--')
            ax.set_ylabel(label, fontsize=9)
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, lw=0.4)

        axes[3].plot(t, err_norm, color='tab:red', lw=1.0)
        axes[3].set_ylabel('|err| [m/s²]', fontsize=9)
        axes[3].grid(True, lw=0.4)

        axes[4].plot(t, T_total, color='tab:green', lw=1.0)
        axes[4].set_ylabel('T_total [N]', fontsize=9)
        axes[4].grid(True, lw=0.4)

        vel_N = np.array([r['vel_N'] for r in rows])
        vel_E = np.array([r['vel_E'] for r in rows])
        vel_D = np.array([r['vel_D'] for r in rows])
        has_vmodel = 'v_model_N' in rows[0]
        axes[5].plot(t, vel_N, color='tab:blue',  lw=1.0, label='GT vN')
        axes[5].plot(t, vel_E, color='tab:orange',lw=1.0, label='GT vE')
        axes[5].plot(t, vel_D, color='tab:green', lw=1.0, label='GT vD')
        if has_vmodel:
            vm_N = np.array([r['v_model_N'] for r in rows])
            vm_E = np.array([r['v_model_E'] for r in rows])
            vm_D = np.array([r['v_model_D'] for r in rows])
            axes[5].plot(t, vm_N, color='tab:blue',  lw=1.0, ls='--', alpha=0.7, label='model vN')
            axes[5].plot(t, vm_E, color='tab:orange',lw=1.0, ls='--', alpha=0.7, label='model vE')
            axes[5].plot(t, vm_D, color='tab:green', lw=1.0, ls='--', alpha=0.7, label='model vD')
        axes[5].set_ylabel('vel NED [m/s]', fontsize=9)
        axes[5].set_xlabel('time [s]', fontsize=9)
        axes[5].legend(fontsize=8, loc='upper right', ncol=2)
        axes[5].grid(True, lw=0.4)

        fig.tight_layout()
        png_path = os.path.splitext(self._mp_shadow_path)[0] + '.png'
        fig.savefig(png_path, dpi=120)
        plt.close(fig)
        print(f'[IMUEKFHandler] shadow plot saved → {png_path}', flush=True)

    def _save_gt_error_shadow(self):
        if self._gt_err_shadow_file is None:
            return
        _writer = self._gt_err_shadow_writer
        _file   = self._gt_err_shadow_file
        self._gt_err_shadow_writer = None
        self._gt_err_shadow_file   = None
        _file.flush()
        _file.close()

        if self._gt_err_shadow_path is None:
            return

        import csv, os
        import numpy as np

        rows = []
        try:
            with open(self._gt_err_shadow_path, newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append({k: float(v) for k, v in row.items()})
        except Exception as e:
            print(f'[IMUEKFHandler] gt_error plot: could not read CSV — {e}', flush=True)
            return

        if len(rows) < 2:
            print('[IMUEKFHandler] gt_error plot: not enough rows to plot', flush=True)
            return

        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            print('[IMUEKFHandler] gt_error plot: matplotlib not available', flush=True)
            return

        t  = np.array([r['t_wall_s'] for r in rows])
        t -= t[0]

        pos_err_N = np.array([r['pos_err_N'] for r in rows])
        pos_err_E = np.array([r['pos_err_E'] for r in rows])
        pos_err_D = np.array([r['pos_err_D'] for r in rows])
        pos_err_norm = np.array([r['pos_err_norm'] for r in rows])
        vel_err_N = np.array([r['vel_err_N'] for r in rows])
        vel_err_E = np.array([r['vel_err_E'] for r in rows])
        vel_err_D = np.array([r['vel_err_D'] for r in rows])
        vel_err_norm = np.array([r['vel_err_norm'] for r in rows])
        roll_err  = np.array([r['roll_err_deg']  for r in rows])
        pitch_err = np.array([r['pitch_err_deg'] for r in rows])
        yaw_err   = np.array([r['yaw_err_deg']   for r in rows])

        fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
        fig.suptitle('EKF vs ground truth error (live sensor-mode estimate)', fontsize=13)

        axes[0].plot(t, pos_err_N, label='N', lw=0.8, alpha=0.8)
        axes[0].plot(t, pos_err_E, label='E', lw=0.8, alpha=0.8)
        axes[0].plot(t, pos_err_D, label='D', lw=0.8, alpha=0.8)
        axes[0].plot(t, pos_err_norm, label='|err|', color='k', lw=1.2)
        axes[0].set_ylabel('pos err [m]', fontsize=9)
        axes[0].legend(fontsize=8, loc='upper left', ncol=4)
        axes[0].grid(True, lw=0.4)

        axes[1].plot(t, vel_err_N, label='N', lw=0.8, alpha=0.8)
        axes[1].plot(t, vel_err_E, label='E', lw=0.8, alpha=0.8)
        axes[1].plot(t, vel_err_D, label='D', lw=0.8, alpha=0.8)
        axes[1].plot(t, vel_err_norm, label='|err|', color='k', lw=1.2)
        axes[1].set_ylabel('vel err [m/s]', fontsize=9)
        axes[1].legend(fontsize=8, loc='upper left', ncol=4)
        axes[1].grid(True, lw=0.4)

        axes[2].plot(t, roll_err,  label='roll',  lw=0.8, alpha=0.8)
        axes[2].plot(t, pitch_err, label='pitch', lw=0.8, alpha=0.8)
        axes[2].set_ylabel('roll/pitch err [deg]', fontsize=9)
        axes[2].legend(fontsize=8, loc='upper left', ncol=2)
        axes[2].grid(True, lw=0.4)

        axes[3].plot(t, yaw_err, color='tab:red', lw=1.0)
        axes[3].set_ylabel('yaw err [deg]', fontsize=9)
        axes[3].set_xlabel('time [s]', fontsize=9)
        axes[3].grid(True, lw=0.4)

        fig.tight_layout()
        png_path = os.path.splitext(self._gt_err_shadow_path)[0] + '.png'
        fig.savefig(png_path, dpi=120)
        plt.close(fig)
        print(f'[IMUEKFHandler] gt_error plot saved → {png_path}', flush=True)
