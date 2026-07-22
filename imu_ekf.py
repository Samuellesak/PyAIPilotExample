"""
imu_ekf.py
==========
IMU-driven EKF: gyro/acc calibration, hover-entry zeroing, model-aided predict,
attitude freeze, post-blip re-init, and vision corrections.

Extracted from the monolithic mavlink_rx.py so mavlink_rx stays a clean
MAVLink transport layer with no physics knowledge.

Usage
-----
    from dyn import load_params
    from mavlink_rx import MAVLinkRX
    from imu_ekf import IMUEKFHandler

    param  = load_params()
    shared = {}
    rx     = MAVLinkRX.create_mavlink_rx(conn, shared, logger=logger)
    ekf_h  = IMUEKFHandler(shared, param, logger=logger)
    ekf_h.register(rx)          # runs inline in the MAVLinkRX thread at 250 Hz

Shared-data keys written
------------------------
    shared['mav_state']                 — EKF pos / vel / quat / rates
    shared['pos_offset_ned']            — world-NED position at hover-entry reset
    shared['post_blip_att_reset_done']  — True after backup attitude re-init fires
    shared['imu_raw']                   — raw IMU sample (legacy key, read by scripts)
    shared['_imu_rates']                — FRD-corrected gyro rates (legacy key)
"""

import time

import numpy as np

from ekf import QuadEKF
from dyn import _quat_to_R as _dyn_quat_to_R


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


class IMUEKFHandler:
    """
    Drives a QuadEKF from HIGHRES_IMU messages forwarded by MAVLinkRX.

    Runs in the MAVLinkRX receive thread via a callback — no extra thread.
    """

    def __init__(self, shared_data, param, logger=None):
        self._data   = shared_data
        self._param  = param
        self._logger = logger

        self._ekf        = QuadEKF()
        self._last_imu_t = None

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
        self._blip_frozen_q            = None
        self._post_blip_att_reset_done = False

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
                't_wall_s', 'T_total_N',
                'vb_x', 'vb_y', 'vb_z',
                'acc_imu_x', 'acc_imu_y', 'acc_imu_z',
                'acc_model_x', 'acc_model_y', 'acc_model_z',
                'err_x', 'err_y', 'err_z', 'err_norm',
                'vel_N', 'vel_E', 'vel_D',
            ])
            print(f'[IMUEKFHandler] model_predict_shadow → {self._mp_shadow_path}', flush=True)

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
            self._blip_frozen_q          = None
            self._post_blip_att_reset_done = False

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

            # Derive attitude from static acc average (prevents ~15° pitch spike at blip).
            if self._wait_acc_cnt > 50:
                _acc_avg = self._wait_acc_sum / self._wait_acc_cnt
                _g       = 9.81
                _theta = float(np.arcsin(np.clip(_acc_avg[0] / _g, -1.0, 1.0)))
                _phi   = float(np.arcsin(
                    np.clip(_acc_avg[1] / (_g * max(np.cos(_theta), 0.1)), -1.0, 1.0)))
                self._ekf.set_attitude(_phi, _theta, self._init_yaw_rad)
                self._blip_frozen_q = self._ekf.x[6:10].copy()
                print(f'[IMUEKFHandler] attitude from static acc: '
                      f'phi={np.rad2deg(_phi):.1f}°  theta={np.rad2deg(_theta):.1f}°  '
                      f'psi={np.rad2deg(self._init_yaw_rad):.1f}°  '
                      f'(n={self._wait_acc_cnt})', flush=True)
            else:
                self._ekf.set_yaw(self._init_yaw_rad)
                self._blip_frozen_q = self._ekf.x[6:10].copy()

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
                _cd, _sd   = np.cos(_delta_psi), np.sin(_delta_psi)
                self._R_align = np.array([[_cd, -_sd, 0.0],
                                          [_sd,  _cd, 0.0],
                                          [0.0,  0.0, 1.0]])
                self._q_align = np.array([np.cos(_delta_psi / 2), 0.0, 0.0,
                                          np.sin(_delta_psi / 2)])
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

        # ── Sim gyro sign correction (FRD convention) ─────────────────────────
        # Sim reports all three axes sign-flipped vs FRD: negate all three.
        gyro = np.array([-gx, -gy, -gz])
        acc  = np.array([ax, ay, az])

        # ── WAIT-phase accumulation ───────────────────────────────────────────
        if not self._hover_reset_done:
            self._wait_gyro_sum += gyro
            self._wait_gyro_cnt += 1
            self._wait_acc_sum  += acc
            self._wait_acc_cnt  += 1
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
        dt  = float(np.clip(dt, 0.0005, 0.05))
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
                    _R_nb      = _dyn_quat_to_R(self._ekf.x[6:10])
                    _vel_b     = _R_nb @ self._ekf.x[3:6]
                    _F_drag    = -self._param['Dv'] @ (np.abs(_vel_b) * _vel_b)
                    acc_model  = ((np.array([0.0, 0.0, -_T_total]) + _F_drag)
                                  / float(self._param['m']))
                    # Shadow log: compare model vs raw IMU (every 10th tick ≈ 25 Hz)
                    if _use_shd and self._hover_reset_done and self._mp_shadow_writer is not None:
                        self._mp_shadow_tick += 1
                        if self._mp_shadow_tick % 10 == 0:
                            _err = acc_model - acc
                            self._mp_shadow_writer.writerow([
                                f'{now:.4f}',
                                f'{_T_total:.2f}',
                                f'{_vel_b[0]:.3f}', f'{_vel_b[1]:.3f}', f'{_vel_b[2]:.3f}',
                                f'{acc[0]:.4f}', f'{acc[1]:.4f}', f'{acc[2]:.4f}',
                                f'{acc_model[0]:.4f}', f'{acc_model[1]:.4f}', f'{acc_model[2]:.4f}',
                                f'{_err[0]:.4f}', f'{_err[1]:.4f}', f'{_err[2]:.4f}',
                                f'{float(np.linalg.norm(_err)):.4f}',
                                f'{self._ekf.x[3]:.3f}', f'{self._ekf.x[4]:.3f}', f'{self._ekf.x[5]:.3f}',
                            ])

        acc_predict = acc_model if (_use_mp and acc_model is not None) else acc
        self._ekf.predict(gyro, acc_predict, dt)
        _acc_norm_upd = float(np.linalg.norm(acc))  # raw IMU for update gate

        # ── Attitude freeze (blip + motor-lag transient) ──────────────────────
        # Pin the quaternion to the hover-reset value for blip_dur + 0.30 s so
        # the motor-ramp acc surge cannot corrupt the EKF attitude.
        _freeze_active = (
            self._blip_frozen_q is not None
            and self._hover_reset_t is not None
            and now - self._hover_reset_t
                < self._param.get('blip_dur_sec', 0.15) + 0.30
        )
        if _freeze_active:
            self._ekf.x[6:10] = self._blip_frozen_q
            _qn = float(np.linalg.norm(self._ekf.x[6:10]))
            if _qn > 1e-9:
                self._ekf.x[6:10] /= _qn

        # ── Accelerometer attitude update ─────────────────────────────────────
        _blip_window = (
            not self._hover_reset_done
            or (self._hover_reset_t is not None
                and now - self._hover_reset_t
                    < self._param.get('blip_dur_sec', 0.15)
                      + self._param.get('blip_acc_extra_sec', 0.50))
        )
        if not _freeze_active and abs(_acc_norm_upd - 9.81) < 2.0 and _blip_window:
            acc_applied, acc_innov = self._ekf.update_accel(acc)
        else:
            acc_applied, acc_innov = False, 0.0

        if self._data.get('zupt_enabled', True):
            zupt_applied, zupt_innov = self._ekf.update_zupt(gyro, acc)
        else:
            zupt_applied, zupt_innov = False, 0.0

        # ── Backup post-blip attitude re-init ─────────────────────────────────
        # One-shot re-init fires immediately after the freeze window ends.
        # Sets attitude from acc=[0,0,-g] → ~0° — a no-op when drone is level,
        # small correction otherwise.
        if (self._hover_reset_done
                and not self._post_blip_att_reset_done
                and not _freeze_active
                and self._hover_reset_t is not None
                and now - self._hover_reset_t
                    > self._param.get('blip_dur_sec', 0.15) + 0.30):
            if abs(_acc_norm_upd - 9.81) < 1.5:
                _g      = 9.81
                _ax_pb  = float(acc[0])
                _ay_pb  = float(acc[1])
                _th_pb  = float(np.arcsin(np.clip(_ax_pb / _g, -1.0, 1.0)))
                _ph_pb  = float(np.arcsin(
                    np.clip(_ay_pb / (_g * max(np.cos(_th_pb), 0.1)), -1.0, 1.0)))
                _qw, _qx, _qy, _qz = self._ekf.x[6:10]
                _psi_pb = float(np.arctan2(
                    2.0 * (_qw * _qz + _qx * _qy),
                    1.0 - 2.0 * (_qy * _qy + _qz * _qz)))
                self._ekf.set_attitude(_ph_pb, _th_pb, _psi_pb)
                print(f'[IMUEKFHandler] post-blip att reset: '
                      f'phi={np.rad2deg(_ph_pb):.1f}°  '
                      f'theta={np.rad2deg(_th_pb):.1f}°  '
                      f'acc_norm={_acc_norm_upd:.3f}  '
                      f'dt={now - self._hover_reset_t:.3f}s', flush=True)
                self._post_blip_att_reset_done = True
                self._data['post_blip_att_reset_done'] = True

        # ── Vision position / velocity / yaw correction ───────────────────────
        vis = self._data.pop('_vision_ekf_update', None)
        if vis is not None:
            if vis.get('pos_ned') is not None:
                # PnP is world-frame; EKF is local-frame (zeroed at hover entry).
                local_pos = np.asarray(vis['pos_ned']) - self._pos_offset_ned
                self._ekf.update_position(
                    local_pos, sigma_pos=vis['sigma_pos'], gate_dist=vis['gate'])
            if vis.get('vel_ned') is not None:
                self._ekf.update_velocity(
                    vis['vel_ned'], sigma_vel=vis['sigma_vel'],
                    gate_dist=vis['vel_gate'])
            if vis.get('yaw_ned') is not None:
                self._ekf.update_yaw(
                    vis['yaw_ned'], sigma_yaw=vis['sigma_yaw'],
                    gate_dist=vis.get('yaw_gate', 1.0))

        # ── EKF log ───────────────────────────────────────────────────────────
        if self._logger is not None:
            try:
                self._logger.log_ekf(
                    t_us, now,
                    self._ekf.x.copy(), np.diag(self._ekf.P).copy(),
                    acc_applied, acc_innov,
                    zupt_applied, zupt_innov,
                    gyro, acc,
                )
            except Exception:
                pass

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
        """Close shadow CSV and generate a comparison plot PNG."""
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
        axes[5].plot(t, vel_N, label='vN', lw=1.0)
        axes[5].plot(t, vel_E, label='vE', lw=1.0)
        axes[5].plot(t, vel_D, label='vD', lw=1.0)
        axes[5].set_ylabel('vel_ned [m/s]', fontsize=9)
        axes[5].set_xlabel('time [s]', fontsize=9)
        axes[5].legend(fontsize=8, loc='upper right')
        axes[5].grid(True, lw=0.4)

        fig.tight_layout()
        png_path = os.path.splitext(self._mp_shadow_path)[0] + '.png'
        fig.savefig(png_path, dpi=120)
        plt.close(fig)
        print(f'[IMUEKFHandler] shadow plot saved → {png_path}', flush=True)
