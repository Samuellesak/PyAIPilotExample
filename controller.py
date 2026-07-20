import time
import numpy as np
from pymavlink import mavutil

from carrot_tracker import CarrotTracker

MAVLINK_CMD_SIM_RESET = 31000
CONTROL_HZ            = 250
DT                    = 1.0 / CONTROL_HZ

# Max yaw slew rate [rad/s] used to prevent instantaneous step-changes in r_des.
# The r_des clamp (±2 rad/s) is the hard limit; this softer limit just shapes the
# transient.  90°/s is fast enough to track direction changes without lag.
PSI_RATE_MAX = np.deg2rad(90)

# ── Launch sequence ────────────────────────────────────────────────────
# 1. WAIT  : motors idle on slope, IMU settles, gyro/acc bias accumulated
# 2. BLIP  : high-thrust burst lifts the drone off the slope
# 3. TRACK : cascade + carrot tracker activate immediately after the blip
WAIT_PHASE_SEC   = 3.5    # seconds to sit on slope before blip fires
INTEGRATE_DELAY    = 2.0    # horizontal (N/E) integrator delay after hover entry
                            # prevents windup: drone hits 3+ m/s during slope release
INTEGRATE_DELAY_V  = 0.0    # vertical (D) integrator delay — activate immediately


def _rot_from_quat(q):
    """Rotation matrix R: v_body = R @ v_NED.  q = [qw, qx, qy, qz]."""
    qw, qx, qy, qz = q
    return np.array([
        [1-2*(qy*qy+qz*qz),   2*(qx*qy+qw*qz),   2*(qx*qz-qw*qy)],
        [  2*(qx*qy-qw*qz), 1-2*(qx*qx+qz*qz),   2*(qy*qz+qw*qx)],
        [  2*(qx*qz+qw*qy),   2*(qy*qz-qw*qx), 1-2*(qx*qx+qy*qy)],
    ])


class Controller:

    def __init__(self, sim_conn, data, system_boot_ms, param, logger=None):
        self.sim_conn         = sim_conn
        self.data             = data
        self.system_boot_ms   = system_boot_ms
        self.param            = param
        self._logger          = logger

        self.T_max  = param['T_max_motor']
        self.T_hover = param['m'] * param['g'] / 4.0

        # Carrot tracker
        self.tracker = CarrotTracker(param)

        print(
            f"[Controller] K_att={param['K_att']}  K_psi={param['K_psi']}",
            flush=True
        )

        # Outer loop gains — per-axis velocity PI
        self._Kp_vN    = float(param['Kp_vN'])
        self._Ki_vN    = float(param['Ki_vN'])
        self._Kp_vE    = float(param['Kp_vE'])
        self._Ki_vE    = float(param['Ki_vE'])
        self._Kp_vD    = float(param['Kp_vD'])
        self._Ki_vD    = float(param['Ki_vD'])
        self._K_att    = float(param['K_att'])
        self._K_psi    = float(param['K_psi'])
        self._Ki_psi   = float(param['Ki_psi'])
        self._MAX_TILT = np.deg2rad(float(param['MAX_TILT_DEG']))
        self._g        = param['g']

        self._hover_only = bool(param.get('hover_only', False))
        self._debug_waypoints_only = bool(param.get('debug_waypoints_only', False))
        if self._debug_waypoints_only:
            print("[Controller] debug_waypoints_only=true — "
                  "gate overrides and visual yaw disabled", flush=True)
        if self._hover_only:
            print("[Controller] hover_only=true — carrot tracker disabled, "
                  "drone will hold altitude indefinitely", flush=True)

        # Cascade integrators
        self.xi_vel      = np.zeros(3)              # NED velocity integrals [m]
        self.xi_psi      = 0.0                      # yaw integral [rad·s]
        self._XI_VEL_LIM = np.array([3.0, 3.0, 2.0])
        self._XI_PSI_LIM = np.pi
        self._VEL_CLAMP  = 5.0                      # m/s — integrator input clamp

        # Low-pass filters to reduce T_collective noise.
        # vD from the EKF is dead-reckoned (no baro/GPS) and carries high-frequency
        # accelerometer noise that goes straight into the vertical P term.
        # T_coll is further modulated by R22 (attitude noise). Both are filtered.
        self._vN_filt      = 0.0                    # filtered EKF vN [m/s]
        self._vE_filt      = 0.0                    # filtered EKF vE [m/s]
        self._vD_filt      = 0.0                    # filtered EKF vD [m/s]
        self._T_coll_filt  = 4.0 * self.T_hover     # filtered collective [N]
        self._TAU_VH       = float(param.get('tau_vel_h', 0.15))   # horizontal velocity filter τ [s]
        self._TAU_VD       = float(param.get('tau_vel_d', 0.15))   # vertical   velocity filter τ [s]
        self._TAU_T_COLL   = 0.30                   # T_coll output filter τ [s]

        # Rate-limited yaw command — initialised at measured yaw on first update
        self._psi_cmd = None

        # Vision-bearing hold: keep last locked tvec for brief YOLO flicker suppression.
        # Cleared when pnp_locked drops (gate truly lost) or age exceeds the limit.
        self._vis_tvec_hold     = None   # last valid locked tvec_cam (3,) or None
        self._vis_tvec_hold_age = 0      # frames since last fresh locked measurement
        self._vis_hold_frames   = int(param.get('vision_hold_frames', 5))

        # PT2 filter on the velocity reference — smooths step-changes when switching
        # between carrot and vision-bearing modes so direction changes are gradual.
        self._pt2_omega0  = float(param.get('vision_ref_omega0', 0.375))   # [rad/s]
        self._pt2_zeta    = float(param.get('vision_ref_zeta',   1.0))
        self._v_ref_pt2_x = np.zeros(3)   # filter output (position state)
        self._v_ref_pt2_v = np.zeros(3)   # filter derivative state

        # Launch-sequence bookkeeping
        self._t_start        = None   # wall-clock time of first valid state
        self._carrot_active  = False  # True after first frame in carrot mode
        self._hover_entered  = False  # True after first HOVER tick (triggers one-shot reset)
        self._hover_entry_t  = None   # wall-clock time of hover entry (for integrate delay)

        # Timing diagnostics
        self._last_loop_t = None   # wall time of previous update() call
        self._sim_t0      = None   # sim timestamp [µs] at first valid state
        self._wall_t0     = None   # wall time at first valid state

        # ── Controller type ────────────────────────────────────────────────
        self._controller_type = int(param.get('controller_type', 1))
        _cname = 'Cascade PI' if self._controller_type == 1 else 'LQI'
        print(f"[CONTROLLER] type={self._controller_type} ({_cname})", flush=True)

        # LQI-specific state (only when controller_type == 2)
        if self._controller_type == 2:
            try:
                _lqi = np.load('lqi_gains.npz', allow_pickle=True)
            except FileNotFoundError:
                raise RuntimeError(
                    "controller_type=2 (LQI) requires lqi_gains.npz — "
                    "run 'python lqi.py' first.")
            self._lqi_K     = _lqi['K']                      # (N, 4, 14)
            self._lqi_ztrim = _lqi['z_trim']                 # (N, 10)
            self._lqi_utrim = _lqi['u_trim']                 # (N,  4)
            self._lqi_C     = _lqi['C_out']                  # (N, 4, 10)
            self._lqi_vneds = np.array(_lqi['v_ned'], dtype=float)   # (N,  3)
            self._lqi_psis  = np.array(_lqi['psi_ref'], dtype=float) # (N,)
            self.xi_lqi     = np.zeros(4)          # [vN, vE, vD, psi] integrators
            self._XI_LQI_LIM = np.array([10.0, 10.0, 10.0, np.pi])
            print(f"[CONTROLLER] LQI: {self._lqi_K.shape[0]} operating points loaded",
                  flush=True)

    # ------------------------------------------------------------------
    # Main update — called at CONTROL_HZ
    # ------------------------------------------------------------------

    def update(self):
        try:
            self._update_inner()
        except Exception as exc:
            print(f"[CONTROLLER ERROR] {type(exc).__name__}: {exc}", flush=True)
            raise

    def _update_inner(self):
        # Measure actual loop period; clamp to [0.5×, 4×] DT to reject outliers.
        _now = time.time()
        if self._last_loop_t is None:
            _actual_dt = DT
        else:
            _actual_dt = np.clip(_now - self._last_loop_t, DT * 0.5, DT * 4.0)
        self._last_loop_t = _now

        state = self.data.get('mav_state')

        if state is None:
            if _now - getattr(self, '_last_nostate_t', 0.0) >= 2.0:
                self._last_nostate_t = _now
                print(f"[CONTROLLER] waiting for mav_state... data keys: {list(self.data.keys())}",
                      flush=True)
            self._send_attitude_target(0.0, 0.0, 0.0, 0.0)
            time.sleep(DT)
            return

        pos_ned  = state['pos_ned']
        quat     = state['quat']       # [qw,qx,qy,qz] NED-to-body
        rates    = state['rates']

        # State age: how old was this IMU packet when the controller read it.
        _wall_t = state.get('wall_t')
        state_age_ms = (_now - _wall_t) * 1000.0 if _wall_t is not None else float('nan')

        # Sim-speed ratio: sim seconds elapsed / wall seconds elapsed (>1 = faster than RT).
        _sim_t_us = state.get('sim_t_us')
        if _sim_t_us is not None:
            if self._sim_t0 is None:
                self._sim_t0  = _sim_t_us
                self._wall_t0 = _now
                sim_speed = 1.0
            else:
                wall_elapsed = max(_now - self._wall_t0, 1e-6)
                sim_elapsed  = (_sim_t_us - self._sim_t0) / 1e6
                sim_speed    = sim_elapsed / wall_elapsed
        else:
            sim_speed = float('nan')

        # --- first-state initialisation --------------------------------
        _, _, psi_meas = quat_to_euler(quat)
        if self._t_start is None:
            # _psi_cmd is set at hover entry (not here) so it reflects the actual
            # heading after the WAIT phase has settled, not the noisy slope reading
            # at t=0.  Use a sentinel non-None value until hover entry sets it.
            self._psi_cmd = psi_meas   # temporary — overwritten at hover entry
            self._t_start = time.time()

        t_elapsed = time.time() - self._t_start

        # ── PHASE 1: WAIT ──────────────────────────────────────────────
        if t_elapsed < WAIT_PHASE_SEC:
            if _now - getattr(self, '_last_wait_diag_t', 0.0) >= 0.5:
                self._last_wait_diag_t = _now
                _phi_w, _theta_w, _ = quat_to_euler(quat)
                phi_deg   = np.degrees(_phi_w)
                theta_deg = np.degrees(_theta_w)
                _, _qx_w, _qy_w, _ = quat
                R22_w = max(0.3, 1.0 - 2*(_qx_w*_qx_w + _qy_w*_qy_w))
                print(f"[WAIT t={t_elapsed:.2f}s]  "
                      f"roll={phi_deg:.1f}°  pitch={theta_deg:.1f}°  R22={R22_w:.3f}  "
                      f"rates=({rates[0]:.2f},{rates[1]:.2f},{rates[2]:.2f})  "
                      f"dt={_actual_dt*1000:.1f}ms", flush=True)
            self._send_attitude_target(0.0, 0.0, 0.0, 4.0 * 0.05 * self.T_max)
            time.sleep(DT)
            return

        # ── PHASE 2: HOVER (attitude hold, outer velocity loop disabled) ──
        # Carrot tracker and velocity reference are inactive.
        # Outer loop runs with v_ref = 0 → commands level flight at T_hover.
        # Yaw is held at the heading measured at hover entry; the yaw P+I loop
        # corrects any gyro-bias drift back to that fixed reference.

        # One-shot at hover entry: snapshot yaw setpoint and zero all integrators.
        # Do NOT reset the EKF — after 3 s of WAIT it has a valid slope estimate
        # and resetting to identity would cause an immediate large tilt error.
        if not self._hover_entered:
            self._hover_entered          = True
            self._hover_entry_t          = time.time()
            self._psi_cmd = np.deg2rad(float(self.param.get('initial_yaw_deg', 0.0)))
            self.xi_vel                  = np.zeros(3)
            self.xi_psi                  = 0.0
            if self._controller_type == 2:
                self.xi_lqi              = np.zeros(4)
            self._vN_filt                = 0.0
            self._vE_filt                = 0.0
            self._vD_filt                = 0.0
            self._T_coll_filt            = 4.0 * self.T_hover
            self.data['reset_vel_flag']  = True   # zero EKF velocity built up on slope
            self.data['zupt_enabled']    = False  # disable ZUPT: hovering drones have
                                                  # acc_norm ≈ g, so ZUPT can't tell
                                                  # "on ground" from "at hover thrust"
                                                  # → it silently zeroes vD in flight
            # Immediately zero vel_ned in shared state so the cascade loop does
            # not read a stale large EKF velocity before the IMU thread processes
            # reset_vel_flag (up to one 4 ms IMU period of lag).
            _s = self.data.get('mav_state')
            if _s is not None:
                _s['vel_ned'] = np.zeros(3)
            print(f"[CONTROLLER] Hover entry — psi_cmd={np.degrees(self._psi_cmd):.1f}°  "
                  f"integrators/filters/EKF-vel reset  ZUPT disabled",
                  flush=True)

        # Hover vs carrot phase.
        hover_elapsed = (time.time() - self._hover_entry_t) if self._hover_entry_t else 0.0
        in_hover = self._hover_only

        # ── Liftoff blip: high-thrust burst right at WAIT→HOVER transition ──
        # All four motors commanded equally (no rate setpoints) for pure vertical force.
        _blip_dur  = float(self.param.get('blip_dur_sec',     0.1))
        _blip_frac = float(self.param.get('blip_thrust_frac', 0.8))
        if hover_elapsed < _blip_dur:
            self._send_motors_norm(np.full(4, _blip_frac))
            time.sleep(DT)
            return

        v_ref_for_gains = np.zeros(3)
        if in_hover:
            # During hover hold, pre-rotate _psi_cmd toward the first waypoint so the
            # drone is already facing the right direction when carrot activates.
            r1 = self.tracker.waypoints[min(1, self.tracker.n_waypoints - 1)]
            to_wp = r1[:2] - pos_ned[:2]   # NE only
            if np.linalg.norm(to_wp) > 0.5:
                psi_to_wp = np.arctan2(to_wp[1], to_wp[0])
                dpsi = _wrap_pi(psi_to_wp - self._psi_cmd)
                self._psi_cmd = _wrap_pi(
                    self._psi_cmd + np.clip(dpsi,
                                            -PSI_RATE_MAX * _actual_dt,
                                             PSI_RATE_MAX * _actual_dt))
        else:
            # Carrot tracking: use EKF position to follow the waypoint path.
            if not self._carrot_active:
                self._carrot_active  = True
                # params.yaml waypoints are in WORLD NED (WP0 = launch point).
                # The EKF has been zeroed to LOCAL NED at hover entry.  Convert
                # all waypoints to local once so the tracker stays frame-consistent.
                _off = self.data.get('pos_offset_ned', np.zeros(3))
                # Only shift N and E axes.  The D (altitude) offset reflects that
                # the launch slope sits ~4-5 m above the track floor (WORLD D=0).
                # Applying the full D offset would place every waypoint 4-5 m below
                # hover, commanding a steep dive that crashes the drone.  Zeroing
                # the D offset keeps the path flat at hover altitude (LOCAL D≈0).
                _off_lateral = np.array([_off[0], _off[1], 0.0])
                self.tracker.waypoints = [np.asarray(wp, dtype=float) - _off_lateral
                                          for wp in self.tracker.waypoints]
                # Resync velocity filter to actual EKF velocity.  The filter was
                # zeroed at hover entry and never updated during the blip (early
                # return), so without this it starts at 0 and takes several seconds
                # to catch up — producing near-zero tilt commands while real velocity
                # is already building from the blip.
                _v0 = state.get('vel_ned', np.zeros(3))
                self._vN_filt = float(_v0[0])
                self._vE_filt = float(_v0[1])
                self._vD_filt = float(_v0[2])
                print(f"[CONTROLLER] Carrot activated  wp={self.tracker.wp}/"
                      f"{self.tracker.n_waypoints-1}  "
                      f"waypoints shifted lateral-only (NE offset={_off[:2]}, D zeroed)  "
                      f"vel_filt resynced to ({_v0[0]:.2f},{_v0[1]:.2f},{_v0[2]:.2f})m/s",
                      flush=True)
                self._v_ref_pt2_x = np.asarray(_v0, dtype=float).copy()
                self._v_ref_pt2_v = np.zeros(3)
            det = self.data.get('gate_detection', {})
            # Refine target waypoint to PnP-measured gate position in local NED.
            # gate_local = pos_ned (EKF local) + R_b2n @ R_cam2body @ tvec_cam
            # This uses only what the camera sees — no track data dependency.
            if not self._debug_waypoints_only:
                if det.get('detected') and det.get('tvec_cam') is not None:
                    _tvec = np.asarray(det['tvec_cam'], dtype=float)
                    # cam→body: body_x=cam_z(fwd), body_y=cam_x(right), body_z=cam_y(down)
                    _t_body = np.array([_tvec[2], _tvec[0], _tvec[1]])
                    R_bn = _rot_from_quat(quat)
                    R_nb = R_bn.T
                    self.tracker.waypoints[self.tracker.wp] = pos_ned + R_nb @ _t_body

            # Update next waypoint from vision-confirmed second-gate NED position.
            # Only applied once position is stable (median over next_gate_min_frames).
            _ng_ned = self.data.get('next_gate_ned')
            if _ng_ned is not None:
                _wp_next = self.tracker.wp + 1
                if _wp_next < self.tracker.n_waypoints:
                    self.tracker.waypoints[_wp_next] = np.asarray(_ng_ned, dtype=float)

            v_ned_ref_carrot, psi_ref_carrot = self.tracker.update(pos_ned)
            v_ref_for_gains = v_ned_ref_carrot

            # Vision-bearing override: replace NE components of v_ned_ref with a
            # direction from the PnP gate bearing — bypasses EKF position drift.
            # Uses last known tvec for up to vision_hold_frames frames so brief YOLO
            # flicker doesn't snap control back to the carrot on every missed frame.
            _fresh_tvec = (det.get('tvec_cam')
                           if (det.get('pnp_locked') and not self._debug_waypoints_only)
                           else None)

            # Publish current mode for the vision debug overlay.
            self.data['vis_ctrl_mode'] = (
                'PNP'    if _fresh_tvec is not None
                else 'HOLD'   if self._vis_tvec_hold is not None
                else 'CARROT'
            )

            if _fresh_tvec is not None:
                # Fresh locked measurement — update cache and reset age.
                self._vis_tvec_hold     = np.asarray(_fresh_tvec, dtype=float)
                self._vis_tvec_hold_age = 0
            elif det.get('pnp_locked') and self._vis_tvec_hold is not None:
                # Gate locked but YOLO flickered this frame — age the cache.
                self._vis_tvec_hold_age += 1
                if self._vis_tvec_hold_age > self._vis_hold_frames:
                    self._vis_tvec_hold = None   # held too long, yield to carrot
            else:
                # Lock dropped (gate truly lost) — discard cache immediately.
                self._vis_tvec_hold     = None
                self._vis_tvec_hold_age = 0

            if self._vis_tvec_hold is not None:
                _tvec_v  = self._vis_tvec_hold
                # cam→body: x_b(fwd)=cam_z, y_b(right)=cam_x
                _horiz_v = float(np.sqrt(_tvec_v[2]**2 + _tvec_v[0]**2))
                if _horiz_v > 1.0:   # gate at least 1 m away horizontally
                    _dir_xb = _tvec_v[2] / _horiz_v   # body-forward component
                    _dir_yb = _tvec_v[0] / _horiz_v   # body-right  component
                    _psi_v  = quat_to_euler(quat)[2]
                    _cp, _sp = np.cos(_psi_v), np.sin(_psi_v)
                    _dir_n  =  _cp * _dir_xb - _sp * _dir_yb
                    _dir_e  =  _sp * _dir_xb + _cp * _dir_yb
                    v_ref_for_gains = np.array([
                        self.tracker.v_ref * _dir_n,
                        self.tracker.v_ref * _dir_e,
                        v_ned_ref_carrot[2],
                    ])

            # Visual yaw correction from orange centroid (bearing-only).
            # When the centroid is available, blend the carrot psi toward the
            # camera bearing so the drone points its nose at the detected orange blob.
            cx_det = None if self._debug_waypoints_only else det.get('centre_px')
            if cx_det is not None:
                dx_px = float(cx_det[0]) - self.param.get('cam_cx', 320.0)
                _cur_yaw = quat_to_euler(quat)[2]
                psi_vis = _wrap_pi(_cur_yaw +
                                   np.arctan2(dx_px, self.param.get('cam_fx', 320.0)))
                dpsi_vis = _wrap_pi(psi_vis - psi_ref_carrot)
                _VIS_BLEND = 0.4   # weight of visual bearing vs carrot waypoint yaw
                psi_ref_carrot = _wrap_pi(psi_ref_carrot + _VIS_BLEND * dpsi_vis)

            # Rate-limit yaw reference
            dpsi = _wrap_pi(psi_ref_carrot - self._psi_cmd)
            self._psi_cmd = _wrap_pi(
                self._psi_cmd + np.clip(dpsi, -PSI_RATE_MAX * _actual_dt,
                                               PSI_RATE_MAX * _actual_dt))
        # PT2 filter on velocity reference — smooths step-changes from mode transitions.
        # Applied in carrot mode only; hover holds v_ref=0 unchanged.
        if not in_hover and self._carrot_active:
            _w0  = self._pt2_omega0
            _z   = self._pt2_zeta
            _err = v_ref_for_gains - self._v_ref_pt2_x
            self._v_ref_pt2_v += _actual_dt * (_w0**2 * _err - 2.0 * _z * _w0 * self._v_ref_pt2_v)
            self._v_ref_pt2_x += _actual_dt * self._v_ref_pt2_v
            v_ref_for_gains    = self._v_ref_pt2_x.copy()

        if not in_hover and self._carrot_active and self._logger is not None:
            self._logger.log_carrot(
                time_ms    = _now * 1e3,
                wp         = self.tracker.wp,
                v_cmd      = v_ref_for_gains,
                carrot_pos = self.tracker.carrot_pos,
                drone_pos  = pos_ned,
            )

        # Advance carrot waypoint when sim signals gate passage.
        if self.data.pop('gate_passed', False):
            # Always advance by at least 1 from current wp.  The old formula
            # new_wp = agi + 1 failed when tracker.wp was already agi + 1 (the
            # common case where sim reports the gate index that was just passed).
            # Additionally sync forward if sim's active_gate_index is ahead.
            agi = self.data.get('active_gate_index', -1)
            new_wp = self.tracker.wp + 1
            if agi >= 0:
                new_wp = max(new_wp, int(agi) + 1)
            new_wp = min(new_wp, self.tracker.n_waypoints - 1)
            if new_wp > self.tracker.wp:
                self.tracker.wp = new_wp
                print(f"[GATE PASSED id={self.data.get('last_gate_id')}] "
                      f"tracker.wp → {self.tracker.wp}", flush=True)

        _, _, psi_meas = quat_to_euler(quat)

        _, qx_c, qy_c, _ = quat
        R22 = max(0.3, 1.0 - 2.0 * (qx_c*qx_c + qy_c*qy_c))
        v_ned = state.get('vel_ned', np.zeros(3)).copy()

        if self._controller_type == 1:
            # ── Cascade PI ────────────────────────────────────────────────
            hover_t = hover_elapsed

            p_des, q_des, r_des, T_coll = self._outer_loop(
                v_ned_ref  = v_ref_for_gains,
                psi_ref    = self._psi_cmd,
                v_ned_meas = v_ned,
                quat       = quat,
                psi_meas   = psi_meas,
                R22        = R22,
                dt         = _actual_dt,
                hover_t    = hover_t,
            )
            if self._logger is not None:
                dbg = getattr(self, '_dbg', {})
                _phase_log = ('HOVER' if in_hover
                              else f'CARROT_wp{self.tracker.wp}' if self._carrot_active
                              else 'INIT')
                _vis_log = self.data.get('vis_ctrl_mode', '') if not in_hover else ''
                self._logger.log_cascade(
                    time_ms       = _now * 1e3,
                    v_ned_ref     = v_ref_for_gains,
                    v_ned_meas    = v_ned,
                    phi_des_deg   = dbg.get('phi_des_deg',   0.0),
                    theta_des_deg = dbg.get('theta_des_deg', 0.0),
                    phi_meas_deg  = dbg.get('phi_meas_deg',  0.0),
                    theta_meas_deg= dbg.get('theta_meas_deg',0.0),
                    psi_meas_deg  = dbg.get('psi_meas_deg',  0.0),
                    psi_ref_deg   = dbg.get('psi_ref_deg',   0.0),
                    p_des=p_des, q_des=q_des, r_des=r_des,
                    rates=rates, T_coll=T_coll, R22=R22,
                    xi_vel=self.xi_vel,
                    flight_phase=_phase_log,
                    vis_mode=_vis_log,
                )

            if _now - getattr(self, '_last_diag_t', 0.0) >= 1.0:
                self._last_diag_t = _now
                rtt  = self.data.get('timesync_rtt_ms', float('nan'))
                dbg  = getattr(self, '_dbg', {})
                intg_h = "ON" if hover_t >= INTEGRATE_DELAY   else f"OFF({INTEGRATE_DELAY-hover_t:.1f}s)"
                intg_v = "ON" if hover_t >= INTEGRATE_DELAY_V else f"OFF({INTEGRATE_DELAY_V-hover_t:.1f}s)"
                _phase = "HOVER" if in_hover else f"CARROT wp{self.tracker.wp}/{self.tracker.n_waypoints-1}"
                print(
                    f"[CASCADE/{_phase} t={t_elapsed:5.1f}s]  "
                    f"dt={_actual_dt*1000:.1f}ms  age={state_age_ms:.1f}ms  "
                    f"sim_spd={sim_speed:.3f}  rtt={rtt:.1f}ms\n"
                    f"  pos=({pos_ned[0]:.1f},{pos_ned[1]:.1f},{pos_ned[2]:.1f})m  "
                    f"v_ned=({v_ned[0]:.2f},{v_ned[1]:.2f},{v_ned[2]:.2f})m/s\n"
                    f"  att_des: phi={dbg.get('phi_des_deg',0):.1f}° theta={dbg.get('theta_des_deg',0):.1f}° "
                    f"psi_ref={dbg.get('psi_ref_deg',0):.1f}°\n"
                    f"  att_meas:phi={dbg.get('phi_meas_deg',0):.1f}° theta={dbg.get('theta_meas_deg',0):.1f}° "
                    f"psi={dbg.get('psi_meas_deg',0):.1f}°\n"
                    f"  rates=({rates[0]:.2f},{rates[1]:.2f},{rates[2]:.2f})rad/s  "
                    f"pdes=({p_des:.2f},{q_des:.2f},{r_des:.2f})  "
                    f"T_coll={T_coll:.2f}N  R22={R22:.3f}\n"
                    f"  xi_vel=({self.xi_vel[0]:.3f},{self.xi_vel[1]:.3f},{self.xi_vel[2]:.3f})  "
                    f"xi_psi={self.xi_psi:.3f}  intg=H={intg_h} V={intg_v}  "
                    f"T_norm={T_coll/(4.0*self.T_max):.3f}",
                    flush=True
                )

            self._send_attitude_target(p_des, q_des, r_des, T_coll)

        else:
            # ── LQI ──────────────────────────────────────────────────────
            u = self._lqi_step(v_ned, quat, rates, v_ref_for_gains, self._psi_cmd, _actual_dt)

            if _now - getattr(self, '_last_diag_t', 0.0) >= 1.0:
                self._last_diag_t = _now
                rtt = self.data.get('timesync_rtt_ms', float('nan'))
                idx = self._lqi_trim_idx(v_ref_for_gains, self._psi_cmd)
                print(
                    f"[LQI t={t_elapsed:5.1f}s  pt={idx}]  "
                    f"dt={_actual_dt*1000:.1f}ms  age={state_age_ms:.1f}ms  "
                    f"sim_spd={sim_speed:.3f}  rtt={rtt:.1f}ms\n"
                    f"  pos=({pos_ned[0]:.1f},{pos_ned[1]:.1f},{pos_ned[2]:.1f})m  "
                    f"v_ned=({v_ned[0]:.2f},{v_ned[1]:.2f},{v_ned[2]:.2f})m/s\n"
                    f"  xi_lqi=({self.xi_lqi[0]:.3f},{self.xi_lqi[1]:.3f},"
                    f"{self.xi_lqi[2]:.3f},{self.xi_lqi[3]:.3f})  "
                    f"u=({u[0]:.2f},{u[1]:.2f},{u[2]:.2f},{u[3]:.2f})N",
                    flush=True
                )

            self._send_motors_norm(u / self.T_max)

        time.sleep(DT)

    # ------------------------------------------------------------------
    # Outer loop: NED velocity + yaw → desired body rates + collective
    # ------------------------------------------------------------------

    def _outer_loop(self, v_ned_ref, psi_ref, v_ned_meas, quat, psi_meas, R22, dt, hover_t):
        # Low-pass filter all three EKF velocity channels before computing errors.
        # Horizontal (vN, vE): longer τ because the cold-start spurious velocity
        # (gravity misattributed as acceleration before attitude converges) is
        # low-frequency and large; the EKF velocity reset at hover entry removes
        # the bulk of it, but the filter handles any residual.
        # Vertical (vD): dead-reckoned from acc-Z with no baro/GPS correction.
        alpha_vH = np.exp(-dt / self._TAU_VH)
        alpha_vD = np.exp(-dt / self._TAU_VD)
        self._vN_filt = alpha_vH * self._vN_filt + (1.0 - alpha_vH) * v_ned_meas[0]
        self._vE_filt = alpha_vH * self._vE_filt + (1.0 - alpha_vH) * v_ned_meas[1]
        self._vD_filt = alpha_vD * self._vD_filt + (1.0 - alpha_vD) * v_ned_meas[2]
        v_ned_meas_filt = np.array([self._vN_filt, self._vE_filt, self._vD_filt])

        e_vel = v_ned_ref - v_ned_meas_filt

        # Velocity integrators with input clamping (anti-windup) and delayed activation.
        # The integrators are held at zero for INTEGRATE_DELAY seconds after hover entry
        # to prevent windup during the liftoff transient: the drone can reach 3+ m/s
        # from slope release within the first second, which saturates xi_vel at full
        # VEL_CLAMP in ~1 s.  The P-term alone stabilises the velocity; the I-term
        # only activates once the drone is near hover.
        e_vel_c = np.clip(e_vel, -self._VEL_CLAMP, self._VEL_CLAMP)
        e_psi   = _wrap_pi(psi_ref - psi_meas)

        # Integrators are gated on two conditions:
        #   1. Ki > 0 in params.yaml  (Ki=0 → state stays zero, no stale buildup)
        #   2. hover_t delay has elapsed (horizontal/yaw delayed to avoid liftoff windup;
        #      vertical starts immediately to correct altitude drift)
        if hover_t >= INTEGRATE_DELAY:
            if self._Ki_vN > 0:
                self.xi_vel[0] = float(np.clip(self.xi_vel[0] + dt * e_vel_c[0],
                                               -self._XI_VEL_LIM[0], self._XI_VEL_LIM[0]))
            if self._Ki_vE > 0:
                self.xi_vel[1] = float(np.clip(self.xi_vel[1] + dt * e_vel_c[1],
                                               -self._XI_VEL_LIM[1], self._XI_VEL_LIM[1]))
            if self._Ki_psi > 0:
                self.xi_psi = np.clip(self.xi_psi + dt * e_psi,
                                      -self._XI_PSI_LIM, self._XI_PSI_LIM)
        if hover_t >= INTEGRATE_DELAY_V and self._Ki_vD > 0:
            self.xi_vel[2] = float(np.clip(self.xi_vel[2] + dt * e_vel_c[2],
                                           -self._XI_VEL_LIM[2], self._XI_VEL_LIM[2]))

        # Desired NED acceleration → desired tilt via small-angle inversion.
        # Use clamped error for proportional path too: unclamped liftoff vz
        # (~11 m/s upward) would otherwise cut collective to ~0.5 N/motor.
        g   = self._g
        a_N = self._Kp_vN * e_vel_c[0] + self._Ki_vN * self.xi_vel[0]
        a_E = self._Kp_vE * e_vel_c[1] + self._Ki_vE * self.xi_vel[1]

        # Rotate desired NED acceleration into body-frame components.
        # At psi=0 (north-facing) this is identity; at other headings it projects
        # a_N/a_E onto the body x_b/y_b axes so tilt angles are always correct.
        R_bn = _rot_from_quat(quat)
        R_nb = R_bn.T

        a_body = R_bn @ np.array([a_N, a_E, 0.0])

        a_xb = a_body[0]
        a_yb = a_body[1]
        theta_des = np.clip( -a_xb / g, -self._MAX_TILT, self._MAX_TILT)
        phi_des   = np.clip( a_yb / g, -self._MAX_TILT, self._MAX_TILT)

        phi_meas, theta_meas, psi_meas = quat_to_euler(quat)

        # Attitude error → desired body rates (clamped to prevent overshooting mixer)
        p_des = np.clip(self._K_att * (phi_des   - phi_meas),   -4.0, 4.0)
        q_des = np.clip(self._K_att * (theta_des - theta_meas), -4.0, 4.0)
        r_des = np.clip(self._K_psi * e_psi + self._Ki_psi * self.xi_psi, -2.0, 2.0)

        # Collective thrust (tilt-corrected); a_z > 0 = NED-down = less lift needed.
        # T_collective is the TOTAL thrust fed into the mixer's first row (T1+T2+T3+T4),
        # so use 4*T_hover (=m*g), not T_hover (=m*g/4) which is per-motor hover thrust.
        a_z = self._Kp_vD * e_vel_c[2] + self._Ki_vD * self.xi_vel[2]
        T_raw = 4.0 * self.T_hover * (1.0 - a_z / g) / R22
        # Output filter: removes R22 (attitude) noise that the vD filter cannot catch.
        alpha_T = np.exp(-dt / self._TAU_T_COLL)
        self._T_coll_filt = alpha_T * self._T_coll_filt + (1.0 - alpha_T) * T_raw
        T_collective = self._T_coll_filt

        # Cache for logger
        self._dbg = {
            'phi_des_deg':   float(np.degrees(phi_des)),
            'theta_des_deg': float(np.degrees(theta_des)),
            'phi_meas_deg':  float(np.degrees(phi_meas)),
            'theta_meas_deg':float(np.degrees(theta_meas)),
            'psi_meas_deg':  float(np.degrees(psi_meas)),
            'psi_ref_deg':   float(np.degrees(psi_ref)),
        }

        return p_des, q_des, r_des, T_collective

    # ------------------------------------------------------------------
    # Attitude-target output (body rates + collective thrust)
    # ------------------------------------------------------------------

    def _send_attitude_target(self, p_des, q_des, r_des, T_collective):
        """Send desired body rates + collective thrust to the sim's onboard rate controller.

        type_mask 0x80: ignore attitude quaternion — sim uses body-rate + thrust setpoints.
        thrust_norm = T_collective [N, total over 4 motors] / (4 * T_max_motor).
        """
        thrust_norm = float(np.clip(T_collective / (4.0 * self.T_max), 0.0, 1.0))
        self.sim_conn.mav.set_attitude_target_send(
            int(time.time() * 1e3) & 0xFFFFFFFF,
            self.sim_conn.target_system,
            self.sim_conn.target_component,
            0x80,                       # ignore attitude quaternion; use rates + thrust
            [1.0, 0.0, 0.0, 0.0],      # attitude quaternion (ignored)
            float(p_des),
            float(q_des),
            float(r_des),
            thrust_norm,
        )
        if self._logger is not None:
            self._logger.log_control(time.time() * 1000.0,
                                     np.full(4, T_collective / 4.0))

    # ------------------------------------------------------------------
    # LQI controller
    # ------------------------------------------------------------------

    def _lqi_trim_idx(self, v_ned_ref, psi_ref):
        """Return index of closest LQI operating point."""
        dv   = self._lqi_vneds - np.asarray(v_ned_ref)
        dpsi = np.abs(_wrap_pi(self._lqi_psis - psi_ref))
        dist = np.sum(dv**2, axis=1) + (dpsi * 2.0)**2
        return int(np.argmin(dist))

    def _lqi_step(self, v_ned, quat, rates, v_ned_ref, psi_ref, dt):
        """
        One LQI control step.

        Reduced state z = [vel_body(3), quat(4), rates(3)]  (10,)
        Augmented state = [z(10), xi(4)]  (14,)
        delta_u = -K @ [delta_z; xi]
        u = clip(u_trim + delta_u, 0, T_max)
        """
        # Body velocity from EKF NED velocity
        R        = _rot_from_quat(quat)
        vel_body = R @ v_ned

        z_meas = np.concatenate([vel_body, quat, rates])   # (10,)

        idx     = self._lqi_trim_idx(v_ned_ref, psi_ref)
        delta_z = z_meas - self._lqi_ztrim[idx]

        # Linearised outputs: [vN, vE, vD, psi]
        y_meas = self._lqi_C[idx] @ z_meas
        y_ref  = np.array([v_ned_ref[0], v_ned_ref[1], v_ned_ref[2], psi_ref])
        e_y    = y_meas - y_ref
        e_y[3] = _wrap_pi(e_y[3])   # keep yaw error in [-π, π]

        self.xi_lqi = np.clip(
            self.xi_lqi + dt * e_y,
            -self._XI_LQI_LIM, self._XI_LQI_LIM
        )

        delta_u = -self._lqi_K[idx] @ np.concatenate([delta_z, self.xi_lqi])
        return np.clip(self._lqi_utrim[idx] + delta_u, 0.0, self.T_max)

    # ------------------------------------------------------------------
    # Motor output
    # ------------------------------------------------------------------

    def _send_motors_norm(self, u_norm):
        """Send 4 normalised [0,1] motor commands to the sim.

        dyn.py motor numbering:  [T1=BR, T2=BL, T3=FL, T4=FR]
        Simulator actuator idx:  [0=FL,  1=FR,  2=BL,  3=BR ]
        Reorder: [T3, T4, T2, T1] → [FL, FR, BL, BR]
        Confirmed by thrust_test.py leveling controller (MORE T3+T4 → NOSE UP).
        """
        cmds = [float(u_norm[2]), float(u_norm[3]),
                float(u_norm[1]), float(u_norm[0])] + [0.0, 0.0, 0.0, 0.0]
        self.sim_conn.mav.set_actuator_control_target_send(
            int(time.time() * 1e6),
            self.sim_conn.target_system,
            self.sim_conn.target_component,
            0,
            cmds
        )
        if self._logger is not None:
            u_n = np.asarray(u_norm)
            self._logger.log_control(time.time() * 1000.0, u_n * self.T_max)

    # ------------------------------------------------------------------
    # Gate helpers
    # ------------------------------------------------------------------

    def _active_gate_ned(self):
        """Return NED position of the currently active gate, or None."""
        gates = self.data.get('track_gates_ned')
        idx   = self.data.get('active_gate_index')
        if gates is None or idx is None:
            return None
        g = gates.get(int(idx))
        return g['ned'].copy() if g is not None else None

    # ------------------------------------------------------------------
    # MAVLink commands
    # ------------------------------------------------------------------

    def arm(self):
        self.sim_conn.mav.command_long_send(
            self.sim_conn.target_system,
            self.sim_conn.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            1, 0, 0, 0, 0, 0, 0
        )

    def send_sim_reset_command(self):
        self.sim_conn.mav.command_long_send(
            self.sim_conn.target_system,
            self.sim_conn.target_component,
            MAVLINK_CMD_SIM_RESET,
            0,
            0, 0, 0, 0, 0, 0, 0
        )


# ------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------

def _wrap_pi(angle):
    """Wrap angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi




# Quaternion → Euler angles (roll, pitch, yaw) to unify rotation representation across the controller.
def quat_to_euler(q):
    qw,qx,qy,qz = q

    roll = np.arctan2(
        2*(qw*qx + qy*qz),
        1-2*(qx*qx+qy*qy)
    )

    pitch = np.arcsin(
        np.clip(2*(qw*qy-qz*qx),-1,1)
    )

    yaw = np.arctan2(
        2*(qw*qz+qx*qy),
        1-2*(qy*qy+qz*qz)
    )

    return roll,pitch,yaw
