import math
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
        # Velocity feedforward coefficients: ff = v_ref / tau_v adds the acceleration
        # needed to hold v_ref against drag without waiting for integrator wind-up.
        _tau_vN = float(param.get('tau_vN', 0.0))
        _tau_vE = float(param.get('tau_vE', 0.0))
        self._ff_vN    = (1.0 / _tau_vN) if _tau_vN > 0.0 else 0.0
        self._ff_vE    = (1.0 / _tau_vE) if _tau_vE > 0.0 else 0.0
        # Reference-derivative feedforward: a_lag = K_dv * dv_ref/dt
        # Pre-commands acceleration for changing reference, reducing tracking lag.
        # Applied to the smoothed reference (post-LP) so noise is not amplified.
        self._K_dv        = float(param.get('K_dv', 0.0))
        self._v_ref_prev  = np.zeros(3)
        # First-order LP on v_ned_ref: smooths carrot oscillation before PI.
        # tau_ref_smooth=0 disables (pass-through). Applied before PT2 and PI.
        self._tau_ref_smooth = float(param.get('tau_ref_smooth', 0.0))
        self._v_ref_smooth   = None   # initialised on first carrot tick
        self._K_att    = float(param['K_att'])
        self._K_psi    = float(param['K_psi'])
        self._Ki_psi   = float(param['Ki_psi'])
        self._MAX_TILT = np.deg2rad(float(param['MAX_TILT_DEG']))
        self._g        = param['g']

        # Velocity gain scheduling: linearly interpolate K_att, Kp_vN/E/D between
        # low-speed (v_lo) and high-speed (v_hi) setpoints.  Scheduling variable is
        # |v_ned_ref| so gains ramp up before speed is reached, not after.
        self._gs_v_lo   = float(param.get('gain_sched_v_lo',  6.0))
        self._gs_v_hi   = float(param.get('gain_sched_v_hi', 20.0))
        self._K_att_hi  = float(param.get('K_att_hi',  self._K_att))
        self._Kp_vN_hi  = float(param.get('Kp_vN_hi',  self._Kp_vN))
        self._Kp_vE_hi  = float(param.get('Kp_vE_hi',  self._Kp_vE))
        self._Kp_vD_hi  = float(param.get('Kp_vD_hi',  self._Kp_vD))

        self._hover_only = bool(param.get('hover_only', False))
        self._debug_waypoints_only = bool(param.get('debug_waypoints_only', False))
        if self._debug_waypoints_only:
            print("[Controller] debug_waypoints_only=true — "
                  "gate overrides and visual yaw disabled", flush=True)
        if self._hover_only:
            print("[Controller] hover_only=true — carrot tracker disabled, "
                  "drone will hold altitude indefinitely", flush=True)

        # Camera tilt matrix (cam→body FRD) — mirrors vision_rx.py, reads cam_tilt_deg from params.
        _tilt = math.radians(float(param.get('cam_tilt_deg', 20.0)))
        _st, _ct = math.sin(_tilt), math.cos(_tilt)
        self._R_cam2body = np.array([[0, _st, _ct],
                                     [1,  0,   0 ],
                                     [0, _ct, -_st]], dtype=float)

        # Cascade integrators
        self.xi_vel      = np.zeros(3)              # NED velocity integrals [m]
        self.xi_psi      = 0.0                      # yaw integral [rad·s]
        self._XI_VEL_LIM = np.array([3.0, 3.0, 2.0])
        self._XI_PSI_LIM = np.pi
        self._VEL_CLAMP  = 10.0                      # m/s — integrator input clamp

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

        # Gate-pass delay: hold wp advance until drone clears the gate by N metres.
        # At v=20 m/s, gate_pass_dist_m=3 → ~0.15 s after collision before turn.
        # A timeout_sec cap prevents deadlock if the drone overshoots without advancing.
        self._gate_pass_dist    = float(param.get('gate_pass_dist_m', 0.0))
        self._gate_pass_timeout = float(param.get('gate_pass_timeout_sec', 2.0))
        self._wp_pending        = None   # (new_wp, gate_wp_ned, ea, t_fired) or None
        # RACE_STATUS's active_gate_index as a second, independent gate-pass
        # trigger alongside COLLISION's gate_passed flag. Confirmed in a flight
        # log: the sim never sent a single COLLISION message for the whole
        # flight, but RACE_STATUS's active_gate_index correctly advanced 0->1
        # right as the drone converged on gate 0 — COLLISION support appears to
        # be absent/disabled in this sim build, so it can't be the sole trigger.
        self._last_seen_agi     = -1

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
        self._level_done_t   = None   # wall-clock time levelling finished (blip clock starts here)

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
        # Motors idle; EKF accumulates static acc/gyro for attitude sysid.
        if t_elapsed < WAIT_PHASE_SEC:
            if _now - getattr(self, '_last_wait_diag_t', 0.0) >= 0.5:
                self._last_wait_diag_t = _now
                _phi_w, _theta_w, _ = quat_to_euler(quat)
                phi_deg   = np.degrees(_phi_w)
                theta_deg = np.degrees(_theta_w)
                _, _qx_w, _qy_w, _ = quat
                R22_w = max(0.3, 1.0 - 2*(_qx_w*_qx_w + _qy_w*_qy_w))
                # Sysid: display ground tilt from static acc average (FRD: θ<0 = nose-down).
                _acc_avg = self.data.get('wait_acc_avg')
                _sysid_str = ''
                if _acc_avg is not None:
                    _g = 9.81
                    _th_s = float(np.degrees(np.arcsin(np.clip(_acc_avg[0] / _g, -1.0, 1.0))))
                    _ph_s = float(np.degrees(np.arcsin(
                        np.clip(_acc_avg[1] / (_g * max(np.cos(np.radians(_th_s)), 0.1)),
                                -1.0, 1.0))))
                    _sysid_str = f'  [SYSID tilt θ={_th_s:.1f}° φ={_ph_s:.1f}°]'
                print(f"[WAIT t={t_elapsed:.2f}s]  "
                      f"roll={phi_deg:.1f}°  pitch={theta_deg:.1f}°  R22={R22_w:.3f}  "
                      f"rates=({rates[0]:.2f},{rates[1]:.2f},{rates[2]:.2f})  "
                      f"dt={_actual_dt*1000:.1f}ms{_sysid_str}", flush=True)
            self._send_attitude_target(0.0, 0.0, 0.0, 4.0 * 0.05 * self.T_max)
            time.sleep(DT)
            return

        # ── PHASE 1.5: LEVEL ──────────────────────────────────────────────
        # Null roll/pitch before the liftoff blip fires. The blip below assumes
        # "pure vertical" thrust from four equal motors — true only if level. On
        # the sloped spawn platform the drone is tilted (~17° pitch, measured
        # during WAIT), so firing the blip as-is injects real horizontal velocity
        # (sin(tilt) fraction of the impulse). The outer loop then chases that
        # velocity error by saturating theta_des to its clamp — confirmed root
        # cause of post-liftoff attitude divergence (theta/phi diverging into the
        # 100s of degrees within a few seconds of hover entry). Level first, at
        # hover-neutral thrust (grounded, no climb yet), then blip once level —
        # this keeps hover_entry_t (and everything timed off it: the blip itself,
        # imu_ekf's attitude-freeze window, its backup post-blip attitude reset,
        # and the vision-update settling window) anchored to when the blip
        # actually fires, so none of that downstream timing needs to change.
        #
        # Tell imu_ekf WAIT has truly ended, every tick from here on (idempotent).
        # imu_ekf's static acc/gyro accumulation (gyro-bias calibration + the
        # ground-tilt reading used at hover-reset) must stop here, not at
        # hover_reset_done — that now fires *after* this whole levelling phase,
        # and letting accumulation run through it would average in real,
        # intentional rotation, corrupting both the bias estimate and the tilt
        # reading with motion that isn't the static "resting on the slope" state
        # they're meant to capture.
        self.data['wait_phase_done'] = True
        if self._level_done_t is None:
            _phi_lv, _theta_lv, _ = quat_to_euler(quat)
            _level_tol      = np.deg2rad(float(self.param.get('level_tol_deg', 2.0)))
            _level_elapsed  = t_elapsed - WAIT_PHASE_SEC
            _level_timeout  = float(self.param.get('level_timeout_sec', 1.5))
            _levelled = abs(_phi_lv) < _level_tol and abs(_theta_lv) < _level_tol
            if _levelled or _level_elapsed > _level_timeout:
                self._level_done_t = time.time()
                print(f"[CONTROLLER] Pre-blip level "
                      f"{'done' if _levelled else 'TIMED OUT'} — "
                      f"phi={np.degrees(_phi_lv):.1f}°  theta={np.degrees(_theta_lv):.1f}°  "
                      f"t={_level_elapsed:.2f}s", flush=True)
            else:
                _p_lv = np.clip(self._K_att * (0.0 - _phi_lv),   -4.0, 4.0)
                _q_lv = np.clip(self._K_att * (0.0 - _theta_lv), -4.0, 4.0)
                self._send_attitude_target(_p_lv, _q_lv, 0.0, 4.0 * self.T_hover)
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
            # hover_target_yaw_deg sets the heading the drone turns to at hover entry
            # (separate from initial_yaw_deg which is the EKF frame alignment at spawn).
            _hover_yaw_deg = float(self.param.get(
                'hover_target_yaw_deg',
                self.param.get('initial_yaw_deg', 0.0)))
            self._psi_cmd = np.deg2rad(_hover_yaw_deg)
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

        # No more liftoff blip or post-blip attitude freeze. The drone is already
        # level (PHASE 1.5) and cascade control now ramps in its own authority
        # (see _outer_loop's _att_ramp, which scales K_att/Kp_vN/Kp_vE and the
        # horizontal reference itself over 2 s) — that ramp does the same job the
        # blip+freeze used to (avoid a violent initial command) without an
        # open-loop thrust burst or a window where the cascade flies blind on a
        # stale attitude estimate. Kp_vD (altitude) was never ramped, so vertical
        # control — and therefore actual liftoff authority — is already at full
        # strength from the very first cascade tick.
        v_ref_for_gains = np.zeros(3)
        if in_hover:
            # hover_only=True: hold psi_cmd at initial_yaw_deg for a stable hover.
            pass
        else:
            # Carrot tracking: use EKF position to follow the waypoint path.
            if not self._carrot_active:
                self._carrot_active  = True
                self._v_ref_smooth   = None   # reset LP so it init from first carrot reference
                # Sync to whatever active_gate_index already is (e.g. 0, before any
                # gate has been passed) so the RACE_STATUS-advance check below only
                # fires on a real *increase* from here, not on this baseline value.
                self._last_seen_agi = int(self.data.get('active_gate_index', -1))
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
            # This path has none of the EKF vision feed's protections (PnP
            # dual-solution hysteresis, yaw smoothing, frame dedup) since it
            # reads tvec_cam directly rather than going through
            # _vision_ekf_update, so a bad frame here would snap the live
            # carrot target straight to a bad position with zero smoothing.
            if not self._debug_waypoints_only:
                if det.get('detected') and det.get('tvec_cam') is not None:
                    _tvec = np.asarray(det['tvec_cam'], dtype=float)
                    _t_body = self._R_cam2body @ _tvec
                    R_bn = _rot_from_quat(quat)
                    R_nb = R_bn.T
                    self.tracker.waypoints[self.tracker.wp] = pos_ned + R_nb @ _t_body

            # Update next waypoint from vision-confirmed second-gate NED position.
            # Only applied once position is stable (median over next_gate_min_frames).
            # Gated on debug_waypoints_only like the other vision overrides above —
            # previously this was unreachable (vision_rx.py never ran detection, so
            # next_gate_ned was never set), but vision_detect_for_logging now lets
            # detection run for logging while debug_waypoints_only stays true, and
            # this block had no gate of its own, so it started actually overwriting
            # the next waypoint with a real (and apparently offset) vision-derived
            # position — causing a beam strike at the second gate.
            if not self._debug_waypoints_only:
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

            # Publish current mode for the vision debug overlay. TRANSITION
            # (the carrot's commit phase — pinned to the gate centre for final
            # approach, see carrot_tracker.py) takes priority over PNP/HOLD/CARROT
            # since it's the more specific, more relevant state near the gate.
            self.data['vis_ctrl_mode'] = (
                'TRANSITION' if self.tracker.committed
                else 'PNP'    if _fresh_tvec is not None
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
                _t_body_v = self._R_cam2body @ _tvec_v
                _horiz_v = float(np.sqrt(_t_body_v[0]**2 + _t_body_v[1]**2))
                if _horiz_v > 1.0:   # gate at least 1 m away horizontally
                    _dir_xb = _t_body_v[0] / _horiz_v   # body-forward component
                    _dir_yb = _t_body_v[1] / _horiz_v   # body-right  component
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
        if not in_hover and self._carrot_active and self._pt2_omega0 > 0:
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
                mode       = self.data.get('vis_ctrl_mode', ''),
            )

        # Advance carrot waypoint when sim signals gate passage — via either
        # COLLISION (gate_passed) or a RACE_STATUS active_gate_index increase.
        # See _last_seen_agi's init comment for why both are needed.
        _agi_now      = int(self.data.get('active_gate_index', -1))
        _agi_advanced = _agi_now > self._last_seen_agi
        self._last_seen_agi = _agi_now
        _via_collision = self.data.pop('gate_passed', False)
        if _via_collision or _agi_advanced:
            # Label which signal actually fired — id=None from COLLISION is
            # normal on sim builds that don't send it; RACE_STATUS (agi=N) is
            # the primary trigger there. See _last_seen_agi's init comment.
            _trigger = (f"COLLISION id={self.data.get('last_gate_id')}"
                        if _via_collision else f"RACE_STATUS agi={_agi_now}")
            agi    = _agi_now
            new_wp = self.tracker.wp + 1
            if agi >= 0:
                new_wp = max(new_wp, int(agi) + 1)
            new_wp = min(new_wp, self.tracker.n_waypoints - 1)
            if new_wp > self.tracker.wp:
                if self._gate_pass_dist > 0.0:
                    # Defer: wait until drone is gate_pass_dist_m past the gate along segment.
                    _r0  = np.asarray(self.tracker.waypoints[self.tracker.wp - 1], dtype=float)
                    _r1  = np.asarray(self.tracker.waypoints[self.tracker.wp],     dtype=float)
                    _d   = _r1 - _r0
                    _len = float(np.linalg.norm(_d))
                    _ea  = _d / _len if _len > 1e-6 else _d
                    self._wp_pending = (new_wp, _r1.copy(), _ea, time.time())
                    print(f"[GATE PASSED {_trigger}] "
                          f"wp advance to {new_wp} pending ({self._gate_pass_dist:.1f} m clearance)",
                          flush=True)
                else:
                    self.tracker.wp = new_wp
                    self._wp_pending = None
                    print(f"[GATE PASSED {_trigger}] "
                          f"tracker.wp → {self.tracker.wp}", flush=True)

        # Resolve pending wp advance: fire when drone clears gate by gate_pass_dist_m
        # or when the timeout elapses (prevents deadlock on missed gate).
        if self._wp_pending is not None:
            _new_wp, _gate_ned, _ea, _t_fired = self._wp_pending
            _elapsed = time.time() - _t_fired
            _along   = float(np.dot(pos_ned - _gate_ned, _ea))
            if _along >= self._gate_pass_dist or _elapsed >= self._gate_pass_timeout:
                self.tracker.wp  = _new_wp
                self._wp_pending = None
                self._wp_transition_reset()
                print(f"[WP ADVANCE] tracker.wp → {_new_wp}  "
                      f"(along={_along:.1f} m  elapsed={_elapsed:.2f} s)", flush=True)

        # No position-based fallback advance anymore — waypoint advance is driven
        # solely by the gate_passed (COLLISION) message above. A geometry-only
        # fallback can fire on noisy/biased position even when the drone hasn't
        # actually passed the gate, which is exactly the kind of spurious
        # advance we don't want feeding into the commit-phase logic below.

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
                    # Post-tau_ref_smooth value (what the loop actually tracks),
                    # not the raw carrot reference — see _dbg's init comment.
                    v_ned_ref     = dbg.get('v_ned_ref_smoothed', v_ref_for_gains),
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
    # Waypoint transition reset
    # ------------------------------------------------------------------

    def _wp_transition_reset(self):
        """Called the instant tracker.wp advances to a new segment.

        Project horizontal integrators (N, E) onto the new segment tangent:
          - Along-track component is preserved  (drag compensation still valid).
          - Cross-track component is discarded  (would fight the turn).
          cos(θ) scaling means: small direction change → near-full preservation;
          90° turn → full zero.  Hard-zero was the prior behaviour (θ treated as 90°
          always), which created a step in controller output at every transition.

        Vertical integrator xi_vel[2] is kept unchanged: it compensates gravity
        imbalance and is independent of horizontal heading.

        Also zero the lag-FF derivative for one tick so dv_ref/dt = 0 on the first
        post-transition call (prevents spike from any residual step in smoothed ref).
        """
        wp  = self.tracker.wp
        r0  = np.asarray(self.tracker.waypoints[wp - 1], dtype=float)
        r1  = np.asarray(self.tracker.waypoints[wp],     dtype=float)
        seg = r1 - r0
        seg_len = float(np.linalg.norm(seg))
        if seg_len > 1e-6:
            ea_h   = seg[:2] / seg_len             # horizontal tangent (N, E)
            ea_h  /= max(float(np.linalg.norm(ea_h)), 1e-9)
            xi_proj = float(np.dot(self.xi_vel[:2], ea_h))
            self.xi_vel[0] = xi_proj * ea_h[0]
            self.xi_vel[1] = xi_proj * ea_h[1]
        else:
            self.xi_vel[0] = 0.0
            self.xi_vel[1] = 0.0
        # xi_vel[2] (vertical) is direction-independent — preserve it

        if self._v_ref_smooth is not None:
            self._v_ref_prev = self._v_ref_smooth.copy()

    # ------------------------------------------------------------------
    # Outer loop: NED velocity + yaw → desired body rates + collective
    # ------------------------------------------------------------------

    def _outer_loop(self, v_ned_ref, psi_ref, v_ned_meas, quat, psi_meas, R22, dt, hover_t):
        # Coordinate convention (body FRD, world NED):
        #   φ>0 right-wing-down   θ<0 nose-down   ψ: 0=N CW+
        #   θ_des = −a_xb/g  →  nose-UP (θ>0) brakes northward drift  ✓
        #   q_des>0 → nose-UP rate to sim (confirmed empirically from cascade logs)
        #   r_des sign: GT sends −r_des (psi_meas = −psi_NED from ATTITUDE);
        #               non-GT sends +r_des (psi_meas standard NED from EKF gyros)

        # ── Gain scheduling ───────────────────────────────────────────────────
        # Interpolate K_att and Kp_v* linearly with |v_ned_ref| so gains ramp up
        # as commanded speed increases, improving attitude bandwidth at high speed.
        _v_sched = float(np.linalg.norm(v_ned_ref))
        if self._gs_v_hi > self._gs_v_lo:
            _gs_alpha = np.clip(
                (_v_sched - self._gs_v_lo) / (self._gs_v_hi - self._gs_v_lo), 0.0, 1.0)
        else:
            _gs_alpha = 0.0
        _K_att_eff = self._K_att + _gs_alpha * (self._K_att_hi - self._K_att)
        _Kp_vN_eff = self._Kp_vN + _gs_alpha * (self._Kp_vN_hi - self._Kp_vN)
        _Kp_vE_eff = self._Kp_vE + _gs_alpha * (self._Kp_vE_hi - self._Kp_vE)
        _Kp_vD_eff = self._Kp_vD + _gs_alpha * (self._Kp_vD_hi - self._Kp_vD)

        # ── Post-hover-entry ramp ────────────────────────────────────────────────
        # No blip/freeze anymore (see PHASE 1.5 / hover-entry in update()) — the
        # drone is level and cascade control starts on the very next tick. Ramp
        # both the attitude gain and the horizontal velocity P gains from 0 to
        # full over 2 s anyway, so theta_des stays near 0 at the start and builds
        # up gradually, preventing the 60°-clip and the resulting hard pitch
        # command that a full-strength response to a stepped-in reference would
        # otherwise produce. Kp_vD (altitude) is NOT ramped — altitude hold must
        # stay active for liftoff.
        _att_ramp = float(np.clip(hover_t / 2.0, 0.0, 1.0))
        _K_att_eff  *= _att_ramp
        _Kp_vN_eff  *= _att_ramp
        _Kp_vE_eff  *= _att_ramp

        # Ramp the horizontal reference itself alongside the gains above, not just
        # the gains. Ramping gain alone leaves e_vel at its full magnitude (carrot
        # hands over a full-value reference, e.g. -5 m/s, from the first tick) —
        # so even a partially-ramped Kp_vN still multiplies a ~full-size error,
        # producing a steadily growing theta_des that reaches MAX_TILT within
        # ~1-1.5s regardless of the gain ramp (confirmed in flight logs: growth
        # tracked _att_ramp almost exactly, with model_predict, the derivative
        # feedforward, and the one-tick kick all independently ruled out first).
        # Scaling v_ned_ref down here means e_vel starts small and grows together
        # with the gain, keeping their product — the commanded tilt — bounded
        # through the same transition instead of just delaying when it saturates.
        # D (altitude) is excluded, matching Kp_vD staying unramped below.
        v_ned_ref = v_ned_ref.copy()
        v_ned_ref[0] *= _att_ramp
        v_ned_ref[1] *= _att_ramp

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

        # ── 1st-order LP on v_ned_ref ─────────────────────────────────────────
        # Smooths carrot oscillation before it reaches the PI and lag FF.
        # tau_ref_smooth=0 (default) disables.
        if self._tau_ref_smooth > 0.0:
            _alpha_ref = np.exp(-dt / self._tau_ref_smooth)
            if self._v_ref_smooth is None:
                self._v_ref_smooth = v_ned_ref.copy()
            self._v_ref_smooth = _alpha_ref * self._v_ref_smooth + (1.0 - _alpha_ref) * v_ned_ref
            v_ned_ref = self._v_ref_smooth.copy()

        # ── Reference-derivative feedforward ──────────────────────────────────
        # a_lag = K_dv * dv_ref/dt: pre-commands the acceleration needed to track
        # a changing reference, reducing velocity tracking lag during transitions.
        if self._K_dv > 0.0:
            _dv = np.clip((v_ned_ref - self._v_ref_prev) / max(dt, 0.001), -20.0, 20.0)
            _a_lag_N = self._K_dv * _dv[0]
            _a_lag_E = self._K_dv * _dv[1]
        else:
            _a_lag_N = _a_lag_E = 0.0
        self._v_ref_prev = v_ned_ref.copy()

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
        a_N = (_Kp_vN_eff * e_vel_c[0] + self._Ki_vN * self.xi_vel[0]
               + self._ff_vN * v_ned_ref[0] + _a_lag_N)
        a_E = (_Kp_vE_eff * e_vel_c[1] + self._Ki_vE * self.xi_vel[1]
               + self._ff_vE * v_ned_ref[1] + _a_lag_E)

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
        p_des = np.clip(_K_att_eff * (phi_des   - phi_meas),   -4.0, 4.0)
        q_des = np.clip(_K_att_eff * (theta_des - theta_meas), -4.0, 4.0)
        r_des = np.clip(self._K_psi * e_psi + self._Ki_psi * self.xi_psi, -2.0, 2.0)

        # Collective thrust (tilt-corrected); a_z > 0 = NED-down = less lift needed.
        # T_collective is the TOTAL thrust fed into the mixer's first row (T1+T2+T3+T4),
        # so use 4*T_hover (=m*g), not T_hover (=m*g/4) which is per-motor hover thrust.
        a_z = _Kp_vD_eff * e_vel_c[2] + self._Ki_vD * self.xi_vel[2]
        T_raw = 4.0 * self.T_hover * (1.0 - a_z / g) / R22
        # Output filter: removes R22 (attitude) noise that the vD filter cannot catch.
        alpha_T = np.exp(-dt / self._TAU_T_COLL)
        self._T_coll_filt = alpha_T * self._T_coll_filt + (1.0 - alpha_T) * T_raw
        T_collective = self._T_coll_filt

        # Cache for logger. v_ned_ref_smoothed is the POST-tau_ref_smooth value
        # actually used below for e_vel/feedforward — distinct from the caller's
        # raw v_ref_for_gains (what log_cascade used to log). Confirmed in a
        # flight log: at a waypoint transition, logged vD_ref jumped -0.20 -> 1.26
        # in one tick with vD_meas responding smoothly and gradually right through
        # it — the "discontinuity" was the log showing the pre-filter carrot
        # target, not an actual step in what the controller commanded.
        self._dbg = {
            'phi_des_deg':   float(np.degrees(phi_des)),
            'theta_des_deg': float(np.degrees(theta_des)),
            'phi_meas_deg':  float(np.degrees(phi_meas)),
            'theta_meas_deg':float(np.degrees(theta_meas)),
            'psi_meas_deg':  float(np.degrees(psi_meas)),
            'psi_ref_deg':   float(np.degrees(psi_ref)),
            'v_ned_ref_smoothed': v_ned_ref.copy(),
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
        # Body rates p/q/r sent as-is to the sim's inner rate controller.
        # r_des is mode-dependent because psi_meas convention differs:
        #   GT   — ATTITUDE yaw is NOT negated in mavlink_rx → psi_meas = −psi_NED
        #          → r_des is sign-inverted → negate before send
        #   non-GT — EKF gyro integration → psi_meas = standard NED → send as-is
        _gt_mode = bool(self.param.get('ground_truth_mode', False))
        _r_cmd   = float(-r_des if _gt_mode else r_des)
        self.sim_conn.mav.set_attitude_target_send(
            int(time.time() * 1e3) & 0xFFFFFFFF,
            self.sim_conn.target_system,
            self.sim_conn.target_component,
            0x80,                       # ignore attitude quaternion; use rates + thrust
            [1.0, 0.0, 0.0, 0.0],      # attitude quaternion (ignored)
            float(p_des),
            float(q_des),
            _r_cmd,
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
