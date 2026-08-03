import time
import numpy as np
from pymavlink import mavutil

from control.carrot_tracker import CarrotTracker
from flight_model import rotations
from vision.pose_estimate import PoseEstimate
from vision.vision_mode import (Mode, VisionModeTracker, VerticalAssist, PursuitGuidance,
                          RecoveryGuard, path_convergence_weight)

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


# v_body = R @ v_NED — see rotations.py (centralized after a real bug was
# traced to this formula's transpose being reused under the same function
# name elsewhere in the codebase).
_rot_from_quat = rotations.quat_to_R_ned2body


class Controller:

    def __init__(self, sim_conn, data, system_boot_ms, param, logger=None):
        self.sim_conn         = sim_conn
        self.data             = data
        self.system_boot_ms   = system_boot_ms
        self.param            = param
        self._logger          = logger
        # GT mode's ATTITUDE-derived yaw is a different convention than
        # standard NED (psi_gt = -psi_NED — see _gt_yaw_convention's
        # docstring). Cached once here rather than re-read per use, and to
        # give every read/write of this flag one canonical name to grep for.
        self._gt_mode         = bool(param.get('ground_truth_mode', False))

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
        # Derivative gains (derivative-on-measurement, not on error) — damps velocity
        # overshoot without reacting to v_ned_ref steps at waypoint transitions.
        # Default 0.0 (off) preserves existing PI-only behaviour.
        self._Kd_vN    = float(param.get('Kd_vN', 0.0))
        self._Kd_vE    = float(param.get('Kd_vE', 0.0))
        self._Kd_vD    = float(param.get('Kd_vD', 0.0))
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
        # Separate D-axis lag-compensator gain: vD was never wired to the N/E lag
        # comp above (only _a_lag_N/_a_lag_E existed), so vD tracked v_ref with pure
        # lag. Independent gain lets vD run more aggressively than K_dv without
        # touching the already-tuned horizontal response.
        self._K_dv_D      = float(param.get('K_dv_D', self._K_dv))
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
        self._v_meas_filt_prev = None                # previous filtered v_ned, for D term
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

        # Vision mode/confidence state machine (vision_mode.py) — replaces
        # the old vis_ctrl_mode string state machine (independently
        # re-derived each tick from a mixture of this-tick-fresh and last-
        # tick-stale sub-signals) with one state driven directly by Layer
        # 1's PoseEstimate.state. Also owns the reacquisition trust ramp
        # (old _reacq_blend_w) — see VisionModeTracker.update() for why no
        # separate edge-detection is needed any more.
        self._vision_mode = VisionModeTracker(
            reacq_tau_s=float(param.get('vision_reacq_blend_tau', 0.4)),
            loss_tau_s=float(param.get('vision_reacq_loss_tau', 0.15)))

        # Vertical assist (vision_mode.py) — merges the old blind-search
        # descend nudge and vertical visual-centering nudge under one owner
        # with one bounded combination instead of two independent,
        # uncoordinated += onto the D reference.
        self._vassist = VerticalAssist(
            search_range_m      = float(param.get('vision_search_range_m', 15.0)),
            search_trigger_s    = float(param.get('vision_search_trigger_sec', 0.5)),
            search_max_m        = float(param.get('vision_search_max_descend_m', 3.0)),
            search_rate_mps     = float(param.get('vision_search_descend_mps', 0.6)),
            vnudge_border_frac  = float(param.get('vision_vnudge_border_frac', 0.6)),
            vnudge_max_mps      = float(param.get('vision_vnudge_max_mps', 1.5)),
            cam_cy              = float(param.get('cam_cy', 180.0)),
        )

        # Pursuit-override direction rate-limiter (vision_mode.py) — damps
        # the overshoot-then-correct swing a large one-off bearing
        # correction (e.g. after a long BLIND coast) otherwise produces in
        # the raw bearing-following pursuit law. See PursuitGuidance's
        # docstring.
        self._pursuit_guidance = PursuitGuidance(
            max_turn_rate=np.deg2rad(float(param.get('vision_pursuit_turn_rate_max_deg', 60.0))))
        # Fixed NED point (not a body-frame vector — see the pursuit-
        # override block for why) computed from the last gate_bearing_body_m
        # seen while gate_id_match was True. The trust_w-decay grace window
        # needs a target to blend toward once gate_id_match itself goes
        # False (same tick mode flips to BLIND on a real loss), since pose
        # no longer asserts one at that point.
        self._last_pursuit_target_ne = None
        # Reacquisition speed cap — see the matching comment where it's
        # applied. Buys convergence time by slowing the approach instead of
        # letting the drone close in at full speed while trust_w is still
        # ramping.
        self._reacq_speed_range_m  = float(param.get('vision_reacq_speed_range_m', 12.0))
        self._reacq_speed_min_frac = float(param.get('vision_reacq_speed_min_frac', 0.5))

        # Recovery guard — vetoes a COMMIT trigger vision evidence directly
        # contradicts and flies back to a known point on the pathway
        # instead of trusting the (possibly-wrong) commit-phase geometry
        # or beelining straight at a now-distant gate. See
        # vision_mode.RecoveryGuard's docstring.
        self._recovery_guard = RecoveryGuard(
            sanity_margin_m=float(param.get('vision_recovery_sanity_margin_m', 5.0)),
            exit_dist_m=float(param.get('vision_recovery_exit_dist_m', 2.5)),
            timeout_s=float(param.get('vision_recovery_timeout_s', 15.0)),
        )
        self._recovery_margin_m  = float(param.get('vision_recovery_margin_m', 8.0))
        self._recovery_speed_mps = float(param.get('vision_recovery_speed_mps', 3.0))

        # Cross-track distance at which the pursuit override's direct-at-
        # the-gate direction is fully suppressed in favor of carrot's own
        # pathway-pursuit — see vision_mode.path_convergence_weight's
        # docstring for why trust_w alone isn't sufficient here.
        self._path_recover_dist_m = float(param.get('vision_path_recover_dist_m', 4.0))

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
                    "run 'python -m flight_model.lqi' first.")
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
            pose = self.data.get('pose_estimate') or PoseEstimate.blind()
            # Single confidence/mode signal (vision_mode.py), driven directly
            # by Layer 1's PoseEstimate.state — replaces the old vis_ctrl_mode
            # string state machine, which was independently re-derived each
            # tick from a mixture of this-tick-fresh and last-tick-stale
            # sub-signals (tracker.committed, _fresh_tvec, and the PREVIOUS
            # tick's _vis_tvec_hold cache). See vision_mode.py's docstring.
            self._vision_mode.update(pose.state, _actual_dt)
            # gate_id_match replaces the old _agi_matches_wp: which
            # active_gate_index THIS pose estimate was actually resolved
            # against (captured at measurement time in vision_rx.py, so it
            # stays correct even across a held/not-fresh tick) rather than
            # re-reading the live shared value — protects every vision-
            # derived write below from a lingering detection of the wrong
            # gate during an agi/tracker.wp desync. Confirmed in a flight
            # log (old mechanism): right after tracker.wp advanced to target
            # G3, a lingering detection of the just-passed G2 dragged a live
            # target ~30+ m off, aiming the drone increasingly sideways over
            # the next ~0.3s.
            gate_id_match = (pose.gate_id is not None
                              and pose.gate_id == self.tracker.wp - 1)

            # Refine target waypoint to PnP-measured gate position in local
            # NED via CarrotTracker.set_live_target (EMA-smoothed inside the
            # tracker — see that method for why raw overwriting freezes a
            # reacquisition jump into the next segment's tangent anchor once
            # tracker.wp advances). Only fed a FRESH measurement, never a
            # held one across a tolerated miss — matches the code this
            # replaces, which only ever used this tick's own detection.
            if (not self._debug_waypoints_only and gate_id_match
                    and pose.fresh and pose.has_pnp):
                _wp_raw = pos_ned + rotations.rotate_body_to_ned(
                    pose.gate_bearing_body_m, quat, self._gt_mode)
                self.tracker.set_live_target(self.tracker.wp, _wp_raw, dt=_actual_dt)

            # Update next waypoint from vision-confirmed second-gate NED position.
            # Only applied once position is stable (median over next_gate_min_frames).
            # Gated on debug_waypoints_only like the live-target update above —
            # previously this was unreachable (vision_rx.py never ran detection, so
            # next_gate_ned was never set), but vision_detect_for_logging now lets
            # detection run for logging while debug_waypoints_only stays true, and
            # this block had no gate of its own, so it started actually overwriting
            # the next waypoint with a real (and apparently offset) vision-derived
            # position — causing a beam strike at the second gate.
            # Also gated on gate_id_match: next_gate_ned is computed in
            # vision_rx.py relative to ITS OWN active_gate_index (the gate
            # after agi's current target) — writing it to tracker.wp+1 is only
            # correct when tracker.wp == agi+1. During a desync this would
            # otherwise land in the wrong slot, same class of bug as the
            # live-target update above.
            if not self._debug_waypoints_only and gate_id_match:
                _ng_ned = self.data.get('next_gate_ned')
                if _ng_ned is not None:
                    _wp_next = self.tracker.wp + 1
                    if _wp_next < self.tracker.n_waypoints:
                        self.tracker.waypoints[_wp_next] = np.asarray(_ng_ned, dtype=float)

            v_ned_ref_carrot, psi_ref_carrot = self.tracker.update(pos_ned, dt=_actual_dt)
            v_ref_for_gains = v_ned_ref_carrot

            # Effective mode needs tracker.committed, only known after
            # tracker.update() just above — commit takes absolute priority
            # over confidence, matching the old TRANSITION > PNP > HOLD >
            # CARROT precedence (see Mode.COMMIT's docstring for why its
            # wire value is still the literal string 'TRANSITION').
            mode = self._vision_mode.effective_mode(pose.state, self.tracker.committed)

            # Recovery point: recovery_margin_m back from the current
            # target gate along the segment's own known endpoints — a
            # point ON the intended pathway, not the gate itself and not
            # wherever carrot_tracker's own internal geometry currently
            # aims (that geometry is driven by the same position estimate
            # RecoveryGuard exists to distrust). Cheap to compute every
            # tick; only acted on when recovery is actually active.
            _seg_r0  = np.asarray(self.tracker.waypoints[self.tracker.wp - 1], dtype=float)
            _seg_r1  = np.asarray(self.tracker.waypoints[self.tracker.wp], dtype=float)
            _seg_vec = _seg_r1 - _seg_r0
            _seg_len = float(np.linalg.norm(_seg_vec))
            _seg_ea  = _seg_vec / _seg_len if _seg_len > 1e-6 else np.array([1.0, 0.0, 0.0])
            _recovery_point = _seg_r1 - self._recovery_margin_m * _seg_ea
            _rec_vec  = _recovery_point - pos_ned
            _rec_dist = float(np.linalg.norm(_rec_vec))
            # How close to the direct line between the two gates the drone
            # currently is — used below to keep the pursuit override from
            # cutting a diagonal at the gate before cross-track error from
            # a BLIND stretch has actually closed. See
            # path_convergence_weight's docstring.
            _path_conv_w = path_convergence_weight(
                pos_ned, _seg_r0, _seg_ea, self._path_recover_dist_m)

            # See RecoveryGuard's docstring: vetoes a COMMIT that vision
            # evidence directly contradicts and takes priority over every
            # other mode — including COMMIT itself — until the drone is
            # back on the pathway or the attempt times out.
            if self._recovery_guard.update(
                    raw_committed=self.tracker.committed, pose=pose,
                    gate_id_match=gate_id_match, commit_dist_m=self.tracker.commit_dist,
                    dist_to_recovery_point=_rec_dist, now=_now):
                mode = Mode.RECOVERY
            self.data['vis_ctrl_mode'] = mode.value

            # Vision-bearing pursuit override: replace NE components of
            # v_ned_ref with a direction from the PnP gate bearing — bypasses
            # EKF position drift. TRACKING/REACQUIRING together cover what
            # the old PNP+HOLD states covered: GateLock's own miss-streak
            # tolerance now absorbs ordinary brief flicker without ever
            # leaving LOCKED, so there's no separate "flickering but still
            # trust it" state left to represent. debug_waypoints_only must
            # stay an explicit check here (not just implied by mode) since
            # vision_detect_for_logging keeps vision_rx.py producing real
            # PoseEstimates even when debug_waypoints_only is true — without
            # this the override would fire on live detections meant only
            # for logging. Suppressed once committed (mode is COMMIT, never
            # TRACKING/REACQUIRING then) — bearing angle ~ atan(offset/range)
            # blows up as range shrinks, so even a tiny positional wobble
            # right at the gate swings the raw bearing violently (confirmed
            # in a flight log: vc_E jumped -1 to +4 m/s in one tick while
            # nominally in TRANSITION).
            if mode == Mode.RECOVERY:
                # Head straight at the recovery point at a capped, gentle
                # speed instead of the ordinary gate-bearing pursuit law.
                # PursuitGuidance's rate limit exists to protect against
                # overshoot on a bearing that swings as range closes on the
                # gate — here the target is a fixed point on the pathway,
                # not a shrinking-range bearing, so a direct heading is
                # enough; the speed cap is what keeps this a controlled
                # recovery rather than another aggressive correction.
                self._pursuit_guidance.reset()
                if _rec_dist > 1e-3:
                    v_ref_for_gains = (_rec_vec / _rec_dist) * self._recovery_speed_mps
            elif (not self._debug_waypoints_only and (
                    (mode in (Mode.TRACKING, Mode.REACQUIRING) and gate_id_match)
                    or (mode == Mode.BLIND and self._vision_mode.trust_w > 1e-3
                        and self._last_pursuit_target_ne is not None))):
                # Two ways in: (a) the ordinary case, a fresh gate_id_match
                # this tick; (b) the trust_w-decay grace window right after
                # a real loss (mode flips to BLIND the SAME tick GateLock
                # leaves LOCKED and vision_rx.py stops asserting gate_id —
                # confirmed in a flight log: agi_match flips 1->0 on the
                # exact tick mode flips to BLIND, well before trust_w has
                # meaningfully decayed). Explicitly scoped to BLIND (not
                # "any mode with trust_w>0") so COMMIT still fully suppresses
                # pursuit as documented below, regardless of trust_w.
                #
                # Case (b) can't read pose.gate_bearing_body_m — vision_rx.py
                # has already stopped asserting a gate_id, so gate_id_match
                # is False and nothing about the current gate is confirmed.
                # The cache below is a fixed NED point, not a body-frame
                # vector: caching the raw body vector and re-rotating it by
                # the CURRENT yaw every tick (an earlier version of this fix)
                # made the "target" swing sideways in lockstep with the
                # drone's own yaw motion during the decay window — confirmed
                # in a flight log: psi_meas rotated ~19deg across the decay
                # window while carrot's cross-track offset swung ~1.8m, right
                # after a wp advance with no real gate motion to justify it.
                # Anchoring to a fixed NED point and re-deriving the body-
                # frame bearing from it every tick (current pos_ned, current
                # attitude) is what set_live_target already does for the
                # tracker's own target — same fix, applied here.
                # Yaw-only 2D rotation throughout this cache (matching the
                # yaw-only convention the rest of this block already uses
                # for dir_n/dir_e below, and deliberately NOT the 3D
                # rotate_body_to_ned helper) — caching with the full 3D
                # rotation but reconstructing with yaw-only would introduce
                # a mismatch proportional to roll/pitch at capture time.
                _psi_v = self._gt_yaw_convention(quat_to_euler(quat)[2])
                _cp, _sp = np.cos(_psi_v), np.sin(_psi_v)
                if gate_id_match:
                    _bx, _by = float(pose.gate_bearing_body_m[0]), float(pose.gate_bearing_body_m[1])
                    _rel_ne = np.array([_cp * _bx - _sp * _by, _sp * _bx + _cp * _by])
                    self._last_pursuit_target_ne = pos_ned[:2] + _rel_ne
                    _t_body_v = pose.gate_bearing_body_m
                else:
                    _rel_ne = self._last_pursuit_target_ne - pos_ned[:2]
                    # Re-derive the equivalent body-frame bearing from the
                    # fixed NED target and the CURRENT attitude — the inverse
                    # of the rotation just above, so this reconstructs
                    # exactly what a fresh detection would report if the
                    # target were still visible at this pose.
                    _t_body_v = np.array([
                        _cp * _rel_ne[0] + _sp * _rel_ne[1],
                        -_sp * _rel_ne[0] + _cp * _rel_ne[1],
                    ])
                _horiz_v = float(np.sqrt(_t_body_v[0]**2 + _t_body_v[1]**2))
                if _horiz_v > 1.0:   # gate at least 1 m away horizontally
                    _dir_xb = _t_body_v[0] / _horiz_v   # body-forward component
                    _dir_yb = _t_body_v[1] / _horiz_v   # body-right  component
                    # Standard NED convention required here (see
                    # _gt_yaw_convention) — this rotates a body-frame bearing
                    # into NED, and GT's raw psi_meas convention would mirror
                    # the result instead of just rotating it wrong. Kept as
                    # its own yaw-only 2D rotation (not the 3D
                    # rotate_body_to_ned helper) — this override deliberately
                    # ignores roll/pitch for a horizontal-only reference,
                    # same as before. _cp/_sp already computed above (also
                    # needed there for case (b)'s inverse rotation).
                    _dir_n  =  _cp * _dir_xb - _sp * _dir_yb
                    _dir_e  =  _sp * _dir_xb + _cp * _dir_yb
                    # Rate-limit the pursuit direction itself (not just the
                    # trust_w blend weight below, which only scales the
                    # magnitude toward this same otherwise-unlimited
                    # direction) — see PursuitGuidance's docstring for why
                    # the raw bearing-following law overshoots on a large
                    # one-off correction.
                    _dir_n, _dir_e = self._pursuit_guidance.update(_dir_n, _dir_e, _actual_dt)
                    # Alignment-scaled pursuit speed: _dir_xb is cos(bearing
                    # angle off the nose), so this ramps the commanded speed
                    # from 0 (gate 90°+ off to the side) to full v_ref (gate
                    # dead ahead) instead of always commanding full v_ref in
                    # whatever direction the raw bearing points. Confirmed in
                    # a flight log this fixed-speed version has a fundamental
                    # instability: a lateral offset barely deflects the
                    # bearing angle at long range (weak correction) but
                    # deflects it sharply at close range (strong correction),
                    # so the loop's effective gain rises sharply as the gate
                    # is approached — this produced a growing, then-reversing
                    # multi-second oscillation on both N/E and (compounded by
                    # the vertical nudge riding along) D, independent of and
                    # not fixed by carrot-side tuning (lookahead_gain,
                    # corner_cut_frac), since those don't apply once pursuit
                    # dominates the reference. Scaling speed by alignment
                    # removes the sharp effective-gain-vs-range dependence:
                    # a large off-axis bearing now commands a gentle approach
                    # instead of a full-speed dash that has to be corrected
                    # again once the bearing swings past it.
                    _align = float(np.clip(_dir_xb, 0.0, 1.0))
                    _pursuit_speed = self.tracker.v_ref * _align
                    # Blend pursuit-at-gate against the carrot tracker's own
                    # cross-track-corrected N/E reference, weighted by the
                    # reacquisition trust ramp — right after a real vision
                    # loss this stays close to the carrot's tangent-following,
                    # self-correcting geometry instead of snapping straight at
                    # wherever the gate now appears (which produced the steep,
                    # unfavorable-angle approach after reacquisition). Ramps to
                    # full pursuit strength as trust rebuilds; already 1.0 during
                    # continuous tracking, so today's working behavior (G0/G1) is
                    # unchanged. Also capped by _path_conv_w: trust_w alone
                    # reaches 1.0 on a fixed vision-confidence schedule that
                    # doesn't know whether the drone has actually rejoined
                    # the direct line between gates yet — without this cap,
                    # a real cross-track offset from a BLIND stretch can
                    # still be present once trust_w says "fully trust
                    # vision," and gate-bearing pursuit would cut a
                    # diagonal straight at the gate instead of closing that
                    # offset first. Once back near the line, _path_conv_w
                    # is 1.0 and this is exactly today's behavior.
                    _w = min(self._vision_mode.trust_w, _path_conv_w)
                    v_ref_for_gains = np.array([
                        _w * _pursuit_speed * _dir_n + (1.0 - _w) * v_ned_ref_carrot[0],
                        _w * _pursuit_speed * _dir_e + (1.0 - _w) * v_ned_ref_carrot[1],
                        v_ned_ref_carrot[2],
                    ])
            else:
                # Pursuit not active this tick (COMMIT, trust_w decayed out,
                # or debug_waypoints_only) — drop the rate-limiter's cached
                # direction AND the bearing cache so neither can leak into
                # the next reacquisition or a later, unrelated BLIND stretch.
                self._pursuit_guidance.reset()
                self._last_pursuit_target_ne = None

            # Vertical assist (vision_mode.py): merges the old blind-search
            # descend nudge and vertical visual-centering nudge under one
            # owner with one bounded combination instead of two independent,
            # uncoordinated += onto the D reference. See VerticalAssist for
            # each sub-term's own trigger conditions (kept distinct — they
            # are not physically mutually exclusive).
            _dist_to_target = float(np.linalg.norm(
                self.tracker.waypoints[self.tracker.wp] - pos_ned))
            # centre_px gated the same way the old cx_det was — shared by the
            # vnudge sub-term below and the yaw blend further down: without
            # this, a lingering detection of the wrong gate during an
            # agi/tracker.wp desync would still bias yaw/altitude toward it
            # even with position/velocity correctly guarded. Also gated on
            # tracker.committed — a near-field bearing/pixel position is too
            # noise-sensitive to trust once committed to the final approach.
            _centre_px_gated = (
                None if (self._debug_waypoints_only or not gate_id_match
                         or self.tracker.committed)
                else pose.centre_px)
            _vassist_bias, _vassist_diag = self._vassist.update(
                mode, _dist_to_target, _centre_px_gated, _now, _actual_dt,
                self._debug_waypoints_only)
            v_ref_for_gains = v_ref_for_gains.copy()
            v_ref_for_gains[2] += _vassist_bias

            # Reacquisition speed cap: while still building trust
            # (REACQUIRING) close to the target, slow the horizontal
            # approach instead of closing in at full speed while the
            # pursuit-guidance correction above is still converging.
            # Confirmed in a flight log: the drone reached the gate
            # mid-correction (reacq_blend_w~0.6) and clipped the frame —
            # PursuitGuidance fixed HOW the correction arrives (smooth, not
            # oscillating) but not WHETHER there's enough time left to
            # finish it before impact; this buys that time back by trading
            # approach speed for it instead of direction smoothness.
            # Scoped to REACQUIRING only (not BLIND, which flies the
            # nominal path at full speed) and scaled by trust_w itself, so
            # it's bounded by the same vision_reacq_blend_tau-driven ramp,
            # not a separate unbounded slow-crawl — a persistently-lost
            # gate still gets flown at normal pace once mode drops to
            # BLIND.
            if (mode == Mode.REACQUIRING and not self._debug_waypoints_only
                    and _dist_to_target < self._reacq_speed_range_m):
                _speed_scale = (self._reacq_speed_min_frac + (1.0 - self._reacq_speed_min_frac)
                                 * self._vision_mode.trust_w)
                v_ref_for_gains[:2] *= _speed_scale

            # Publish transition/reacquisition-stage diagnostics for the
            # logger and the vision debug overlay (see log_carrot below and
            # vision_rx.py's _draw_overlay) — the full locked/transition/
            # search/reacquisition pipeline has enough interacting pieces
            # that they need to be visible together, live and in the log,
            # not just inferred after the fact from position traces.
            self.data['agi_matches_wp']        = gate_id_match
            self.data['reacq_blend_w']         = self._vision_mode.trust_w
            self.data['search_active']         = _vassist_diag['search_active']
            self.data['search_descend_offset'] = _vassist_diag['search_offset']
            self.data['vnudge_bias']           = _vassist_diag['vnudge_bias']

            # Visual yaw correction from orange centroid (bearing-only).
            # When the centroid is available, blend the carrot psi toward the
            # camera bearing so the drone points its nose at the detected
            # orange blob. Shares _centre_px_gated's guards above.
            if _centre_px_gated is not None:
                dx_px = float(_centre_px_gated[0]) - self.param.get('cam_cx', 320.0)
                # Standard NED convention required here (see _gt_yaw_convention)
                # — psi_ref_carrot below is always standard NED, and blending
                # it against a GT-convention _cur_yaw would corrupt it with a
                # mirrored delta instead of a correctly-signed one.
                _cur_yaw = self._gt_yaw_convention(quat_to_euler(quat)[2])
                psi_vis = _wrap_pi(_cur_yaw +
                                   np.arctan2(dx_px, self.param.get('cam_fx', 320.0)))
                dpsi_vis = _wrap_pi(psi_vis - psi_ref_carrot)
                _VIS_BLEND = 0.4   # weight of visual bearing vs carrot waypoint yaw
                # Scaled by the same reacquisition trust ramp as the velocity
                # pursuit blend above, so heading also stays tangent-following
                # right after a real vision loss instead of snapping toward
                # "point nose at gate."
                psi_ref_carrot = _wrap_pi(psi_ref_carrot
                                           + _VIS_BLEND * self._vision_mode.trust_w * dpsi_vis)

            # Rate-limit yaw reference
            dpsi = _wrap_pi(psi_ref_carrot - self._psi_cmd)
            self._psi_cmd = _wrap_pi(
                self._psi_cmd + np.clip(dpsi, -PSI_RATE_MAX * _actual_dt,
                                               PSI_RATE_MAX * _actual_dt))
        if not in_hover and self._carrot_active and self._logger is not None:
            self._logger.log_carrot(
                time_ms    = _now * 1e3,
                wp         = self.tracker.wp,
                v_cmd      = v_ref_for_gains,
                carrot_pos = self.tracker.carrot_pos,
                drone_pos  = pos_ned,
                mode       = self.data.get('vis_ctrl_mode', ''),
                agi_match      = self.data.get('agi_matches_wp', True),
                reacq_blend_w  = self.data.get('reacq_blend_w', 1.0),
                search_active  = self.data.get('search_active', False),
                search_offset  = self.data.get('search_descend_offset', 0.0),
                vnudge_bias    = self.data.get('vnudge_bias', 0.0),
            )

        # Advance carrot waypoint when the sim reports a gate passage through
        # the RACE_STATUS active_gate_index update.
        _agi_now      = int(self.data.get('active_gate_index', -1))
        _agi_advanced = _agi_now > self._last_seen_agi
        self._last_seen_agi = _agi_now
        if _agi_advanced:
            _trigger = f"RACE_STATUS agi={_agi_now}"
            ekf_pos = self.data.get('mav_state', {}).get('pos_ned')
            if ekf_pos is not None:
                pos_str = f"[{ekf_pos[0]:.6f}, {ekf_pos[1]:.6f}, {ekf_pos[2]:.6f}]"
            else:
                pos_str = 'unknown'
            print(f"[GATE PASSED {_trigger}] ekf_pos={pos_str}", flush=True)
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
                    # Keep active_gate_index in lockstep with tracker.wp instead
                    # of waiting for the sim's own RACE_STATUS message to catch
                    # up. Confirmed in a flight log: an active-gate-index-triggered
                    # advance can arrive before the next update cycle, leaving
                    # _agi_matches_wp (and vision_rx.py's own gate_info resolution,
                    # which reads this same shared value) stuck on the just-passed
                    # gate for a brief window. new_wp-1 >= agi always holds (from
                    # the max() above), so this only ever advances the index,
                    # never regresses it if RACE_STATUS already agrees or is ahead.
                    self.data['active_gate_index'] = new_wp - 1
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
                # See the immediate-advance branch above for why this is needed.
                self.data['active_gate_index'] = _new_wp - 1
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

    def _gt_yaw_convention(self, psi):
        """Flip between standard NED yaw (0=North, +CW towards East — the
        convention carrot_tracker.py and every psi_ref in this file uses)
        and the sim's GT-mode ATTITUDE convention (psi_gt = -psi_NED; GT's
        ATTITUDE yaw is not corrected in mavlink_rx.py, see _outer_loop's
        r_des sign note). A pure negation, so the same call converts either
        direction. Identity outside GT mode, where quat_to_euler(quat)[2] is
        already standard NED.

        Every quat_to_euler(quat)[2] read from mav_state's quaternion (i.e.
        every psi_meas) needs this before it's compared against or combined
        with a standard-convention angle like psi_ref — confirmed via a real
        GT flight log that skipping it drives the true heading to -psi_ref
        (a left/right mirror of the intended direction) while the raw
        logged error still reads ~0, since the loop faithfully converges
        its own (mis-defined) error term. Found and fixed three call sites
        that needed this in one pass (_outer_loop's e_psi, the vision-
        bearing velocity override, and the visual-yaw-blend-from-centroid).

        Same underlying issue also turned up in vision_rx.py, which builds
        rotation matrices straight from mav_state['quat'] with no GT
        awareness at all — see rotations.gt_yaw_flip/gt_correct_quat (this
        method is now a thin wrapper around gt_yaw_flip). Grep for those two
        names, not just this method, before adding a fourth site.
        """
        return rotations.gt_yaw_flip(psi, self._gt_mode)

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
        #
        # psi_ref (from carrot_tracker, atan2(ea_E, ea_N)) is always standard-
        # convention NED — it has no GT awareness. Convert it into psi_meas's
        # convention (see _gt_yaw_convention) before using it below, so both
        # the error term and this function's own psi_ref_deg debug log are
        # self-consistent with psi_meas. Confirmed against a real GT flight
        # log: without this, logged psi_err tracked ~0 throughout (the loop
        # faithfully converges its own error term) while the actual camera
        # view was consistently offset to the mirror-opposite side of the gate.
        psi_ref = self._gt_yaw_convention(psi_ref)

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
        # D (altitude) reference is ramped too, same as N/E, so vD_ref starts at
        # 0 on the first carrot-activation tick instead of stepping straight to
        # whatever the carrot/tracker hands over. Kp_vD/Ki_vD themselves stay
        # unramped (see below) so the vertical loop still has full authority to
        # track that ramping target from tick 1 — liftoff isn't delayed, only the
        # target it's chasing eases in.
        v_ned_ref = v_ned_ref.copy()
        v_ned_ref[0] *= _att_ramp
        v_ned_ref[1] *= _att_ramp
        v_ned_ref[2] *= _att_ramp

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

        # ── Derivative term (derivative-on-measurement) ─────────────────────────
        # d(v_meas_filt)/dt, not d(error)/dt: differentiating the filtered
        # measurement (rather than v_ref - v_meas) means a stepped v_ned_ref
        # (waypoint transitions, mode switches) doesn't inject a derivative
        # kick — only actual velocity change feeds the D term.
        if self._v_meas_filt_prev is None:
            self._v_meas_filt_prev = v_ned_meas_filt.copy()
        dv_meas = (v_ned_meas_filt - self._v_meas_filt_prev) / max(dt, 1e-4)
        self._v_meas_filt_prev = v_ned_meas_filt.copy()

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
        # D axis uses its own gain (K_dv_D) — vD's dynamics/noise floor differ from
        # N/E, so it needs independent tuning rather than sharing K_dv.
        if self._K_dv > 0.0 or self._K_dv_D > 0.0:
            _dv = np.clip((v_ned_ref - self._v_ref_prev) / max(dt, 0.001), -20.0, 20.0)
            _a_lag_N = self._K_dv * _dv[0]
            _a_lag_E = self._K_dv * _dv[1]
            _a_lag_D = self._K_dv_D * _dv[2]
        else:
            _a_lag_N = _a_lag_E = _a_lag_D = 0.0
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
               - self._Kd_vN * dv_meas[0]
               + self._ff_vN * v_ned_ref[0] + _a_lag_N)
        a_E = (_Kp_vE_eff * e_vel_c[1] + self._Ki_vE * self.xi_vel[1]
               - self._Kd_vE * dv_meas[1]
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
        a_z = (_Kp_vD_eff * e_vel_c[2] + self._Ki_vD * self.xi_vel[2]
               - self._Kd_vD * dv_meas[2] + _a_lag_D)
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
        _r_cmd   = float(-r_des if self._gt_mode else r_des)
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




# Quaternion → Euler angles (roll, pitch, yaw) — see rotations.py.
quat_to_euler = rotations.quat_to_euler
