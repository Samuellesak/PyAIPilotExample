import dataclasses
import math
import queue
import socket
import struct
import threading
import time
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO as _YOLO

from dyn import load_params
import rotations
import pose_disambiguation
import gate_lock
from pose_estimate import LockState, PoseEstimate

SIM_SERVER_UDP_IP   = "0.0.0.0"
SIM_SERVER_UDP_PORT = 5600


class VisionRX:

    def __init__(self, data, logger=None):
        self.data   = data
        self.logger = logger

        _p = load_params("params.yaml")
        # GT mode's mav_state['quat'] carries the sim's raw ATTITUDE yaw
        # convention (psi_gt = -psi_NED, not standard NED — see
        # rotations.gt_yaw_flip/gt_correct_quat's module docstring). Cached
        # once so every rotation built from that quaternion in this file can
        # be corrected consistently.
        self._gt_mode = bool(_p.get('ground_truth_mode', False))
        self._cam_K = np.array([
            [_p.get('cam_fx', 320.), 0.,                   _p.get('cam_cx', 320.)],
            [0.,                   _p.get('cam_fy', 320.), _p.get('cam_cy', 240.)],
            [0.,                   0.,                     1.                    ],
        ], dtype=np.float64)
        self._ekf_vis_sigma     = float(_p.get('ekf_vision_sigma',     0.5))
        self._ekf_vis_sigma_k   = float(_p.get('ekf_vision_sigma_range_k', 0.08))
        self._ekf_vis_gate      = float(_p.get('ekf_vision_gate',      10.0))
        # track_gates_ned's raw NED marks the gate's bottom beam, not the opening
        # centre — see the correction applied where gate_info is built from it below.
        self._gate_beam_to_center_d = float(_p.get('gate_beam_to_center_offset_m', 1.0))
        self._ekf_vis_vel_sigma = float(_p.get('ekf_vision_vel_sigma', 1.0))
        self._ekf_vis_vel_gate  = float(_p.get('ekf_vision_vel_gate',  5.0))
        self._ekf_vis_yaw_sigma = float(_p.get('ekf_vision_yaw_sigma', 0.1))
        self._ekf_vis_yaw_gate  = float(_p.get('ekf_vision_yaw_gate',  1.0))
        self._ekf_vis_att_sigma = float(_p.get('ekf_vision_att_sigma', 0.15))
        self._ekf_vis_att_gate  = float(_p.get('ekf_vision_att_gate',  5.0))
        # Range-scaled, mirroring ekf_vision_sigma_range_k's exact pattern for
        # position. Confirmed in a flight log: attitude/yaw error consistently
        # grows during the long-range, mid-approach portion of a segment and
        # self-corrects once the gate is close (short range) — the same
        # low-SNR-at-range effect already motivating position's range-scaled
        # sigma, just not yet applied to the rotation solve extracted from the
        # same corner detections. 0 = disabled (flat sigma at all ranges,
        # today's behaviour) until a fitted value is available.
        self._ekf_vis_yaw_sigma_k = float(_p.get('ekf_vision_yaw_sigma_range_k', 0.0))
        self._ekf_vis_att_sigma_k = float(_p.get('ekf_vision_att_sigma_range_k', 0.0))
        # Minimum YOLO confidence to accept a detection — rejects low-confidence
        # false positives (background clutter mistaken for a gate) from ever
        # reaching PnP/EKF.
        self._conf_min          = float(_p.get('vision_conf_min',      0.6))
        # Fallback gate size used for PnP when track data hasn't arrived yet.
        # Gives a distance readout and yaw estimate even during the WAIT phase.
        self._gate_w_default    = float(_p.get('gate_width_default',   2.5))
        self._gate_h_default    = float(_p.get('gate_height_default',  2.5))
        # Fallback gate NED positions: wp[0]=start, wp[1]=gate0, wp[2]=gate1, …
        # Used for EKF position/velocity updates when track data never arrives from sim.
        self._waypoints = _p.get('waypoints', None)

        # Blue-disturbance suppression: zero out blue pixels before YOLO inference.
        # The 41×41 dilation lets blue bleed into the YOLO input through the mask
        # boundary — suppressing it at source prevents keypoint confusion.
        self._debug_waypoints_only = bool(_p.get('debug_waypoints_only', False))
        # Separate opt-in from debug_waypoints_only: controller.py gates all of
        # its OWN vision-influenced control paths (gate-NED override, bearing
        # override, visual yaw blend) behind its own debug_waypoints_only check
        # independent of whether real detection data exists — so running YOLO/
        # PnP here for logging (ekf_shadow.py's vision_fix.csv) doesn't change
        # flight behavior at all as long as controller.py's flag stays true.
        self._vision_detect_for_logging = bool(_p.get('vision_detect_for_logging', False))
        if self._debug_waypoints_only and not self._vision_detect_for_logging:
            print("[VisionRX] debug_waypoints_only=true — YOLO disabled", flush=True)
        elif self._debug_waypoints_only:
            print("[VisionRX] debug_waypoints_only=true but vision_detect_for_logging=true "
                  "— YOLO/PnP still runs for logging; controller.py ignores it as before",
                  flush=True)

        self._yolo_enabled = bool(_p.get('yolo_enabled', True))
        if not self._yolo_enabled:
            print("[VisionRX] yolo_enabled=false — YOLO inference skipped", flush=True)

        self._suppress_blue = bool(_p.get('suppress_blue', True))
        self._blue_lo = np.array([
            int(_p.get('suppress_blue_h_lo', 100)),
            int(_p.get('suppress_blue_s_lo',  80)),
            int(_p.get('suppress_blue_v_lo',  50)),
        ], dtype=np.uint8)
        self._blue_hi = np.array([
            int(_p.get('suppress_blue_h_hi', 140)),
            255, 255,
        ], dtype=np.uint8)

        # Gate distance lock: require N consecutive PnP frames before trusting
        # the distance, then reject any measurement farther than the lock + tolerance.
        # This drops spurious detections of more distant gates that spike the distance.
        self._max_gate_dist = float(_p.get('vision_max_gate_dist', 50.0))
        self._v_ref         = float(_p.get('v_ref', 2.0))
        self._vel_max_factor = float(_p.get('vision_vel_max_factor', 2.5))
        self._lock_frames   = int(_p.get('vision_lock_frames', 5))
        self._lock_miss_max = int(_p.get('vision_lock_miss',   30))
        self._spike_tol     = float(_p.get('vision_spike_tol', 5.0))
        # "Largest visible box wins" (below) assumes the closest visible gate
        # IS the current target. That breaks the instant the current gate goes
        # out of view while the NEXT gate (farther down-track) is still
        # visible: with no valid current-gate box, the next gate's box becomes
        # "largest" by default and gets misattributed as the current gate.
        # gate_info['ned'] (known from track data, independent of vision) plus
        # mav_state's own position estimate already give an expected range to
        # the ACTUAL current gate at zero extra cost — reject a detection
        # whose measured range disagrees with that by more than ordinary
        # EKF/PnP noise would explain, rather than feeding a misidentified
        # detection into lock/EKF/bearing override. Default is well under
        # typical gate-to-gate spacing (~20m+ on this track) so it can't
        # confuse adjacent gates for each other, but generous enough to
        # tolerate EKF position error during a blind spell. Applied at lock
        # acquisition on every frame, and periodically (every
        # vision_gate_id_recheck_interval_s) while already locked — see the
        # check's own site for why the acquisition-only version wasn't
        # enough: a wrong-gate lock that happens to pass the one-time
        # acquisition check (e.g. right at a gate handoff, before the
        # drone has moved far past the previous gate) was then tracked
        # indefinitely, since vision_spike_tol only checks self-consistency
        # against the lock's own history, not correctness against the real
        # target. Confirmed in a flight log: ground-truth position (not just
        # the EKF estimate) diverged in the wrong lateral direction for an
        # entire approach following a gate handoff. The periodic recheck
        # uses the same trusting-mav_state['pos_ned'] logic as acquisition,
        # so it carries the same self-reinforcing-lockout risk from EKF
        # drift in principle — but a genuine wrong-gate lock is off by tens
        # of metres (gate-to-gate spacing), while EKF drift measured this
        # session tops out around a few metres, so the existing tolerance
        # discriminates the two cleanly without needing to be tightened.
        # Lock/confidence state machine (gate_lock.py) — UNLOCKED -> ACQUIRING
        # -> LOCKED, with the gate-identity plausibility recheck (comment
        # above explains why it exists and why it's periodic, not just at
        # acquisition) as a guarded transition inside the machine rather
        # than a side channel that mutates lock state from outside it.
        self._lock = gate_lock.GateLock(
            lock_frames=self._lock_frames, miss_max=self._lock_miss_max,
            spike_tol_m=self._spike_tol,
            id_tol_m=float(_p.get('vision_gate_id_tol_m', 10.0)),
            id_recheck_period_s=float(_p.get('vision_gate_id_recheck_interval_s', 1.5)),
        )
        self._prev_agi      = None   # previous active_gate_index for change detection
        # Layer1->Layer2 pose contract (pose_estimate.py). Holds the last
        # ACCEPTED full pose payload so it can be republished (fresh=False)
        # across a tolerated-miss tick while self._lock is still ACQUIRING/
        # LOCKED — the relocated replacement for controller.py's old
        # _vis_tvec_hold cache, now owned by the layer that actually knows
        # whether a value is still trustworthy. Cleared back to .blind()
        # whenever self._lock drops to UNLOCKED.
        self._last_pose = PoseEstimate.blind()
        # gate_index → last gate_info dict received from track data (includes ned, width, height, quat).
        # Used as fallback when track_gates_ned is temporarily unavailable.
        self._last_gate_info = {}

        # Image preprocessing: unsharp mask (sharpening) + optional Gaussian denoise.
        # Sharpening restores orange saturation lost to motion blur at speed, improving
        # HSV detection rate without widening the colour gate (no false-positive increase).
        self._sharpen_k     = int(_p.get('preproc_sharpen_k',      5))
        self._sharpen_alpha = float(_p.get('preproc_sharpen_alpha', 0.8))
        self._gauss_k       = int(_p.get('preproc_gauss_k',         0))

        # cam→FRD body with upward tilt; cam_tilt_deg from params.yaml (positive = nose up)
        self._R_cam2body = rotations.cam_to_body_matrix(_p.get('cam_tilt_deg', 20.0))

        # PnP velocity estimation: sliding window of (wall_t, t_gate_body, gyro_q)
        # tuples — BODY-frame drone->gate vector (fixed R_cam2body only, no live
        # attitude), not world-NED position. See the regression site for why body
        # frame, and for why a gyro-integrated quaternion is buffered alongside it.
        # Averaging over N frames reduces differentiation noise by ~N× vs 2-frame diff.
        _vel_win = int(_p.get('vision_vel_window_frames', 10))
        self._pnp_vel_buf = deque(maxlen=max(2, _vel_win))
        # Parallel buffer of (wall_t, drone_ned) — the already-computed absolute
        # NED position, one attitude-rotation per SAMPLE rather than one per
        # regression. See the combination site below for why this is run
        # alongside (not instead of) the body-frame buffer above.
        self._pnp_vel_ned_buf = deque(maxlen=max(2, _vel_win))
        self._vel_cross_check_tol = float(_p.get('vision_vel_cross_check_tol', 2.0))
        # Minimum sample count / time span before a regression is trusted at
        # all. Briefly lowered to 5/0.15s to reduce latency, on the theory that
        # the per-tick dynamic sigma below (_ols_slope_sigma) would tell the
        # EKF how much to trust a shorter, noisier fit instead of vision_rx
        # pre-smoothing to a single fixed noise level via window size alone.
        # Reverted after a real crash traced this exact combination: replaying
        # the raw per-frame drone_ned samples from that flight (ordinary PnP
        # scatter, ~0.3-0.4m, no bad samples) through a 5-9 sample/~0.15-0.27s
        # window reproduced a spurious -3 to -4 m/s slope from what was
        # actually a small, smooth wiggle in position — a short window's OLS
        # slope is highly sensitive to exactly this kind of ordinary noise
        # shape, and the fit's own residuals stayed small throughout (a
        # straight line still fits a short wiggle "well"), so the dynamic
        # sigma never flagged it. That spurious velocity got fed to the EKF
        # near the floor sigma, and directly correlates with the onset of a
        # real, GT-confirmed roll divergence that crashed the drone into a
        # gate's inner beam. The dynamic sigma still helps characterize
        # residual scatter on an adequately-sized window; it does not, on its
        # own, substitute for the window being long enough that ordinary
        # position noise averages out rather than dominating the fit — see
        # this file's vision_vel_window_frames param for the Monte-Carlo data
        # this reversion restores the margin from.
        self._vel_min_frames = int(_p.get('vision_vel_min_frames', 8))
        self._vel_min_span_s = float(_p.get('vision_vel_min_span_s', 0.3))
        # Sanity cap only — real de-rotation (below) handles ordinary flight
        # rotation, this just bails out of the small-angle regime for genuinely
        # extreme spins where first-order quaternion integration breaks down.
        self._vel_max_rotation_rad = math.radians(
            float(_p.get('vision_vel_max_rotation_deg', 90.0)))
        # Gyro-only dead-reckoned attitude, purely for de-rotating buffered
        # samples into the current body frame before regressing (see below) —
        # deliberately NOT the EKF's own attitude, so this can't inherit any of
        # the attitude-estimation-error bugs fixed elsewhere this session. Gyro
        # bias drifts over tens of seconds, but the velocity window only spans
        # ~1s, so bias contributes a negligible rotation error over that span.
        self._gyro_q      = np.array([1.0, 0.0, 0.0, 0.0])   # wxyz, body->some-fixed-ref
        self._gyro_q_last_t = None
        # Detects a hover-reset (imu_ekf.py sets pos_offset_ned exactly once,
        # at reset) so the velocity buffer/gyro tracker can be cleared then —
        # see the reset-detection block in process_frame for why: the LEVEL
        # phase snaps the drone from its resting ramp tilt to level right at
        # reset (confirmed in a flight log: theta changed ~9.5deg in one 66ms
        # tick, gy~3.9 rad/s), too fast for gyro_q's ~30fps-sampled first-order
        # integration to track accurately, leaking a residual rotation error
        # into any buffered sample that straddles the transient. Resetting
        # the buffer here means it can only ever contain post-transient,
        # already-settled samples.
        self._last_pos_offset_ned = None

        # Shared disambiguation policy (pose_disambiguation.py) — see that
        # module's docstring for the full rationale. Neither _pnp_gate's
        # IPPE 2-solution ambiguity nor _attitude_from_pnp's 4-fold
        # corner-labelling ambiguity is ever resolved against the EKF's own
        # live attitude any more (that was self-referential and confirmed
        # to produce a wrong pick that then drifted the EKF further next
        # frame, in two separate incidents this session at two different
        # layers) — only against an independent position anchor
        # (mav_state['pos_ned']) when available, or rate-limited continuity
        # against THIS PIPELINE's own last-accepted rotation otherwise.
        self._pnp_continuity_max_rate = float(
            _p.get('vision_pnp_continuity_max_rate', 3.0))          # [rad/s]
        self._pos_degenerate_tol_m = float(
            _p.get('vision_pose_position_degenerate_tol_m', 1.0))   # [m]

        # Landmark aid: when a detection fails the current-target identity
        # check, see if it matches a course-sequence NEIGHBOUR gate instead
        # (self._waypoints) and if so use it as a position-only EKF fix —
        # see _landmark_gate_fix. Tolerance is deliberately tighter than the
        # primary gate-identity check (vision_gate_id_tol_m): this is a
        # confirmatory cross-check against the full static layout, not the
        # single expected-range comparison the primary check makes, so a
        # loose match here risks anchoring the EKF to the wrong gate
        # entirely. Sigma is inflated on top of the normal range-scaled
        # sigma to reflect that extra cross-gate-identity uncertainty.
        self._landmark_aid_enabled  = bool(_p.get('vision_landmark_aid_enabled', True))
        self._landmark_match_tol_m  = float(_p.get('vision_landmark_match_tol_m', 5.0))
        self._landmark_sigma_mult   = float(_p.get('vision_landmark_sigma_mult', 2.0))

        # _pnp_gate's continuity anchors: last-accepted rotation (+ its
        # timestamp, for the rate-limit above) per target. Kept separate per
        # target — the primary/current-gate call, the next-gate-candidate
        # call, and the landmark-aid call each need their own anchor, or
        # they'd clobber each other's state every frame more than one fires.
        self._last_pnp_R_primary   = None
        self._last_pnp_R_primary_t = None
        self._last_pnp_R_next      = None
        self._last_pnp_R_next_t    = None
        self._last_pnp_R_landmark   = None
        self._last_pnp_R_landmark_t = None

        # _attitude_from_pnp's continuity anchor. Deliberately NOT cleared on
        # an active_gate_index change (unlike the two anchors above) — those
        # are gate-relative (R_gate2cam), but this one is the drone's own
        # attitude, which doesn't become invalid just because the tracked
        # gate index advanced; clearing it would only force an unnecessary
        # fallback to the coarser cold-start reference more often than
        # needed. Same reasoning exempts it from the gate-identity-recheck
        # reset below: a wrong-gate lock doesn't imply a wrong attitude
        # either. Cold-start reference is the drone's known spawn heading
        # (initial_yaw_deg), not identity — a fixed, physically-motivated
        # reference that never changes in response to this pipeline's own
        # past output, so it can't self-reinforce the way scoring against a
        # live estimate does.
        self._last_pnp_att_R = None
        self._last_pnp_att_t = None
        self._att_nominal_R  = rotations.quat_to_R_body2ned(rotations.euler_to_quat(
            0.0, 0.0, math.radians(float(_p.get('initial_yaw_deg', 0.0))),
        ))

        # PnP yaw smoothing: same rationale as the velocity window above, but for
        # yaw_ned. A single bad-frame yaw reading (oblique angle / keypoint noise)
        # used to go straight into update_yaw() with a tight sigma (0.1 rad) and a
        # loose gate (1.0 rad ≈ 57°), so the EKF absorbed it almost at full
        # confidence. Confirmed in a flight log: the very first post-settle-window
        # yaw update (t=vision_settle_sec after reset, to the tick) carried a ~27°
        # error, which the tight sigma let through, corrupting psi — and since
        # position is computed by rotating tvec through this same (now-bad) yaw
        # every subsequent frame, the ~9m position corruption that followed was a
        # downstream symptom of the same single bad yaw sample, not a separate bug.
        # Circular-mean over a short window smooths single-frame noise, and a frame
        # that disagrees with that mean by more than the outlier threshold is
        # dropped rather than fed to the EKF.
        _yaw_win = int(_p.get('vision_yaw_window_frames', 5))
        self._pnp_yaw_buf = deque(maxlen=max(2, _yaw_win))
        self._yaw_outlier_rad = math.radians(float(_p.get('vision_yaw_outlier_deg', 12.0)))

        # Next-gate candidate: accumulate PnP-derived NED positions of the second-largest
        # YOLO detection across many frames.  Confirmed position = median of buffer.
        # Buffer clears on gate-index change to discard stale measurements.
        self._next_gate_min_frames = int(_p.get('next_gate_min_frames', 15))
        self._next_gate_ned_buf    = deque(maxlen=self._next_gate_min_frames)
        self._next_gate_ned        = None    # median NED once buffer is full, else None

        # Live debug overlay window (enabled via vision_debug_overlay: true in params.yaml).
        self._debug_overlay     = bool(_p.get('vision_debug_overlay', False))
        self._overlay_last_ctr  = None   # last known gate centre_px for hold/transition display

        # Diagnostics counters (reset every 5 s)
        self._stat_recv           = 0
        self._stat_proc           = 0
        self._stat_det            = 0
        self._stat_pnp            = 0     # frames with a valid PnP position fix
        self._stat_dist_sum       = 0.0   # cumulative PnP distance [m]
        self._stat_t0             = time.time()
        self._stat_pnp_skip_reason = None

        # Frame queue between receive thread and inference thread.
        # maxsize=1: inference always gets the most recent frame; older ones are dropped.
        self._frame_q = queue.Queue(maxsize=1)

        # Guards against reprocessing the same camera frame twice (UDP can
        # duplicate a packet/chunk-set, or the sim can resend one). frame_id
        # is a monotonically increasing counter, so anything <= the last
        # queued id is a dup or stale reorder. Confirmed in a flight log: the
        # same frame_id (bit-identical keypoints and tvec) was processed
        # twice ~33ms apart; drone_ned is computed from live mav_state, which
        # had drifted between the two passes, so the identical PnP solve
        # produced two different positions — an 8m position (and matching
        # PnP-velocity) jump the EKF absorbed as a real, sudden motion.
        self._last_queued_frame_id = -1

        self._model = _YOLO("YOLO/best3.pt")
        self._model.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
        print("[VisionRX] YOLO model loaded and warmed up.", flush=True)

        self.is_running = True
        self._recv_thread  = threading.Thread(target=self._recv_loop,  daemon=False, name="vision-recv")
        self._infer_thread = threading.Thread(target=self._infer_loop, daemon=False, name="vision-infer")
        self._recv_thread.start()
        self._infer_thread.start()

    def get_thread_for_join(self):
        self.is_running = False
        # Wake the inference thread so it sees is_running=False without waiting 0.5 s
        try:
            self._frame_q.put_nowait(None)
        except queue.Full:
            pass
        self._infer_thread.join(timeout=2.0)
        return self._recv_thread

    # ── Receive loop (lightweight — just UDP reassembly) ───────────────────

    def _recv_loop(self):
        header_format = "<IHHIIQ"
        header_sz     = struct.calcsize(header_format)
        frames        = {}

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(0.5)
        sock.bind((SIM_SERVER_UDP_IP, SIM_SERVER_UDP_PORT))
        print("Listening for camera frames...", flush=True)

        while self.is_running:
            try:
                packet, _ = sock.recvfrom(65536)
            except socket.timeout:
                continue

            header  = packet[:header_sz]
            payload = packet[header_sz:]
            frame_id, chunk_id, total_chunks, jpeg_size, payload_size, sim_time_ns = \
                struct.unpack(header_format, header)

            if frame_id not in frames:
                frames[frame_id] = {"chunks": {}, "total": total_chunks,
                                     "sim_time_ns": sim_time_ns}
            frames[frame_id]["chunks"][chunk_id] = payload

            if len(frames[frame_id]["chunks"]) == total_chunks:
                jpeg_bytes = bytearray()
                ok = True
                for i in range(total_chunks):
                    if i not in frames[frame_id]["chunks"]:
                        ok = False; break
                    jpeg_bytes.extend(frames[frame_id]["chunks"][i])

                if ok:
                    img = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8),
                                       cv2.IMREAD_COLOR)
                    if img is not None and frame_id > self._last_queued_frame_id:
                        self._last_queued_frame_id = frame_id
                        self._stat_recv += 1
                        frame_ts = frames[frame_id]["sim_time_ns"]
                        # Stamp vis_ctrl_mode at capture time, not at whatever
                        # time _infer_loop eventually gets around to processing
                        # this frame. Inference lags capture by up to ~170ms
                        # under queueing backlog (confirmed in flight logs), and
                        # controller.py flips vis_ctrl_mode asynchronously on its
                        # own control-loop thread. Reading it live inside
                        # process_frame let a frame captured mid-TRANSITION be
                        # evaluated after the mode had already flipped back to
                        # CARROT, defeating the TRANSITION suppression below and
                        # letting a bad detection (small gate seen inside the
                        # near gate's beams) through as a real PnP fix.
                        ctrl_mode_at_capture = self.data.get('vis_ctrl_mode')
                        # Drop oldest frame if inference is lagging; keep latest.
                        if self._frame_q.full():
                            try:
                                self._frame_q.get_nowait()
                            except queue.Empty:
                                pass
                        try:
                            self._frame_q.put_nowait((frame_id, img, frame_ts, ctrl_mode_at_capture))
                        except queue.Full:
                            pass

                del frames[frame_id]

            # Evict stale incomplete frames (frame_id much older than current)
            if len(frames) > 30:
                oldest = min(frames)
                del frames[oldest]

    # ── Inference loop (YOLO + PnP — runs at whatever speed GPU/CPU allows) ──

    def _infer_loop(self):
        while self.is_running:
            try:
                item = self._frame_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:   # shutdown sentinel
                break
            frame_id, img, sim_time_ns, ctrl_mode_at_capture = item
            self.process_frame(frame_id, img, sim_time_ns, ctrl_mode_at_capture)

    # ── Helpers ────────────────────────────────────────────────────────────

    def _apply_blue_suppression(self, img_bgr):
        """Replace blue/cyan pixels with grey (equal B=G=R=luminance).
        Avoids HSV-round-trip artefacts by operating directly on BGR channels."""
        if not self._suppress_blue:
            return img_bgr
        hsv     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        blue_px = cv2.inRange(hsv, self._blue_lo, self._blue_hi)
        if not np.any(blue_px):
            return img_bgr
        grey     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        grey_3ch = cv2.merge([grey, grey, grey])
        return np.where(blue_px[:, :, np.newaxis] > 0,
                        grey_3ch, img_bgr).astype(img_bgr.dtype)

    def _orange_mask(self, img_bgr):
        # img_bgr has blue/cyan pixels desaturated (S=0) upstream.
        # S threshold lowered 100→80 to catch gate pixels mixed with cyan interference.
        hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        lo   = np.array([ 5,  80,  80], dtype=np.uint8)
        hi   = np.array([25, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lo, hi)
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask    = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
        return cv2.bitwise_and(img_bgr, img_bgr, mask=mask), mask

    def _orange_centroid(self, mask):
        """
        Find the centroid of the largest orange blob in the tight mask.
        Returns (cx, cy, area) in pixels, or (None, None, 0) if no blob found.
        Used as a low-cost fallback when YOLO fails: the centroid gives the
        horizontal/vertical bearing to the gate center.
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None, 0
        best = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(best)
        if area < 50:   # too small — noise
            return None, None, 0
        M = cv2.moments(best)
        if M['m00'] < 1e-6:
            return None, None, 0
        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']
        return float(cx), float(cy), float(area)

    def _preprocess(self, img):
        """Unsharp-mask sharpening + optional Gaussian denoise before orange masking."""
        if self._sharpen_k > 0:
            k = self._sharpen_k | 1   # ensure odd
            blur = cv2.GaussianBlur(img, (k, k), 0)
            img  = cv2.addWeighted(img, 1.0 + self._sharpen_alpha,
                                   blur, -self._sharpen_alpha, 0)
        if self._gauss_k > 0:
            gk = self._gauss_k | 1
            img = cv2.GaussianBlur(img, (gk, gk), 0)
        return img

    def _draw_overlay(self, img, tvec_cam, centre_px, corners, vel_ned_pnp,
                       pnp_skip_reason=None, gate_id_rejected=False):
        """Draw live debug overlay onto a copy of img and return it."""
        vis = img.copy()
        h, w = vis.shape[:2]
        ic_x, ic_y = w // 2, h // 2   # image centre

        ctrl_mode = self.data.get('vis_ctrl_mode', 'CARROT')

        # ── Mode badge ────────────────────────────────────────────────────────
        # Mode names/values from vision_mode.py's Mode enum — TRANSITION is
        # the one wire value deliberately kept unchanged from before the
        # gate-transition rewrite (see Mode.COMMIT's docstring); the others
        # are free-form names (TRACKING/REACQUIRING/BLIND/RECOVERY). Falls
        # back gracefully to an unrecognized-string display for anything else.
        _MODE_CFG = {
            'TRACKING':    ((0,  200,  0),  'TRACK'),
            'REACQUIRING': ((0,  200, 220), 'REACQ'),
            'TRANSITION':  ((30, 140, 255), 'TRANSIT'),
            'BLIND':       ((120,120, 120), 'BLIND'),
            'RECOVERY':    ((0,   0, 220),  'RECOVER'),
        }
        badge_color, badge_label = _MODE_CFG.get(ctrl_mode, ((100,100,100), ctrl_mode))
        cv2.rectangle(vis, (5, 5), (165, 38), badge_color, -1)
        cv2.putText(vis, badge_label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

        # ── Lock status ───────────────────────────────────────────────────────
        det     = self.data.get('gate_detection', {})
        locked  = det.get('pnp_locked', False)
        lock_lbl = 'LOCKED' if locked else 'SEARCHING'
        lock_col = (0, 200, 0) if locked else (60, 60, 200)
        cv2.putText(vis, lock_lbl, (5, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, lock_col, 2, cv2.LINE_AA)

        # ── PnP velocity status ───────────────────────────────────────────────
        if vel_ned_pnp is not None:
            vel_mag  = float(np.linalg.norm(vel_ned_pnp[:2]))   # horizontal only
            vel_lbl  = f'VEL PNP  {vel_mag:.1f} m/s'
            vel_col  = (0, 220, 0)
        else:
            vel_lbl = 'VEL PNP  --'
            vel_col = (120, 120, 120)
        cv2.putText(vis, vel_lbl, (5, 88),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, vel_col, 2, cv2.LINE_AA)

        # ── Distance ──────────────────────────────────────────────────────────
        if tvec_cam is not None:
            dist_m   = float(tvec_cam[2])
            dist_lbl = f'DIST  {dist_m:.1f} m'
            dist_col = (0, 220, 0)
        else:
            dist_lbl = 'DIST  --'
            dist_col = (120, 120, 120)
        cv2.putText(vis, dist_lbl, (5, 114),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, dist_col, 2, cv2.LINE_AA)

        # ── Next-gate status ──────────────────────────────────────────────────
        ng_ned    = self.data.get('next_gate_ned')
        ng_frames = self.data.get('next_gate_buf_frames', 0)
        ng_min    = self.data.get('next_gate_min_frames', self._next_gate_min_frames)
        if ng_ned is not None:
            ng_lbl = 'NEXT GATE: CONFIRMED'
            ng_col = (0, 220, 0)
        elif ng_frames > 0:
            ng_lbl = f'NEXT GATE: {ng_frames}/{ng_min}'
            ng_col = (0, 200, 220)
        else:
            ng_lbl = 'NEXT GATE: --'
            ng_col = (120, 120, 120)
        cv2.putText(vis, ng_lbl, (5, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, ng_col, 2, cv2.LINE_AA)

        # ── Transition/reacquisition pipeline diagnostics ───────────────────
        # Surfaces every stage added this session (locked -> transition ->
        # search/fallback -> reacquisition) in one place, live, so a failure
        # mode doesn't need post-flight log archaeology to spot. Controller-
        # side fields (agi_matches_wp, reacq_blend_w, search/vnudge state) are
        # published to self.data each control tick; skip_reason/
        # gate_id_rejected are this vision frame's own local diagnostics,
        # passed in directly.
        _y = 166
        _agi_ok = self.data.get('agi_matches_wp', True)
        agi_lbl = 'AGI/WP: OK' if _agi_ok else 'AGI/WP: MISMATCH'
        agi_col = (0, 220, 0) if _agi_ok else (0, 0, 220)
        cv2.putText(vis, agi_lbl, (5, _y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, agi_col, 2, cv2.LINE_AA)
        _y += 24

        _reacq = float(self.data.get('reacq_blend_w', 1.0))
        reacq_lbl = f'REACQ BLEND: {_reacq:.2f}'
        reacq_col = (0, 220, 0) if _reacq > 0.99 else (30, 140, 255)
        cv2.putText(vis, reacq_lbl, (5, _y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, reacq_col, 2, cv2.LINE_AA)
        _y += 24

        if self.data.get('search_active', False):
            search_lbl = f'SEARCH: {self.data.get("search_descend_offset", 0.0):+.2f}m'
            search_col = (0, 200, 220)
        else:
            search_lbl = 'SEARCH: off'
            search_col = (120, 120, 120)
        cv2.putText(vis, search_lbl, (5, _y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, search_col, 2, cv2.LINE_AA)
        _y += 24

        _vnudge = float(self.data.get('vnudge_bias', 0.0))
        if abs(_vnudge) > 1e-3:
            vnudge_lbl = f'VNUDGE: {_vnudge:+.2f}m/s'
            vnudge_col = (0, 200, 220)
        else:
            vnudge_lbl = 'VNUDGE: off'
            vnudge_col = (120, 120, 120)
        cv2.putText(vis, vnudge_lbl, (5, _y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, vnudge_col, 2, cv2.LINE_AA)
        _y += 24

        if gate_id_rejected:
            _lm_wp = self.data.get('landmark_fix_wp')
            _reject_lbl = (f'ID REJECT: landmark fix wp {_lm_wp}' if _lm_wp is not None
                           else 'ID REJECT: wrong gate?')
            cv2.putText(vis, _reject_lbl, (5, _y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 220), 2, cv2.LINE_AA)
            _y += 24
        elif pnp_skip_reason:
            cv2.putText(vis, f'SKIP: {pnp_skip_reason}', (5, _y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1, cv2.LINE_AA)
            _y += 22

        # ── Gate keypoints + quad outline ─────────────────────────────────────
        if corners is not None:
            pts = corners.astype(int)
            for i in range(4):
                cv2.circle(vis, tuple(pts[i]), 7, (255, 0, 220), -1)
            for i in range(4):
                cv2.line(vis, tuple(pts[i]), tuple(pts[(i + 1) % 4]),
                         (255, 0, 220), 2, cv2.LINE_AA)

        # ── Bearing arrow ─────────────────────────────────────────────────────
        # Update cached gate centre when we have a live detection.
        if centre_px is not None:
            self._overlay_last_ctr = (int(centre_px[0]), int(centre_px[1]))

        # Choose arrow target: live centre, or cached centre when holding/
        # transit. No live centre_px this tick directly means "holding" —
        # checking that data-availability condition rather than a mode
        # string is more direct than trying to map a specific mode name to
        # it (the old 'HOLD' mode this mirrored no longer exists as a
        # distinct state: GateLock's own miss-streak tolerance now absorbs
        # an ordinary brief miss without ever leaving LOCKED, so it shows up
        # as an ordinary TRACKING/REACQUIRING tick with no fresh centre_px,
        # not a separate mode).
        _holding = (centre_px is None) or (ctrl_mode == 'TRANSITION')
        if _holding and self._overlay_last_ctr is not None:
            arrow_target = self._overlay_last_ctr
        elif centre_px is not None:
            arrow_target = (int(centre_px[0]), int(centre_px[1]))
        else:
            arrow_target = None

        if arrow_target is not None:
            arrow_col = badge_color
            # Dashed style while holding: draw segmented line then arrowhead.
            if _holding:
                dx = arrow_target[0] - ic_x
                dy = arrow_target[1] - ic_y
                dist_px = max(1, int(np.sqrt(dx*dx + dy*dy)))
                segs = 8
                for s in range(segs):
                    if s % 2 == 0:
                        p1 = (int(ic_x + dx * s / segs),
                              int(ic_y + dy * s / segs))
                        p2 = (int(ic_x + dx * (s + 1) / segs),
                              int(ic_y + dy * (s + 1) / segs))
                        cv2.line(vis, p1, p2, arrow_col, 2, cv2.LINE_AA)
                # Arrowhead at tip
                cv2.arrowedLine(vis,
                                (int(ic_x + dx * 0.85), int(ic_y + dy * 0.85)),
                                arrow_target, arrow_col, 2, tipLength=0.25,
                                line_type=cv2.LINE_AA)
            else:
                cv2.arrowedLine(vis, (ic_x, ic_y), arrow_target,
                                arrow_col, 3, tipLength=0.15, line_type=cv2.LINE_AA)

        # ── EKF velocity arrow (white) ────────────────────────────────────────
        # Shows where the drone is actually going according to the EKF.
        # Arrow direction = body lateral (right) + vertical (down) components of
        # vel_ned rotated to camera frame.  Fixed pixel length so it is purely
        # directional; speed is printed next to the tip.
        _mav_ov = self.data.get('mav_state')
        if _mav_ov is not None:
            _vel_n = np.asarray(_mav_ov['vel_ned'], dtype=float)
            _spd_ov = float(np.linalg.norm(_vel_n))
            if _spd_ov > 0.3:
                _Rb2n = rotations.quat_to_R_body2ned(_mav_ov['quat'])
                _vb = _Rb2n.T @ _vel_n          # body frame: x=fwd, y=right, z=down
                _VLEN = 80                       # fixed arrow length in pixels
                _vtx = ic_x + int(_VLEN * _vb[1] / _spd_ov)   # body-right → cam-x
                _vty = ic_y + int(_VLEN * _vb[2] / _spd_ov)   # body-down  → cam-y
                cv2.arrowedLine(vis, (ic_x, ic_y), (_vtx, _vty),
                                (220, 220, 220), 2, tipLength=0.2, line_type=cv2.LINE_AA)
                cv2.putText(vis, f'EKF {_spd_ov:.1f}m/s', (_vtx + 4, _vty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

        # ── Image centre crosshair ─────────────────────────────────────────────
        cv2.drawMarker(vis, (ic_x, ic_y), (200, 200, 200),
                       cv2.MARKER_CROSS, 20, 1, cv2.LINE_AA)

        return vis

    def _pnp_gate(self, corners_px, gate_width, gate_height,
                  continuity_attr='_last_pnp_R_primary', gate_ned=None):
        """Returns (tvec, rvec) both as flat (3,) arrays, or (None, None) on failure.

        IPPE produces two solutions with nearly equal reprojection error for
        near-frontal views (a front/back flip about the gate plane). Which
        one is correct is resolved by pose_disambiguation.disambiguate() —
        see that module's docstring for the full policy. In short:
        position-consistency against mav_state['pos_ned'] (independent,
        reliable) whenever gate_ned and a live position estimate are both
        available, otherwise rate-limited continuity against this call
        site's own last-accepted rotation. Deliberately never scored against
        the EKF's own live attitude any more — confirmed in a flight log
        that a wrong candidate can win with a clear, non-tied margin once
        the EKF's roll/pitch has already drifted, entrenching the same
        drift that caused the wrong pick.

        gate_ned, when given (the gate's known absolute NED position), is
        what lets position-consistency run at all.

        continuity_attr selects which of this instance's two independent
        continuity anchors (primary/current-gate vs. next-gate-candidate)
        to read and update — see their __init__ comment for why they're
        kept separate.
        """
        hw = gate_width  / 2.0
        hh = gate_height / 2.0
        obj_pts = np.array([
            [-hw, -hh, 0.0],
            [ hw, -hh, 0.0],
            [ hw,  hh, 0.0],
            [-hw,  hh, 0.0],
        ], dtype=np.float64)
        img_pts = corners_px.astype(np.float64).reshape(4, 1, 2)

        n, rvecs, tvecs, _ = cv2.solvePnPGeneric(
            obj_pts, img_pts, self._cam_K, np.zeros((4, 1)),
            flags=cv2.SOLVEPNP_IPPE,
        )
        if n < 1:
            return None, None

        mav = self.data.get('mav_state')
        mav_pos, R_b2n_meas = None, None
        if mav is not None and mav.get('pos_ned') is not None:
            mav_pos = np.asarray(mav['pos_ned'])
            # gt_correct_quat: rotates a body-frame bearing into NED, so GT
            # mode's raw yaw convention would rotate it the wrong way (see
            # rotations.py's module note) if left uncorrected.
            R_b2n_meas = rotations.quat_to_R_body2ned(
                rotations.gt_correct_quat(mav['quat'], self._gt_mode))
        have_position_anchor = gate_ned is not None and R_b2n_meas is not None

        candidates = []
        for rv, tv in zip(rvecs, tvecs):
            tv_f = tv.flatten()
            rv_f = rv.flatten()
            # A degenerate corner configuration can make solvePnPGeneric return
            # NaN/Inf instead of failing outright. Must reject explicitly here:
            # `nan < 0.1` is also False in numpy, so the "behind camera" filter
            # below would NOT catch it either, and a NaN tvec/rvec would flow
            # all the way through to the logged vision fix and (previously)
            # crash/corrupt the EKF via update_position's gate check.
            if not (np.all(np.isfinite(tv_f)) and np.all(np.isfinite(rv_f))):
                continue
            if tv_f[2] < 0.1:          # gate behind camera — physically impossible
                continue
            R_sol, _ = cv2.Rodrigues(rv)
            if have_position_anchor:
                t_gate_body = self._R_cam2body @ tv_f
                t_gate_ned  = R_b2n_meas @ t_gate_body
                implied_pos = gate_ned - t_gate_ned
            else:
                implied_pos = np.zeros(3)   # unused: mav_pos_ned=None below skips tier 1 entirely
            candidates.append(pose_disambiguation.PoseCandidate(
                R_b2n=R_sol, implied_pos_ned=implied_pos, payload=(rv_f, tv_f)))

        if not candidates:
            return None, None

        now = time.time()
        result = pose_disambiguation.disambiguate(
            candidates,
            mav_pos_ned=(mav_pos if have_position_anchor else None),
            last_accepted_R=getattr(self, continuity_attr, None),
            last_accepted_t=getattr(self, continuity_attr + '_t', None),
            now=now,
            position_degenerate_tol_m=self._pos_degenerate_tol_m,
            max_rotation_rate_rad_s=self._pnp_continuity_max_rate,
        )
        chosen = candidates[result.index]
        best_rvec, best_tvec = chosen.payload

        if best_tvec[2] < 0.5:
            return None, None
        # Only persist this pick as the new continuity anchor when it was
        # rate-plausible. Confirmed in a flight log: persisting an
        # implausible pick unconditionally lets a single wrong candidate
        # (e.g. an IPPE front/back flip) become self-perpetuating, since
        # every subsequent frame then scores against that now-wrong anchor
        # — see pose_disambiguation.disambiguate()'s docstring for the full
        # mechanism. Using this frame's pick as the OUTPUT is still fine
        # (best available among a small discrete set); just don't let it
        # poison what the NEXT frame compares against.
        if result.method != 'continuity_implausible':
            setattr(self, continuity_attr, chosen.R_b2n)
            setattr(self, continuity_attr + '_t', now)
        return best_tvec, best_rvec

    def _landmark_gate_fix(self, corners_px, agi):
        """When a detection fails the current-target identity check, see
        whether it matches a course-sequence NEIGHBOUR gate instead
        (self._waypoints, the same static layout track_gates_ned refines at
        runtime) — a gate seen in the distance while the current target is
        out of view still tells us exactly where the drone is, via the same
        gate_ned -> drone_ned inversion the primary path uses (see the main
        PnP position solve above), just anchored on a different, already-
        known gate position instead of the current target's.

        Deliberately position-only (no yaw/velocity): this path only sees a
        given gate intermittently, so it can't build the same-gate frame-to-
        frame continuity vel-regression/yaw-smoothing depend on. Restricted
        to the two course-sequence neighbours of the current target (not the
        full waypoint list) to keep the match unambiguous, and disambiguates
        its own PnP solution via a dedicated continuity anchor
        (_last_pnp_R_landmark) so it can't disturb the primary/next-gate
        anchors.

        Feeds self.data['_vision_ekf_update'] directly and nothing else —
        never touches gate_detection/pose_estimate, so it cannot reach
        controller.py's pursuit-override or live-target-write paths (both
        gated on gate_id_match against the CURRENT target only, independent
        of this). A wrong match here can bias the EKF position estimate but
        can never redirect the drone at the wrong gate.

        Returns (drone_ned_fix, matched_waypoint_idx, range_m), or None on
        no match — the common case, since most rejected detections are
        genuine noise/false positives, not a neighbour gate.
        """
        if agi is None or self._waypoints is None:
            return None
        mav = self.data.get('mav_state')
        if mav is None or mav.get('pos_ned') is None:
            return None
        tvec, _ = self._pnp_gate(corners_px, self._gate_w_default, self._gate_h_default,
                                  continuity_attr='_last_pnp_R_landmark')
        if tvec is None or tvec[2] > self._max_gate_dist:
            return None

        R_b2n = rotations.quat_to_R_body2ned(
            rotations.gt_correct_quat(mav['quat'], self._gt_mode))
        t_gate_body = self._R_cam2body @ tvec
        t_gate_ned  = R_b2n @ t_gate_body
        guess_ned   = np.asarray(mav['pos_ned']) + t_gate_ned

        best_idx, best_dist = None, self._landmark_match_tol_m
        for wp_idx in (int(agi), int(agi) + 2):   # course-sequence neighbours of agi+1 (current target)
            if 1 <= wp_idx < len(self._waypoints):
                d = float(np.linalg.norm(np.asarray(self._waypoints[wp_idx]) - guess_ned))
                if d < best_dist:
                    best_idx, best_dist = wp_idx, d
        if best_idx is None:
            return None

        drone_ned_fix = np.asarray(self._waypoints[best_idx]) - t_gate_ned
        return drone_ned_fix, best_idx, float(tvec[2])

    def _attitude_from_pnp(self, rvec, gate_quat_wxyz):
        """
        Derive drone (roll, pitch, yaw) NED, radians, from PnP rvec + known
        gate orientation. The gate quaternion from track data gives the gate
        frame orientation in NED, completing the chain:
            R_b2n = R_gate2ned @ R_gate2cam.T @ R_cam2body.T
        R_b2n is a full attitude, so roll/pitch fall out of the same matrix
        at no extra PnP cost — used by ekf.py's update_attitude as vision's
        only *ongoing* roll/pitch correction that doesn't have to go silent
        during real linear acceleration the way the accelerometer-based
        update does.

        Square gates have a 4-fold corner-labelling ambiguity: solvePnP can
        return a pose rotated by any multiple of 90 degrees about the gate's
        own normal axis (relabelling which detected corner is "top-left") —
        for how these gates are actually mounted (normal roughly horizontal,
        pointing back down the track, not aligned with the world yaw/
        vertical axis), a 90-degree mislabelling shifts YAW by only a few
        degrees but ROLL by ~90-180 degrees (confirmed via a synthetic
        forward-model test and matched exactly against real flight logs:
        roll_ned spanned the full +-180 degrees essentially at random while
        yaw_ned tracked GT with only ~13 degree mean error, when this wasn't
        resolved jointly) — so all three angles are always extracted from
        the same winning candidate, never patched independently.

        Relabelling doesn't move tvec at all, so position can never
        discriminate between these 4 candidates — pose_disambiguation.
        disambiguate() always resolves this via rate-limited continuity
        against this method's own last-accepted rotation (or a fixed
        cold-start reference), NEVER the EKF's live attitude — see
        pose_disambiguation.py's module docstring and this file's
        _last_pnp_att_R init comment for why scoring against a live,
        vision-correctable state is self-reinforcing (confirmed in a flight
        log: a wrong candidate got picked and dragged EKF yaw error from
        ~2 to ~22 degrees over ~4s before this fix).
        """
        R_gate2cam, _ = cv2.Rodrigues(rvec.reshape(3, 1))
        R_gate2ned = rotations.quat_to_R_body2ned(gate_quat_wxyz)

        candidates = []
        for n in range(4):
            c, s = np.cos(n * np.pi / 2.0), np.sin(n * np.pi / 2.0)
            Rz_n = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
            R_cam2ned_cand = R_gate2ned @ (R_gate2cam @ Rz_n).T
            R_b2n_cand = R_cam2ned_cand @ self._R_cam2body.T
            candidates.append(pose_disambiguation.PoseCandidate(
                R_b2n=R_b2n_cand, implied_pos_ned=np.zeros(3), payload=None))

        now = time.time()
        result = pose_disambiguation.disambiguate(
            candidates, mav_pos_ned=None,   # position-degenerate by construction — see docstring
            last_accepted_R=self._last_pnp_att_R, last_accepted_t=self._last_pnp_att_t,
            now=now, position_degenerate_tol_m=self._pos_degenerate_tol_m,
            max_rotation_rate_rad_s=self._pnp_continuity_max_rate,
            nominal_R=self._att_nominal_R,
        )
        chosen_R = candidates[result.index].R_b2n
        roll, pitch, yaw = rotations.euler_from_R_body2ned(chosen_R)

        # See the matching comment in _pnp_gate: only persist as the new
        # continuity anchor when the pick was rate-plausible, so a single
        # wrong 90-degree-rotated candidate can't entrench itself as the
        # reference every subsequent frame gets scored against.
        if result.method != 'continuity_implausible':
            self._last_pnp_att_R = chosen_R
            self._last_pnp_att_t = now
        return roll, pitch, yaw

    # ── Gyro-only attitude tracking (velocity de-rotation helper) ───────────

    @staticmethod
    def _quat_mul(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ])

    # v_ned = R @ v_body — see rotations.py.
    _quat_to_R = staticmethod(rotations.quat_to_R_body2ned)

    @staticmethod
    def _ols_slope_sigma(A, y, coef):
        """Standard error of an OLS slope (column 0 of `coef`), as a single
        scalar summarizing all columns of `y`. Used to give the EKF a per-tick
        velocity sigma that reflects how good THIS window's fit actually was
        (short/noisy vs. long/settled), instead of one fixed sigma for every
        fit regardless of quality.

        sigma_slope^2 = residual_variance * (AtA)^-1[0,0] per column (standard
        OLS slope-covariance result); averaging that variance across columns
        before rotation gives the same trace/3 as after rotation by any
        orthogonal R (v_ned = R @ v_body), since trace(R Cov R^T) = trace(Cov)
        — so this can be computed once in whichever frame is convenient.
        """
        n, p = A.shape
        dof = max(n - p, 1)
        resid = y - A @ coef
        rss = np.sum(resid**2, axis=0)              # (ncols,)
        AtA_inv_00 = np.linalg.inv(A.T @ A)[0, 0]
        slope_var = (rss / dof) * AtA_inv_00         # (ncols,)
        return float(np.sqrt(np.mean(slope_var)))

    def _update_gyro_q(self, now):
        """First-order-integrate the gyro-only attitude to `now` and return it.

        Deliberately independent of the EKF's own attitude estimate — see the
        buffer init comment for why. Uses mav_state['rates'] (body rates) and
        the elapsed wall time since the last call.
        """
        gyro = self.data.get('mav_state', {}).get('rates')
        if gyro is not None and self._gyro_q_last_t is not None:
            dt = now - self._gyro_q_last_t
            if 0.0 < dt < 0.5:   # skip absurd gaps (startup, stalls)
                wx, wy, wz = gyro
                _dq = np.array([1.0, 0.5*wx*dt, 0.5*wy*dt, 0.5*wz*dt])
                self._gyro_q = self._quat_mul(self._gyro_q, _dq)
                self._gyro_q /= np.linalg.norm(self._gyro_q)
        self._gyro_q_last_t = now
        return self._gyro_q.copy()

    # ── Main frame processing ───────────────────────────────────────────────

    def process_frame(self, frame_id, img, sim_time_ns=None, ctrl_mode_at_capture=None):
        if self._debug_waypoints_only and not self._vision_detect_for_logging:
            self.data['gate_detection'] = {
                'detected': False, 'centre_px': None, 'conf': 0.0,
                'tvec_cam': None, 'rvec_cam': None, 'frame_id': frame_id,
            }
            if self.logger:
                self.logger.log_frame(frame_id, img, sim_time_ns=sim_time_ns)
            return

        img = self._preprocess(img)
        # Desaturate blue/cyan pixels (S=0 → grey) before YOLO and orange mask.
        # Desaturation preserves luminance so YOLO keeps structural context from
        # buildings and the track beam, while removing the hue that would
        # contaminate the orange mask or produce cyan false-positive detections.
        img = self._apply_blue_suppression(img)
        masked, mask_raw = self._orange_mask(img)
        orange_px = int(np.count_nonzero(mask_raw))

        results = self._model.predict(img, verbose=False, conf=self._conf_min) if self._yolo_enabled else None

        detected  = False
        centre_px = None
        conf      = 0.0
        corners   = None
        best_box  = None
        tvec_cam  = None
        rvec_cam  = None
        gate_info = None
        drone_ned = None
        _pnp_skip_reason = None   # diagnostic: why PnP was skipped this frame
        _gate_id_rejected = False   # this frame's detection failed the gate-identity check
        _landmark_wp_hit = None   # set if a rejected detection matched a neighbour gate instead
        # Use the mode stamped at frame *capture* time (see _recv_loop), not a
        # live read here — process_frame can run up to ~170ms after capture
        # under queueing backlog, by which point controller.py's async control
        # loop may have already flipped vis_ctrl_mode back out of TRANSITION.
        _in_transit = ctrl_mode_at_capture == 'TRANSITION'

        # ── Gate distance lock reset ──────────────────────────────────────────
        # Reset lock when the sim advances to the next gate (active_gate_index
        # changes). Moved here (before detection/gate-identity processing
        # below) from its previous position after the next-gate-candidate
        # block: that ordering left the lock holding the OLD gate's state for
        # the entire frame where agi actually changes, so the gate-identity
        # plausibility check inside GateLock.on_frame (gated on "not yet
        # LOCKED") silently skipped its very first, most-needed frame — the
        # one right at the transition. Resetting first means the check sees
        # the correct (reset) lock state from the transition frame onward,
        # with no one-frame gap.
        agi = self.data.get('active_gate_index')
        if agi != self._prev_agi and self._prev_agi is not None:
            self._lock.reset()
            self._pnp_vel_buf.clear()          # stale relative vectors from old gate are invalid
            self._pnp_vel_ned_buf.clear()      # same for the parallel NED-position buffer
            self._pnp_yaw_buf.clear()          # stale yaw readings from old gate are invalid
            self._last_pnp_R_primary   = None  # stale rotation continuity from old gate is invalid
            self._last_pnp_R_primary_t = None
            self._last_pnp_R_next      = None
            self._last_pnp_R_next_t    = None
            self._last_pnp_R_landmark   = None  # "other gate" identity shifts too as agi advances
            self._last_pnp_R_landmark_t = None
            # _last_pnp_att_R/_t deliberately NOT cleared here — see their
            # __init__ comment: that anchor is the drone's own attitude, not
            # gate-relative, so it doesn't become invalid just because the
            # tracked gate index advanced.
            self._next_gate_ned_buf.clear()    # next-gate buffer also invalid after advance
            self._next_gate_ned = None
            print(f"[VISION] gate index {self._prev_agi}→{agi}: lock reset", flush=True)
        self._prev_agi = agi

        h_img, w_img = mask_raw.shape
        if results is not None and results[0].boxes is not None \
                and results[0].keypoints is not None and len(results[0].boxes) > 0:
            r = results[0]
            boxes = r.boxes.xywh.cpu().numpy()   # (N,4): cx,cy,w,h
            confs = r.boxes.conf.cpu().numpy()
            kpts  = r.keypoints.xy.cpu().numpy() # (N,4,2)

            # Two-pass selection: filter → nearest by area.
            # Pass 1: reject boxes with conf < vision_conf_min, zero keypoints, or
            #         orange_frac < 0.1% (removes false positives: blue beam,
            #         background structures, unlabelled distant objects).
            # Pass 2: among survivors pick LARGEST bbox area = nearest gate.
            #         Confidence is not a reliable gate-distance discriminator —
            #         a farther gate can score higher than the near one; apparent
            #         size (bbox area) correctly selects the closest visible gate.
            # Bbox is expanded by _PAD pixels on each side before the orange_frac check.
            # At 40 m YOLO's bbox can be misaligned by 10–15 px from the orange gate frame;
            # padding bridges that gap without requiring pixel-perfect localisation.
            # Threshold 0.1% (not 0.5%) because the padded area is larger.
            _PAD = 15
            valid_idx = []
            for _i in range(len(boxes)):
                if confs[_i] < self._conf_min:
                    continue
                if kpts[_i].shape != (4, 2):
                    continue
                if np.any(np.all(kpts[_i] < 2.0, axis=1)):
                    continue
                _bx1 = max(0, int(boxes[_i][0] - boxes[_i][2] / 2) - _PAD)
                _by1 = max(0, int(boxes[_i][1] - boxes[_i][3] / 2) - _PAD)
                _bx2 = min(w_img, int(boxes[_i][0] + boxes[_i][2] / 2) + _PAD)
                _by2 = min(h_img, int(boxes[_i][1] + boxes[_i][3] / 2) + _PAD)
                _ba  = max(1, (_bx2 - _bx1) * (_by2 - _by1))
                if np.count_nonzero(mask_raw[_by1:_by2, _bx1:_bx2]) / _ba < 0.001:
                    continue
                valid_idx.append(_i)

            # Suppress detection during TRANSITION (the carrot tracker's final-
            # approach/commit phase, pinned to the gate centre while passing
            # through it): at close range and steep viewing angles the camera
            # can see smaller structures inside/beyond the gate's own beams
            # (e.g. a further gate framed by the near one), and YOLO's largest-
            # bbox-wins selection can mistake one of those for the target,
            # producing a discontinuous tvec_cam that corrupts _pnp_vel_buf's
            # regression (the reported PnP velocity spikes at gate transitions)
            # as well as gate_info/gate-NED overrides. The real target gate is
            # already being flown through open-loop by the carrot at this
            # point, so nothing is lost by ignoring vision here.
            if valid_idx and _in_transit:
                _pnp_skip_reason = "transit phase (suppressed)"
            if valid_idx and not _in_transit:
                areas       = boxes[:, 2] * boxes[:, 3]
                _sorted_v   = sorted(valid_idx, key=lambda i: areas[i], reverse=True)
                best        = _sorted_v[0]
                _second_idx = _sorted_v[1] if len(_sorted_v) >= 2 else None

                corners   = kpts[best]
                centre_px = corners.mean(axis=0)
                conf      = float(confs[best])
                best_box  = tuple(float(v) for v in boxes[best])
                detected  = True

                gates = self.data.get('track_gates_ned', {})
                # Default to 0: before first RACE_STATUS the fallback
                # still points at waypoints[1] (first gate).
                agi   = self.data.get('active_gate_index', 0)
                if agi is not None and int(agi) in gates:
                    # track_gates_ned's raw NED marks the gate's bottom beam, not the
                    # opening centre the drone flies through (~1 m above the beam) —
                    # same correction already baked into params.yaml's waypoints: list,
                    # which is why the fallback branch below (using self._waypoints)
                    # doesn't need it but this live-track-data branch does.
                    #
                    # gate_info['width']/['height'] from live track data are the gate's
                    # OUTER frame (2.7 m) — the physical structure size, broadcast by the
                    # sim. YOLO's keypoints measure the INNER opening (1.5 m) the drone
                    # flies through, which is what the PnP object model must match.
                    # Using the outer size here computed tvec_z ~1.8x too far (confirmed:
                    # 40.4 m vs the true ~22.4 m for an identical corner set), silently
                    # rejected by vision_max_gate_dist and blocking PnP almost entirely.
                    # Keep the true inner size (gate_width/height_default) for the PnP
                    # call; only 'ned' (position) comes from the live track message.
                    gate_info = dict(gates[int(agi)])
                    gate_info['ned'] = gate_info['ned'].copy()
                    gate_info['ned'][2] -= self._gate_beam_to_center_d
                    gate_info['width']  = self._gate_w_default
                    gate_info['height'] = self._gate_h_default
                    # Cache so later frames can use it as a fallback when
                    # track_gates_ned is temporarily unavailable.
                    self._last_gate_info[int(agi)] = gate_info
                    tvec_cam, rvec_cam = self._pnp_gate(
                        corners, gate_info['width'], gate_info['height'],
                        gate_ned=gate_info['ned'])
                    if tvec_cam is None:
                        _pnp_skip_reason = "solvePnP failed"
                else:
                    # No live track data for this gate.
                    # Fallback priority:
                    #   1. Last known gate_info from track data (accurate NED; width/height
                    #      are already the corrected inner-opening size, see above)
                    #   2. Approximate NED from params.yaml waypoints
                    #   3. Distance-only (no position update)
                    # Gate dims are always the fixed inner-opening constant — never taken
                    # from track data's outer-frame width/height (see note above).
                    _last = self._last_gate_info.get(int(agi)) if agi is not None else None
                    # Resolve a fallback NED (independent of whether PnP below
                    # succeeds) so _pnp_gate can use it for its position-
                    # consistency ambiguity check too.
                    _wp_ned = None
                    if _last is None and agi is not None and self._waypoints is not None:
                        wp_idx = int(agi) + 1
                        if wp_idx < len(self._waypoints):
                            _wp_ned = self._waypoints[wp_idx]
                    _fallback_ned = _last['ned'] if _last is not None else _wp_ned
                    tvec_cam, rvec_cam = self._pnp_gate(
                        corners, self._gate_w_default, self._gate_h_default,
                        gate_ned=_fallback_ned)
                    gate_info = None
                    if tvec_cam is not None and agi is not None:
                        if _last is not None:
                            gate_info = _last
                        elif _wp_ned is not None:
                            gate_info = {
                                'ned':    _wp_ned.copy(),
                                'width':  self._gate_w_default,
                                'height': self._gate_h_default,
                                'quat':   None,
                            }
                    if tvec_cam is None:
                        _pnp_skip_reason = "solvePnP failed (fallback dims)"
                    elif gate_info is None:
                        if not gates:
                            _pnp_skip_reason = "no track_gates_ned, no cache, no agi (dist only)"
                        else:
                            _pnp_skip_reason = "no active_gate_index (dist only)"
                    elif _last is not None:
                        _pnp_skip_reason = "last-known gate NED (no track_gates_ned)"
                    else:
                        _pnp_skip_reason = "no track_gates_ned (waypoint NED fallback)"

                # Gate-identity plausibility is now checked as part of the
                # unified lock state machine below (self._lock.on_frame),
                # alongside the distance-spike check — see that call site.

                # ── Next-gate candidate (second-largest valid box) ────────────
                # Accumulates PnP-derived NED positions across many frames and
                # publishes a confirmed median position once the buffer is full.
                # Guard: second gate must be farther than current gate by ≥5 m so
                # duplicate detections of the same gate are rejected.
                _cur_dist = float(tvec_cam[2]) if tvec_cam is not None else 0.0
                if _second_idx is not None:
                    _sec_corners = kpts[_second_idx]
                    # Gate dims for PnP are always the fixed inner-opening constant —
                    # never track data's outer-frame width/height (see note above).
                    _tvec_ng, _ = self._pnp_gate(
                        _sec_corners, self._gate_w_default, self._gate_h_default,
                        continuity_attr='_last_pnp_R_next')
                    if (_tvec_ng is not None
                            and _tvec_ng[2] > _cur_dist + 5.0   # farther than current gate
                            and _tvec_ng[2] < self._max_gate_dist):
                        _mav_ng = self.data.get('mav_state')
                        if _mav_ng is not None:
                            _R_b2n_ng = rotations.quat_to_R_body2ned(_mav_ng['quat'])
                            _t_ng_ned = _R_b2n_ng @ (self._R_cam2body @ _tvec_ng)
                            _gate2_ned = np.asarray(_mav_ng['pos_ned']) + _t_ng_ned
                            self._next_gate_ned_buf.append(_gate2_ned)
                            if len(self._next_gate_ned_buf) >= self._next_gate_min_frames:
                                _buf = np.array(list(self._next_gate_ned_buf))
                                self._next_gate_ned = np.median(_buf, axis=0)

        # Publish next-gate state for controller and overlay.
        self.data['next_gate_ned']        = self._next_gate_ned
        self.data['next_gate_buf_frames'] = len(self._next_gate_ned_buf)
        self.data['next_gate_min_frames'] = self._next_gate_min_frames

        # Hard range gate: the next gate is never more than 50 m away.
        # Detections beyond this are background noise or a gate from a later lap.
        if tvec_cam is not None and tvec_cam[2] > self._max_gate_dist:
            tvec_cam = None
            rvec_cam = None
            gate_info = None

        # ── Hover-reset detection ────────────────────────────────────────────
        # pos_offset_ned is set exactly once, at hover-reset (imu_ekf.py). A
        # change means a reset just happened — clear the velocity buffer and
        # resync the gyro tracker so it can't straddle the LEVEL-phase snap
        # transient (see _gyro_q's init comment).
        _pos_off = self.data.get('pos_offset_ned')
        if _pos_off is not None:
            _pos_off_key = tuple(np.round(np.asarray(_pos_off, dtype=float), 6))
            # Also fires on the very first appearance (None -> value) — that
            # transition IS the first (and usually only) hover-reset each
            # flight, which is exactly the one that matters here.
            if _pos_off_key != self._last_pos_offset_ned:
                self._pnp_vel_buf.clear()
                self._pnp_vel_ned_buf.clear()
                self._gyro_q        = np.array([1.0, 0.0, 0.0, 0.0])
                self._gyro_q_last_t = None
                print("[VISION] hover-reset detected: velocity buffer/gyro tracker reset",
                      flush=True)
                self._last_pos_offset_ned = _pos_off_key

        # ── Gate lock / identity / confidence state machine (gate_lock.py) ──
        # Consolidates the old gate-identity plausibility recheck and the
        # distance-spike/hit-miss bookkeeping into ONE state machine
        # (UNLOCKED -> ACQUIRING -> LOCKED) instead of two separately-
        # orchestrated blocks — see gate_lock.py's module docstring for the
        # full policy and gate_lock.GateLock.on_frame for the invariants it
        # preserves from the code this replaces.
        #
        # Minor, deliberate ordering change from the code this replaces: the
        # identity check used to run BEFORE the hard-range-gate above, so a
        # detection that failed both simultaneously was always attributed to
        # "wrong gate" (_gate_id_rejected=True); it now runs after (folded
        # into this one call), so that rare double-failure case is
        # attributed to "no measurement" instead. In practice the hard-range
        # ceiling (~32-50m) and identity tolerance (~10m) rarely overlap —
        # anything wrong enough to fail identity almost always does so well
        # inside the hard-range ceiling — so this doesn't change behaviour
        # for the wrong-gate-handoff case the identity check exists for.
        _mav_lock = self.data.get('mav_state')
        _expected_range = (
            float(np.linalg.norm(np.asarray(gate_info['ned']) - np.asarray(_mav_lock['pos_ned'])))
            if (gate_info is not None and _mav_lock is not None) else None
        )
        _measured_range = float(tvec_cam[2]) if tvec_cam is not None else None
        _state_before   = self._lock.state
        _lock_accepted  = self._lock.on_frame(_measured_range, _expected_range, time.time())

        if _lock_accepted and _state_before != LockState.LOCKED and self._lock.state == LockState.LOCKED:
            print(f"[VISION] gate locked at {self._lock.last_good_range_m:.1f}m "
                  f"after {self._lock.hit_count} frames", flush=True)

        if not _lock_accepted:
            _reason = self._lock.last_reject_reason
            if _reason == 'identity':
                _gate_id_rejected = True
                _pnp_skip_reason = "range mismatch vs expected gate range (likely next gate)"
                if _state_before == LockState.LOCKED:
                    print(f"[VISION] gate-identity recheck failed while locked "
                          f"(range {_measured_range:.1f}m "
                          f"vs expected {_expected_range:.1f}m) — lock dropped", flush=True)
                # Landmark aid: before discarding this detection outright,
                # check whether it's actually a NEIGHBOUR gate rather than
                # noise — see _landmark_gate_fix. Uses corners (not yet
                # nulled below) — position-only, feeds the EKF directly, and
                # is otherwise fully independent of everything nulled here.
                if self._landmark_aid_enabled and corners is not None:
                    _landmark = self._landmark_gate_fix(corners, agi)
                    if _landmark is not None:
                        _lm_ned, _landmark_wp_hit, _lm_range = _landmark
                        _lm_sigma = ((self._ekf_vis_sigma
                                      + self._ekf_vis_sigma_k * _lm_range)
                                     * self._landmark_sigma_mult)
                        self.data['_vision_ekf_update'] = {
                            'pos_ned':   _lm_ned,
                            'vel_ned':   None,
                            'yaw_ned':   None,
                            'roll_ned':  None,
                            'pitch_ned': None,
                            'sigma_pos': _lm_sigma,
                            'sigma_vel': self._ekf_vis_vel_sigma,
                            'sigma_yaw': self._ekf_vis_yaw_sigma,
                            'sigma_att': self._ekf_vis_att_sigma,
                            'yaw_gate':  self._ekf_vis_yaw_gate,
                            'att_gate':  self._ekf_vis_att_gate,
                            'gate':      self._ekf_vis_gate,
                            'vel_gate':  self._ekf_vis_vel_gate,
                            'wall_t':    time.time(),
                        }
                        print(f"[VISION] landmark fix: rejected detection matches "
                              f"waypoint {_landmark_wp_hit} (current target agi={agi}) "
                              f"at {_lm_range:.1f}m — position-only EKF update", flush=True)
                tvec_cam  = None
                rvec_cam  = None
                gate_info = None
                detected  = False
                centre_px = None
                conf      = 0.0
                best_box  = None
            elif _reason == 'spike':
                # Distance spike only invalidates the PnP fix itself, not the
                # detection — still show the box/centroid this tick (matches
                # the code this replaces, which never nulled detected/
                # centre_px/conf/best_box on a spike, only tvec_cam/rvec_cam).
                tvec_cam = None
                rvec_cam = None
            # 'no_measurement': tvec_cam was already None going in (solvePnP
            # failure or the hard-range cutoff above) — nothing further to null.

        self.data['landmark_fix_wp'] = _landmark_wp_hit

        # ── Orange-centroid fallback ──────────────────────────────────────────
        # When YOLO fails, find the centroid of the orange blob in the tight mask.
        # This gives a bearing to the gate center even without keypoints or PnP.
        # The centroid is published as centre_px so the controller can use it for
        # visual yaw correction (keep gate centered).
        # Skipped when this frame's YOLO detection was itself rejected by the
        # gate-identity check above: the same wrong (next) gate's blob is still
        # the dominant orange region in the mask, so this fallback would just
        # rediscover it via a cruder path and steer at it anyway. Treat this
        # frame as genuinely blind instead — the reacquisition/search-nudge
        # logic in controller.py is built to handle that gracefully.
        centroid_area = 0
        if not detected and not _gate_id_rejected:
            ccx, ccy, centroid_area = self._orange_centroid(mask_raw)
            if ccx is not None:
                centre_px = np.array([ccx, ccy])
                detected  = True   # bearing-only detection

        self.data['gate_detection'] = {
            'detected':      detected,
            'centre_px':     centre_px,
            'conf':          conf,
            'tvec_cam':      tvec_cam,
            'centroid_only': tvec_cam is None and centroid_area > 0,
            'pnp_locked':    self._lock.state == LockState.LOCKED,
            'frame_id':      frame_id,
        }

        # EKF vision update: position + velocity + yaw from PnP, from a fresh
        # YOLO detection only (no optical-flow bridging — removed after being
        # a repeat source of position/velocity corruption: a tracked-but-stale
        # keypoint set has no way to tell it's no longer on the gate once the
        # gate leaves frame, e.g. right after flythrough. model_predict plus
        # the current detection robustness make bridging brief YOLO misses
        # unnecessary).
        vel_ned     = None
        vel_ned_pnp = None
        _vel_body   = None   # body-frame-regression PnP velocity estimate (see below)
        _vel_pos    = None   # NED-position-regression PnP velocity estimate (see below)
        _sigma_body = None   # per-tick fit-quality sigma for _vel_body [m/s]
        _sigma_pos_vel = None   # per-tick fit-quality sigma for _vel_pos [m/s]
        _sigma_vel_dyn = None   # combined dynamic sigma for vel_ned_pnp [m/s]
        drone_ned   = None
        yaw_ned     = None
        roll_ned    = None
        pitch_ned   = None

        if tvec_cam is not None:
            # Attitude estimate: requires gate quaternion from track data.
            # Unavailable in the pure-fallback path (gate_info=None).
            if gate_info is not None and gate_info.get('quat') is not None:
                try:
                    # The square-gate 4-fold corner-labelling ambiguity (solvePnP
                    # can return a pose off by any multiple of 90° about the
                    # gate's own normal) is now resolved *inside*
                    # _attitude_from_pnp, jointly for roll/pitch/yaw against the
                    # EKF's full current attitude — see that method's docstring.
                    # tvec/position is unaffected by this ambiguity either way
                    # (confirmed empirically: relabelling which corner is "TL"
                    # rotates the recovered orientation but leaves the object's
                    # centre unchanged), so it's used as-is regardless.
                    roll_ned, pitch_ned, yaw_ned = self._attitude_from_pnp(
                        rvec_cam, gate_info['quat'])
                except Exception:
                    yaw_ned = None
                    roll_ned = None
                    pitch_ned = None

                if yaw_ned is not None:
                    # Circular-mean smoothing + outlier rejection (see buffer init comment).
                    self._pnp_yaw_buf.append(yaw_ned)
                    if len(self._pnp_yaw_buf) >= 3:
                        _yaws = np.array(self._pnp_yaw_buf)
                        _yaw_mean = float(np.arctan2(np.mean(np.sin(_yaws)),
                                                      np.mean(np.cos(_yaws))))
                        _dev = abs((yaw_ned - _yaw_mean + np.pi) % (2*np.pi) - np.pi)
                        yaw_ned = None if _dev > self._yaw_outlier_rad else _yaw_mean
                    else:
                        yaw_ned = None   # not enough samples yet to trust a fix

            # Position + velocity: require known gate NED (from track data or waypoint fallback).
            if gate_info is not None:
                mav = self.data.get('mav_state')
                if mav is not None:
                    # gt_correct_quat: this is the main PnP position solve —
                    # rotating a body-frame gate bearing into NED. GT mode's
                    # raw yaw convention would rotate it the wrong way (see
                    # rotations.py's module note) if left uncorrected.
                    R_b2n = rotations.quat_to_R_body2ned(
                        rotations.gt_correct_quat(mav['quat'], self._gt_mode))
                    t_gate_body = self._R_cam2body @ tvec_cam
                    t_gate_ned  = R_b2n @ t_gate_body
                    drone_ned   = gate_info['ned'] - t_gate_ned

                    now      = time.time()
                    _gyro_q  = self._update_gyro_q(now)
                    # Velocity buffer holds t_gate_body (drone->gate, BODY frame) —
                    # a direct PnP output via the fixed R_cam2body calibration, not
                    # world-NED position. drone_ned above needs R_b2n (live attitude)
                    # to rotate into world frame, and regressing across samples that
                    # each carry a *different* historical attitude estimate bakes
                    # every attitude error along the way into the velocity — this is
                    # exactly the mechanism behind the position/yaw corruption bugs
                    # found this session (LK-bridge, PnP dual-solution flips, frame
                    # duplication), so the old vN swings (std 1.38 m/s while
                    # stationary) were inheriting that same noise source, not just
                    # PnP pixel jitter. Regressing the body-frame vector instead
                    # needs attitude only once, at the very end, to rotate the
                    # resulting body-frame velocity into NED — so historical
                    # attitude error can no longer accumulate into the estimate.
                    #
                    # Body-frame differencing assumes the body axes don't rotate over
                    # the window — if they do, the gate's direction sweeps past in
                    # body frame purely from rotation, with zero real translation.
                    # Confirmed in a flight log: right at hover-reset the drone
                    # pitches hard into forward flight (a real, large rotation), and
                    # while pos_N/pos_E/tvec_z all stayed smooth and small, the
                    # regression produced a spurious vD ramping to -9.6 m/s over
                    # ~0.4s purely from that pitch sweep. A first attempt gated the
                    # whole fit on peak-rate x window-duration, but that rejected
                    # ordinary flight maneuvering too (any sustained ~10 deg/s turn
                    # exceeds an 8 deg budget over the ~0.8s window), killing PnP
                    # velocity for almost the entire active-flight portion of a log —
                    # exactly when it's needed most. Instead, de-rotate each buffered
                    # sample into the CURRENT body frame using gyro_q (gyro-only,
                    # integrated independently of the EKF's attitude estimate — see
                    # _update_gyro_q) before regressing, so real rotation is
                    # compensated rather than just detected-and-rejected.
                    # A detection gap (YOLO miss, occlusion) leaves a stale old
                    # cluster of samples in the buffer; appending one fresh sample
                    # after the gap makes the fit lean almost entirely on that one
                    # point relative to the stale cluster — functionally an
                    # endpoint difference again, with all the noise-sensitivity
                    # that the windowed regression exists to avoid. Confirmed in a
                    # flight log: a ~0.3s detection gap was immediately followed by
                    # a single-frame 9.3 m/s spurious speed reading. Drop the whole
                    # buffer instead of letting old and new samples mix once the
                    # gap since the last sample is large enough to matter.
                    if self._pnp_vel_buf and (now - self._pnp_vel_buf[-1][0]) > 0.15:
                        self._pnp_vel_buf.clear()
                        self._pnp_vel_ned_buf.clear()
                    self._pnp_vel_buf.append((now, t_gate_body.copy(), _gyro_q))
                    self._pnp_vel_ned_buf.append((now, drone_ned.copy()))
                    # Windowed velocity: least-squares slope over all buffered samples,
                    # not an endpoint difference. Endpoint-difference only ever uses 2 of
                    # the N buffered positions (the oldest and newest), so raising
                    # vision_vel_window_frames barely helped — the two endpoints are just
                    # as individually noisy regardless of how many frames sit between
                    # them. A regression slope uses every sample, averaging out
                    # per-frame noise instead of inheriting it from whichever 2 frames
                    # happen to be the endpoints.
                    # Minimum count/span raised from 3/0.05s: those were fine when the
                    # buffer was always full (25 samples), but clearing it on hover-reset
                    # and now on detection gaps too means it also needs to survive the
                    # refill period right after a clear. A 2-3 point fit spanning
                    # ~60-90ms turns ordinary PnP position noise (a few tenths of a
                    # metre, normal at 20m+ range) into several m/s of apparent
                    # velocity — confirmed in the same flight log, immediately after a
                    # hover-reset buffer clear. Better to report no PnP velocity for the
                    # ~0.3s refill period than a noise-dominated one.
                    if len(self._pnp_vel_buf) >= self._vel_min_frames:
                        _ts   = np.array([t for t, _, _ in self._pnp_vel_buf])
                        _dt_win = float(_ts[-1] - _ts[0])
                        if _dt_win >= self._vel_min_span_s:
                            _R_now = self._quat_to_R(_gyro_q)
                            # Sanity cap: if the total rotation implied is extreme
                            # (near-tumbling), first-order integration and the
                            # small-window-translation model both break down —
                            # skip rather than trust a degenerate fit.
                            _q_oldest = self._pnp_vel_buf[0][2]
                            _R_oldest = self._quat_to_R(_q_oldest)
                            _cos = (np.trace(_R_now.T @ _R_oldest) - 1.0) / 2.0
                            _total_rot = float(np.arccos(np.clip(_cos, -1.0, 1.0)))
                            if _total_rot <= self._vel_max_rotation_rad:
                                # q represents body(i)->reference (same convention as
                                # mav_state['quat'] elsewhere in this file). To express
                                # a body(i)-frame vector in the CURRENT body frame:
                                # body(i) -> reference -> body(now), i.e.
                                # R(q_now).T @ R(q_i) @ p_i.
                                _relpos = np.array([
                                    _R_now.T @ (self._quat_to_R(_q) @ _p)
                                    for _, _p, _q in self._pnp_vel_buf
                                ])   # each sample de-rotated into the current body frame
                                _ts_c   = _ts - _ts[0]
                                _A      = np.vstack([_ts_c, np.ones_like(_ts_c)]).T
                                _coef   = np.linalg.lstsq(_A, _relpos, rcond=None)[0]  # (2,3): [slope; intercept] per axis
                                _v_gate_rel_body = _coef[0]   # d(t_gate_body)/dt, current-body-frame
                                # Gate is stationary: v_drone_ned = -R_b2n @ d(t_gate_body)/dt,
                                # rotated by the CURRENT attitude only (see comment above).
                                _vel_body = -(R_b2n @ _v_gate_rel_body)
                                _sigma_body = self._ols_slope_sigma(_A, _relpos, _coef)

                    # Second, independent estimate: regress the already-computed
                    # drone_ned (world-NED position) directly. No de-rotation
                    # needed here — these samples are already in a common frame
                    # — so this is exactly as simple as it looks, at the cost of
                    # the file-header comment's concern: each sample carries its
                    # OWN tick's attitude estimate baked in, so if attitude drifts
                    # or glitches between samples, this regression inherits it
                    # directly (unlike the body-frame one above, which only uses
                    # attitude once, at the end). That tradeoff is exactly why
                    # this runs ALONGSIDE the body-frame estimate rather than
                    # replacing it: the two have different, largely independent
                    # failure modes (in-window rotation sweep vs. per-sample
                    # attitude/position glitches), so cross-checking them against
                    # each other catches whichever one a single method would have
                    # silently trusted. Confirmed against a real flight log: a
                    # single bad PnP position sample produced an 18 m/s spurious
                    # body-frame-regression velocity that only decayed back to
                    # correct over ~0.3s as the buffer refilled — the same bad
                    # sample would have thrown the NED-position regression off by
                    # a comparable amount at the same instant, which is exactly
                    # the kind of disagreement the check below is meant to catch.
                    if len(self._pnp_vel_ned_buf) >= self._vel_min_frames:
                        _ts2 = np.array([t for t, _ in self._pnp_vel_ned_buf])
                        if float(_ts2[-1] - _ts2[0]) >= self._vel_min_span_s:
                            _pos2   = np.array([p for _, p in self._pnp_vel_ned_buf])
                            _ts2_c  = _ts2 - _ts2[0]
                            _A2     = np.vstack([_ts2_c, np.ones_like(_ts2_c)]).T
                            _coef2  = np.linalg.lstsq(_A2, _pos2, rcond=None)[0]
                            _vel_pos = _coef2[0]   # d(drone_ned)/dt
                            _sigma_pos_vel = self._ols_slope_sigma(_A2, _pos2, _coef2)

            # Combine the two independent estimates. Agreement is evidence
            # neither buffer currently contains a corrupted sample — average
            # them for a less noisy result. Disagreement beyond
            # vision_vel_cross_check_tol means at least one is compromised
            # (bad position sample, in-window rotation sweep, gate-reference
            # mismatch, ...) and — critically — the two methods' failure modes
            # are different enough that we have no principled way to tell
            # which one to believe, so reject both rather than guess. Falls
            # back to whichever single estimate is available if the other
            # method's buffer hasn't filled yet (they normally fill in lockstep
            # since both append once per frame, so this is rare in practice).
            if _vel_body is not None and _vel_pos is not None:
                if float(np.linalg.norm(_vel_body - _vel_pos)) <= self._vel_cross_check_tol:
                    # Inverse-variance weighted average, using each regression's
                    # own per-tick fit-quality sigma (see _ols_slope_sigma)
                    # instead of a flat 50/50 mean — whichever fit was actually
                    # better (longer/cleaner window) counts for more.
                    _w_body = 1.0 / max(_sigma_body**2, 1e-6)
                    _w_pos  = 1.0 / max(_sigma_pos_vel**2, 1e-6)
                    vel_ned_pnp = (_w_body * _vel_body + _w_pos * _vel_pos) / (_w_body + _w_pos)
                    _sigma_vel_dyn = float(np.sqrt(1.0 / (_w_body + _w_pos)))
                else:
                    vel_ned_pnp = None
            else:
                vel_ned_pnp = _vel_body if _vel_body is not None else _vel_pos
                _sigma_vel_dyn = _sigma_body if _vel_body is not None else _sigma_pos_vel

            if vel_ned_pnp is not None:
                # Gate PnP velocity against max(current_speed, v_ref) × factor.
                # Dynamic: at high flight speeds the gate scales with actual speed so
                # valid high-velocity estimates are not rejected.
                _v_now = float(np.linalg.norm(
                    self.data.get('mav_state', {}).get('vel_ned', np.zeros(3))))
                _vel_max = max(_v_now, self._v_ref) * self._vel_max_factor
                if float(np.linalg.norm(vel_ned_pnp)) > _vel_max:
                    vel_ned_pnp = None
                    _sigma_vel_dyn = None

            if drone_ned is not None or yaw_ned is not None or vel_ned_pnp is not None:
                # Position sigma grows with range: PnP lateral accuracy is sub-metre
                # right at the gate but degrades over distance (keypoint pixel noise
                # and any residual attitude error both scale into lateral position
                # error roughly linearly with range — confirmed via check_yolo.py
                # replay analysis: ~0.1-0.6 m error at 5-30 m once the GT-mode yaw
                # sign bug was fixed, vs. several metres of raw PnP scatter beyond
                # ~15-20 m before that). sigma_pos = base + k*range lets the EKF
                # trust close-range fixes tightly while discounting far ones instead
                # of applying one fixed sigma to both regimes.
                _sigma_pos_dyn = (self._ekf_vis_sigma
                                   + self._ekf_vis_sigma_k * float(tvec_cam[2]))
                # Same range-scaling idea applied to yaw/attitude (see
                # ekf_vision_yaw_sigma_range_k/ekf_vision_att_sigma_range_k in
                # params.yaml): the rotation solve comes from the same corner
                # detections as position, so it degrades with range the same
                # way. k=0 (default) reproduces today's flat-sigma behaviour.
                _sigma_yaw_dyn = (self._ekf_vis_yaw_sigma
                                   + self._ekf_vis_yaw_sigma_k * float(tvec_cam[2]))
                _sigma_att_dyn = (self._ekf_vis_att_sigma
                                   + self._ekf_vis_att_sigma_k * float(tvec_cam[2]))
                # Per-tick fit-quality sigma (see _ols_slope_sigma), floored at
                # ekf_vision_vel_sigma so a near-singular fit can't produce an
                # unrealistically small sigma. Lets update_velocity's Kalman
                # gain trust a long/clean-window fit more and a short/noisy one
                # (e.g. right after a reacquisition, before the buffer refills)
                # less, instead of one fixed sigma for every fit regardless of
                # actual quality.
                _sigma_vel_out = (max(_sigma_vel_dyn, self._ekf_vis_vel_sigma)
                                   if _sigma_vel_dyn is not None else self._ekf_vis_vel_sigma)
                self.data['_vision_ekf_update'] = {
                    'pos_ned':   drone_ned,
                    'vel_ned':   vel_ned_pnp,
                    'yaw_ned':   yaw_ned,
                    'roll_ned':  roll_ned,
                    'pitch_ned': pitch_ned,
                    'sigma_pos': _sigma_pos_dyn,
                    'sigma_vel': _sigma_vel_out,
                    'sigma_yaw': _sigma_yaw_dyn,
                    'sigma_att': _sigma_att_dyn,
                    'yaw_gate':  self._ekf_vis_yaw_gate,
                    'att_gate':  self._ekf_vis_att_gate,
                    'gate':      self._ekf_vis_gate,
                    'vel_gate':  self._ekf_vis_vel_gate,
                    'wall_t':    time.time(),   # item 4: capture timestamp for latency compensation
                }
                # Log the raw fix regardless of ground_truth_mode (which only
                # affects whether imu_ekf.py consumes it) — feeds ekf_shadow.py's
                # offline replay.
                if self.logger:
                    _vu = self.data['_vision_ekf_update']
                    self.logger.log_vision_fix(
                        wall_t=_vu['wall_t'], pos_ned=_vu['pos_ned'], vel_ned=_vu['vel_ned'],
                        yaw_ned=_vu['yaw_ned'], sigma_pos=_vu['sigma_pos'], sigma_vel=_vu['sigma_vel'],
                        sigma_yaw=_vu['sigma_yaw'], gate=_vu['gate'], vel_gate=_vu['vel_gate'],
                        yaw_gate=_vu['yaw_gate'],
                        roll_ned=_vu['roll_ned'], pitch_ned=_vu['pitch_ned'],
                        sigma_att=_vu['sigma_att'], att_gate=_vu['att_gate'])

            if drone_ned is not None:
                _dist_m = float(tvec_cam[2])
                self._stat_pnp      += 1
                self._stat_dist_sum += _dist_m
                self.data['_vision_pnp_record'] = {
                    'drone_ned':   drone_ned.copy(),
                    'vel_ned_pnp': vel_ned_pnp.copy() if vel_ned_pnp is not None else None,
                    'dist_m':      _dist_m,
                    'conf':        conf,
                    't_wall':      time.time(),
                    'frame_id':    frame_id,
                }

        # ── Unified pose estimate (pose_estimate.py) ────────────────────────
        # The Layer1(perception)->Layer2(reference-shaping) contract for the
        # gate-transition rewrite. Built from the SAME underlying values as
        # gate_detection/_vision_ekf_update/_vision_pnp_record above
        # (drone_ned, vel_ned_pnp, yaw_ned, roll_ned, pitch_ned) — this is an
        # additional, additive view onto that data, not a replacement for
        # those dicts' existing shapes/consumers, which see zero change.
        #
        # Pose fields persist (with fresh=False) across a tolerated-miss
        # tick while self._lock is still ACQUIRING/LOCKED — the relocated
        # replacement for controller.py's old _vis_tvec_hold cache, now
        # owned by the layer that actually knows whether a value is still
        # trustworthy — and clear entirely once the lock drops to UNLOCKED
        # (nothing left worth holding).
        _gate_bearing_body_m = (self._R_cam2body @ tvec_cam) if tvec_cam is not None else None
        _pose_fresh = bool(_lock_accepted and drone_ned is not None)
        if _pose_fresh:
            self._last_pose = PoseEstimate(
                state=self._lock.state, fresh=True,
                t_capture=time.time(), frame_id=frame_id,
                gate_id=self.data.get('active_gate_index'),
                position_ned=drone_ned, velocity_ned=vel_ned_pnp,
                yaw=yaw_ned, roll=roll_ned, pitch=pitch_ned,
                gate_bearing_body_m=_gate_bearing_body_m,
                range_m=float(tvec_cam[2]), centre_px=centre_px,
            )
        elif self._lock.state == LockState.UNLOCKED:
            self._last_pose = PoseEstimate.blind()
        else:
            # Tolerated miss (or an accepted-but-position-less frame, e.g.
            # distance-only fallback with no gate_info) while still
            # ACQUIRING/LOCKED — hold the last known pose, marked not-fresh,
            # with state/gate_id refreshed to this tick's current values.
            self._last_pose = dataclasses.replace(
                self._last_pose, state=self._lock.state, fresh=False,
                gate_id=self.data.get('active_gate_index'))
        self.data['pose_estimate'] = self._last_pose

        # Diagnostics: print frame rate + detection rate every 5 s
        self._stat_proc += 1
        if detected:
            self._stat_det += 1
        if _pnp_skip_reason is not None:
            self._stat_pnp_skip_reason = _pnp_skip_reason
        now_s = time.time()
        elapsed = now_s - self._stat_t0
        if elapsed >= 5.0:
            recv_fps  = self._stat_recv / elapsed
            proc_fps  = self._stat_proc / elapsed
            det_fps   = self._stat_det  / elapsed
            det_pct   = 100.0 * self._stat_det / max(1, self._stat_proc)
            pnp_pct   = 100.0 * self._stat_pnp  / max(1, self._stat_proc)
            avg_dist  = (self._stat_dist_sum / self._stat_pnp
                         if self._stat_pnp > 0 else float('nan'))
            conf_str  = f"{conf:.2f}" if detected else "—"
            dist_str  = (f"{avg_dist:.1f}m avg ({self._stat_pnp}frames)"
                         if self._stat_pnp > 0 else "no PnP")
            orange_kpx = orange_px / 1000
            skip_str = (f"  pnp_skip={self._stat_pnp_skip_reason}"
                        if getattr(self, '_stat_pnp_skip_reason', None) else "")
            agi_str  = str(self.data.get('active_gate_index', '?'))
            ned_str  = (f"[{drone_ned[0]:.1f},{drone_ned[1]:.1f},{drone_ned[2]:.1f}]"
                        if drone_ned is not None else "no-pos")
            vel_str  = (f"{np.linalg.norm(vel_ned_pnp):.1f}m/s"
                        if vel_ned_pnp is not None else "no-vel")
            print(
                f"[VISION] recv={recv_fps:.1f}fps  proc={proc_fps:.1f}fps  "
                f"det={det_fps:.1f}fps ({det_pct:.0f}%)  "
                f"pnp={pnp_pct:.0f}%  dist={dist_str}  "
                f"conf={conf_str}  agi={agi_str}  "
                f"pos={ned_str}  vel={vel_str}  "
                f"orange={orange_kpx:.0f}kpx{skip_str}",
                flush=True,
            )
            self._stat_recv     = 0
            self._stat_proc     = 0
            self._stat_det      = 0
            self._stat_pnp      = 0
            self._stat_dist_sum = 0.0
            self._stat_t0       = now_s
            self._stat_pnp_skip_reason = None

        # Log and annotate
        if self.logger:
            self.logger.log_vision(
                wall_t    = now_s,
                frame_id  = frame_id,
                detected  = detected,
                conf      = conf,
                bb        = best_box,
                corners   = corners,
                tvec      = tvec_cam,
                drone_ned = drone_ned,
                vel_ned   = vel_ned_pnp,
                gate_ned  = gate_info['ned'] if gate_info is not None else None,
                skip_reason       = _pnp_skip_reason,
                gate_id_rejected  = _gate_id_rejected,
            )

            annotated = img.copy()
            if detected and corners is not None and centre_px is not None:
                for pt in corners:
                    cv2.circle(annotated, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)
                for i in range(4):
                    p1 = (int(corners[i][0]),          int(corners[i][1]))
                    p2 = (int(corners[(i+1)%4][0]),    int(corners[(i+1)%4][1]))
                    cv2.line(annotated, p1, p2, (0, 255, 0), 2)
                cx, cy = int(centre_px[0]), int(centre_px[1])
                cv2.drawMarker(annotated, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 15, 2)
                dist_txt = f"{tvec_cam[2]:.1f}m" if tvec_cam is not None else "no PnP"
                vel_txt  = (f" v={np.linalg.norm(vel_ned):.1f}m/s"
                            if vel_ned is not None else "")
                cv2.putText(annotated,
                            f"conf={conf:.2f} d={dist_txt}{vel_txt}",
                            (max(0, cx - 70), max(12, cy - 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (0, 255, 255), 1, cv2.LINE_AA)

            if self._debug_overlay:
                annotated = self._draw_overlay(annotated, tvec_cam, centre_px, corners, vel_ned_pnp,
                                                pnp_skip_reason=_pnp_skip_reason,
                                                gate_id_rejected=_gate_id_rejected)

            # Always save frames with a detection; rate-limit background frames
            self.logger.log_frame(frame_id, annotated, force=detected,
                                  sim_time_ns=sim_time_ns)

