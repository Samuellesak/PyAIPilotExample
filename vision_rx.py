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

SIM_SERVER_UDP_IP   = "0.0.0.0"
SIM_SERVER_UDP_PORT = 5600


class VisionRX:

    def __init__(self, data, logger=None):
        self.data   = data
        self.logger = logger

        _p = load_params("params.yaml")
        self._cam_K = np.array([
            [_p.get('cam_fx', 320.), 0.,                   _p.get('cam_cx', 320.)],
            [0.,                   _p.get('cam_fy', 320.), _p.get('cam_cy', 240.)],
            [0.,                   0.,                     1.                    ],
        ], dtype=np.float64)
        self._ekf_vis_sigma     = float(_p.get('ekf_vision_sigma',     0.5))
        self._ekf_vis_gate      = float(_p.get('ekf_vision_gate',      10.0))
        self._ekf_vis_vel_sigma = float(_p.get('ekf_vision_vel_sigma', 1.0))
        self._ekf_vis_vel_gate  = float(_p.get('ekf_vision_vel_gate',  5.0))
        self._ekf_vis_yaw_sigma = float(_p.get('ekf_vision_yaw_sigma', 0.1))
        self._ekf_vis_yaw_gate  = float(_p.get('ekf_vision_yaw_gate',  1.0))
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
        if self._debug_waypoints_only:
            print("[VisionRX] debug_waypoints_only=true — YOLO disabled", flush=True)

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
        self._locked_dist   = None   # None = lock not yet acquired
        self._consec_det    = 0      # consecutive frames with valid PnP
        self._consec_miss   = 0      # consecutive frames without valid PnP
        self._prev_agi      = None   # previous active_gate_index for change detection
        # gate_index → last gate_info dict received from track data (includes ned, width, height, quat).
        # Used as fallback when track_gates_ned is temporarily unavailable.
        self._last_gate_info = {}

        # Image preprocessing: unsharp mask (sharpening) + optional Gaussian denoise.
        # Sharpening restores orange saturation lost to motion blur at speed, improving
        # HSV detection rate without widening the colour gate (no false-positive increase).
        self._sharpen_k     = int(_p.get('preproc_sharpen_k',      5))
        self._sharpen_alpha = float(_p.get('preproc_sharpen_alpha', 0.8))
        self._gauss_k       = int(_p.get('preproc_gauss_k',         0))

        # cam→FRD body: x_b(fwd)=cam_z, y_b(right)=cam_x, z_b(down)=cam_y
        self._R_cam2body = np.array([[0, 0, 1],
                                     [1, 0, 0],
                                     [0, 1, 0]], dtype=float)

        # PnP velocity estimation: sliding window of (wall_t, drone_ned) pairs.
        # Velocity = (last_ned - first_ned) / (last_t - first_t) over the window.
        # Averaging over N frames reduces differentiation noise by ≈ N× vs 2-frame diff.
        _vel_win = int(_p.get('vision_vel_window_frames', 10))
        self._pnp_vel_buf = deque(maxlen=max(2, _vel_win))

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

        self._model = _YOLO("YOLO/best.pt")
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
            frame_id, chunk_id, total_chunks, jpeg_size, payload_size, _ = \
                struct.unpack(header_format, header)

            if frame_id not in frames:
                frames[frame_id] = {"chunks": {}, "total": total_chunks}
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
                    if img is not None:
                        self._stat_recv += 1
                        # Drop oldest frame if inference is lagging; keep latest.
                        if self._frame_q.full():
                            try:
                                self._frame_q.get_nowait()
                            except queue.Empty:
                                pass
                        try:
                            self._frame_q.put_nowait((frame_id, img))
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
            frame_id, img = item
            self.process_frame(frame_id, img)

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

    def _draw_overlay(self, img, tvec_cam, centre_px, corners, vel_ned_pnp):
        """Draw live debug overlay onto a copy of img and return it."""
        vis = img.copy()
        h, w = vis.shape[:2]
        ic_x, ic_y = w // 2, h // 2   # image centre

        ctrl_mode = self.data.get('vis_ctrl_mode', 'CARROT')

        # ── Mode badge ────────────────────────────────────────────────────────
        _MODE_CFG = {
            'PNP':        ((0,  200,  0),  'PNP'),
            'HOLD':       ((0,  200, 220), 'HOLD'),
            'TRANSITION': ((30, 140, 255), 'TRANSIT'),
            'CARROT':     ((120,120, 120), 'CARROT'),
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

        # Choose arrow target: live centre, or cached centre when holding/transit.
        if ctrl_mode in ('HOLD', 'TRANSITION') and self._overlay_last_ctr is not None:
            arrow_target = self._overlay_last_ctr
        elif centre_px is not None:
            arrow_target = (int(centre_px[0]), int(centre_px[1]))
        else:
            arrow_target = None

        if arrow_target is not None:
            arrow_col = badge_color
            # Dashed style for HOLD/TRANSITION: draw segmented line then arrowhead.
            if ctrl_mode in ('HOLD', 'TRANSITION'):
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
                _qw, _qx, _qy, _qz = _mav_ov['quat']
                _Rb2n = np.array([
                    [1-2*(_qy*_qy+_qz*_qz),  2*(_qx*_qy-_qw*_qz),  2*(_qx*_qz+_qw*_qy)],
                    [  2*(_qx*_qy+_qw*_qz),1-2*(_qx*_qx+_qz*_qz),  2*(_qy*_qz-_qw*_qx)],
                    [  2*(_qx*_qz-_qw*_qy),  2*(_qy*_qz+_qw*_qx),1-2*(_qx*_qx+_qy*_qy)],
                ], dtype=float)
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

    def _pnp_gate(self, corners_px, gate_width, gate_height):
        """Returns (tvec, rvec) both as flat (3,) arrays, or (None, None) on failure.

        IPPE produces two solutions with nearly equal reprojection error for
        near-frontal views.  The horizontal-gate constraint resolves the ambiguity:
        the gate Y-axis (downward direction in gate frame) must point toward NED-down.
        We fetch both solutions and pick the one whose gate_Y_cam best aligns with
        NED-down in camera frame (derived from EKF attitude if available).
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

        # NED-down direction expressed in camera frame.
        # For a level drone with forward-facing camera: cam_Y = NED-down.
        # Use EKF attitude to correct for any drone tilt.
        ned_down_cam = np.array([0.0, 1.0, 0.0])    # cam_Y fallback (level drone)
        mav = self.data.get('mav_state')
        if mav is not None:
            qw, qx, qy, qz = mav['quat']
            R_b2n = np.array([
                [1-2*(qy*qy+qz*qz),   2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)],
                [  2*(qx*qy+qw*qz), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qw*qx)],
                [  2*(qx*qz-qw*qy),   2*(qy*qz+qw*qx), 1-2*(qx*qx+qy*qy)],
            ], dtype=float)
            ned_down_body = R_b2n.T @ np.array([0.0, 0.0, 1.0])
            ned_down_cam  = self._R_cam2body.T @ ned_down_body

        # Pick the solution where gate_Y (col 1 of R_gate2cam) most aligns with
        # NED-down in camera frame.  Gate Y = downward in gate frame = NED-down
        # in the world when the gate is horizontal.
        best_rvec, best_tvec = rvecs[0].flatten(), tvecs[0].flatten()
        best_score = -np.inf
        for rv, tv in zip(rvecs, tvecs):
            tv_f = tv.flatten()
            if tv_f[2] < 0.1:          # gate behind camera — physically impossible
                continue
            R_sol, _ = cv2.Rodrigues(rv)
            score = float(np.dot(R_sol[:, 1], ned_down_cam))
            if score > best_score:
                best_score = score
                best_rvec, best_tvec = rv.flatten(), tv_f

        if best_tvec[2] < 0.5:
            return None, None
        return best_tvec, best_rvec

    def _yaw_from_pnp(self, rvec, gate_quat_wxyz):
        """
        Derive drone yaw (NED, radians) from PnP rvec + known gate orientation.

        The gate y-axis is world-Down (gate is perfectly horizontal), and with
        roll=0 the drone body-z is also world-Down, so the only free attitude
        degree of freedom is yaw.  The gate quaternion from track data gives the
        gate frame orientation in NED, completing the chain:
            R_b2n = R_gate2ned @ R_gate2cam.T @ R_cam2body.T
        """
        R_gate2cam, _ = cv2.Rodrigues(rvec.reshape(3, 1))
        qw, qx, qy, qz = gate_quat_wxyz
        R_gate2ned = np.array([
            [1 - 2*(qy*qy + qz*qz),  2*(qx*qy - qw*qz),  2*(qx*qz + qw*qy)],
            [    2*(qx*qy + qw*qz),  1 - 2*(qx*qx + qz*qz),  2*(qy*qz - qw*qx)],
            [    2*(qx*qz - qw*qy),  2*(qy*qz + qw*qx),  1 - 2*(qx*qx + qy*qy)],
        ], dtype=float)
        R_cam2ned = R_gate2ned @ R_gate2cam.T
        R_b2n     = R_cam2ned  @ self._R_cam2body.T
        return float(np.arctan2(R_b2n[1, 0], R_b2n[0, 0]))

    # ── Main frame processing ───────────────────────────────────────────────

    def process_frame(self, frame_id, img):
        if self._debug_waypoints_only:
            self.data['gate_detection'] = {
                'detected': False, 'centre_px': None, 'conf': 0.0,
                'tvec_cam': None, 'rvec_cam': None, 'frame_id': frame_id,
            }
            if self.logger:
                self.logger.log_frame(frame_id, img)
            return

        img = self._preprocess(img)
        # Desaturate blue/cyan pixels (S=0 → grey) before YOLO and orange mask.
        # Desaturation preserves luminance so YOLO keeps structural context from
        # buildings and the track beam, while removing the hue that would
        # contaminate the orange mask or produce cyan false-positive detections.
        img = self._apply_blue_suppression(img)
        masked, mask_raw = self._orange_mask(img)
        orange_px = int(np.count_nonzero(mask_raw))

        results = self._model.predict(img, verbose=False, conf=0.03) if self._yolo_enabled else None

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

        h_img, w_img = mask_raw.shape
        if results is not None and results[0].boxes is not None \
                and results[0].keypoints is not None and len(results[0].boxes) > 0:
            r = results[0]
            boxes = r.boxes.xywh.cpu().numpy()   # (N,4): cx,cy,w,h
            confs = r.boxes.conf.cpu().numpy()
            kpts  = r.keypoints.xy.cpu().numpy() # (N,4,2)

            # Two-pass selection: filter → nearest by area.
            # Pass 1: reject boxes with conf < 0.03, zero keypoints, or
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
                if confs[_i] < 0.03:
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

            if valid_idx:
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
                    gate_info = gates[int(agi)]
                    # Cache so later frames can use it as a fallback when
                    # track_gates_ned is temporarily unavailable.
                    self._last_gate_info[int(agi)] = gate_info
                    tvec_cam, rvec_cam = self._pnp_gate(
                        corners, gate_info['width'], gate_info['height'])
                    if tvec_cam is None:
                        _pnp_skip_reason = "solvePnP failed"
                else:
                    # No live track data for this gate.
                    # Fallback priority:
                    #   1. Last known gate_info from track data (accurate dims + NED)
                    #   2. Approximate NED from params.yaml waypoints
                    #   3. Distance-only (no position update)
                    _last = self._last_gate_info.get(int(agi)) if agi is not None else None
                    _w = _last['width']  if _last else self._gate_w_default
                    _h = _last['height'] if _last else self._gate_h_default
                    tvec_cam, rvec_cam = self._pnp_gate(corners, _w, _h)
                    gate_info = None
                    if tvec_cam is not None and agi is not None:
                        if _last is not None:
                            gate_info = _last
                        elif self._waypoints is not None:
                            wp_idx = int(agi) + 1
                            if wp_idx < len(self._waypoints):
                                gate_info = {
                                    'ned':    self._waypoints[wp_idx].copy(),
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

                # ── Next-gate candidate (second-largest valid box) ────────────
                # Accumulates PnP-derived NED positions across many frames and
                # publishes a confirmed median position once the buffer is full.
                # Guard: second gate must be farther than current gate by ≥5 m so
                # duplicate detections of the same gate are rejected.
                _cur_dist = float(tvec_cam[2]) if tvec_cam is not None else 0.0
                if _second_idx is not None:
                    _sec_corners = kpts[_second_idx]
                    # Gate dimensions for the next gate index
                    _agi_next = (int(agi) + 1) if agi is not None else None
                    _ng_cache = self._last_gate_info.get(_agi_next) if _agi_next is not None else None
                    if _agi_next is not None and _agi_next in gates:
                        _ng_w = gates[_agi_next]['width']
                        _ng_h = gates[_agi_next]['height']
                    elif _ng_cache is not None:
                        _ng_w, _ng_h = _ng_cache['width'], _ng_cache['height']
                    else:
                        _ng_w, _ng_h = self._gate_w_default, self._gate_h_default
                    _tvec_ng, _ = self._pnp_gate(_sec_corners, _ng_w, _ng_h)
                    if (_tvec_ng is not None
                            and _tvec_ng[2] > _cur_dist + 5.0   # farther than current gate
                            and _tvec_ng[2] < self._max_gate_dist):
                        _mav_ng = self.data.get('mav_state')
                        if _mav_ng is not None:
                            _qw, _qx, _qy, _qz = _mav_ng['quat']
                            _R_b2n_ng = np.array([
                                [1-2*(_qy*_qy+_qz*_qz),  2*(_qx*_qy-_qw*_qz),  2*(_qx*_qz+_qw*_qy)],
                                [  2*(_qx*_qy+_qw*_qz),1-2*(_qx*_qx+_qz*_qz),  2*(_qy*_qz-_qw*_qx)],
                                [  2*(_qx*_qz-_qw*_qy),  2*(_qy*_qz+_qw*_qx),1-2*(_qx*_qx+_qy*_qy)],
                            ], dtype=float)
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

        # ── Gate distance lock ────────────────────────────────────────────────
        # Reset lock when the sim advances to the next gate (active_gate_index changes).
        agi = self.data.get('active_gate_index')
        if agi != self._prev_agi and self._prev_agi is not None:
            self._locked_dist = None
            self._consec_det  = 0
            self._consec_miss = 0
            self._pnp_vel_buf.clear()        # stale positions from old gate are invalid
            self._next_gate_ned_buf.clear()  # next-gate buffer also invalid after advance
            self._next_gate_ned = None
            print(f"[VISION] gate index {self._prev_agi}→{agi}: lock reset", flush=True)
        self._prev_agi = agi

        if tvec_cam is not None:
            dist = tvec_cam[2]
            if self._locked_dist is not None and dist > self._locked_dist + self._spike_tol:
                # Distance spike — almost certainly a different, farther gate.
                # Nullify this frame's PnP output so EKF and carrot target are not corrupted.
                tvec_cam = None
                rvec_cam = None
                self._consec_miss += 1
                if self._consec_miss >= self._lock_miss_max:
                    self._locked_dist = None
                    self._consec_det  = 0
                    self._consec_miss = 0
            else:
                self._consec_miss = 0
                self._consec_det += 1
                if self._locked_dist is None:
                    if self._consec_det >= self._lock_frames:
                        self._locked_dist = dist
                        print(f"[VISION] gate locked at {dist:.1f}m "
                              f"after {self._consec_det} frames", flush=True)
                else:
                    self._locked_dist = dist   # track distance as drone approaches
        else:
            self._consec_miss += 1
            if self._consec_miss >= self._lock_miss_max:
                self._locked_dist = None
                self._consec_det  = 0
                self._consec_miss = 0

        # ── Orange-centroid fallback ──────────────────────────────────────────
        # When YOLO fails, find the centroid of the orange blob in the tight mask.
        # This gives a bearing to the gate center even without keypoints or PnP.
        # The centroid is published as centre_px so the controller can use it for
        # visual yaw correction (keep gate centered).
        centroid_area = 0
        if not detected:
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
            'pnp_locked':    self._locked_dist is not None,
            'frame_id':      frame_id,
        }

        # EKF vision update: position + velocity + yaw from PnP.
        vel_ned     = None
        vel_ned_pnp = None
        drone_ned   = None
        yaw_ned     = None

        if tvec_cam is not None:
            # Yaw estimate: requires gate quaternion from track data.
            # Unavailable in the pure-fallback path (gate_info=None).
            _pnp_flipped = False   # True when rvec was from the wrong PnP branch
            if gate_info is not None and gate_info.get('quat') is not None:
                try:
                    yaw_ned = self._yaw_from_pnp(rvec_cam, gate_info['quat'])
                    # Resolve PnP rotational ambiguity for square gates (4-fold symmetry).
                    # solvePnP can return a pose off by any multiple of 90°. Round the
                    # innovation to the nearest 90° step and subtract it to recover the
                    # correct yaw.  Safe as long as the drone's true yaw deviation from
                    # the EKF is < 45° — which holds for normal gate-approach geometry.
                    # IMPORTANT: when a flip is detected, tvec is also from the wrong
                    # PnP branch → position/velocity updates for this frame are skipped.
                    _mav = self.data.get('mav_state')
                    if _mav is not None:
                        _qw, _qx, _qy, _qz = _mav['quat']
                        _yaw_ekf = np.arctan2(2*(_qw*_qz + _qx*_qy),
                                              1 - 2*(_qy*_qy + _qz*_qz))
                        _innov = (yaw_ned - _yaw_ekf + np.pi) % (2*np.pi) - np.pi
                        _n = round(_innov / (np.pi / 2))
                        if _n != 0:
                            yaw_ned = (yaw_ned - _n * np.pi / 2 + np.pi) % (2*np.pi) - np.pi
                            _pnp_flipped = True
                except Exception:
                    yaw_ned = None

            # Position + velocity: require known gate NED (from track data or waypoint fallback).
            # Skip when the PnP rotation was flipped — tvec from the wrong branch is also wrong.
            if gate_info is not None and not _pnp_flipped:
                mav = self.data.get('mav_state')
                if mav is not None:
                    qw, qx, qy, qz = mav['quat']
                    R_b2n = np.array([
                        [1-2*(qy*qy+qz*qz),  2*(qx*qy-qw*qz),  2*(qx*qz+qw*qy)],
                        [  2*(qx*qy+qw*qz),1-2*(qx*qx+qz*qz),  2*(qy*qz-qw*qx)],
                        [  2*(qx*qz-qw*qy),  2*(qy*qz+qw*qx),1-2*(qx*qx+qy*qy)],
                    ])
                    t_gate_body = self._R_cam2body @ tvec_cam
                    t_gate_ned  = R_b2n @ t_gate_body
                    drone_ned   = gate_info['ned'] - t_gate_ned

                    now = time.time()
                    self._pnp_vel_buf.append((now, drone_ned.copy()))
                    # Windowed velocity: endpoint difference over all buffered samples.
                    # Using first vs last (not pairwise average) is equivalent and cheaper.
                    # Requires ≥ 2 entries and a sane time span to guard against stale data.
                    if len(self._pnp_vel_buf) >= 2:
                        _t0, _p0 = self._pnp_vel_buf[0]
                        _t1, _p1 = self._pnp_vel_buf[-1]
                        _dt_win = _t1 - _t0
                        if _dt_win >= 0.05:
                            vel_ned_pnp = (_p1 - _p0) / _dt_win

            if vel_ned_pnp is not None:
                # Gate PnP velocity against max(current_speed, v_ref) × factor.
                # Dynamic: at high flight speeds the gate scales with actual speed so
                # valid high-velocity estimates are not rejected.
                _v_now = float(np.linalg.norm(
                    self.data.get('mav_state', {}).get('vel_ned', np.zeros(3))))
                _vel_max = max(_v_now, self._v_ref) * self._vel_max_factor
                if float(np.linalg.norm(vel_ned_pnp)) > _vel_max:
                    vel_ned_pnp = None

            if yaw_ned is not None or vel_ned_pnp is not None:
                # Position updates disabled: PnP pos noise causes EKF path jumps.
                # Dead-reckoning (IMU velocity) is accurate enough over gate distances;
                # the controller gate-waypoint update handles navigation from PnP.
                # Velocity injection is enabled when consecutive PnP frames are available.
                self.data['_vision_ekf_update'] = {
                    'pos_ned':   None,
                    'vel_ned':   vel_ned_pnp,
                    'yaw_ned':   yaw_ned,
                    'sigma_pos': self._ekf_vis_sigma,
                    'sigma_vel': self._ekf_vis_vel_sigma,
                    'sigma_yaw': self._ekf_vis_yaw_sigma,
                    'yaw_gate':  self._ekf_vis_yaw_gate,
                    'gate':      self._ekf_vis_gate,
                    'vel_gate':  self._ekf_vis_vel_gate,
                }

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
                annotated = self._draw_overlay(annotated, tvec_cam, centre_px, corners, vel_ned_pnp)

            # Always save frames with a detection; rate-limit background frames
            self.logger.log_frame(frame_id, annotated, force=detected)

