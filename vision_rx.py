import queue
import socket
import struct
import threading
import time

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
        _v_ref              = float(_p.get('v_ref', 2.0))
        _vel_factor         = float(_p.get('vision_vel_max_factor', 2.5))
        self._vis_vel_max   = _v_ref * _vel_factor
        self._lock_frames   = int(_p.get('vision_lock_frames', 5))
        self._lock_miss_max = int(_p.get('vision_lock_miss',   30))
        self._spike_tol     = float(_p.get('vision_spike_tol', 5.0))
        self._locked_dist   = None   # None = lock not yet acquired
        self._consec_det    = 0      # consecutive frames with valid PnP
        self._consec_miss   = 0      # consecutive frames without valid PnP
        self._prev_agi      = None   # previous active_gate_index for change detection

        # Rotation from OpenCV camera frame (x=right,y=down,z=fwd) to FRD body
        self._R_cam2body = np.array([[0, 0, 1],
                                     [1, 0, 0],
                                     [0, 1, 0]], dtype=float)

        # State for PnP velocity estimation — consecutive-frame guard
        self._prev_drone_ned = None
        self._prev_vis_t     = None
        self._prev_vis_fid   = None

        # Diagnostics counters (reset every 5 s)
        self._stat_recv           = 0
        self._stat_proc           = 0
        self._stat_det            = 0
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

    def _orange_mask(self, img_bgr):
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        # Zero out the blue disturbance before orange masking so it cannot
        # bleed into the dilated YOLO input through the mask boundary.
        if self._suppress_blue:
            blue_px = cv2.inRange(hsv, self._blue_lo, self._blue_hi)
            img_bgr = img_bgr.copy()
            img_bgr[blue_px > 0] = 0
            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        lo   = np.array([ 5, 100,  80], dtype=np.uint8)
        hi   = np.array([25, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lo, hi)
        # Close small holes in the gate frame
        k_close  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask     = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
        # Dilate to add context around the gate corners so YOLO keypoint heads
        # can anchor on edge structure even when corners are at the mask boundary.
        k_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (41, 41))
        mask_inf = cv2.dilate(mask, k_dilate)
        # mask_raw (tight) used for orange-pixel count; mask_inf used for YOLO input
        return cv2.bitwise_and(img_bgr, img_bgr, mask=mask_inf), mask

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

    def _pnp_gate(self, corners_px, gate_width, gate_height):
        """Returns (tvec, rvec) both as flat (3,) arrays, or (None, None) on failure."""
        hw = gate_width  / 2.0
        hh = gate_height / 2.0
        obj_pts = np.array([
            [-hw, -hh, 0.0],
            [ hw, -hh, 0.0],
            [ hw,  hh, 0.0],
            [-hw,  hh, 0.0],
        ], dtype=np.float64)
        img_pts = corners_px.astype(np.float64).reshape(4, 1, 2)
        ok, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts, self._cam_K, np.zeros((4, 1)),
            flags=cv2.SOLVEPNP_IPPE,
        )
        if not ok:
            return None, None
        return tvec.flatten(), rvec.flatten()

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
        masked, mask_raw = self._orange_mask(img)
        orange_px = int(np.count_nonzero(mask_raw))

        # YOLO runs on the dilated-mask image: background distractors are blacked
        # out (detection robustness) but the mask is expanded by 41 px so that
        # gate corners sitting at the orange boundary get enough texture context
        # for keypoint heads to anchor correctly.
        results = self._model.predict(masked, verbose=False, conf=0.3)

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

        r = results[0]
        if r.boxes is not None and r.keypoints is not None and len(r.boxes) > 0:
            boxes = r.boxes.xywh.cpu().numpy()   # (N,4): cx,cy,w,h
            confs = r.boxes.conf.cpu().numpy()
            kpts  = r.keypoints.xy.cpu().numpy() # (N,4,2)

            # Pick detection with LARGEST bounding-box area (most prominent gate)
            areas = boxes[:, 2] * boxes[:, 3]
            best  = int(np.argmax(areas))

            if confs[best] > 0.3 and kpts[best].shape == (4, 2):
                # Require ≥15 % of the detection bbox to be orange (tight mask).
                # This rejects detections that landed on dilated-mask fringe area
                # with little actual orange content.
                h_img, w_img = mask_raw.shape
                bx1 = max(0, int(boxes[best][0] - boxes[best][2] / 2))
                by1 = max(0, int(boxes[best][1] - boxes[best][3] / 2))
                bx2 = min(w_img, int(boxes[best][0] + boxes[best][2] / 2))
                by2 = min(h_img, int(boxes[best][1] + boxes[best][3] / 2))
                bbox_area    = max(1, (bx2 - bx1) * (by2 - by1))
                orange_frac  = np.count_nonzero(mask_raw[by1:by2, bx1:bx2]) / bbox_area
                has_orange   = orange_frac >= 0.10

                if has_orange:
                    raw_corners = kpts[best]   # (4,2) pixel coords

                    # Reject if any corner is at (0,0) — YOLO sets undetected
                    # keypoints to origin, which makes solvePnP degenerate.
                    any_zero = np.any(np.all(raw_corners < 2.0, axis=1))

                    if not any_zero:
                        corners   = raw_corners
                        centre_px = corners.mean(axis=0)
                        conf      = float(confs[best])
                        best_box  = tuple(float(v) for v in boxes[best])
                        detected  = True

                        gates = self.data.get('track_gates_ned', {})
                        # Default to 0: before first RACE_STATUS the fallback
                        # still points at waypoints[1] (first gate).
                        agi   = self.data.get('active_gate_index', 0)
                        if agi is not None and int(agi) in gates:
                            gate_info           = gates[int(agi)]
                            tvec_cam, rvec_cam  = self._pnp_gate(
                                corners, gate_info['width'], gate_info['height'])
                            if tvec_cam is None:
                                _pnp_skip_reason = "solvePnP failed"
                        else:
                            # Fallback: solve PnP with default dims for distance readout.
                            tvec_cam, rvec_cam = self._pnp_gate(
                                corners, self._gate_w_default, self._gate_h_default)
                            gate_info = None
                            # When active_gate_index is known, supply approximate gate NED
                            # from waypoints so EKF position/velocity updates can run.
                            # Yaw update is skipped (gate orientation unknown without track data).
                            if tvec_cam is not None and agi is not None and self._waypoints is not None:
                                wp_idx = int(agi) + 1  # gate 0 → waypoint 1, gate 1 → waypoint 2, …
                                if wp_idx < len(self._waypoints):
                                    gate_info = {
                                        'ned':    self._waypoints[wp_idx].copy(),
                                        'width':  self._gate_w_default,
                                        'height': self._gate_h_default,
                                        'quat':   None,   # unknown → yaw update skipped
                                    }
                            if tvec_cam is None:
                                _pnp_skip_reason = "solvePnP failed (fallback dims)"
                            elif gate_info is None:
                                if not gates:
                                    _pnp_skip_reason = "no track_gates_ned, no agi (dist only)"
                                else:
                                    _pnp_skip_reason = "no active_gate_index (dist only)"
                            else:
                                _pnp_skip_reason = "no track_gates_ned (waypoint NED fallback)"

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
            'frame_id':      frame_id,
        }

        # EKF vision update: position + velocity + yaw from PnP.
        vel_ned   = None
        drone_ned = None
        yaw_ned   = None

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

                    # Velocity: allow up to 3 missed detections between valid PnP frames.
                    # Strict consecutive-frame check (==fid-1) was dropping all velocity
                    # updates whenever YOLO missed a single frame; the dt guard
                    # (0.01–0.15 s) still prevents stale or too-fast differences.
                    now = time.time()
                    if (self._prev_drone_ned is not None
                            and self._prev_vis_t   is not None
                            and 0 < frame_id - self._prev_vis_fid <= 3):
                        dt = now - self._prev_vis_t
                        if 0.01 < dt < 0.15:
                            _v = (drone_ned - self._prev_drone_ned) / dt
                            if np.linalg.norm(_v) <= self._vis_vel_max:
                                vel_ned = _v
                    self._prev_drone_ned = drone_ned
                    self._prev_vis_t     = now
                    self._prev_vis_fid   = frame_id

            if drone_ned is not None or yaw_ned is not None:
                self.data['_vision_ekf_update'] = {
                    'pos_ned':   drone_ned,    # None → position update skipped
                    'vel_ned':   vel_ned,
                    'yaw_ned':   yaw_ned,
                    'sigma_pos': self._ekf_vis_sigma,
                    'sigma_vel': self._ekf_vis_vel_sigma,
                    'sigma_yaw': self._ekf_vis_yaw_sigma,
                    'yaw_gate':  self._ekf_vis_yaw_gate,
                    'gate':      self._ekf_vis_gate,
                    'vel_gate':  self._ekf_vis_vel_gate,
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
            recv_fps = self._stat_recv / elapsed
            proc_fps = self._stat_proc / elapsed
            det_fps  = self._stat_det  / elapsed
            det_pct  = 100.0 * self._stat_det / max(1, self._stat_proc)
            conf_str = f"{conf:.2f}" if detected else "—"
            dist_str = f"{tvec_cam[2]:.1f}m" if tvec_cam is not None else "no PnP"
            orange_kpx = orange_px / 1000
            skip_str = (f"  pnp_skip={self._stat_pnp_skip_reason}"
                        if getattr(self, '_stat_pnp_skip_reason', None) else "")
            agi_str  = str(self.data.get('active_gate_index', '?'))
            ned_str  = (f"[{drone_ned[0]:.1f},{drone_ned[1]:.1f},{drone_ned[2]:.1f}]"
                        if drone_ned is not None else "no-pos")
            vel_str  = (f"{np.linalg.norm(vel_ned):.1f}m/s"
                        if vel_ned is not None else "no-vel")
            print(
                f"[VISION] recv={recv_fps:.1f}fps  proc={proc_fps:.1f}fps  "
                f"det={det_fps:.1f}fps ({det_pct:.0f}%)  "
                f"conf={conf_str}  dist={dist_str}  agi={agi_str}  "
                f"pos={ned_str}  vel={vel_str}  "
                f"orange={orange_kpx:.0f}kpx{skip_str}",
                flush=True,
            )
            self._stat_recv = 0
            self._stat_proc = 0
            self._stat_det  = 0
            self._stat_t0   = now_s
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
                vel_ned   = vel_ned,
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

            # Always save frames with a detection; rate-limit background frames
            self.logger.log_frame(frame_id, annotated, force=detected)

            # Save the orange mask alongside every saved annotated frame for debugging
            if detected:
                self.logger.log_frame(frame_id, masked, suffix="_mask")
