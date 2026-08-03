"""
check_yolo.py  —  Offline YOLO gate-detection analysis on captured flight frames.

Replays every JPEG frame saved during the last (or specified) flight, runs the
same YOLO + PnP pipeline as vision_rx.py, and compares the estimated drone NED
position to the EKF ground truth.

Output (saved next to the log):
  yolo_check.csv        per-frame results
  yolo_check.png        6-panel comparison plot
  annotated/            annotated frames (--save-annotated only)

Usage (from the repo root):
  python -m vision.check_yolo                        # auto-find latest log
  python -m vision.check_yolo logs/20260722_164557   # specific log dir
  python -m vision.check_yolo --save-annotated       # also write annotated frames
"""

import sys
import os
import glob
import csv
import re
import math

import cv2
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ultralytics import YOLO as _YOLO


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Locate log dir
# ─────────────────────────────────────────────────────────────────────────────

def find_latest_log():
    candidates = glob.glob('logs/*/')
    if not candidates:
        raise FileNotFoundError('No subdirectories found under logs/')
    return max(candidates, key=os.path.getmtime).rstrip('/\\')


save_annotated = '--save-annotated' in sys.argv
_model_arg = next((a.split('=', 1)[1] for a in sys.argv[1:] if a.startswith('--model=')), None)
_tag_arg   = next((a.split('=', 1)[1] for a in sys.argv[1:] if a.startswith('--tag=')),   None)
positional = [a for a in sys.argv[1:] if not a.startswith('--')]
log_dir = positional[0] if positional else find_latest_log()
print(f'Log dir : {log_dir}')


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Load params.yaml
# ─────────────────────────────────────────────────────────────────────────────

with open('params.yaml') as f:
    raw_yaml = f.read()
param = yaml.safe_load(raw_yaml)

cam_K = np.array([
    [param.get('cam_fx', 320.0), 0.0,                      param.get('cam_cx', 320.0)],
    [0.0,                        param.get('cam_fy', 320.0), param.get('cam_cy', 180.0)],
    [0.0,                        0.0,                      1.0],
], dtype=np.float64)
dist_coef = np.zeros((4, 1))

gate_w           = float(param.get('gate_width_default',  1.5))   # inner opening [m]
gate_h           = float(param.get('gate_height_default', 1.5))
max_gate_dist    = float(param.get('vision_max_gate_dist', 50.0))
ground_truth_mode = bool(param.get('ground_truth_mode', False))

# cam→body with upward tilt; reads cam_tilt_deg from params.yaml (positive = nose up)
_tilt = math.radians(float(param.get('cam_tilt_deg', 20.0)))
_st, _ct = math.sin(_tilt), math.cos(_tilt)
R_cam2body = np.array([[0, _st, _ct],
                        [1,  0,   0 ],
                        [0, _ct, -_st]], dtype=float)

# Blue-suppression HSV range (from params.yaml)
_blue_lo = np.array([
    int(param.get('suppress_blue_h_lo', 70)),
    int(param.get('suppress_blue_s_lo',  0)),
    int(param.get('suppress_blue_v_lo', 50)),
], dtype=np.uint8)
_blue_hi = np.array([int(param.get('suppress_blue_h_hi', 140)), 255, 255], dtype=np.uint8)

_sharpen_k     = int(param.get('preproc_sharpen_k',      5))
_sharpen_alpha = float(param.get('preproc_sharpen_alpha', 0.8))


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Parse TRACK gate NEDs from params.yaml comments
#     Format: #[TRACK] gate N: NED=[n  e  d]  WxHm
# ─────────────────────────────────────────────────────────────────────────────

_TRACK_RE = re.compile(
    r'#\[TRACK\]\s+gate\s+(\d+):\s+NED=\[([-\d\.\s]+)\]\s+([\d\.]+)x([\d\.]+)')
gate_neds  = {}   # gi → np.array([N, E, D])
gate_outer = {}   # gi → (width_m, height_m) outer frame

# The sim's raw TRACK NED marks the gate's bottom beam, not the opening centre
# the drone actually flies through — the centre sits ~1 m above the beam.
# params.yaml's `waypoints:` list already has this baked in (each entry is the
# matching [TRACK] D minus the offset); the [TRACK] comment tags themselves are
# the raw/uncorrected beam position, so apply the same correction here. Shared
# with vision_rx.py's live PnP-to-NED conversion via the same params.yaml key.
GATE_BEAM_TO_CENTER_D_OFFSET = float(param.get('gate_beam_to_center_offset_m', 1.0))

for line in raw_yaml.splitlines():
    m = _TRACK_RE.search(line)
    if m:
        gi  = int(m.group(1))
        ned = np.array([float(x) for x in m.group(2).split()])
        ned[2] -= GATE_BEAM_TO_CENTER_D_OFFSET
        gate_neds[gi]  = ned
        gate_outer[gi] = (float(m.group(3)), float(m.group(4)))

if not gate_neds:
    print('[WARN] No [TRACK] entries in params.yaml — using waypoints[1:] as gate NEDs '
          '(already beam→centre corrected there).')
    for i, wp in enumerate(param.get('waypoints', [])[1:]):
        gate_neds[i]  = np.array(wp, dtype=float)
        gate_outer[i] = (2.7, 2.7)

n_gates = len(gate_neds)
if n_gates == 0:
    raise ValueError('No gate positions found. Check params.yaml for [TRACK] entries.')

print(f'Gates   : {n_gates}')
for gi, ned in sorted(gate_neds.items()):
    ow, oh = gate_outer.get(gi, (2.7, 2.7))
    print(f'  Gate {gi}: NED=[{ned[0]:8.4f}, {ned[1]:7.4f}, {ned[2]:7.4f}]  '
          f'outer {ow:.1f}×{oh:.1f}m  (PnP inner {gate_w}×{gate_h}m)')


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Load EKF ground truth  (position + attitude over time)
# ─────────────────────────────────────────────────────────────────────────────

ekf_path = os.path.join(log_dir, 'ekf.csv')
if not os.path.exists(ekf_path):
    raise FileNotFoundError(f'ekf.csv not found in {log_dir}')

ekf_t, ekf_pN, ekf_pE, ekf_pD = [], [], [], []
ekf_qw, ekf_qx, ekf_qy, ekf_qz = [], [], [], []
ekf_wall_t = []

with open(ekf_path, newline='') as f:
    for row in csv.DictReader(f):
        try:
            ekf_t.append(float(row['time_s']))
            ekf_pN.append(float(row['pN']));  ekf_pE.append(float(row['pE']))
            ekf_pD.append(float(row['pD']))
            ekf_qw.append(float(row['qw']));  ekf_qx.append(float(row['qx']))
            ekf_qy.append(float(row['qy']));  ekf_qz.append(float(row['qz']))
            ekf_wall_t.append(float(row['wall_t']))
        except (ValueError, KeyError):
            continue

ekf_t  = np.array(ekf_t)
ekf_pN = np.array(ekf_pN);  ekf_pE = np.array(ekf_pE);  ekf_pD = np.array(ekf_pD)
ekf_qw = np.array(ekf_qw);  ekf_qx = np.array(ekf_qx)
ekf_qy = np.array(ekf_qy);  ekf_qz = np.array(ekf_qz)
ekf_wall_t = np.array(ekf_wall_t)
t_start, t_end = float(ekf_t[0]), float(ekf_t[-1])
print(f'EKF     : {len(ekf_t)} rows  t=[{t_start:.2f}, {t_end:.2f}] s')


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Enumerate frames and build frame_id → time mapping
# ─────────────────────────────────────────────────────────────────────────────

frames_dir = os.path.join(log_dir, 'frames')
if not os.path.isdir(frames_dir):
    raise FileNotFoundError(f'frames/ directory not found in {log_dir}')

frame_files = sorted(
    glob.glob(os.path.join(frames_dir, 'frame_*.jpg')),
    key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split('_')[1]))
n_frames = len(frame_files)
if n_frames == 0:
    raise ValueError(f'No frame_*.jpg files in {frames_dir}')

frame_ids = np.array([
    int(os.path.splitext(os.path.basename(p))[0].split('_')[1])
    for p in frame_files], dtype=int)
fid_min, fid_max = int(frame_ids[0]), int(frame_ids[-1])

# Frame → EKF-time mapping, from the recorded per-frame sim_time_ns (frame_timestamps.csv).
# NOTE: the old approach linearly stretched [fid_min, fid_max] onto [t_start, t_end] —
# i.e. it assumed frame recording exactly spans the EKF log. It doesn't: the camera starts
# recording before EKF logging begins and keeps going after it stops, so that stretch
# compresses/shifts every frame time and desyncs GT vs PnP more as time goes on.
# sim_time_ns and ekf.csv's wall_t are the same wall-clock epoch (both ~time.time()-based),
# so frame_time_s = sim_time_ns/1e9 - ekf_wall_t[0] lines up exactly with ekf.csv's time_s.
ts_path = os.path.join(log_dir, 'frame_timestamps.csv')
if os.path.exists(ts_path):
    ts_fid, ts_ns = [], []
    with open(ts_path, newline='') as f:
        for row in csv.DictReader(f):
            try:
                ts_fid.append(int(row['frame_id']));  ts_ns.append(int(row['sim_time_ns']))
            except (ValueError, KeyError):
                continue
    ts_fid = np.array(ts_fid);  ts_ns = np.array(ts_ns, dtype=np.float64)
    order = np.argsort(ts_fid)
    ts_fid, ts_ns = ts_fid[order], ts_ns[order]
    # A handful of saved frames can be missing a timestamp row (e.g. one at session
    # shutdown); interpolate those against neighboring known (frame_id, sim_time_ns) pairs.
    frame_ns = np.interp(frame_ids, ts_fid, ts_ns)
    frame_t  = frame_ns / 1e9 - float(ekf_wall_t[0])
    n_missing = int(np.sum(~np.isin(frame_ids, ts_fid)))
    if n_missing:
        print(f'[WARN] {n_missing} frame(s) missing from frame_timestamps.csv — interpolated.')
else:
    print('[WARN] frame_timestamps.csv not found — falling back to linear frame_id/EKF-span '
          'interpolation (inaccurate if frame recording does not exactly span the EKF log).')
    span = float(fid_max - fid_min) if fid_max > fid_min else 1.0
    frame_t = t_start + (frame_ids - fid_min) / span * (t_end - t_start)
est_fps = n_frames / (t_end - t_start) if t_end > t_start else 0.0
print(f'Frames  : {n_frames}  id=[{fid_min}, {fid_max}]  '
      f't=[{frame_t[0]:.2f}, {frame_t[-1]:.2f}] s  ≈{est_fps:.1f} fps (est.)')


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Helper functions  (match vision_rx.py exactly)
# ─────────────────────────────────────────────────────────────────────────────

def quat_to_R(qw, qx, qy, qz):
    """Body → NED rotation matrix from [qw, qx, qy, qz]."""
    return np.array([
        [1 - 2*(qy*qy + qz*qz),     2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy)],
        [    2*(qx*qy + qw*qz), 1 - 2*(qx*qx + qz*qz),     2*(qy*qz - qw*qx)],
        [    2*(qx*qz - qw*qy),     2*(qy*qz + qw*qx), 1 - 2*(qx*qx + qy*qy)],
    ], dtype=float)


def interp_pose(t):
    """Return (pos_ned [3], R_b2n [3×3]) interpolated from EKF at time t.

    In ground_truth_mode, ekf.csv's quaternion comes from mavlink_rx.py's
    on_attitude(), which deliberately leaves the sim's yaw sign inverted
    (see its docstring: "yaw — sign-inverted vs standard ZYX; NOT negated
    here (GT mode relies on the inverted sign; see _send_attitude_target
    r_des for correction)"). That's fine for the live controller, which has
    its own compensating negation on the GT branch of r_des — but it means
    ekf.csv's logged attitude has yaw = -yaw_true. Left uncorrected here,
    R_b2n rotates the PnP tvec by the wrong-sign yaw, which is invisible at
    zero range but grows linearly with distance to the gate and flips sign
    with whichever way the drone is actually crabbing — exactly the pattern
    that was showing up as PnP East diverging from GT East while North/Down
    stayed fine (yaw error is ~perpendicular to a mostly-North flight path).
    Confirmed empirically: negating yaw here cut median clean-frame 3D PnP
    error from 1.4 m to 0.19 m on a 6-gate test log. So: extract roll/pitch/
    yaw, flip yaw's sign, rebuild the quaternion with the same ZYX formula
    on_attitude() uses.
    """
    pos = np.array([np.interp(t, ekf_t, ekf_pN),
                    np.interp(t, ekf_t, ekf_pE),
                    np.interp(t, ekf_t, ekf_pD)])
    qw = float(np.interp(t, ekf_t, ekf_qw))
    qx = float(np.interp(t, ekf_t, ekf_qx))
    qy = float(np.interp(t, ekf_t, ekf_qy))
    qz = float(np.interp(t, ekf_t, ekf_qz))
    mag = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    if mag > 1e-9:
        qw, qx, qy, qz = qw/mag, qx/mag, qy/mag, qz/mag
    if ground_truth_mode:
        roll  = math.atan2(2*(qw*qx+qy*qz), 1-2*(qx*qx+qy*qy))
        pitch = math.asin(max(-1.0, min(1.0, 2*(qw*qy-qz*qx))))
        yaw   = -math.atan2(2*(qw*qz+qx*qy), 1-2*(qy*qy+qz*qz))
        cr, sr = math.cos(roll/2),  math.sin(roll/2)
        cp, sp = math.cos(pitch/2), math.sin(pitch/2)
        cy, sy = math.cos(yaw/2),   math.sin(yaw/2)
        qw = cr*cp*cy + sr*sp*sy
        qx = sr*cp*cy - cr*sp*sy
        qy = cr*sp*cy + sr*cp*sy
        qz = cr*cp*sy - sr*sp*cy
    return pos, quat_to_R(qw, qx, qy, qz)


def nearest_forward_gate(drone_ned, R_b2n):
    """Return (gate_idx, gate_ned, dist_3d, dist_fwd_body_x) for nearest in-front gate.
    'Forward' means the gate has a positive body-X (camera-Z) component."""
    best_gi, best_ned, best_3d, best_fwd = None, None, np.inf, float('nan')
    for gi in sorted(gate_neds):
        v_ned  = gate_neds[gi] - drone_ned
        v_body = R_b2n.T @ v_ned
        if v_body[0] <= 0.5:                    # gate must be ahead (body +X = forward)
            continue
        d3d = float(np.linalg.norm(v_ned))
        if d3d < best_3d:
            best_gi, best_ned, best_3d, best_fwd = gi, gate_neds[gi], d3d, float(v_body[0])
    return best_gi, best_ned, best_3d, best_fwd


def preprocess(img_bgr):
    """Unsharp mask then blue suppression — matches vision_rx.py defaults."""
    if _sharpen_k > 0:
        k    = _sharpen_k | 1
        blur = cv2.GaussianBlur(img_bgr, (k, k), 0)
        img  = cv2.addWeighted(img_bgr, 1.0 + _sharpen_alpha, blur, -_sharpen_alpha, 0)
    else:
        img  = img_bgr.copy()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    bm  = cv2.inRange(hsv, _blue_lo, _blue_hi)
    if np.any(bm):
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g3   = cv2.merge([grey, grey, grey])
        img  = np.where(bm[:, :, np.newaxis] > 0, g3, img).astype(img.dtype)
    return img


_ORN_LO = np.array([ 5,  80,  80], dtype=np.uint8)
_ORN_HI = np.array([25, 255, 255], dtype=np.uint8)
_ORN_K  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def orange_mask(img_bgr):
    hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, _ORN_LO, _ORN_HI)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _ORN_K)


def pnp_gate(corners_px, width, height, R_b2n):
    """IPPE PnP with attitude-based disambiguation.

    The two IPPE solutions differ in gate orientation.  The physical constraint
    is that the gate must be upright: gate object-Y ([0,1,0] = gate-down in the
    corner layout [-w,-h,0] / [w,-h,0] / [w,h,0] / [-w,h,0] — the corners are
    ordered TL,TR,BR,BL, so +Y runs from the top edge to the bottom edge) should
    map to world-DOWN in NED ([0,0,+1], i.e. positive D). Matches vision_rx.py's
    _pnp_gate, which scores np.dot(R_sol[:, 1], ned_down_cam) — same direction,
    same sign.

    Score each solution by how well gate-Y lands in the world-down hemisphere:
        gate_Y_ned = R_b2n @ R_cam2body @ (R_sol @ [0,1,0])
        score      = gate_Y_ned[2]          # D>0 = downward → positive score
    Pick the solution with the highest score.

    Returns (tvec [3], rvec [3]) or (None, None).
    """
    hw, hh  = width / 2.0, height / 2.0
    obj_pts = np.array([[-hw, -hh, 0.],
                        [ hw, -hh, 0.],
                        [ hw,  hh, 0.],
                        [-hw,  hh, 0.]], dtype=np.float64)
    img_pts = corners_px.astype(np.float64).reshape(4, 1, 2)
    n, rvecs, tvecs, _ = cv2.solvePnPGeneric(
        obj_pts, img_pts, cam_K, dist_coef, flags=cv2.SOLVEPNP_IPPE)
    if n < 1:
        return None, None

    _gate_Y = np.array([0., 1., 0.])
    best_tv, best_rv, best_sc = None, None, -np.inf
    for rv, tv in zip(rvecs, tvecs):
        tv_f = tv.flatten()
        if tv_f[2] < 0.1:
            continue
        R_sol, _ = cv2.Rodrigues(rv)
        gate_Y_ned = R_b2n @ (R_cam2body @ (R_sol @ _gate_Y))
        score = gate_Y_ned[2]    # D>0 in NED = downward = gate right-side up
        if score > best_sc:
            best_tv, best_rv, best_sc = tv_f, rv.flatten(), score
    if best_tv is None:
        return None, None

    return (None, None) if best_tv[2] < 0.5 else (best_tv, best_rv)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Load YOLO model
# ─────────────────────────────────────────────────────────────────────────────

print('Loading YOLO …', end='', flush=True)
model = _YOLO(_model_arg if _model_arg else 'YOLO/best3.pt')
model.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
print(' done.')


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Process frames
# ─────────────────────────────────────────────────────────────────────────────

if save_annotated:
    ann_dir = os.path.join(log_dir, 'annotated')
    os.makedirs(ann_dir, exist_ok=True)

PAD      = 15      # bbox padding for orange-fraction check [px]
CONF_THR = float(param.get('vision_conf_min', 0.6))   # minimum YOLO confidence (matches vision_rx.py)
ORN_MIN  = 0.001   # minimum orange pixel fraction in padded bbox

rows = []
print_step = max(1, n_frames // 20)
print(f'Processing {n_frames} frames …')

# Sequential gate tracker: advance when drone passes through each gate in order.
# Uses distance monotonicity (rebounds > _GATE_ADVANCE from minimum) rather than
# body-frame angle, so it works even when the gate isn't squarely ahead.
_gate_seq     = sorted(gate_neds.keys())
_gate_seq_idx = 0        # current target gate index into _gate_seq
_gate_min_d   = np.inf   # minimum distance to current target gate seen so far [m]
_GATE_ADVANCE = 2.0      # advance to next gate when dist exceeds min by this [m]

# Snap / outlier detection — three independent checks with different suppression rules:
#   GT-error check (GT mode):    err_3d > _OUTLIER_THRESH — always active, never suppressed.
#   Dist-mismatch check (GT mode): |tvec[2] - dist_fwd_gt| > _DIST_MISMATCH_THRESH
#                                  — catches wrong IPPE depth (PnP picks near solution when
#                                    gate is actually far, or vice versa); always active.
#   Jump check (both modes):     |drone_est - _last_pnp_est| > _PNP_SNAP_DIST
#                                  — suppressed during gate-transition window (_gate_trans_cnt>0)
#                                    because the gate-NED change makes the reference stale.
# _last_pnp_est updates only on clean frames OUTSIDE the transition window.
_OUTLIER_THRESH       = 25.0   # m; EKF-error above this → snap (GT mode only)
_DIST_MISMATCH_THRESH =  2.0   # m; |PnP depth - GT depth| above this → snap (GT mode only)
_PNP_SNAP_DIST        =  2.0   # m; minimum jump distance that constitutes a snap
_GATE_TRANS_SKIP      =  5     # frames after gate change: suppress jump check only
_prev_gate_idx   = None
_gate_trans_cnt  = 0
_last_pnp_est    = None

for fi, (fpath, fid, ft) in enumerate(zip(frame_files, frame_ids, frame_t)):
    if fi % print_step == 0:
        print(f'  {fi:4d}/{n_frames}  t_est={ft:.1f}s', flush=True)

    img_raw = cv2.imread(fpath)
    if img_raw is None:
        continue
    h_img, w_img = img_raw.shape[:2]

    # EKF pose at estimated frame time.
    # In GT mode ekf.csv pN/pE/pD == true position (EKF is overridden with GT).
    # In non-GT mode this is the EKF estimate, which may drift from truth.
    ekf_pos, R_b2n = interp_pose(ft)

    # Sequential gate tracker.
    # Advance the moment the drone crosses the current target's gate plane
    # (dist_fwd goes negative), with the old rebound-from-minimum check kept as a
    # fallback for off-axis passes where the forward projection doesn't cleanly
    # cross zero. Checking crossing FIRST matters: at the moment of passage the
    # camera frequently already sees the NEXT gate through the current one's
    # opening, so if we don't advance immediately, 1-2 frames get PnP'd against
    # the stale (just-passed) gate's NED while actually looking at the next gate
    # ~20+ m further down the corridor — producing spurious large-looking "wrong
    # solution" position snaps that are really just a gate-index attribution lag.
    if _gate_seq:
        _gi  = _gate_seq[min(_gate_seq_idx, len(_gate_seq) - 1)]
        _gnl = gate_neds[_gi]
        _v   = _gnl - ekf_pos
        _d   = float(np.linalg.norm(_v))
        _vb  = R_b2n.T @ _v
        _dist_fwd_cur = float(np.dot(_vb, R_cam2body[:, 2]))
        if _d < _gate_min_d:
            _gate_min_d = _d
        if (_dist_fwd_cur < 0.0 or _d > _gate_min_d + _GATE_ADVANCE) \
                and _gate_seq_idx < len(_gate_seq) - 1:
            _gate_seq_idx += 1
            _gate_min_d    = np.inf
            _gi  = _gate_seq[_gate_seq_idx]
            _gnl = gate_neds[_gi]
            _v   = _gnl - ekf_pos
            _d   = float(np.linalg.norm(_v))
            _vb  = R_b2n.T @ _v
        gate_idx = _gi
        gnl      = _gnl
        dist_3d  = _d
        # Project gate vector onto camera-Z axis (tilted 20° above body-X) so
        # dist_fwd matches what tvec[2] measures, not the body-X component.
        dist_fwd = float(np.dot(_vb, R_cam2body[:, 2]))
    else:
        gate_idx = None
        gnl      = None
        dist_3d  = np.inf
        dist_fwd = float('nan')

    # Gate transition: reset jump reference and suppress jump check for _GATE_TRANS_SKIP frames.
    # The absolute GT-error check is NOT suppressed — wrong IPPE solutions during transition
    # still get flagged as snaps.  Only the jump check is suppressed because the gate-NED
    # change makes the pre-transition _last_pnp_est an unreliable reference.
    if gate_idx != _prev_gate_idx:
        _gate_trans_cnt = _GATE_TRANS_SKIP
        _prev_gate_idx  = gate_idx
        _last_pnp_est   = None   # clear jump reference; post-transition frames use GT check only
    elif _gate_trans_cnt > 0:
        _gate_trans_cnt -= 1

    # Preprocessing (matches vision_rx.py)
    img  = preprocess(img_raw)
    mask = orange_mask(img)

    # YOLO inference
    res = model.predict(img, verbose=False, conf=CONF_THR)

    detected  = False
    pnp_ok    = False
    conf_best = 0.0
    tvec      = None
    corners   = None
    drone_est = None

    if (res and res[0].boxes is not None and res[0].keypoints is not None
            and len(res[0].boxes) > 0):
        boxes = res[0].boxes.xywh.cpu().numpy()    # (N,4): cx,cy,w,h
        confs = res[0].boxes.conf.cpu().numpy()    # (N,)
        kpts  = res[0].keypoints.xy.cpu().numpy()  # (N,4,2): TL,TR,BR,BL

        valid = []
        for i in range(len(boxes)):
            if confs[i] < CONF_THR:
                continue
            if kpts[i].shape != (4, 2):
                continue
            if np.any(np.all(kpts[i] < 2.0, axis=1)):  # keypoint at image edge → invalid
                continue
            x1 = max(0,     int(boxes[i][0] - boxes[i][2] / 2) - PAD)
            y1 = max(0,     int(boxes[i][1] - boxes[i][3] / 2) - PAD)
            x2 = min(w_img, int(boxes[i][0] + boxes[i][2] / 2) + PAD)
            y2 = min(h_img, int(boxes[i][1] + boxes[i][3] / 2) + PAD)
            area = max(1, (x2 - x1) * (y2 - y1))
            if np.count_nonzero(mask[y1:y2, x1:x2]) / area < ORN_MIN:
                continue
            valid.append(i)

        if valid:
            # Largest bbox area = nearest gate (same as vision_rx.py)
            best_i    = max(valid, key=lambda i: boxes[i][2] * boxes[i][3])
            corners   = kpts[best_i]   # (4,2): TL TR BR BL
            conf_best = float(confs[best_i])
            detected  = True

            tvec, _rvec = pnp_gate(corners, gate_w, gate_h, R_b2n)
            if tvec is not None and tvec[2] <= max_gate_dist:
                pnp_ok = True
                if gate_idx is not None:
                    # drone_ned = gate_ned - R_b2n @ R_cam2body @ tvec  (vision_rx.py)
                    t_gate_ned = R_b2n @ (R_cam2body @ tvec)
                    drone_est  = gnl - t_gate_ned

    # Position error vs GT + snap / outlier detection.
    # Three independent checks:
    #   1. GT-error check: err_3d > _OUTLIER_THRESH — ALWAYS active in GT mode.
    #   2. Dist-mismatch check: |tvec[2] - dist_fwd| > _DIST_MISMATCH_THRESH — ALWAYS active
    #      in GT mode. Catches wrong IPPE depth (e.g. PnP picks 25 m when gate is 35 m away).
    #   3. Jump check: |drone_est - _last_pnp_est| > _PNP_SNAP_DIST — SUPPRESSED during
    #      gate transition (gate-NED change makes the reference stale).
    # _last_pnp_est updates only on clean frames OUTSIDE the transition window so it
    # stays anchored to the last trustworthy estimate.
    snap = 0
    if drone_est is not None:
        err    = drone_est - ekf_pos
        err_3d = float(np.linalg.norm(err))
        # GT-error outlier: not suppressed during transition.
        if ground_truth_mode and err_3d > _OUTLIER_THRESH:
            snap = 1
        # Distance-consistency: compare PnP depth (tvec[2]) against GT forward distance.
        # Wrong IPPE solution picks the wrong depth branch; this catches it even when the
        # 3-D position error happens to stay below _OUTLIER_THRESH.
        if snap == 0 and ground_truth_mode and tvec is not None and math.isfinite(dist_fwd) and dist_fwd > 0.5:
            if abs(float(tvec[2]) - dist_fwd) > _DIST_MISMATCH_THRESH:
                snap = 1
        # Jump check: only outside the transition window and only when we have a reference.
        if snap == 0 and _gate_trans_cnt == 0 and _last_pnp_est is not None:
            if float(np.linalg.norm(drone_est - _last_pnp_est)) > _PNP_SNAP_DIST:
                snap = 1
        # Reference updates only on clean frames outside the transition window.
        if snap == 0 and _gate_trans_cnt == 0:
            _last_pnp_est = drone_est.copy()
    else:
        err    = np.full(3, float('nan'))
        err_3d = float('nan')

    # dist_fwd_gt: body-X component to gate — directly comparable to tvec[2] (cam-Z = body-X)
    rows.append({
        'frame_idx':   fi,
        'frame_id':    int(fid),
        't_est':       round(float(ft), 4),
        'detected':    int(detected),
        'conf':        round(float(conf_best), 4),
        'pnp_ok':      int(pnp_ok),
        'gate_idx':    int(gate_idx) if gate_idx is not None else -1,
        'dist_est':    round(float(tvec[2]),    3) if pnp_ok              else float('nan'),
        'dist_3d_gt':  round(float(dist_3d),    3) if dist_3d < np.inf   else float('nan'),
        'dist_fwd_gt': round(float(dist_fwd),   3) if math.isfinite(dist_fwd) else float('nan'),
        'pos_N':       round(float(drone_est[0]), 3) if drone_est is not None else float('nan'),
        'pos_E':       round(float(drone_est[1]), 3) if drone_est is not None else float('nan'),
        'pos_D':       round(float(drone_est[2]), 3) if drone_est is not None else float('nan'),
        'ekf_N':       round(float(ekf_pos[0]), 3),
        'ekf_E':       round(float(ekf_pos[1]), 3),
        'ekf_D':       round(float(ekf_pos[2]), 3),
        'err_N':       round(float(err[0]), 3),
        'err_E':       round(float(err[1]), 3),
        'err_D':       round(float(err[2]), 3),
        'err_3d':      round(err_3d, 3),
        'snap':        snap,
    })

    if save_annotated and detected:
        ann = img_raw.copy()
        if corners is not None:
            for pt in corners:
                cv2.circle(ann, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)
            for i in range(4):
                cv2.line(ann, tuple(corners[i].astype(int)),
                         tuple(corners[(i + 1) % 4].astype(int)), (0, 255, 0), 2)
        if pnp_ok:
            lbl = f'G{gate_idx} d={tvec[2]:.1f}m conf={conf_best:.2f} t={ft:.1f}s'
        else:
            lbl = f'det conf={conf_best:.2f} no-PnP t={ft:.1f}s'
        cv2.putText(ann, lbl, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.imwrite(os.path.join(ann_dir, f'frame_{fid:06d}.jpg'), ann)

print(f'  {n_frames}/{n_frames}  done.')


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Write CSV
# ─────────────────────────────────────────────────────────────────────────────

_out_stem = f'yolo_check_{_tag_arg}' if _tag_arg else 'yolo_check'
csv_path = os.path.join(log_dir, f'{_out_stem}.csv')
with open(csv_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f'\nCSV → {csv_path}')


# ─────────────────────────────────────────────────────────────────────────────
# 10. Summary statistics
# ─────────────────────────────────────────────────────────────────────────────

det_arr  = np.array([r['detected']    for r in rows], dtype=bool)
pnp_arr  = np.array([r['pnp_ok']      for r in rows], dtype=bool)
snap_arr = np.array([r['snap']         for r in rows], dtype=bool)
clean    = pnp_arr & ~snap_arr   # PnP frames that passed both snap checks
conf_arr = np.array([r['conf']         for r in rows], dtype=float)
d_est    = np.array([r['dist_est']     for r in rows], dtype=float)
d_fwd    = np.array([r['dist_fwd_gt']  for r in rows], dtype=float)
d_3d     = np.array([r['dist_3d_gt']   for r in rows], dtype=float)
gi_arr   = np.array([r['gate_idx']     for r in rows], dtype=int)
err_3d   = np.array([r['err_3d']       for r in rows], dtype=float)
t_arr    = np.array([r['t_est']        for r in rows], dtype=float)
gt_N     = np.array([r['ekf_N']        for r in rows], dtype=float)
gt_E     = np.array([r['ekf_E']        for r in rows], dtype=float)
gt_D     = np.array([r['ekf_D']        for r in rows], dtype=float)
pos_N    = np.array([r['pos_N']        for r in rows], dtype=float)
pos_E    = np.array([r['pos_E']        for r in rows], dtype=float)
pos_D    = np.array([r['pos_D']        for r in rows], dtype=float)
err_N    = np.array([r['err_N']        for r in rows], dtype=float)
err_D    = np.array([r['err_D']        for r in rows], dtype=float)

near_mask = d_3d < max_gate_dist    # frames where a gate is within range
d_err     = d_est - d_fwd           # forward-distance error (cam-Z vs body-X to gate)


def _med(a, m):
    v = a[m & np.isfinite(a)]
    return float(np.median(v)) if len(v) > 0 else float('nan')

def _std(a, m):
    v = a[m & np.isfinite(a)]
    return float(np.std(v)) if len(v) > 0 else float('nan')

def _p95(a, m):
    v = np.abs(a[m & np.isfinite(a)])
    return float(np.percentile(v, 95)) if len(v) > 0 else float('nan')


print('\n── Overall ─────────────────────────────────────────')
print(f'  Frames processed      : {len(rows)}')
print(f'  Frames near a gate    : {near_mask.sum()} ({100*near_mask.mean():.0f}%)')
det_r = 100 * det_arr[near_mask].mean() if near_mask.any() else 0.
pnp_r = 100 * pnp_arr[near_mask].mean() if near_mask.any() else 0.
print(f'  Detection rate        : {det_r:.1f}%')
print(f'  PnP rate              : {pnp_r:.1f}%')
if np.isfinite(d_err[pnp_arr]).any():
    print(f'  Distance error        : '
          f'median={_med(d_err, pnp_arr):+.2f}m  '
          f'std={_std(d_err, pnp_arr):.2f}m  '
          f'|p95|={_p95(d_err, pnp_arr):.2f}m')
if np.isfinite(err_3d[pnp_arr]).any():
    print(f'  Position error (3D)   : '
          f'median={_med(err_3d, pnp_arr):.2f}m  '
          f'std={_std(err_3d, pnp_arr):.2f}m  '
          f'p95={_p95(err_3d, pnp_arr):.2f}m')
    print(f'  Error breakdown       : '
          f'N={_med(err_N, pnp_arr):+.2f}m  '
          f'E={_med(np.array([r["err_E"] for r in rows], dtype=float), pnp_arr):+.2f}m  '
          f'D={_med(err_D, pnp_arr):+.2f}m  (median)')
snap_r = 100 * snap_arr[pnp_arr].mean() if pnp_arr.any() else 0.
_snap_criteria = (f'EKF-err>{_OUTLIER_THRESH:.0f}m or dist-mismatch>{_DIST_MISMATCH_THRESH:.0f}m (GT, always) or '
                  f'jump>{_PNP_SNAP_DIST:.0f}m (non-trans)') if ground_truth_mode else f'jump>{_PNP_SNAP_DIST:.0f}m (non-trans)'
print(f'  Snaps ({_snap_criteria}): {snap_arr.sum()} frames ({snap_r:.1f}% of PnP)')
if np.isfinite(err_3d[clean]).any():
    print(f'  Clean PnP err (3D)    : '
          f'median={_med(err_3d, clean):.2f}m  '
          f'std={_std(err_3d, clean):.2f}m  '
          f'p95={_p95(err_3d, clean):.2f}m')

print(f'\n── By gate ─────────────────────────────────────────')
print(f'  {"Gate":>4}  {"Frames":>6}  {"Det%":>5}  {"PnP%":>5}  {"Snaps":>5}  '
      f'{"d_err med":>9}  {"clean err med":>13}')
for gi in sorted(gate_neds):
    gm = near_mask & (gi_arr == gi)
    if not gm.any():
        continue
    dr  = 100 * det_arr[gm].mean()
    pr  = 100 * pnp_arr[gm].mean()
    snc = int(snap_arr[gm & pnp_arr].sum()) if (gm & pnp_arr).any() else 0
    de  = _med(d_err,  gm & pnp_arr)
    pe  = _med(err_3d, gm & clean)
    ned = gate_neds[gi]
    print(f'  {gi:>4}  {gm.sum():>6}  {dr:>5.1f}  {pr:>5.1f}  {snc:>5}  '
          f'{de:>+9.2f}m  {pe:>13.2f}m   NED=[{ned[0]:.1f},{ned[1]:.1f},{ned[2]:.1f}]')

print(f'\n── Detection rate by distance ──────────────────────')
for lo, hi in [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50)]:
    bm = near_mask & (d_3d >= lo) & (d_3d < hi)
    if bm.any():
        dr = 100 * det_arr[bm].mean()
        pr = 100 * pnp_arr[bm].mean()
        print(f'  {lo:>2}–{hi:>2} m : {bm.sum():>4} frames  '
              f'det={dr:>5.1f}%  pnp={pr:>5.1f}%')


# ─────────────────────────────────────────────────────────────────────────────
# 11. Plots
# ─────────────────────────────────────────────────────────────────────────────

GCOLS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle(f'YOLO Gate-Detection Analysis — {os.path.basename(log_dir)}', fontsize=12)


# Panel 1: YOLO confidence over time, coloured by gate (X = snap)
ax = axes[0, 0]
for gi in sorted(gate_neds):
    m = clean & (gi_arr == gi)
    if m.any():
        ax.scatter(t_arr[m], conf_arr[m], s=14,
                   color=GCOLS[gi % len(GCOLS)], label=f'G{gi} (PnP)', alpha=0.8, zorder=3)
    ms = snap_arr & (gi_arr == gi)
    if ms.any():
        ax.scatter(t_arr[ms], conf_arr[ms], s=22,
                   color=GCOLS[gi % len(GCOLS)], marker='x', linewidths=1.2,
                   alpha=0.9, zorder=4, label=f'G{gi} snap')
m_nopnp = det_arr & ~pnp_arr
if m_nopnp.any():
    ax.scatter(t_arr[m_nopnp], conf_arr[m_nopnp],
               s=8, color='grey', alpha=0.4, label='det (no PnP)', zorder=2)
ax.axhline(CONF_THR, color='k', ls='--', lw=0.8, label=f'conf thr={CONF_THR}')
ax.set_xlabel('time [s]');  ax.set_ylabel('YOLO confidence')
ax.set_title('Detection confidence');  ax.legend(fontsize=7, ncol=2)
ax.set_ylim(-0.02, 1.05);  ax.grid(True, lw=0.3)


# Panel 2: Gate distance — GT (solid) vs PnP estimate (scatter)
ax = axes[0, 1]
ax.plot(t_arr, d_3d,  color='steelblue', lw=0.9, alpha=0.8, label='GT 3D dist')
ax.plot(t_arr, d_fwd, color='k',         lw=0.7, alpha=0.6, ls='--', label='GT fwd dist (body-X)')
if pnp_arr.any():
    ax.scatter(t_arr[pnp_arr], d_est[pnp_arr], s=10, c='tab:orange',
               alpha=0.8, label='PnP tvec_z', zorder=3)
ax.set_xlabel('time [s]');  ax.set_ylabel('distance [m]')
ax.set_title('Gate distance: GT vs PnP');  ax.legend(fontsize=7);  ax.grid(True, lw=0.3)


# Panel 3: Distance error vs GT forward distance (coloured by time)
ax = axes[0, 2]
pm = pnp_arr & np.isfinite(d_err) & np.isfinite(d_fwd)
if pm.any():
    sc = ax.scatter(d_fwd[pm], d_err[pm], s=12, c=t_arr[pm], cmap='plasma', alpha=0.75)
    plt.colorbar(sc, ax=ax, label='time [s]')
ax.axhline( 0, color='k',          lw=0.8)
ax.axhline(+1, color='tab:green',  ls='--', lw=0.8)
ax.axhline(-1, color='tab:green',  ls='--', lw=0.8, label='±1 m')
ax.set_xlabel('GT forward distance [m]')
ax.set_ylabel('dist_est − dist_fwd_gt [m]')
ax.set_title('Distance error vs gate range');  ax.legend(fontsize=8);  ax.grid(True, lw=0.3)


# Panel 4: Drone N position over time
ax = axes[1, 0]
_ref_label = 'GT North' if ground_truth_mode else 'EKF North'
ax.plot(t_arr, gt_N, color='steelblue', lw=1.0, label=_ref_label)
if clean.any():
    ax.scatter(t_arr[clean], pos_N[clean], s=10, c='tab:orange',
               alpha=0.75, label='PnP North', zorder=3)
if snap_arr.any():
    ax.scatter(t_arr[snap_arr], pos_N[snap_arr], s=22, c='tab:red',
               marker='x', linewidths=1.2, alpha=0.85, label='snap', zorder=4)
for gi, ned in sorted(gate_neds.items()):
    ax.axhline(ned[0], color=GCOLS[gi % len(GCOLS)], lw=0.6, ls=':', alpha=0.7,
               label=f'G{gi} N={ned[0]:.0f}m')
ax.set_xlabel('time [s]');  ax.set_ylabel('North [m]')
_ref_str = 'GT' if ground_truth_mode else 'EKF'
ax.set_title(f'Drone N: {_ref_str} vs PnP');  ax.legend(fontsize=6, ncol=2);  ax.grid(True, lw=0.3)


# Panel 5: Drone E and D over time
ax = axes[1, 1]
ax.plot(t_arr, gt_E, color='tab:blue',  lw=1.0,
        label='GT East' if ground_truth_mode else 'EKF East')
ax.plot(t_arr, gt_D, color='tab:green', lw=1.0,
        label='GT Down' if ground_truth_mode else 'EKF Down')
if clean.any():
    ax.scatter(t_arr[clean], pos_E[clean], s=10,
               c='tab:cyan',  alpha=0.75, label='PnP East', zorder=3)
    ax.scatter(t_arr[clean], pos_D[clean], s=10,
               c='tab:olive', alpha=0.75, label='PnP Down', zorder=3)
if snap_arr.any():
    ax.scatter(t_arr[snap_arr], pos_E[snap_arr], s=22, c='tab:red',
               marker='x', linewidths=1.2, alpha=0.85, label='snap E/D', zorder=4)
    ax.scatter(t_arr[snap_arr], pos_D[snap_arr], s=22, c='tab:red',
               marker='+', linewidths=1.2, alpha=0.85, zorder=4)
ax.set_xlabel('time [s]');  ax.set_ylabel('East / Down [m]')
ax.set_title(f'Drone E/D: {_ref_str} vs PnP');  ax.legend(fontsize=7, ncol=2);  ax.grid(True, lw=0.3)


# Panel 6: 2D overhead path (East vs North)
ax = axes[1, 2]
ax.plot(gt_E, gt_N, 'b-', lw=1.2,
        label='GT path' if ground_truth_mode else 'EKF path', zorder=2)
ax.plot(gt_E[0], gt_N[0], 'bs', ms=8, label='start', zorder=5)
if clean.any():
    sc6 = ax.scatter(pos_E[clean], pos_N[clean], s=14, c=t_arr[clean],
                     cmap='YlOrRd', alpha=0.65, label='PnP pos', zorder=3)
    plt.colorbar(sc6, ax=ax, label='time [s]')
if snap_arr.any():
    ax.scatter(pos_E[snap_arr], pos_N[snap_arr], s=22, c='tab:red',
               marker='x', linewidths=1.2, alpha=0.8, label='snap', zorder=4)
for gi, ned in sorted(gate_neds.items()):
    ax.plot(ned[1], ned[0], 'k^', ms=9, zorder=4)
    ax.text(ned[1] + 0.5, ned[0], f'G{gi}', fontsize=7, va='center')
ax.set_xlabel('East [m]');  ax.set_ylabel('North [m]')
ax.set_title('2D path: GT vs PnP');  ax.legend(fontsize=7)
ax.set_aspect('equal', adjustable='datalim');  ax.grid(True, lw=0.3)


fig.tight_layout()
png_path = os.path.join(log_dir, f'{_out_stem}.png')
fig.savefig(png_path, dpi=120)
plt.close(fig)
print(f'Plot  → {png_path}')
