"""
check_yolo.py  —  Offline YOLO gate-detection analysis on captured flight frames.

Replays every JPEG frame saved during the last (or specified) flight, runs the
same YOLO + PnP pipeline as vision_rx.py, and compares the estimated drone NED
position to the EKF ground truth.

Output (saved next to the log):
  yolo_check.csv        per-frame results
  yolo_check.png        6-panel comparison plot
  annotated/            annotated frames (--save-annotated only)

Usage:
  python check_yolo.py                        # auto-find latest log
  python check_yolo.py logs/20260722_164557   # specific log dir
  python check_yolo.py --save-annotated       # also write annotated frames
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

gate_w        = float(param.get('gate_width_default',  1.5))   # inner opening [m]
gate_h        = float(param.get('gate_height_default', 1.5))
max_gate_dist = float(param.get('vision_max_gate_dist', 50.0))

# cam→body: x_b(fwd)=cam_Z, y_b(right)=cam_X, z_b(down)=cam_Y  (vision_rx.py)
R_cam2body = np.array([[0, 0, 1],
                        [1, 0, 0],
                        [0, 1, 0]], dtype=float)

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

for line in raw_yaml.splitlines():
    m = _TRACK_RE.search(line)
    if m:
        gi  = int(m.group(1))
        ned = np.array([float(x) for x in m.group(2).split()])
        gate_neds[gi]  = ned
        gate_outer[gi] = (float(m.group(3)), float(m.group(4)))

if not gate_neds:
    print('[WARN] No [TRACK] entries in params.yaml — using waypoints[1:] as gate NEDs '
          '(may have ~1 m altitude offset from true gate center).')
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

with open(ekf_path, newline='') as f:
    for row in csv.DictReader(f):
        try:
            ekf_t.append(float(row['time_s']))
            ekf_pN.append(float(row['pN']));  ekf_pE.append(float(row['pE']))
            ekf_pD.append(float(row['pD']))
            ekf_qw.append(float(row['qw']));  ekf_qx.append(float(row['qx']))
            ekf_qy.append(float(row['qy']));  ekf_qz.append(float(row['qz']))
        except (ValueError, KeyError):
            continue

ekf_t  = np.array(ekf_t)
ekf_pN = np.array(ekf_pN);  ekf_pE = np.array(ekf_pE);  ekf_pD = np.array(ekf_pD)
ekf_qw = np.array(ekf_qw);  ekf_qx = np.array(ekf_qx)
ekf_qy = np.array(ekf_qy);  ekf_qz = np.array(ekf_qz)
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

# Linear map: frame_id range → EKF time range.
# frame_id is a monotonic sim-step counter, so relative spacing tracks real time.
span = float(fid_max - fid_min) if fid_max > fid_min else 1.0
frame_t = t_start + (frame_ids - fid_min) / span * (t_end - t_start)
est_fps = n_frames / (t_end - t_start) if t_end > t_start else 0.0
print(f'Frames  : {n_frames}  id=[{fid_min}, {fid_max}]  ≈{est_fps:.1f} fps (est.)')


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
    """Return (pos_ned [3], R_b2n [3×3]) interpolated from EKF at time t."""
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
    """IPPE PnP; picks solution where gate-Y aligns with NED-down in camera frame.
    Uses GT attitude for disambiguation (more accurate than live EKF noise).
    Returns (tvec [3], rvec [3]) or (None, None)."""
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

    # NED-down direction in camera frame (using GT attitude — better than live system)
    ned_down_cam = R_cam2body.T @ (R_b2n.T @ np.array([0., 0., 1.]))

    best_tv, best_rv, best_sc = rvecs[0].flatten(), tvecs[0].flatten(), -np.inf
    for rv, tv in zip(rvecs, tvecs):
        tv_f = tv.flatten()
        if tv_f[2] < 0.1:
            continue
        R_sol, _ = cv2.Rodrigues(rv)
        score = float(np.dot(R_sol[:, 1], ned_down_cam))
        if score > best_sc:
            best_tv, best_rv, best_sc = tv_f, rv.flatten(), score

    return (None, None) if best_tv[2] < 0.5 else (best_tv, best_rv)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Load YOLO model
# ─────────────────────────────────────────────────────────────────────────────

print('Loading YOLO …', end='', flush=True)
model = _YOLO('YOLO/best.pt')
model.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
print(' done.')


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Process frames
# ─────────────────────────────────────────────────────────────────────────────

if save_annotated:
    ann_dir = os.path.join(log_dir, 'annotated')
    os.makedirs(ann_dir, exist_ok=True)

PAD      = 15      # bbox padding for orange-fraction check [px]
CONF_THR = 0.03    # minimum YOLO confidence (matches vision_rx.py)
ORN_MIN  = 0.001   # minimum orange pixel fraction in padded bbox

rows = []
print_step = max(1, n_frames // 20)
print(f'Processing {n_frames} frames …')

for fi, (fpath, fid, ft) in enumerate(zip(frame_files, frame_ids, frame_t)):
    if fi % print_step == 0:
        print(f'  {fi:4d}/{n_frames}  t_est={ft:.1f}s', flush=True)

    img_raw = cv2.imread(fpath)
    if img_raw is None:
        continue
    h_img, w_img = img_raw.shape[:2]

    # GT pose at estimated frame time
    gt_pos, R_b2n = interp_pose(ft)
    gate_idx, gnl, dist_3d, dist_fwd = nearest_forward_gate(gt_pos, R_b2n)

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

    # Position error vs GT
    if drone_est is not None:
        err    = drone_est - gt_pos
        err_3d = float(np.linalg.norm(err))
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
        'gt_N':        round(float(gt_pos[0]), 3),
        'gt_E':        round(float(gt_pos[1]), 3),
        'gt_D':        round(float(gt_pos[2]), 3),
        'err_N':       round(float(err[0]), 3),
        'err_E':       round(float(err[1]), 3),
        'err_D':       round(float(err[2]), 3),
        'err_3d':      round(err_3d, 3),
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

csv_path = os.path.join(log_dir, 'yolo_check.csv')
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
conf_arr = np.array([r['conf']         for r in rows], dtype=float)
d_est    = np.array([r['dist_est']     for r in rows], dtype=float)
d_fwd    = np.array([r['dist_fwd_gt']  for r in rows], dtype=float)
d_3d     = np.array([r['dist_3d_gt']   for r in rows], dtype=float)
gi_arr   = np.array([r['gate_idx']     for r in rows], dtype=int)
err_3d   = np.array([r['err_3d']       for r in rows], dtype=float)
t_arr    = np.array([r['t_est']        for r in rows], dtype=float)
gt_N     = np.array([r['gt_N']         for r in rows], dtype=float)
gt_E     = np.array([r['gt_E']         for r in rows], dtype=float)
gt_D     = np.array([r['gt_D']         for r in rows], dtype=float)
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

print(f'\n── By gate ─────────────────────────────────────────')
print(f'  {"Gate":>4}  {"Frames":>6}  {"Det%":>5}  {"PnP%":>5}  '
      f'{"d_err med":>9}  {"pos_err med":>11}')
for gi in sorted(gate_neds):
    gm = near_mask & (gi_arr == gi)
    if not gm.any():
        continue
    dr = 100 * det_arr[gm].mean()
    pr = 100 * pnp_arr[gm].mean()
    de = _med(d_err,  gm & pnp_arr)
    pe = _med(err_3d, gm & pnp_arr)
    ned = gate_neds[gi]
    print(f'  {gi:>4}  {gm.sum():>6}  {dr:>5.1f}  {pr:>5.1f}  '
          f'{de:>+9.2f}m  {pe:>11.2f}m   NED=[{ned[0]:.1f},{ned[1]:.1f},{ned[2]:.1f}]')

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


# Panel 1: YOLO confidence over time, coloured by gate
ax = axes[0, 0]
for gi in sorted(gate_neds):
    m = pnp_arr & (gi_arr == gi)
    if m.any():
        ax.scatter(t_arr[m], conf_arr[m], s=14,
                   color=GCOLS[gi % len(GCOLS)], label=f'G{gi} (PnP)', alpha=0.8, zorder=3)
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
ax.plot(t_arr, gt_N, color='steelblue', lw=1.0, label='GT North')
if pnp_arr.any():
    ax.scatter(t_arr[pnp_arr], pos_N[pnp_arr], s=10, c='tab:orange',
               alpha=0.75, label='PnP North', zorder=3)
for gi, ned in sorted(gate_neds.items()):
    ax.axhline(ned[0], color=GCOLS[gi % len(GCOLS)], lw=0.6, ls=':', alpha=0.7,
               label=f'G{gi} N={ned[0]:.0f}m')
ax.set_xlabel('time [s]');  ax.set_ylabel('North [m]')
ax.set_title('Drone N: GT vs PnP');  ax.legend(fontsize=6, ncol=2);  ax.grid(True, lw=0.3)


# Panel 5: Drone E and D over time
ax = axes[1, 1]
ax.plot(t_arr, gt_E, color='tab:blue',  lw=1.0, label='GT East')
ax.plot(t_arr, gt_D, color='tab:green', lw=1.0, label='GT Down')
if pnp_arr.any():
    ax.scatter(t_arr[pnp_arr], pos_E[pnp_arr], s=10,
               c='tab:cyan',  alpha=0.75, label='PnP East', zorder=3)
    ax.scatter(t_arr[pnp_arr], pos_D[pnp_arr], s=10,
               c='tab:olive', alpha=0.75, label='PnP Down', zorder=3)
ax.set_xlabel('time [s]');  ax.set_ylabel('East / Down [m]')
ax.set_title('Drone E/D: GT vs PnP');  ax.legend(fontsize=7, ncol=2);  ax.grid(True, lw=0.3)


# Panel 6: 2D overhead path (East vs North)
ax = axes[1, 2]
ax.plot(gt_E, gt_N, 'b-', lw=1.2, label='GT path', zorder=2)
ax.plot(gt_E[0], gt_N[0], 'bs', ms=8, label='start', zorder=5)
if pnp_arr.any():
    sc6 = ax.scatter(pos_E[pnp_arr], pos_N[pnp_arr], s=14, c=t_arr[pnp_arr],
                     cmap='YlOrRd', alpha=0.65, label='PnP pos', zorder=3)
    plt.colorbar(sc6, ax=ax, label='time [s]')
for gi, ned in sorted(gate_neds.items()):
    ax.plot(ned[1], ned[0], 'k^', ms=9, zorder=4)
    ax.text(ned[1] + 0.5, ned[0], f'G{gi}', fontsize=7, va='center')
ax.set_xlabel('East [m]');  ax.set_ylabel('North [m]')
ax.set_title('2D path: GT vs PnP');  ax.legend(fontsize=7)
ax.set_aspect('equal', adjustable='datalim');  ax.grid(True, lw=0.3)


fig.tight_layout()
png_path = os.path.join(log_dir, 'yolo_check.png')
fig.savefig(png_path, dpi=120)
plt.close(fig)
print(f'Plot  → {png_path}')
