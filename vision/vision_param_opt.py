"""
vision_param_opt.py
====================
Offline parameter optimizer for stationary-drone gate detection at 40 m.

The drone sits stationary with gate 0 ahead at ~40 m.  The script:
  1. CAPTURE — collects raw camera frames from UDP for CAPTURE_SEC seconds
               (no YOLO inference during capture — pure frame store)
  2. SWEEP   — for every preprocessing × confidence combination, runs the full
               orange-mask → YOLO → PnP pipeline on all stored frames
  3. REPORT  — ranks combos by detection rate; prints best config and plots

Parameter grid swept:
  sharpen_k     : [0, 3, 5, 7, 9]      — unsharp-mask Gaussian kernel (0 = off)
  sharpen_alpha : [0.3, 0.5, 0.8, 1.2] — sharpening strength
  gauss_k       : [0, 3]               — Gaussian denoise kernel (0 = off)
  conf_thresh   : [0.2, 0.3, 0.4, 0.5] — YOLO confidence gate (applied post-hoc)
  orange_frac   : [0.05, 0.10, 0.15]  — min orange fraction inside bbox

When sharpen_k == 0, sharpen_alpha has no effect — those duplicates are collapsed.

Outputs (logs/param_opt_<timestamp>/):
  results.csv       — per-combo: det_rate, pnp_rate, mean_dist, dist_err, …
  heatmap.png       — detection-rate grid (sharpen_k × sharpen_alpha)
  conf_sweep.png    — det rate vs conf for top-3 combos
  best_params.yaml  — copy-paste block for params.yaml
"""

import os
import sys
import time
import struct
import socket
import queue
import threading
import msvcrt
from datetime import datetime
from itertools import product

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pymavlink import mavutil

from flight_model.dyn import load_params
from comms.mavlink_rx import MAVLinkRX
from ultralytics import YOLO as _YOLO

# ── Camera UDP ────────────────────────────────────────────────────────────────
CAM_IP   = "0.0.0.0"
CAM_PORT = 5600

# ── MAVLink ───────────────────────────────────────────────────────────────────
SIM_IP   = "127.0.0.1"
SIM_PORT = 14550

# ── Capture ───────────────────────────────────────────────────────────────────
CAPTURE_SEC = 10     # seconds of frame collection
MAX_FRAMES  = 400    # hard cap on stored frames

# ── Target distance for accuracy scoring ─────────────────────────────────────
TARGET_DIST_M   = 25
DIST_WINDOW_M   = 8.0   # PnP reads within [TARGET ± WINDOW] count as "accurate"

# ── Parameter grid ────────────────────────────────────────────────────────────
SHARPEN_K_LIST      = [0, 3, 5, 7, 9]
SHARPEN_ALPHA_LIST  = [0.3, 0.5, 0.8, 1.2]
GAUSS_K_LIST        = [0, 3]
CONF_THRESH_LIST    = [0.1, 0.2, 0.3, 0.4]
ORANGE_FRAC_LIST    = [0.0005, 0.001, 0.002, 0.005, 0.01]

YOLO_INFERENCE_CONF = 0.05   # run YOLO at very low conf; filter post-hoc

MAVLINK_CMD_SIM_RESET = 31000


# ── Preprocessing ─────────────────────────────────────────────────────────────

def _preprocess(img, sharpen_k, sharpen_alpha, gauss_k):
    if sharpen_k > 0:
        k    = sharpen_k | 1
        blur = cv2.GaussianBlur(img, (k, k), 0)
        img  = cv2.addWeighted(img, 1.0 + sharpen_alpha, blur, -sharpen_alpha, 0)
    if gauss_k > 0:
        gk  = gauss_k | 1
        img = cv2.GaussianBlur(img, (gk, gk), 0)
    return img


def _apply_blue_suppression(img_bgr, suppress_blue, blue_lo, blue_hi):
    if not suppress_blue:
        return img_bgr
    hsv     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    blue_px = cv2.inRange(hsv, blue_lo, blue_hi)
    if not np.any(blue_px):
        return img_bgr
    grey     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    grey_3ch = cv2.merge([grey, grey, grey])
    return np.where(blue_px[:, :, np.newaxis] > 0,
                    grey_3ch, img_bgr).astype(img_bgr.dtype)


def _orange_mask(img_bgr, suppress_blue, blue_lo, blue_hi):
    # img_bgr has blue/cyan desaturated (S=0) upstream.
    # S lowered 100→80 to catch gate pixels mixed with cyan interference.
    hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lo   = np.array([ 5,  80,  80], dtype=np.uint8)
    hi   = np.array([25, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lo, hi)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask    = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    return cv2.bitwise_and(img_bgr, img_bgr, mask=mask), mask


def _pnp_gate(corners_px, gate_width, gate_height, cam_K):
    hw = gate_width  / 2.0
    hh = gate_height / 2.0
    obj_pts = np.array([[-hw,-hh,0], [hw,-hh,0], [hw,hh,0], [-hw,hh,0]], dtype=np.float64)
    img_pts = corners_px.astype(np.float64).reshape(4, 1, 2)
    n, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj_pts, img_pts, cam_K, np.zeros((4, 1)),
                                              flags=cv2.SOLVEPNP_IPPE)
    if n < 1:
        return None
    # Horizontal-gate constraint: gate Y-axis must align with NED-down in camera frame.
    # Level-drone fallback: NED-down in cam frame = cam_Y = [0, 1, 0].
    ned_down_cam = np.array([0.0, 1.0, 0.0])
    best_tv = None
    best_score = -np.inf
    for rv, tv in zip(rvecs, tvecs):
        tv_f = tv.flatten()
        if tv_f[2] < 0.1:
            continue
        R_sol, _ = cv2.Rodrigues(rv)
        score = float(np.dot(R_sol[:, 1], ned_down_cam))
        if score > best_score:
            best_score = score
            best_tv = tv_f
    if best_tv is None:
        return None
    return float(best_tv[2])   # forward distance [m]


# ── Frame capture via UDP ─────────────────────────────────────────────────────

def _capture_frames(n_max, duration_sec):
    """Receive JPEG frames from the sim camera UDP stream.  Returns list of numpy images."""
    header_format = "<IHHIIQ"
    header_sz     = struct.calcsize(header_format)
    frames_raw    = {}
    out_frames    = []

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(0.5)
    sock.bind((CAM_IP, CAM_PORT))

    t_end = time.time() + duration_sec
    print(f"  Capturing frames for {duration_sec:.0f} s (max {n_max})…", flush=True)
    last_print = time.time()

    while time.time() < t_end and len(out_frames) < n_max:
        try:
            packet, _ = sock.recvfrom(65536)
        except socket.timeout:
            continue

        header  = packet[:header_sz]
        payload = packet[header_sz:]
        frame_id, chunk_id, total_chunks, _, _, _ = struct.unpack(header_format, header)

        if frame_id not in frames_raw:
            frames_raw[frame_id] = {"chunks": {}, "total": total_chunks}
        frames_raw[frame_id]["chunks"][chunk_id] = payload

        if len(frames_raw[frame_id]["chunks"]) == total_chunks:
            jpeg = bytearray()
            ok   = True
            for i in range(total_chunks):
                if i not in frames_raw[frame_id]["chunks"]:
                    ok = False; break
                jpeg.extend(frames_raw[frame_id]["chunks"][i])
            if ok:
                img = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    out_frames.append(img)
                    if time.time() - last_print >= 2.0:
                        last_print = time.time()
                        print(f"    {len(out_frames)} frames captured…", flush=True)
            del frames_raw[frame_id]

        # evict stale incomplete frames
        if len(frames_raw) > 30:
            del frames_raw[min(frames_raw)]

    sock.close()
    return out_frames


# ── Single-combo sweep ────────────────────────────────────────────────────────

def _run_combo(frames, model, cam_K, gate_w, gate_h, suppress_blue, blue_lo, blue_hi,
               sharpen_k, sharpen_alpha, gauss_k, conf_thresh, orange_frac_min):
    """Run full pipeline on all frames for one parameter combo.
    Returns dict with detection stats."""
    det    = 0
    pnp_ok = 0
    dists  = []

    for img in frames:
        pre      = _preprocess(img, sharpen_k, sharpen_alpha, gauss_k)
        pre_clean = _apply_blue_suppression(pre, suppress_blue, blue_lo, blue_hi)
        masked, mask_raw = _orange_mask(pre_clean, suppress_blue, blue_lo, blue_hi)

        # YOLO on blue-suppressed preprocessed frame (no masking).
        results = model.predict(pre_clean, verbose=False, conf=YOLO_INFERENCE_CONF)
        r = results[0]
        if r.boxes is None or r.keypoints is None or len(r.boxes) == 0:
            continue

        boxes = r.boxes.xywh.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        kpts  = r.keypoints.xy.cpu().numpy()

        h_img, w_img = mask_raw.shape

        # Filter → pick nearest by area (mirrors live pipeline):
        # 1. Reject: conf < thresh, zero keypoints, or orange_frac below floor.
        # 2. Among survivors pick largest bbox area (= nearest gate).
        _PAD = 15
        valid_idx = []
        for _i in range(len(boxes)):
            if float(confs[_i]) < conf_thresh:
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
            if np.count_nonzero(mask_raw[_by1:_by2, _bx1:_bx2]) / _ba < orange_frac_min:
                continue
            valid_idx.append(_i)

        if not valid_idx:
            continue

        areas  = boxes[:, 2] * boxes[:, 3]
        best   = valid_idx[int(np.argmax(areas[valid_idx]))]
        corners = kpts[best]

        det += 1

        dist = _pnp_gate(corners, gate_w, gate_h, cam_K)
        if dist is not None and 1.0 < dist < 120.0:
            pnp_ok += 1
            dists.append(dist)

    n = len(frames)
    det_rate  = det    / max(1, n)
    pnp_rate  = pnp_ok / max(1, n)
    mean_dist = float(np.mean(dists))  if dists else float('nan')
    dist_err  = float(abs(mean_dist - TARGET_DIST_M)) if dists else float('nan')
    acc_rate  = sum(1 for d in dists if abs(d - TARGET_DIST_M) <= DIST_WINDOW_M) / max(1, n)
    return {
        'det_rate':  det_rate,
        'pnp_rate':  pnp_rate,
        'mean_dist': mean_dist,
        'dist_err':  dist_err,
        'acc_rate':  acc_rate,     # frames with pnp dist within ±DIST_WINDOW of target
        'n_det':     det,
        'n_pnp':     pnp_ok,
        'n_frames':  n,
    }


# ── Reporting & plots ─────────────────────────────────────────────────────────

def _score(r):
    """Primary sort key: 0.7 × det_rate + 0.3 × acc_rate (higher is better)."""
    return 0.7 * r['det_rate'] + 0.3 * r['acc_rate']


def _write_csv(rows, out_dir):
    path = os.path.join(out_dir, "results.csv")
    with open(path, 'w', encoding='utf-8') as f:
        f.write("sharpen_k,sharpen_alpha,gauss_k,conf_thresh,orange_frac,"
                "det_rate,pnp_rate,mean_dist,dist_err,acc_rate,n_det,n_pnp,n_frames\n")
        for combo, res in rows:
            sk, sa, gk, ct, of = combo
            f.write(f"{sk},{sa},{gk},{ct},{of},"
                    f"{res['det_rate']:.4f},{res['pnp_rate']:.4f},"
                    f"{res['mean_dist']:.2f},{res['dist_err']:.2f},"
                    f"{res['acc_rate']:.4f},"
                    f"{res['n_det']},{res['n_pnp']},{res['n_frames']}\n")
    print(f"Results → {path}", flush=True)


def _plot_heatmap(rows, out_dir):
    # Fix: best gauss_k and best conf and best orange_frac — show k × alpha grid
    # Filter to gauss_k=0 (usually better), best conf overall, best orange_frac overall
    best_combo, best_res = max(rows, key=lambda x: _score(x[1]))
    best_gk = best_combo[2]
    best_ct = best_combo[3]
    best_of = best_combo[4]

    k_vals     = sorted({c[0] for c in [r[0] for r in rows] if c[0] > 0})
    alpha_vals = sorted({c[1] for c in [r[0] for r in rows]})

    # Build grid: rows=sharpen_alpha (y), cols=sharpen_k (x)
    grid = np.full((len(alpha_vals), len(k_vals)), np.nan)
    for combo, res in rows:
        sk, sa, gk, ct, of = combo
        if sk == 0 or gk != best_gk or ct != best_ct or of != best_of:
            continue
        if sk in k_vals and sa in alpha_vals:
            xi = k_vals.index(sk)
            yi = alpha_vals.index(sa)
            cur = grid[yi, xi]
            grid[yi, xi] = res['det_rate'] if np.isnan(cur) else max(cur, res['det_rate'])

    # Also add k=0 row as a reference column or annotation
    k0_rates = []
    for combo, res in rows:
        if combo[0] == 0 and combo[2] == best_gk and combo[3] == best_ct and combo[4] == best_of:
            k0_rates.append(res['det_rate'])
    k0_rate = float(np.mean(k0_rates)) if k0_rates else float('nan')

    fig, ax = plt.subplots(figsize=(max(6, len(k_vals) * 1.5), max(4, len(alpha_vals) * 1.2)))
    im = ax.imshow(grid, aspect='auto', vmin=0, vmax=1,
                   cmap='RdYlGn', origin='lower')
    ax.set_xticks(range(len(k_vals)));  ax.set_xticklabels(k_vals)
    ax.set_yticks(range(len(alpha_vals))); ax.set_yticklabels(alpha_vals)
    ax.set_xlabel("sharpen_k")
    ax.set_ylabel("sharpen_alpha")
    ax.set_title(f"Detection rate — gauss_k={best_gk}  conf={best_ct}  "
                 f"orange_frac={best_of}\n(k=0 baseline: {k0_rate:.0%})")
    for yi in range(len(alpha_vals)):
        for xi in range(len(k_vals)):
            v = grid[yi, xi]
            if not np.isnan(v):
                ax.text(xi, yi, f"{v:.0%}", ha='center', va='center',
                        fontsize=9, color='black' if v > 0.5 else 'white')
    plt.colorbar(im, ax=ax, label="detection rate")
    plt.tight_layout()
    path = os.path.join(out_dir, "heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Heatmap → {path}", flush=True)


def _plot_conf_sweep(rows, out_dir):
    """For the top-3 preprocessing combos (by best det_rate across all conf), plot det_rate vs conf."""
    # Group by preprocessing combo (sk, sa, gk, of)
    from collections import defaultdict
    groups = defaultdict(list)
    for combo, res in rows:
        sk, sa, gk, ct, of = combo
        groups[(sk, sa, gk, of)].append((ct, res['det_rate']))

    # Pick top-3 preprocessing combos by their peak det_rate
    top3 = sorted(groups.items(), key=lambda x: max(v for _, v in x[1]), reverse=True)[:3]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    for (pre_combo, points), color in zip(top3, colors):
        sk, sa, gk, of = pre_combo
        pts_sorted = sorted(points)
        conf_vals  = [p[0] for p in pts_sorted]
        det_rates  = [p[1] for p in pts_sorted]
        label = f"k={sk} a={sa} gauss={gk} of={of}"
        ax.plot(conf_vals, [r * 100 for r in det_rates], 'o-', color=color, label=label, lw=1.5)

    ax.set_xlabel("conf_thresh")
    ax.set_ylabel("detection rate (%)")
    ax.set_title(f"Detection rate vs confidence — top-3 preprocessing combos")
    ax.axhline(100, color='red', lw=0.8, ls='--', alpha=0.5, label='100% target')
    ax.set_ylim(0, 110)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "conf_sweep.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Conf sweep → {path}", flush=True)


def _write_best_params(best_combo, out_dir):
    sk, sa, gk, ct, of = best_combo
    path = os.path.join(out_dir, "best_params.yaml")
    with open(path, 'w', encoding='utf-8') as f:
        f.write("# ── Optimal preprocessing params (from vision_param_opt.py) ──\n")
        f.write(f"preproc_sharpen_k:     {sk}\n")
        f.write(f"preproc_sharpen_alpha: {sa}\n")
        f.write(f"preproc_gauss_k:       {gk}\n")
        f.write(f"# YOLO confidence threshold (update conf= in _orange_mask call)\n")
        f.write(f"# optimal conf_thresh: {ct}\n")
        f.write(f"# optimal orange_frac_min: {of}\n")
    print(f"Best params → {path}", flush=True)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    param  = load_params()
    cam_K  = np.array([
        [param.get('cam_fx', 320.), 0.,                    param.get('cam_cx', 320.)],
        [0.,                        param.get('cam_fy', 320.), param.get('cam_cy', 180.)],
        [0.,                        0.,                    1.],
    ], dtype=np.float64)
    suppress_blue = bool(param.get('suppress_blue', True))
    blue_lo = np.array([
        int(param.get('suppress_blue_h_lo', 100)),
        int(param.get('suppress_blue_s_lo',  80)),
        int(param.get('suppress_blue_v_lo',  50)),
    ], dtype=np.uint8)
    blue_hi = np.array([int(param.get('suppress_blue_h_hi', 140)), 255, 255], dtype=np.uint8)
    gate_w_def = float(param.get('gate_width_default',  1.5))
    gate_h_def = float(param.get('gate_height_default', 1.5))

    out_dir = os.path.join("logs", f"param_opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)

    # ── Connect MAVLink for gate track data ──────────────────────────────────
    print("Connecting MAVLink…", flush=True)
    conn = mavutil.mavlink_connection(f"udpin:{SIM_IP}:{SIM_PORT}")
    conn.wait_heartbeat()
    print(f"Connected (sys {conn.target_system})", flush=True)

    shared = {}
    rx = MAVLinkRX.create_mavlink_rx(conn, shared, logger=None)

    print("\nPress 's' to start frame capture (ensure sim is loaded, drone stationary)…",
          flush=True)
    while True:
        if msvcrt.kbhit() and msvcrt.getwch().lower() == 's':
            break
        time.sleep(0.05)

    # Wait up to 5 s for gate track data
    t_wait = time.time()
    gate_w, gate_h = gate_w_def, gate_h_def
    while time.time() - t_wait < 5.0:
        gates = shared.get('track_gates_ned', {})
        if 0 in gates:
            gate_w = gates[0]['width']
            gate_h = gates[0]['height']
            print(f"Gate 0: w={gate_w:.2f}m h={gate_h:.2f}m  NED={gates[0]['ned']}",
                  flush=True)
            break
        time.sleep(0.1)
    else:
        print(f"No track data — using defaults {gate_w:.2f}×{gate_h:.2f}m", flush=True)

    # ── CAPTURE ──────────────────────────────────────────────────────────────
    print(f"\n[CAPTURE] {CAPTURE_SEC}s, max {MAX_FRAMES} frames…", flush=True)
    frames = _capture_frames(MAX_FRAMES, CAPTURE_SEC)
    print(f"Captured {len(frames)} frames.", flush=True)

    if len(frames) < 5:
        print("Too few frames — check sim camera UDP on port 5600.", flush=True)
        return

    # ── Load YOLO ─────────────────────────────────────────────────────────────
    print("\nLoading YOLO…", flush=True)
    model = _YOLO("YOLO/best.pt")
    model.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
    print("YOLO ready.", flush=True)

    # ── Build parameter grid ──────────────────────────────────────────────────
    combos = []
    for sk in SHARPEN_K_LIST:
        for sa in (SHARPEN_ALPHA_LIST if sk > 0 else [SHARPEN_ALPHA_LIST[0]]):
            for gk in GAUSS_K_LIST:
                for ct in CONF_THRESH_LIST:
                    for of in ORANGE_FRAC_LIST:
                        combos.append((sk, sa, gk, ct, of))

    # Deduplicate (k=0 rows with different alpha are identical)
    seen  = set()
    uniq  = []
    for c in combos:
        key = (c[0] if c[0] > 0 else (0, 0), c[2], c[3], c[4])
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    combos = uniq

    total = len(combos)
    print(f"\n[SWEEP] {total} combos × {len(frames)} frames…", flush=True)

    rows     = []
    t_sweep  = time.time()
    for i, combo in enumerate(combos):
        sk, sa, gk, ct, of = combo
        res = _run_combo(frames, model, cam_K, gate_w, gate_h,
                         suppress_blue, blue_lo, blue_hi,
                         sk, sa, gk, ct, of)
        rows.append((combo, res))
        if (i + 1) % max(1, total // 10) == 0 or i == total - 1:
            elapsed  = time.time() - t_sweep
            eta      = elapsed / (i + 1) * (total - i - 1)
            print(f"  {i+1}/{total}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s  "
                  f"last: k={sk} a={sa} gk={gk} ct={ct} of={of} → "
                  f"det={res['det_rate']:.0%}  dist={res['mean_dist']:.1f}m",
                  flush=True)

    # ── Sort and report ───────────────────────────────────────────────────────
    rows_sorted = sorted(rows, key=lambda x: _score(x[1]), reverse=True)

    print(f"\n{'─'*80}", flush=True)
    print(f"{'RANK':>4}  {'sk':>3} {'sa':>5} {'gk':>3} {'ct':>5} {'of':>6}  "
          f"{'det%':>6}  {'pnp%':>6}  {'dist(m)':>8}  {'err(m)':>7}  {'acc%':>6}",
          flush=True)
    print(f"{'─'*80}", flush=True)
    for rank, (combo, res) in enumerate(rows_sorted[:20], 1):
        sk, sa, gk, ct, of = combo
        print(f"{rank:>4}  {sk:>3} {sa:>5} {gk:>3} {ct:>5} {of:>6}  "
              f"{res['det_rate']:>6.0%}  {res['pnp_rate']:>6.0%}  "
              f"{res['mean_dist']:>8.1f}  {res['dist_err']:>7.2f}  "
              f"{res['acc_rate']:>6.0%}",
              flush=True)
    print(f"{'─'*80}", flush=True)

    best_combo, best_res = rows_sorted[0]
    print(f"\nBEST COMBO:", flush=True)
    sk, sa, gk, ct, of = best_combo
    print(f"  preproc_sharpen_k:     {sk}", flush=True)
    print(f"  preproc_sharpen_alpha: {sa}", flush=True)
    print(f"  preproc_gauss_k:       {gk}", flush=True)
    print(f"  conf_thresh:           {ct}", flush=True)
    print(f"  orange_frac_min:       {of}", flush=True)
    print(f"  → det_rate={best_res['det_rate']:.0%}  pnp_rate={best_res['pnp_rate']:.0%}  "
          f"mean_dist={best_res['mean_dist']:.1f}m  dist_err={best_res['dist_err']:.2f}m",
          flush=True)

    # ── Plots + CSV ───────────────────────────────────────────────────────────
    _write_csv(rows_sorted, out_dir)
    _plot_heatmap(rows_sorted, out_dir)
    _plot_conf_sweep(rows_sorted, out_dir)
    _write_best_params(best_combo, out_dir)

    print(f"\nAll outputs in: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
