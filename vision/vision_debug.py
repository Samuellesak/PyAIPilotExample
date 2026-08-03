"""
vision_debug.py
===============
One-shot visual debugger for the YOLO detection pipeline.

Captures N_FRAMES frames from the sim camera, then for each frame saves:
  debug_raw_NNN.png   — what the camera sees (no processing)
  debug_yolo_NNN.png  — YOLO on raw frame with boxes + keypoints annotated

Per-frame console report:
  - Orange pixel count (tight mask) and % of image
  - All YOLO box confidences
  - Best box (highest conf): centre, size, orange_frac in bbox, keypoints + any_zero flag

Usage (from the repo root): python -m vision.vision_debug
  Press 's' once sim is loaded and drone is stationary facing the gate.
"""

import os
import sys
import time
import struct
import socket
import msvcrt
from datetime import datetime

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import cv2
import numpy as np
from ultralytics import YOLO as _YOLO

from flight_model.dyn import load_params

CAM_IP   = "0.0.0.0"
CAM_PORT = 5600
N_FRAMES = 5      # number of frames to capture and analyse
YOLO_CONF = 0.01  # very low — we want to see ALL boxes including weak ones


def _capture_n_frames(n):
    header_format = "<IHHIIQ"
    header_sz     = struct.calcsize(header_format)
    frames_raw    = {}
    out           = []

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(1.0)
    sock.bind((CAM_IP, CAM_PORT))
    print(f"  Listening on :{CAM_PORT} for {n} frames…", flush=True)

    while len(out) < n:
        try:
            packet, _ = sock.recvfrom(65536)
        except socket.timeout:
            print("  (timeout — no camera packets yet)", flush=True)
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
                    out.append(img)
                    print(f"  Frame {len(out)}/{n} — shape={img.shape}", flush=True)
            del frames_raw[frame_id]

        if len(frames_raw) > 30:
            del frames_raw[min(frames_raw)]

    sock.close()
    return out


def _draw_boxes(img, results, label_prefix=""):
    """Draw all YOLO boxes on a copy of img regardless of conf."""
    vis = img.copy()
    r   = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        cv2.putText(vis, f"{label_prefix} NO BOXES", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return vis, 0

    boxes = r.boxes.xywh.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    kpts  = r.keypoints.xy.cpu().numpy() if r.keypoints is not None else None

    for i, (box, conf) in enumerate(zip(boxes, confs)):
        cx, cy, w, h = box
        x1 = int(cx - w/2); y1 = int(cy - h/2)
        x2 = int(cx + w/2); y2 = int(cy + h/2)
        color = (0, 255, 0) if conf >= 0.3 else (0, 165, 255)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, f"{conf:.2f}", (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        if kpts is not None and i < len(kpts):
            for pt in kpts[i]:
                if pt[0] > 1 or pt[1] > 1:
                    cv2.circle(vis, (int(pt[0]), int(pt[1])), 5, (255, 0, 0), -1)

    cv2.putText(vis, f"{label_prefix} {len(boxes)} box(es)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return vis, len(boxes)


def main():
    param = load_params()
    suppress_blue = bool(param.get('suppress_blue', True))
    blue_lo = np.array([
        int(param.get('suppress_blue_h_lo', 100)),
        int(param.get('suppress_blue_s_lo',  80)),
        int(param.get('suppress_blue_v_lo',  50)),
    ], dtype=np.uint8)
    blue_hi = np.array([int(param.get('suppress_blue_h_hi', 140)), 255, 255], dtype=np.uint8)

    # HSV orange range (same as vision_rx.py) — S lowered to 80 to catch cyan-mixed gate pixels
    hsv_lo = np.array([ 5,  80,  80], dtype=np.uint8)
    hsv_hi = np.array([25, 255, 255], dtype=np.uint8)

    out_dir = os.path.join("logs", f"debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output → {out_dir}", flush=True)

    print("\nPress 's' to start capture…", flush=True)
    while True:
        if msvcrt.kbhit() and msvcrt.getwch().lower() == 's':
            break
        time.sleep(0.05)

    frames = _capture_n_frames(N_FRAMES)
    if not frames:
        print("No frames received — check sim camera UDP.", flush=True)
        return

    print("\nLoading YOLO…", flush=True)
    model = _YOLO("YOLO/best.pt")
    model.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
    print("YOLO ready.\n", flush=True)

    print(f"{'─'*70}", flush=True)
    print(f"{'Frame':>6}  {'Orange px':>10}  {'Orange%':>8}  {'YOLO boxes':>12}", flush=True)
    print(f"{'─'*70}", flush=True)

    for idx, img in enumerate(frames):
        h_img, w_img = img.shape[:2]
        tag = f"{idx:03d}"

        # ── Raw frame ───────────────────────────────────────────────────────
        cv2.imwrite(os.path.join(out_dir, f"debug_raw_{tag}.png"), img)

        # ── Blue/cyan suppression: replace matched pixels with grey (no HSV round-trip) ──
        if suppress_blue:
            hsv_raw  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            blue_px  = cv2.inRange(hsv_raw, blue_lo, blue_hi)
            grey     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            grey_3ch = cv2.merge([grey, grey, grey])
            img_proc = np.where(blue_px[:, :, np.newaxis] > 0,
                                grey_3ch, img).astype(img.dtype)
        else:
            img_proc = img

        # ── Orange mask (tight, for validation only) ─────────────────────────
        hsv_proc   = cv2.cvtColor(img_proc, cv2.COLOR_BGR2HSV)
        tight_mask = cv2.inRange(hsv_proc, hsv_lo, hsv_hi)
        k_close    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        tight_mask = cv2.morphologyEx(tight_mask, cv2.MORPH_CLOSE, k_close)

        orange_px  = int(np.count_nonzero(tight_mask))
        orange_pct = 100.0 * orange_px / max(1, h_img * w_img)

        # ── YOLO on blue-suppressed frame ─────────────────────────────────────
        res_raw        = model.predict(img_proc, verbose=False, conf=YOLO_CONF)
        vis_raw, n_raw = _draw_boxes(img_proc, res_raw, "")
        cv2.imwrite(os.path.join(out_dir, f"debug_yolo_{tag}.png"), vis_raw)

        # ── Per-frame report ─────────────────────────────────────────────────
        print(f"{idx:>6}  {orange_px:>10}  {orange_pct:>7.2f}%  {n_raw:>8} boxes",
              flush=True)

        r2 = res_raw[0]
        if r2.boxes is not None and len(r2.boxes) > 0:
            boxes2 = r2.boxes.xywh.cpu().numpy()
            confs2 = r2.boxes.conf.cpu().numpy()
            kpts2  = r2.keypoints.xy.cpu().numpy() if r2.keypoints is not None else None

            print(f"         all confs: {[f'{c:.3f}' for c in sorted(confs2, reverse=True)]}",
                  flush=True)

            # Mirror live pipeline: filter by conf + orange_frac (padded) + no-zero-kpts,
            # then pick largest area (= nearest gate).
            # Reports per-box rejection reason so we know which condition fails.
            _PAD = 15
            valid_idx = []
            for _i in range(len(boxes2)):
                _c = confs2[_i]
                if _c < 0.03:
                    print(f"         box{_i} REJECT  conf={_c:.3f} < 0.03", flush=True)
                    continue
                if kpts2 is None or kpts2[_i].shape != (4, 2):
                    print(f"         box{_i} REJECT  bad keypoints shape", flush=True)
                    continue
                _kp = kpts2[_i]
                if np.any(np.all(_kp < 2.0, axis=1)):
                    bad = [j for j in range(len(_kp)) if np.all(_kp[j] < 2.0)]
                    print(f"         box{_i} REJECT  zero keypoint(s) at indices {bad}  "
                          f"kpts={[(f'{p[0]:.1f}',f'{p[1]:.1f}') for p in _kp]}", flush=True)
                    continue
                _bx1 = max(0, int(boxes2[_i][0] - boxes2[_i][2] / 2) - _PAD)
                _by1 = max(0, int(boxes2[_i][1] - boxes2[_i][3] / 2) - _PAD)
                _bx2 = min(w_img, int(boxes2[_i][0] + boxes2[_i][2] / 2) + _PAD)
                _by2 = min(h_img, int(boxes2[_i][1] + boxes2[_i][3] / 2) + _PAD)
                _ba  = max(1, (_bx2 - _bx1) * (_by2 - _by1))
                _of  = np.count_nonzero(tight_mask[_by1:_by2, _bx1:_bx2]) / _ba
                if _of < 0.001:
                    print(f"         box{_i} REJECT  orange_frac={_of:.4f} ({_of*100:.2f}%) < 0.1%  "
                          f"padded_bbox=({_bx1},{_by1})-({_bx2},{_by2}) area={_ba}px²", flush=True)
                    continue
                print(f"         box{_i} PASS    conf={_c:.3f}  orange_frac={_of:.4f} "
                      f"(padded +{_PAD}px)", flush=True)
                valid_idx.append(_i)

            if valid_idx:
                areas2 = boxes2[:, 2] * boxes2[:, 3]
                best2  = valid_idx[int(np.argmax(areas2[valid_idx]))]
                bx, by, bw, bh = boxes2[best2]
                print(f"         best (largest area, {len(valid_idx)} valid): "
                      f"cx={bx:.0f} cy={by:.0f} "
                      f"w={bw:.1f} h={bh:.1f}  area={bw*bh:.0f}px²  conf={confs2[best2]:.3f}",
                      flush=True)
                kp       = kpts2[best2]
                any_zero = np.any(np.all(kp < 2.0, axis=1))
                print(f"         keypoints: {[(f'{p[0]:.1f}',f'{p[1]:.1f}') for p in kp]}"
                      f"  any_zero={any_zero}", flush=True)
            else:
                print(f"         no box passed all filters", flush=True)

        # Orange pixel locations (centroid) for debugging
        M = cv2.moments(tight_mask)
        if M['m00'] > 0:
            cx_orange = int(M['m10'] / M['m00'])
            cy_orange = int(M['m01'] / M['m00'])
            print(f"         Orange centroid: ({cx_orange}, {cy_orange})  "
                  f"image size: {w_img}×{h_img}", flush=True)
        else:
            print(f"         No orange pixels found. Image: {w_img}×{h_img}", flush=True)

    print(f"{'─'*70}", flush=True)
    print(f"\nDiagnosis guide:", flush=True)
    print(f"  Orange%=0 & no boxes → gate not orange or not in frame", flush=True)
    print(f"  Orange%>0 & no boxes → YOLO model issue or gate too small / occluded",
          flush=True)
    print(f"  boxes>0 & any_zero=True → YOLO keypoints at origin → PnP will fail",
          flush=True)
    print(f"  boxes>0 & orange_frac<0.005 → likely false positive (no orange in bbox)",
          flush=True)
    print(f"  boxes>0 & orange_frac≥0.005 & any_zero=False → pipeline OK",
          flush=True)
    print(f"\nSaved: debug_raw_NNN.png  debug_yolo_NNN.png  →  {out_dir}", flush=True)


if __name__ == "__main__":
    main()
